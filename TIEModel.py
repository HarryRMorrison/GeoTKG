import torch
import torch.nn as nn
from transformers import AutoModel
from seqeval.metrics import f1_score as seqeval_f1, classification_report as seqeval_cr
from sklearn.metrics import f1_score as sk_f1, classification_report as sk_cr
from globals import EVENT_BI, TIMEX_BI, LABEL2ID_EVNER, LABEL2ID_EE, ID2LABEL_EVNER, ID2LABEL_EE

# --------------------------------
# Max Span Pooling
# --------------------------------
def MaxSpanPool(H, starts, ends):
    B, L, d = H.shape
    K = starts.size(1) # number of spans

    # Clamp indices to safe range
    starts = starts.clamp(min=0, max=L-1)
    # ensure end >= start+1 to avoid empty spans; clamp to L
    ends = torch.maximum(ends, starts + 1).clamp(min=1, max=L)

    # Build [B,K,L] mask where True = inside span
    rng = torch.arange(L, device=H.device).view(1, 1, L)  # [1,1,L]
    span_mask = (rng >= starts.unsqueeze(-1)) & (rng < ends.unsqueeze(-1))  # [B,K,L]

    # Expand H to [B,K,L,d]
    H_exp = H.unsqueeze(1).expand(-1, K, -1, -1)

    # Mask out tokens outside spans with -inf so max ignores them
    neg_inf = torch.finfo(H.dtype).min
    masked = H_exp.masked_fill(~span_mask.unsqueeze(-1), neg_inf)  # [B,K,L,d]

    # Max over sequence length
    H_span = masked.amax(dim=2)  # [B,K,d]

    # Replace -inf (could happen if a row was fully masked) with zeros,
    # and zero-out padded spans if a mask was provided.
    is_all_masked = ~span_mask.any(dim=2)  # [B,K]
    H_span = torch.where(is_all_masked.unsqueeze(-1), torch.zeros_like(H_span), H_span)

    return H_span

# --------------------------------
# Event Time NER Head
# --------------------------------
class NERHead(nn.Module):
    """
    Minimal NER head: dropout + Linear -> logits.
    Uses CrossEntropyLoss with ignore_index=-100 (for subword padding etc).
    """
    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, H):
        """
        H: [B, L, d] encoder token reps
        labels: [B, L] (int), -100 where you want to ignore (e.g., non-first subwords)
        attention_mask: [B, L] (optional; not required for loss)
        returns dict with logits and optional loss
        """
        logits = self.classifier(self.drop(H))  # [B, L, C]
        out = {"logits": logits}
        return out

    @torch.no_grad()    
    def decode(logits):
        # Need to update to take a batch instead
        decoded = logits.argmax(-1)
        bi_map = {**EVENT_BI, **TIMEX_BI}
        Bs = bi_map.keys()
        ti_starts, ti_ends = [], []
        ev_starts, ev_ends = [], []
        for example in decoded:
            time_starts, time_ends = [], []
            event_starts, event_ends = [], []
            i = 0
            while i < example.shape[0]:
                if example[i].item() in Bs:
                    start = i
                    ent_type = example[i].item()
                    is_timex = True if ent_type in TIMEX_BI else False
                    i += 1
                    while i < len(example) and example[i] == bi_map[ent_type]:
                        i += 1
                    if is_timex: 
                        time_starts.append(start)
                        time_ends.append(i)
                    else: 
                        event_starts.append(start)
                        event_ends.append(i)
                else:
                    i += 1
            ti_starts.append(torch.tensor(time_starts, dtype=torch.long))
            ti_ends.append(torch.tensor(time_ends, dtype=torch.long))
            ev_starts.append(torch.tensor(event_starts, dtype=torch.long))
            ev_ends.append(torch.tensor(event_ends, dtype=torch.long))
        ev_starts = torch.nn.utils.rnn.pad_sequence(ev_starts, batch_first=True, padding_value=-1)
        ev_ends   = torch.nn.utils.rnn.pad_sequence(ev_ends,   batch_first=True, padding_value=-1)
        ti_starts = torch.nn.utils.rnn.pad_sequence(ti_starts, batch_first=True, padding_value=-1)
        ti_ends   = torch.nn.utils.rnn.pad_sequence(ti_ends,   batch_first=True, padding_value=-1)
        return ev_starts, ev_ends, ti_starts, ti_ends

# ----------------------------
# Cross-attention
# ----------------------------
class CrossAttention(nn.Module):
    def __init__(self, d, heads=6, dropout=0.0):  # set dropout=0.0 for clean probs
        super().__init__()
        self.mha = nn.MultiheadAttention(d, heads, batch_first=True, dropout=dropout)
        self.none_token = nn.Parameter(torch.zeros(1, 1, d))  # learned NONE
        self.ln = nn.LayerNorm(d)

    def forward(self, hE, hT, attn_bias=None):
        """
        hE: [B, Ne, d] events -> queries
        hT: [B, Nt, d] times  -> keys/values
        key_padding_mask_T: [B, Nt] (True = pad) for real time slots
        attn_bias: optional float mask to add to logits, shape:
                   [B, Ne, Nt+1] or [B*heads, Ne, Nt+1] (PyTorch 2.x)
        """
        B = hE.size(0)
        # append NONE
        none = self.none_token.expand(B, 1, -1)          # [B,1,d]
        T_aug = torch.cat([hT, none], dim=1)             # [B, Nt+1, d]

        # attn_mask: additive bias (float) or bool; if float, negative values downweight
        am = None
        if attn_bias is not None:        # [B,Ne,Nt]
            zero = attn_bias.new_zeros(B, hE.size(1), 1)
            attn_bias = torch.cat([attn_bias, zero], -1)            # [B,Ne,Nt+1]
            H = self.mha.num_heads
            am = attn_bias.repeat_interleave(H, dim=0).to(hE.dtype) # [B*H,Ne,Nt+1]

        out, attn = self.mha(hE, T_aug, T_aug, key_padding_mask=None, attn_mask=am, need_weights=True, average_attn_weights=True)
        out = self.ln(out + hE)
        # out:  [B, Ne, d]  (refined events)
        # attn:[B, Ne, Nt+1] (avg across heads); with dropout=0, this sums to 1 → pointer probs

        ptr_probs = attn                      # treat as p(t|e)
        ptr_idx   = ptr_probs.argmax(-1)      # [B, Ne]
        h_time_exp = ptr_probs @ T_aug        # [B, Ne, d] expected time embedding

        return {"ptr_probs": ptr_probs, "ptr_idx": ptr_idx,
                "h_time_exp": h_time_exp, "hE_refined": out}

# ----------------------------
# Event-Event Temporal Relation Head
# ----------------------------
class EEHead(nn.Module):
    def __init__(self, d, n_labels, hidden=256*2, dropout=0.1):
        super().__init__()
        in_dim = 8*d
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), 
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_labels)
        )

    def forward(self, hE, hT_exp, pairs):
        """
        hE:      [B, Ne, d]     (event embeddings)
        hT_exp:  [B, Ne, d]     (expected time embeddings per event)
        pairs:   [B, M, 2]      (indices into event set)
        Returns:
          logits: [B, M, n_labels]
        """
        B, Ne, d = hE.shape
        M = pairs.size(1)
        e1 = pairs[:,:,0]  # [B,M]
        e2 = pairs[:,:,1]  # [B,M]
        # Gather
        he1 = torch.gather(hE, 1, e1.unsqueeze(-1).expand(-1,-1,d))      # [B,M,d]
        he2 = torch.gather(hE, 1, e2.unsqueeze(-1).expand(-1,-1,d))      # [B,M,d]
        ht1 = torch.gather(hT_exp, 1, e1.unsqueeze(-1).expand(-1,-1,d))  # [B,M,d]
        ht2 = torch.gather(hT_exp, 1, e2.unsqueeze(-1).expand(-1,-1,d))  # [B,M,d]

        x = torch.cat([he1, he2, ht1, ht2, he1*he2, ht1*ht2, (he1- he2), (ht1- ht2)], dim=-1) # [B,M, 8d]
        logits = self.mlp(x)                                             # [B,M,C]
        return logits

class TIEModel(nn.Module):
    def __init__(self, base="roberta-base",
                 num_ner=len(LABEL2ID_EVNER),
                 ee_labels=len(LABEL2ID_EE),
                 heads=6,
                 et_feat_dim=0,       # φ(e,t) size
                 ee_feat_dim=0):      # φ(e1,e2) size
        super().__init__()
        self.enc = AutoModel.from_pretrained(base)
        d = self.enc.config.hidden_size
        self.ner = NERHead(d, num_ner)
        self.span_pool = MaxSpanPool
        self.ca = CrossAttention(d=d, heads=heads)
        self.ee = EEHead(d, n_labels=ee_labels)
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=-100)  # for NER and EE

    def save(self, save_path):
        torch.save({'model_state_dict':self.state_dict()}, save_path)

    def forward(self, input_ids, attention_mask,
                # --- Event and Time Locations
                ev_starts=None, ev_ends=None, ev_mask=None, ti_starts=None, ti_ends=None, ti_mask=None, e_sent_ids=None, t_sent_ids=None,
                # --- Gold Labels
                ner_gold_labels=None, ev_ti_gold=None, ee_rel_gold=None, ee_mask=None):

        out = {}
        H = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # 1) NER head
        ner_out = self.ner(H)
        out["ner_logits"] = ner_out["logits"]
        if ner_gold_labels is not None:
            ner_loss = self.loss_ce(ner_out["logits"].view(-1, ner_out["logits"].size(-1)), 
                                    ner_gold_labels.view(-1))
            out["ner_loss"] = ner_loss
        else:
            ner_loss = 0

        # 2) Span Max Pooling
        if None in [ev_starts, ev_ends, ti_starts, ti_ends]:
            ev_starts, ev_ends, ti_starts, ti_ends = NERHead.decode(ner_out["logits"])

        hE = self.span_pool(H, ev_starts, ev_ends)  # [B, K, d] for events
        hT = self.span_pool(H, ti_starts, ti_ends)  # [B, K, d] for times

        # 3) Cross-Attention (events + times)
        ca_out = self.ca(hE, hT)
        ptr_probs = ca_out["ptr_probs"]  # [B, Ne, Nt+1]
        hE_ref = ca_out["hE_refined"]
        hT_e = ca_out["h_time_exp"]
        out["ptr_probs"] = ptr_probs
        if ev_ti_gold is not None:
            ptr_loss = self.loss_ce(ptr_probs.view(-1, ptr_probs.size(-1)), 
                                    ev_ti_gold.view(-1))
            out["ptr_loss"] = ptr_loss
        else:
            ptr_loss = 0

        # 5) Event-Event Temporal Relation Head
        if ee_rel_gold is not None:
            ee_pairs  = ee_rel_gold[:, :, [0, 2]]           # [B, M, 2]
            ee_logits = self.ee(hE, hT_e, ee_pairs)         # [B, M, C] (optionally use hE — instead of hE_ref)
            out["ee_logits"] = ee_logits
            ee_labels = ee_rel_gold[:, :, 1].clone()        # [B, M]
            ee_labels[~ee_mask] = -100                      # <-- mask padded pairs
            ee_loss = self.loss_ce(ee_logits.view(-1, ee_logits.size(-1)),
                                    ee_labels.view(-1))
            out["ee_loss"] = ee_loss
        else:
            B, Ne, d = hE.shape
            row_idx, col_idx = torch.triu_indices(Ne, Ne, offset=1)
            try:
                batch_idx = torch.arange(B).unsqueeze(1).expand(B, row_idx.size(1))
            except:
                batch_idx = torch.arange(B).unsqueeze(1).expand(B, row_idx.size(0))
            ee_pairs = torch.stack([row_idx.expand(B, -1), col_idx.expand(B, -1)], dim=-1)
            ee_logits = self.ee(hE_ref, hT_e, ee_pairs)   
            out["ee_logits"] = ee_logits
            ee_loss = 0

        out["loss"] = (ner_loss + ptr_loss + ee_loss)/3

        return out
    
    def evaluate_dataloader(self, dev_loader, id2label_ner, id2label_ee, *, ee_average="micro"):
        self.eval()
        device = next(self.parameters()).device

        # NER (seqeval expects list[list[str]])
        ner_true_seqs, ner_pred_seqs = [], []
        # Pointer metrics
        et_correct, et_total = 0, 0
        # EE F1 collections
        ee_true_all, ee_pred_all = [], []
        # Eval Loss
        loss = []
        ner_loss, ptr_loss, ee_loss = 0, 0, 0

        with torch.no_grad():
            for batch in dev_loader:
                # move to device
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

                out = self.forward(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    ev_starts=batch["ev_starts"], ev_ends=batch["ev_ends"], ev_mask=batch["ev_mask"], e_sent_ids=batch["e_sent_ids"],
                    ti_starts=batch["ti_starts"], ti_ends=batch["ti_ends"], ti_mask=batch["ti_mask"], t_sent_ids=batch["t_sent_ids"],
                    ner_gold_labels=batch["ner_labels"],
                    ev_ti_gold=batch["ev_ti_gold"],
                    ee_rel_gold=batch["ee_triples"],
                    ee_mask=batch["ee_mask"],
                )

                # -------- NER F1 (seqeval) --------
                ner_logits   = out["ner_logits"]              # [B,L,C]
                ner_pred_ids = ner_logits.argmax(-1)          # [B,L]
                ner_gold_ids = batch["ner_labels"]            # [B,L]
                B, L = ner_gold_ids.shape

                for i in range(B):
                    ti = ner_gold_ids[i].tolist()
                    pi = ner_pred_ids[i].tolist()
                    true_seq, pred_seq = [], []
                    for t_id, p_id in zip(ti, pi):
                        if t_id == -100:
                            continue
                        true_seq.append(id2label_ner[t_id])
                        pred_seq.append(id2label_ner[p_id])
                    ner_true_seqs.append(true_seq)
                    ner_pred_seqs.append(pred_seq)

                # -------- Event→Time pointer@1 --------
                ptr_pred = out["ptr_probs"].argmax(-1)       # [B,Ne]
                ev_mask  = batch["ev_mask"]                   # [B,Ne] (bool)
                gold_ptr = batch["ev_ti_gold"]                # [B,Ne] (long)

                valid = ev_mask & (gold_ptr != -100)
                et_correct += (ptr_pred[valid] == gold_ptr[valid]).sum().item()
                et_total   += valid.sum().item()                

                # -------- EE F1 --------
                ee_logits = out["ee_logits"]                  # [B,M,C]
                ee_pred   = ee_logits.argmax(-1)              # [B,M]
                ee_gold   = batch["ee_triples"][:, :, 1]      # [B,M]
                ee_mask   = batch["ee_mask"]                  # [B,M] (bool)

                if ee_mask.any():
                    ee_true_all.extend(ee_gold[ee_mask].tolist())
                    ee_pred_all.extend(ee_pred[ee_mask].tolist())

                loss.append(out["loss"].item())
                ner_loss += out.get("ner_loss", 0).item()
                ptr_loss += out.get("ptr_loss", 0).item()
                ee_loss += out.get("ee_loss", 0).item()

        metrics = {}
        metrics["ner_f1"]  = seqeval_f1(ner_true_seqs, ner_pred_seqs) if ner_true_seqs else 0.0
        print(seqeval_cr(ner_true_seqs, ner_pred_seqs, digits=4))
        metrics["ptr_acc"] = (et_correct / et_total) if et_total > 0 else 0.0
        metrics["ee_f1"] = sk_f1(ee_true_all, ee_pred_all, average=ee_average)
        print(sk_cr(ee_true_all, ee_pred_all, target_names=id2label_ee.values(), digits=4))
        metrics["eval_loss"] = sum(loss) / len(loss)
        metrics["ner_loss"] = ner_loss / len(loss)
        metrics["ptr_loss"] = ptr_loss / len(loss)
        metrics["ee_loss"] = ee_loss / len(loss)
        return metrics
    
    @torch.no_grad()
    def predict(self, batch):
        self.eval()
        out = self.forward(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        preds = {}

        # --- NER decode ---
        ner_ids = out["ner_logits"].argmax(-1).cpu()   # [B,L]
        ner_tags = [
            [ID2LABEL_EVNER[i.item()] for i in row if i.item() != -100]
            for row in ner_ids
        ]
        preds["ner_tags"] = ner_tags

        # --- Event→Time pointer decode ---
        ptr_ids = out["ptr_probs"].argmax(-1).cpu()     # [B,Ne]
        preds["event_time_idx"] = ptr_ids

        # --- Event–Event temporal relation decode ---
        ee_ids = out["ee_logits"].argmax(-1).cpu() # [B,M]
        ee_labels = [
            [ID2LABEL_EE[i.item()] for i in row]
            for row in ee_ids
        ]
        preds["ee_relations"] = ee_labels

        return preds
    
if __name__=="__main__":
    # model = TIEModel().to("cuda")
    # model.save("results/tie_model/")

    load = torch.load("D:\\GeoTKG\\results\\tie_model\\tie_model.pt")
    model = TIEModel()  
    model.load_state_dict(load['model_state_dict'])
    from TIEUtils import TemporalDataset, collator
    from torch.utils.data import DataLoader 
    from transformers import AutoTokenizer
    ex = {"text": [["Eleven", "people", "were", "Eleven", "people", "were", "confirmed", "confirmed", "blast", "blast", "Wednesday", "morning", "Wednesday", "morning", "said", "said"]], 
          "instances": [{"offset": [6, 7], "type": "EVENT", "sent_id": 0, "text": "confirmed", "id": 0}, 
                        {"offset": [8, 9], "type": "EVENT", "sent_id": 0, "text": "blast", "id": 1}, 
                        {"offset": [14, 15], "type": "EVENT", "sent_id": 0, "text": "said", "id": 2}, 
                        {"value": "2006-11-29", "type": "TIME", "offset": [1, 0], "id": 0}, 
                        {"value": "2006-11-22TMO", "type": "DATE", "sent_id": 0, "offset": [10, 12], "text": "Wednesday morning", "id": 1}], 
                        "event_times": [{"event": 1, "time": 1}, 
                                        {"event": 0, "time": "NONE"}, 
                                        {"event": 2, "time": "NONE"}], 
                        "ee_temprels": [{"e1": 0, "e2": 2, "rel": "DURING"}], 
                        "bio_tags": [["O", "O", "O", "O", "O", "O", "B-EVENT", "O", "B-EVENT", "O", "B-DATE", "I-DATE", "O", "O", "B-EVENT", "O"]]}
    TOKENIZER = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
    encod = TOKENIZER(ex["text"][0], is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
    batch = {"input_ids":torch.tensor(encod["input_ids"]), "attention_mask":torch.tensor(encod["attention_mask"])}

    preds = model.predict(batch)
    temprels = preds["ee_relations"]
    ets = preds['event_time_idx']
    ner = preds["ner_tags"]
    print(temprels, ets, ner)
