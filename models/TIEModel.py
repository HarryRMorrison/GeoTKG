import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from seqeval.metrics import f1_score as seqeval_f1, classification_report as seqeval_cr
from sklearn.metrics import f1_score as sk_f1, classification_report as sk_cr
from models.globals import EVENT_BI, TIMEX_BI, LABEL2ID_EVNER, LABEL2ID_EE, ID2LABEL_EVNER, ID2LABEL_EE
TOKENIZER = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)

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
# Classifing Head
# --------------------------------
class ClassifingHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        logits = self.classifier(self.drop(x))  # [B, L, C]
        out = {"logits": logits}
        return out

# ----------------------------
# Cross-attention
# ----------------------------
class CrossAttention(nn.Module):
    def __init__(self, d, heads=6, dropout=0.0, none_token=True):  # set dropout=0.0 for clean probs
        super().__init__()
        self.mha = nn.MultiheadAttention(d, heads, batch_first=True, dropout=dropout)
        self.has_none = none_token
        if none_token:
            self.none_token = nn.Parameter(torch.zeros(1, 1, d))
        self.ln = nn.LayerNorm(d)

    def forward(self, Q, KV, key_padding_mask):
        '''
        hE: [B, Ne, d]
        hT: [B, Nt, d]
        key_padding_mask: [B, Nt]
        attn_mask: [B, Ne]
        '''
        B, Nq, d = Q.shape
        _, Nkv, _ = KV.shape
        if self.has_none:
            none = self.none_token.expand(B, 1, d)                 # [B,1,d]
            KV_aug = torch.cat([KV, none], dim=1)                     # [B, Nt+1, d]

            # Applying None to key_padding_mask
            kp_none = torch.zeros(B, 1, dtype=torch.bool, device=key_padding_mask.device)
            key_padding_mask = torch.cat([~key_padding_mask, kp_none], dim=1)
        else:
            KV_aug = KV
            key_padding_mask = ~key_padding_mask

        # Get output
        out, attn = self.mha(Q, KV_aug, KV_aug, key_padding_mask=key_padding_mask, need_weights=True, average_attn_weights=True)        
        out = self.ln(out + Q)
        # out:  [B, Ne, d]  (refined events)
        # attn:[B, Ne, Nt] (avg across heads); with dropout=0, this sums to 1 → pointer probs

        h_KV_exp = attn @ KV_aug           # [B, Ne, d] expected time embedding treating attn as P(t|e)

        return {"h_KV_exp": h_KV_exp, "Q_refined": out}
        
class TIEModel(nn.Module):
    def __init__(self, base="roberta-base",
                 num_ner=len(LABEL2ID_EVNER),
                 ee_labels=len(LABEL2ID_EE),
                 use_ca=True):
        super().__init__()
        self.enc = AutoModel.from_pretrained(base)
        d = self.enc.config.hidden_size
        self.ner = ClassifingHead(d, num_ner)
        self.span_pool = MaxSpanPool

        if use_ca:
            self.ev_to_ti_ca = CrossAttention(d=d, heads=6, none_token=True)
            self.ti_to_ev_ca = CrossAttention(d=d, heads=6, none_token=False)
            self.et = ClassifingHead(d*4, num_labels=2, is_et_linker=True)
            self.ee = ClassifingHead(d*6, n_labels=ee_labels)
        else:
            self.et = ClassifingHead(d*2, num_labels=2, is_et_linker=True)
            self.ee = ClassifingHead(d*2, n_labels=ee_labels)
        self.sig = nn.Sigmoid()
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=-100)  # for NER and EE
        self.loss_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([22]))
        self.is_using_ca = use_ca

    def save(self, save_path):
        torch.save({'model_state_dict':self.state_dict()}, save_path)

    @torch.no_grad()
    def decode_et_link(self, logits):
        out = self.sig(logits)
        return (out > 0.5).float()

    @torch.no_grad()    
    def decode_ner(logits):
        decoded = logits.argmax(-1)
        bi_map = {**EVENT_BI, **TIMEX_BI}
        Bs = bi_map.keys()
        ti_starts, ti_ends, ti_types = [], [], []
        ev_starts, ev_ends = [], []
        for example in decoded:
            i = 0
            temp_ts, temp_te, temp_es, temp_ee = [], [], [], []
            temp_ti_types = []
            while i < example.shape[0]:
                if example[i].item() in Bs:
                    start = i
                    ent_type = example[i].item()
                    is_timex = True if ent_type in TIMEX_BI else False
                    i += 1
                    while i < len(example) and example[i] == bi_map[ent_type]:
                        i += 1
                    if is_timex: 
                        temp_ts.append(start)
                        temp_te.append(i)
                        temp_ti_types.append(ID2LABEL_EVNER[ent_type][2:])
                    else: 
                        temp_es.append(start)
                        temp_ee.append(i)
                else:
                    i += 1
            ti_starts.append(torch.tensor(temp_ts))
            ti_ends.append(torch.tensor(temp_te))
            ev_starts.append(torch.tensor(temp_es))
            ev_ends.append(torch.tensor(temp_ee))
            ti_types.append(temp_ti_types)
        ev_starts = torch.nn.utils.rnn.pad_sequence(ev_starts, batch_first=True, padding_value=-1)
        ev_ends   = torch.nn.utils.rnn.pad_sequence(ev_ends,   batch_first=True, padding_value=-1)
        ti_starts = torch.nn.utils.rnn.pad_sequence(ti_starts, batch_first=True, padding_value=-1)
        ti_ends   = torch.nn.utils.rnn.pad_sequence(ti_ends,   batch_first=True, padding_value=-1)
        return ev_starts, ev_ends, ti_starts, ti_ends, ti_types

    @torch.no_grad()
    def create_ee_input_pairs(self, hE_tensor, pairs, hT_tensor=None):
        B, Ne, d = hE_tensor.shape
        M = pairs.size(1)
        e1 = pairs[:,:,0]  # [B,M]
        e2 = pairs[:,:,1]  # [B,M]
        # Gather
        he1 = torch.gather(hE_tensor, 1, e1.unsqueeze(-1).expand(-1,-1,d))      # [B,M,d]
        he2 = torch.gather(hE_tensor, 1, e2.unsqueeze(-1).expand(-1,-1,d))      # [B,M,d]
        if hT_tensor is not None:
            ht1 = torch.gather(hT_tensor, 1, e1.unsqueeze(-1).expand(-1,-1,d))      # [B,M,d]
            ht2 = torch.gather(hT_tensor, 1, e2.unsqueeze(-1).expand(-1,-1,d))      # [B,M,d]

            # Build pair vector + logits
            x = torch.cat([he1, he2, ht1, ht2, he1 * he2, (ht1 - ht2).abs()], dim=-1)  # [B,M,6d+2]
        else:
            x = torch.cat([he1, he2])

    @torch.no_grad()
    def create_et_input(self, E, T, uses_ca=True):
        e = E.unsqueeze(2).expand(-1, -1, T.size(1), -1)
        t = T.unsqueeze(1).expand(-1, E.size(1), -1, -1)
        if uses_ca:
            return torch.cat([e, t, e*t, (e - t).abs()], dim=-1) # [B, Ne, Nt, De+Dt]
        else:
            return torch.cat([e, t], dim=-1)

    def forward(self, input_ids, attention_mask,
                # --- Event and Time Locations
                ev_starts=None, ev_ends=None, ev_mask=None, ti_starts=None, ti_ends=None, ti_mask=None, e_sent_ids=None, t_sent_ids=None,
                # --- Gold Labels
                ner_gold_labels=None, ev_ti_gold=None, ee_rel_gold=None, ee_mask=None):

        out = {}

        # 1) Encoding
        H = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # 2) NER head
        ner_out = self.ner(H)
        ner_logits = ner_out["logits"]
        ner_loss = self.loss_ce(ner_logits.view(-1, ner_logits.size(-1)), 
                                ner_gold_labels.view(-1))
        out["ner_loss"] = ner_loss
        out["ner_logits"] = ner_logits

        # 3) Span Max Pooling
        hE = self.span_pool(H, ev_starts, ev_ends)  # [B, K, d] for events
        hT = self.span_pool(H, ti_starts, ti_ends)  # [B, K, d] for times

        if self.is_using_ca:
            # 4a) Cross-Attention (Event->Time)
            e_to_t_ca_out = self.ev_to_ti_ca(hE, hT, key_padding_mask=ti_mask)
            hE_ref = e_to_t_ca_out["Q_refined"]
            hT_e = e_to_t_ca_out["h_KV_exp"]

            # 4b) Cross-Attention (Time->Event)
            t_to_e_ca_out = self.ti_to_ev_ca(hT, hE, key_padding_mask=ev_mask)
            hT_ref = t_to_e_ca_out["Q_refined"]

            et_input = self.create_et_input(hE_ref, hT_ref, uses_ca=True)
            ee_input = self.create_ee_input_pairs(hE_ref, ee_rel_gold[:, :, [0, 2]], hT_e)
        else:
            et_input = self.create_et_input(hE_ref, hT_ref, uses_ca=False)
            ee_input = self.create_ee_input_pairs(hE_ref, ee_rel_gold[:, :, [0, 2]])        

        # 5a) Event Time Linking
        et_out = self.et(et_input)
        et_logits = et_out['logits']
        mask = ev_ti_gold != -100
        et_loss = self.loss_bce(et_logits[mask].view(-1), 
                                ev_ti_gold[mask].view(-1))
        out["et_loss"] = et_loss
        out["et_logits"] = et_logits

        # 5b) Event-Event Temporal Relation Head
        ee_logits = self.ee(ee_input)     # [B, M, C] (optionally use hE — instead of hE_ref)
        ee_labels = ee_rel_gold[:, :, 1].clone()        # [B, M]
        ee_labels[~ee_mask] = -100                      # <-- mask padded pairs
        ee_loss = self.loss_ce(ee_logits.view(-1, ee_logits.size(-1)),
                                ee_labels.view(-1))
        out["ee_loss"] = ee_loss
        out["ee_logits"] = ee_logits

        out["loss"] = ner_loss + et_loss + ee_loss
        return out
    
    def evaluate_dataloader(self, dev_loader, id2label_ner=ID2LABEL_EVNER, id2label_ee=ID2LABEL_EE, average="micro", return_ner_tags = False):
        self.eval()
        device = next(self.parameters()).device
        
        # NER (seqeval expects list[list[str]])
        et_ner, all_ner_pred_ids = {"truth":[], "pred":[]}, []
        # ET Linker metrics
        et_link = {"truth":[], "pred":[]}
        # EE Temp Rels
        ee_temprel = {"truth":[], "pred":[]}
        # Eval Loss
        loss = []
        ev_ner_loss, et_loss, ee_loss = 0, 0, 0
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

                # -------- EV NER F1 (seqeval) --------
                et_ner_logits   = out["ner_logits"]              # [B,L,C]
                et_ner_pred_ids = et_ner_logits.argmax(-1)          # [B,L]
                et_ner_gold_ids = batch["ner_labels"]            # [B,L]
                B, L = et_ner_gold_ids.shape
                all_ner_pred_ids.extend(et_ner_pred_ids)

                for i in range(B):
                    ti = et_ner_gold_ids[i].tolist()
                    pi = et_ner_pred_ids[i].tolist()
                    true_seq, pred_seq = [], []
                    for t_id, p_id in zip(ti, pi):
                        if t_id == -100:
                            continue
                        true_seq.append(ID2LABEL_EVNER[t_id])
                        pred_seq.append(ID2LABEL_EVNER[p_id])
                    et_ner['truth'].append(true_seq)
                    et_ner['pred'].append(pred_seq)

                # -------- Event→Time Link --------
                et_pred = self.et.decode(out["et_logits"])                    # [B,Ne]
                gold_et = batch["ev_ti_gold"]                # [B,Ne] (long)

                valid = gold_et != -100
                et_link['pred'].extend(et_pred[valid].tolist())
                et_link['truth'].extend(gold_et[valid].tolist())

                # -------- EE TempRel --------
                ee_logits = out["ee_logits"]                  # [B,M,C]
                ee_pred   = ee_logits.argmax(-1)              # [B,M]
                ee_gold   = batch["ee_triples"][:, :, 1]      # [B,M]
                ee_mask   = batch["ee_mask"]                  # [B,M] (bool)

                if ee_mask.any():
                    ee_temprel['truth'].extend(ee_gold[ee_mask].tolist())
                    ee_temprel['pred'].extend(ee_pred[ee_mask].tolist())

                loss.append(out["loss"].item())
                ev_ner_loss += out.get("ner_loss", 0).item()
                et_loss += out.get("et_loss", 0).item()
                ee_loss += out.get("ee_loss", 0).item()

        metrics = {}
        metrics["ner_f1"]  = seqeval_f1(et_ner['truth'], et_ner['pred'])
        print(seqeval_cr(et_ner['truth'], et_ner['pred'], digits=4))
        metrics["et_f1"] = sk_f1(et_link['truth'], et_link['pred'], average=average)
        print(sk_cr(et_link['truth'], et_link['pred'], digits=4))
        metrics["ee_f1"] = sk_f1(ee_temprel['truth'], ee_temprel['pred'], average=average)
        print(sk_cr(ee_temprel['truth'], ee_temprel['pred'], target_names=id2label_ee.values(), digits=4))
        metrics["eval_loss"] = sum(loss) / len(loss)
        metrics["ner_loss"] = ev_ner_loss / len(loss)
        metrics["et_loss"] = et_loss / len(loss)
        metrics["ee_loss"] = ee_loss / len(loss)
        if return_ner_tags:
            return metrics, et_ner, et_link, ee_temprel 
        return metrics
    
    def predict(self, text_batch):
        self.eval()
        tokens = TOKENIZER(text_batch, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt")
        model_device = next(self.parameters()).device

        tokens.to(model_device)
        H = self.enc(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask']).last_hidden_state
        H.to(model_device)
        ner_logits = self.ner(H)["logits"]
        ev_starts, ev_ends, ti_starts, ti_ends, ti_types = TIEModel.decode_ner(ner_logits)
        ev_mask = ev_starts != -1
        ti_mask = ti_starts != -1

        ev_starts, ev_ends, ti_starts, ti_ends, ev_mask, ti_mask = ev_starts.to(model_device), ev_ends.to(model_device), ti_starts.to(model_device), ti_ends.to(model_device), ev_mask.to(model_device), ti_mask.to(model_device) 

        hE = self.span_pool(H, ev_starts, ev_ends).to(model_device)  # [B, K, d]
        hT = self.span_pool(H, ti_starts, ti_ends).to(model_device)

        ee_pairs = []
        ee_mask = []
        for i, text_events in enumerate(ev_starts):
            events = text_events[ev_mask[i]]
            combos = torch.combinations(torch.arange(events.shape[0]), r=2)
            ee_pairs.append(combos)
            ee_mask.append(torch.ones(combos.shape[0], dtype=torch.bool))

        ee_pairs = torch.nn.utils.rnn.pad_sequence(ee_pairs, batch_first=True, padding_value=0).to(model_device)
        ee_mask = torch.nn.utils.rnn.pad_sequence(ee_mask, batch_first=True, padding_value=False).to(model_device)

        if self.is_using_ca:
            # 4a) Cross-Attention (Event->Time)
            e_to_t_ca_out = self.ev_to_ti_ca(hE, hT, key_padding_mask=ti_mask)
            hE_ref = e_to_t_ca_out["Q_refined"].to(model_device)
            hT_e = e_to_t_ca_out["h_KV_exp"].to(model_device)

            # 4b) Cross-Attention (Time->Event)
            t_to_e_ca_out = self.ti_to_ev_ca(hT, hE, key_padding_mask=ev_mask)
            hT_ref = t_to_e_ca_out["Q_refined"].to(model_device)

            et_input = self.create_et_input(hE_ref, hT_ref, uses_ca=True)
            ee_input = self.create_ee_input_pairs(hE_ref, ee_pairs, hT_e)
        else:
            et_input = self.create_et_input(hE_ref, hT_ref, uses_ca=False)
            ee_input = self.create_ee_input_pairs(hE_ref, ee_pairs)

        et_out = self.et(et_input)
        et_preds = self.et.decode(et_out['logits'])

        ee_preds = self.ee(ee_input).argmax(-1)
        ee_triples = torch.cat([ee_pairs, ee_preds.unsqueeze(-1)], -1)

        events = []
        for i, event_batch in enumerate(ev_starts):
            temp = []
            for start, end in zip(event_batch, ev_ends[i]):
                if start == -1:
                    continue
                else:
                    temp.append((start.item(), end.item(), "B-EVENT"))
            events.append(temp)
        times = []
        for i, time_batch in enumerate(ti_starts):
            temp = []
            for start, end, type_ in zip(time_batch, ti_ends[i], ti_types[i]):
                if start == -1:
                    continue
                else:
                    temp.append((start.item(), end.item(), type_))
            times.append(temp)

        et_preds = et_preds * ti_mask.unsqueeze(1)

        return tokens, events, times, et_preds, ee_triples, ee_mask
        


if __name__=="__main__":
    model = CrossAttention(3*4, 4)

    torch.manual_seed(42)

    # 3D embeddings
    hE = torch.randn(4, 5, 3)   # [B=4, Ne=5, d=3]
    hT = torch.randn(4, 3, 3)   # [B=4, Nt=3, d=3]
    kpm = torch.full((4,3), True, dtype=torch.bool)
    print(kpm)

    out = model(hE, hT, key_padding_mask=kpm)

    print(hE.shape)
    print(hT.shape)
    print(out)


    
    