import torch
import torch.nn as nn
from transformers import AutoModel
from seqeval.metrics import f1_score as seqeval_f1, classification_report as seqeval_cr
from sklearn.metrics import f1_score as sk_f1, classification_report as sk_cr
import contextlib

EVENT_TIME_NER_LABS = ["B-DATE", "B-DURATION", "B-EVENT", "B-SET", "B-TIME", "I-DATE", "I-DURATION", "I-EVENT", "I-SET", "I-TIME", "O"]
EE_TEMPREL_LABS = ["BEFORE", "AFTER", "DURING", "OVERLAPS", "CONTAINS", "SIMULTANEOUS", "IDENTITY"]

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
    def decode(self, logits):
        # Greedy tag IDs per token. Shape: [B, L]
        return logits.argmax(dim=-1)
    
# ----------------------------
# Cross-attention
# ----------------------------
class CrossAttention(nn.Module):
    def __init__(self, d, heads=6, dropout=0.0):  # set dropout=0.0 for clean probs
        super().__init__()
        self.mha = nn.MultiheadAttention(d, heads, batch_first=True, dropout=dropout)
        self.none_token = nn.Parameter(torch.zeros(1, 1, d))  # learned NONE
        self.ln = nn.LayerNorm(d)
        # optional: small linear to map ϕ(e,t) -> additive bias via attn_mask
        # self.bias = nn.Linear(k_feats, 1)

    def forward(self, hE, hT, key_padding_mask_T=None, attn_bias=None):
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

        # build masks
        if key_padding_mask_T is not None:
            kpm = torch.cat([key_padding_mask_T, torch.zeros(B,1, dtype=torch.bool, device=hE.device)], dim=1)
        else:
            kpm = None

        # attn_mask: additive bias (float) or bool; if float, negative values downweight
        am = None
        if attn_bias is not None:
            # For simplicity, average across heads → use [B, Ne, Nt+1] then flatten to [Ne, Nt+1] per batch
            # prior_b: [B, Ne, Nt] negative distances (or any additive bias)
            # append NONE column of zeros
            prior_b = torch.cat([attn_bias, attn_bias.new_zeros(attn_bias.size(0), attn_bias.size(1), 1)], dim=2)  # [B, Ne, Nt+1]

            # expand to (B * num_heads, Ne, Nt+1)
            H = self.mha.num_heads  # don't hardcode
            am = prior_b.repeat_interleave(H, dim=0)  # [B*H, Ne, Nt+1]


        # run MHA: queries=hE, keys=T_aug, values=T_aug
        out, attn = self.mha(hE, T_aug, T_aug, key_padding_mask=kpm, attn_mask=am, need_weights=True, average_attn_weights=True)
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
    """
    Classifies temporal relation between two events using:
      [h_e1 ; h_e2 ; h_tbar(e1) ; h_tbar(e2) ; φ(e1,e2)]
    Optionally include elementwise products if you like.
    Could include dropout
    """
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

class JointTemporalModel(nn.Module):
    def __init__(self, base="roberta-base",
                 num_ner=len(EVENT_TIME_NER_LABS),
                 ee_labels=len(EE_TEMPREL_LABS),
                 heads=6,
                 et_feat_dim=0,       # φ(e,t) size
                 ee_feat_dim=0):      # φ(e1,e2) size
        super().__init__()
        self.enc = AutoModel.from_pretrained(base)
        d = self.enc.config.hidden_size

        # 1) Token classification (BIO)
        self.ner = NERHead(d, num_ner)

        # 2) Span pooling
        self.span_pool = MaxSpanPool

        # 3) Cross-attention-pointer (events -> times)
        self.ca = CrossAttention(d=d, heads=heads)

        # 4) Event–Event relation head
        self.ee = EEHead(d, n_labels=ee_labels)

        # Loss functions
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=-100)  # for NER and EE
        self.loss_nll = nn.NLLLoss(ignore_index=-100)          # for CA

    def forward(self, input_ids, attention_mask,
                # --- Event and Time Locations
                ev_starts, ev_ends, ev_mask, ti_starts, ti_ends, ti_mask,
                # --- Gold Labels
                ner_gold_labels, ev_ti_gold, ee_rel_gold, ee_mask):
        
        out = {}
        H = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # 1) NER head
        ner_out = self.ner(H)
        out["ner_logits"] = ner_out["logits"]
        if ner_gold_labels is not None:
            ner_loss = self.loss_ce(ner_out["logits"].view(-1, ner_out["logits"].size(-1)), 
                                    ner_gold_labels.view(-1))
            out["ner_loss"] = ner_loss

        # 2) Span Max Pooling
        hE = self.span_pool(H, ev_starts, ev_ends)  # [B, K, d] for events
        hT = self.span_pool(H, ti_starts, ti_ends)  # [B, K, d] for times

        # 3) Cross-Attention-Pointer (events -> times)
        ev_ti_gold = ev_ti_gold.clone()
        ev_ti_gold[~ev_mask] = -100   # ignore padded events
        # centers in token space for each span
        e_center = (ev_starts + ev_ends - 1) / 2.0     # [B,Ne]
        t_center = (ti_starts + ti_ends - 1) / 2.0     # [B,Nt]
        dist = (e_center.unsqueeze(2) - t_center.unsqueeze(1)).abs()   # [B,Ne,Nt]
        tau = 20.0  # temperature in tokens (tune 10–40)
        prior = -dist / tau                                              # [B,Ne,Nt]

        ca_out = self.ca(hE, hT, key_padding_mask_T=~ti_mask, attn_bias=prior)
        ca_probs = ca_out["ptr_probs"]
        logp = (ca_probs + 1e-12).log() 
        ca_loss = self.loss_nll(logp.view(-1, logp.size(-1)), 
                                ev_ti_gold.view(-1))
        out["ptr_idx"] = ca_out["ptr_idx"]
        out['ptr_probs'] = ca_probs
        out["ca_loss"] = ca_loss

        # 4) Event-Event Temporal Relation Head
        ee_pairs  = ee_rel_gold[:, :, [0, 2]]           # [B, M, 2]
        ee_labels = ee_rel_gold[:, :, 1].clone()        # [B, M]
        ee_labels[~ee_mask] = -100                      # <-- mask padded pairs
        hT_e = ca_out["h_time_exp"]                     # [B, Ne, d]
        # (optionally use refined hE — see below)
        hE_for_ee = ca_out["hE_refined"]
        ee_logits = self.ee(hE_for_ee, hT_e, ee_pairs)         # [B, M, C]
        out["ee_logits"] = ee_logits
        ee_loss = self.loss_ce(ee_logits.view(-1, ee_logits.size(-1)),
                            ee_labels.view(-1))
        out["ee_loss"] = ee_loss

        out["loss"] = ner_loss + ca_loss + ee_loss

        return out
    
    def evaluate_dataloader(self, dev_loader, id2label_ner, id2label_ee, *, ee_average="micro"):
        """
        Evaluate over a dataloader.
        - NER: seqeval F1 (entity-level over BIO tags)
        - Event→Time: pointer@1 accuracy
        - EE: F1 (macro by default; set ee_average='micro' or 'weighted' if desired)

        Args:
        id2label_ner: dict[int->str] for BIO tags
        id2label_ee : dict[int->str] for EE labels (not strictly needed for F1 on ids)
        ee_average  : 'macro' | 'micro' | 'weighted'
        """
        self.eval()
        device = next(self.parameters()).device

        # NER (seqeval expects list[list[str]])
        ner_true_seqs, ner_pred_seqs = [], []

        # Pointer metrics
        et_correct, et_total = 0, 0

        # EE F1 collections
        ee_true_all, ee_pred_all = [], []

        with torch.no_grad():
            for batch in dev_loader:
                # move to device
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

                out = self.forward(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    ev_starts=batch["ev_starts"], ev_ends=batch["ev_ends"], ev_mask=batch["ev_mask"],
                    ti_starts=batch["ti_starts"], ti_ends=batch["ti_ends"], ti_mask=batch["ti_mask"],
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
                ptr_pred = out["ptr_idx"]       # [B,Ne]
                ev_mask  = batch["ev_mask"]                   # [B,Ne] (bool)
                gold_ptr = batch["ev_ti_gold"]                # [B,Ne] (long)

                valid = ev_mask & (gold_ptr != -100)
                et_correct += (ptr_pred[valid] == gold_ptr[valid]).sum().item()
                et_total   += valid.sum().item()

                ti_mask = batch["ti_mask"]                  # [B,Nt] (bool)
                ev_ti_gold = batch["ev_ti_gold"]            # [B,Nt] (long)
                none_idx = ti_mask.sum(-1)   # [B], each element is Nt for that example
                valid = (ev_mask) & (ev_ti_gold != -100)
                count = 0
                whole = 0

                # -------- EE F1 --------
                ee_logits = out["ee_logits"]                  # [B,M,C]
                ee_pred   = ee_logits.argmax(-1)              # [B,M]
                ee_gold   = batch["ee_triples"][:, :, 1]      # [B,M]
                ee_mask   = batch["ee_mask"]                  # [B,M] (bool)

                if ee_mask.any():
                    ee_true_all.extend(ee_gold[ee_mask].tolist())
                    ee_pred_all.extend(ee_pred[ee_mask].tolist())

        metrics = {}
        # NER seqeval F1
        metrics["ner_f1"]  = seqeval_f1(ner_true_seqs, ner_pred_seqs) if ner_true_seqs else 0.0
        print(seqeval_cr(ner_true_seqs, ner_pred_seqs, digits=4))
        # Pointer@1 accuracy
        metrics["ptr_acc"] = (et_correct / et_total) if et_total > 0 else 0.0
        # EE F1 (scikit-learn, on label ids)
        if ee_true_all:
            metrics["ee_f1"] = sk_f1(ee_true_all, ee_pred_all, average=ee_average)
            print(sk_cr(ee_true_all, ee_pred_all, target_names=id2label_ee.values(), digits=4))
        else:
            metrics["ee_f1"] = 0.0

        return metrics
    
if __name__=="__main__":
    from Utils import collator
    ex1 = {
        "tokens": [["Alpha", "won", "on", "Friday", "at", "noon", "."]],
        "bio_tags": [["O","B-EVENT","O","B-DATE","O","B-TIME","O"]],
        "instances": [
            {"type":"EVENT","sent_id":0,"offset":[1,2],"id":0},           # "won"
            {"type":"DATE","sent_id":0,"offset":[3,4],"id":10},           # "Friday"
            {"type":"TIME","sent_id":0,"offset":[5,6],"id":11},           # "noon"
            {"type":"EVENT","sent_id":0,"offset":[0,1],"id":1},           # "Alpha" (treat as event for demo)
        ],
        "event_times": [
            {"event":0,"time":10},     # won -> Friday
            {"event":1,"time":"NONE"}  # Alpha has no time
        ],
        "ee_temprels":[
            {"e1":1,"e2":0,"rel":"BEFORE"}  # Alpha BEFORE won (directional)
        ]
    }
    ex2 = {
        "tokens": [
            ["China","bagged","gold","on","Saturday","."],
            ["Final","starts","Monday","morning","."]
        ],
        "bio_tags": [
            ["O","B-EVENT","B-EVENT","O","B-DATE","O"],
            ["O","B-EVENT","B-DATE","O","O"]
        ],
        "instances": [
            {"type":"EVENT","sent_id":0,"offset":[1,2],"id":0},     # bagged
            {"type":"EVENT","sent_id":0,"offset":[2,3],"id":1},     # gold
            {"type":"DATE","sent_id":0,"offset":[4,5],"id":100},    # Saturday
            {"type":"EVENT","sent_id":1,"offset":[1,2],"id":2},     # starts
            {"type":"DATE","sent_id":1,"offset":[2,3],"id":101}     # Monday
        ],
        "event_times": [
            {"event":0,"time":100},   # bagged -> Saturday
            {"event":1,"time":"NONE"},
            {"event":2,"time":101}    # starts -> Monday
        ],
        "ee_temprels":[
            {"e1":0,"e2":1,"rel":"AFTER"},
            {"e1":1,"e2":2,"rel":"BEFORE"}
        ]
    }

    batch=collator([ex1, ex2], {"B-DATE":0, "B-DURATION":1, "B-EVENT": 2, "B-TIME": 3, "I-DATE":4, "I-DURATION":5, "I-EVENT":6, "I-TIME":7, "O":8}, {"AFTER": 0, "BEFORE": 1, "CONTAINS": 2, "DURING":3, "EQUALS":4, "IDENTITY":5, "OVERLAPS":6})

    model = JointTemporalModel()
    out   = model(input_ids=batch["input_ids"],
                  attention_mask=batch["attention_mask"],
                  ev_starts=batch["ev_starts"], ev_ends=batch["ev_ends"], ev_mask=batch["ev_mask"],
                  ti_starts=batch["ti_starts"], ti_ends=batch["ti_ends"], ti_mask=batch["ti_mask"],
                  ner_gold_labels=batch["ner_labels"],
                  ev_ti_gold=batch["ev_ti_gold"],
                  ee_rel_gold=batch["ee_triples"],)
    print(out["loss"], out["ner_logits"].shape, out["ca_probs"].shape, out["ee_logits"].shape)

