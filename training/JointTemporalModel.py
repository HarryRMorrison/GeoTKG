import torch
import torch.nn as nn
from transformers import AutoModel

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
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

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
            am = attn_bias

        # run MHA: queries=hE, keys=T_aug, values=T_aug
        out, attn = self.mha(hE, T_aug, T_aug, key_padding_mask=kpm, attn_mask=am, need_weights=True, average_attn_weights=True)
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
    def __init__(self, d, n_labels, hidden=256, dropout=0.1):
        super().__init__()
        in_dim = 4*d
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), 
            nn.ReLU(),
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

        x = torch.cat([he1, he2, ht1, ht2], dim=-1)                      # [B,M,4d]
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
                # --- Number of Labels
                ner_labels=EVENT_TIME_NER_LABS,  ee_labels=EE_TEMPREL_LABS,
                # --- Event and Time Locations
                ev_starts=None, ev_ends=None,
                ti_starts=None, ti_ends=None,
                # --- Gold Labels
                ner_gold_labels=None, ev_ti_gold=None, ee_gold_triples=None):
        out = {}
        H = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # 1) NER head
        ner_out = self.ner(H)
        out["ner_logits"] = ner_out["logits"]
        if ner_gold_labels is not None:
            ner_loss = self.loss_fn(ner_out["logits"].view(-1, ner_out["logits"].size(-1)), ner_gold_labels.view(-1))
            out["ner_loss"] = ner_loss

        # 2) Span Max Pooling
        hE = self.span_pool(H, ev_starts, ev_ends)  # [B, K, d] for events
        hT = self.span_pool(H, ti_starts, ti_ends)  # [B, K, d] for times

        # 3) Cross-Attention-Pointer (events -> times)
        ca_out = self.ca(hE, hT)
        out["ca_logits"] = ca_out["ptr_probs"]
        if ev_ti_gold_pairs is not None:
            logp = (ca_out["ptr_probs"] + 1e-12).log() 
            ca_loss = self.nll_loss(logp.view(-1, logp.size(-1)), ev_ti_gold_pairs.view(-1), ignore_index=-100, reduction='mean')
            out["ca_loss"] = ca_loss

        # 4) Event-Event Temporal Relation Head
        ee_pairs = ee_gold_triples[:, [0, 2]]   # [B, M, 2]
        ee_triples_labels = ee_gold_triples[:, 1]  # [B, M]
        e_idx = ev_ti_gold_pairs[..., 0].long()  # [B, M] event indices
        t_idx = ev_ti_gold_pairs[..., 1].long()  # [B, M] time indices
        t_emb = torch.gather(hT, 1, t_idx.unsqueeze(-1).expand(-1, -1, hT.size(-1)))  # [B, M, d]
        hT_e = torch.zeros(hE.size(0), hE.size(1), hT.size(-1), device=hE.device)  # [B, Ne, d]
        batch = torch.arange(hE.size(0), device=hT.device).unsqueeze(-1).expand_as(e_idx)  # [B, M]
        hT_e[batch, e_idx, :] = t_emb   # [B, Ne, d]
        ee_logits = self.ee(hE, hT_e, ee_pairs)  # [B, M, C]
        out["ee_logits"] = ee_logits
        ee_loss = self.loss_ce(ee_logits.view(-1, ee_logits.size(-1)), ee_triples_labels.view(-1))
        out["ee_loss"] = ee_loss

        out["loss"] = ner_loss + ca_loss + ee_loss

        return out

if __name__=="__main__":
    test = MaxSpanPool()
    toks = torch.tensor([[[ 1.,  0.,  2.],   # token 0
                        [ 0.,  3.,  1.],   # token 1
                        [-1.,  4.,  0.],   # token 2
                        [ 2.,  2.,  2.],   # token 3
                        [ 1.,  5., -3.],   # token 4
                        [ 0., -1.,  7.]]])
    starts = torch.tensor([[0, 2, 4, 5]])
    ends = torch.tensor([[2, 5, 6, 5]])
    print(test.forward(toks, starts, ends))
