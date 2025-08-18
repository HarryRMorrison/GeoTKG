import torch
import torch.nn as nn
from transformers import AutoModel

class MaxSpanPool(nn.Module):
    def forward(self, H, starts, ends, mask=None):
        B, L, d = H.shape
        K = starts.size(1)

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

        if mask is not None:
            H_span = torch.where(mask.unsqueeze(-1), H_span, torch.zeros_like(H_span))

        return H_span    

class JointTemporalModel(nn.Module):
    def __init__(self, base="roberta-base",
                 num_ner=5,           # e.g., B-ev/I-ev/B-tx/I-tx/O
                 ee_labels=7,         # BEFORE/AFTER/OVERLAP/INCLUDES/IS_INCLUDED/SIMUL/VAGUE
                 heads=4,
                 et_feat_dim=0,       # φ(e,t) size
                 ee_feat_dim=0):      # φ(e1,e2) size
        super().__init__()
        self.enc = AutoModel.from_pretrained(base)
        d = self.enc.config.hidden_size

        # 1) Token classification (BIO)
        self.ner = nn.Linear(d, num_ner)

        # 2) Span pooling
        self.span_pool = SpanPool(d)

        # 3) Cross-attention-pointer (events -> times)
        self.ptr = CrossPointer(d, heads=heads, feat_dim=et_feat_dim)

        # 4) Event–Event relation head
        self.ee = EEHead(d, feat_dim=ee_feat_dim, n_labels=ee_labels)

        # Loss functions
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=-100)  # for NER and EE

    def forward():
        return
