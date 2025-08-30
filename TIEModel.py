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
    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, H):
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
        self.none_token = nn.Parameter(torch.zeros(1, 1, d))
        self.ln = nn.LayerNorm(d)

    def forward(self, hE, hT, key_padding_mask, attn_mask):
        '''
        hE: [B, Ne, d]
        hT: [B, Nt, d]
        key_padding_mask: [B, Nt]
        attn_mask: [B, Ne]
        '''
        B, Ne, d = hE.shape
        _, Nt, _ = hT.shape

        # Adding None to hT
        none = self.none_token.expand(B, 1, d)                 # [B,1,d]
        T_aug = torch.cat([hT, none], dim=1)                     # [B, Nt+1, d]

        # Applying None to key_padding_mask
        kp_none = torch.zeros(B, 1, dtype=torch.bool, device=key_padding_mask.device)
        key_padding_mask = torch.cat([~key_padding_mask, kp_none], dim=1)
        
        # Switching to correct syntax
        #atn_mask = ~attn_mask.unsqueeze(-1).repeat(1, 1, Nt).repeat(self.mha.num_heads, 1, 1)

        out, attn = self.mha(hE, T_aug, T_aug, key_padding_mask=key_padding_mask, need_weights=True, average_attn_weights=True)
        out = self.ln(out + hE)
        # out:  [B, Ne, d]  (refined events)
        # attn:[B, Ne, Nt] (avg across heads); with dropout=0, this sums to 1 → pointer probs

        h_time_exp = attn @ T_aug           # [B, Ne, d] expected time embedding treating attn as P(t|e)

        return {"h_time_exp": h_time_exp, "hE_refined": out}

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
    
# ----------------------------
# Event-Time Matching Head
# ----------------------------
class ETHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)
        self.sig = nn.Sigmoid()

    @torch.no_grad()
    def _pair_concat(self, E, T):
        e = E.unsqueeze(2).expand(-1, -1, T.size(1), -1)
        t = T.unsqueeze(1).expand(-1, E.size(1), -1, -1)
        return torch.cat([e, t], dim=-1) # [B, Ne, Nt, De+Dt]

    def forward(self, hT, hE):
        x = self._pair_concat(hE, hT)
        output = self.linear(self.drop(x)) # [B, Ne, Nt, 1]
        return {"logits": output.squeeze(-1)}
    
class TIEModel(nn.Module):
    def __init__(self, base="roberta-base",
                 num_ner=len(LABEL2ID_EVNER),
                 ee_labels=len(LABEL2ID_EE),
                 heads=6):
        super().__init__()
        self.enc = AutoModel.from_pretrained(base)
        d = self.enc.config.hidden_size
        self.ner = NERHead(d, num_ner)
        self.span_pool = MaxSpanPool
        self.ca = CrossAttention(d=d, heads=heads)
        self.et = ETHead(d*2)
        self.ee = EEHead(d, n_labels=ee_labels)
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=-100)  # for NER and EE
        self.loss_bce = nn.BCEWithLogitsLoss()

    def save(self, save_path):
        torch.save({'model_state_dict':self.state_dict()}, save_path)

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

        # 4) Cross-Attention (events + times)
        ca_out = self.ca(hE, hT, key_padding_mask=ti_mask, attn_mask = ev_mask)
        hE_ref = ca_out["hE_refined"]
        hT_e = ca_out["h_time_exp"]

        # 5a) Event Time Linking
        et_out = self.et(hT, hE_ref)
        et_logits = et_out['logits']
        mask = ev_ti_gold != -100
        et_loss = self.loss_bce(et_logits[mask].view(-1), 
                                ev_ti_gold[mask].view(-1))
        out["et_loss"] = et_loss
        out["et_logits"] = et_logits

        # 5b) Event-Event Temporal Relation Head
        ee_pairs  = ee_rel_gold[:, :, [0, 2]]           # [B, M, 2]
        ee_logits = self.ee(hE_ref, hT_e, ee_pairs)     # [B, M, C] (optionally use hE — instead of hE_ref)
        ee_labels = ee_rel_gold[:, :, 1].clone()        # [B, M]
        ee_labels[~ee_mask] = -100                      # <-- mask padded pairs
        ee_loss = self.loss_ce(ee_logits.view(-1, ee_logits.size(-1)),
                                ee_labels.view(-1))
        out["ee_loss"] = ee_loss
        out["ee_logits"] = ee_logits

        out["loss"] = ner_loss + et_loss + ee_loss
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
        ner_loss, et_loss, ee_loss = 0, 0, 0
        real_corr, real_none = 0, 0
        tot = 0
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
                et_pred = out["et_logits"]                    # [B,Ne]
                ev_mask  = batch["ev_mask"]                   # [B,Ne] (bool)
                gold_et = batch["ev_ti_gold"]                # [B,Ne] (long)

                valid = ev_mask & (gold_et != -100)
                et_correct += (et_pred[valid] == gold_et[valid]).sum().item()
                et_total   += valid.sum().item()

                nts = []
                fou = False
                for exa in batch["ti_mask"]:
                    for i, bol in enumerate(exa):
                        if not fou and bol == False:
                            nts.append(i+1)
                            fou = True
                    if not fou:
                        nts.append(i+1)
                    fou = False
                for i, ex in enumerate(out["et_pred"]):
                    for j, ans in enumerate(ex.argmax(-1)):
                        if valid[i,j] == True:
                            if ans==gold_et[i,j]:
                                if ans != nts[i]:
                                    real_corr += 1
                                else:
                                    real_none += 1
                            tot += 1

                # -------- EE F1 --------
                ee_logits = out["ee_preds"]                  # [B,M,C]
                ee_pred   = ee_logits.argmax(-1)              # [B,M]
                ee_gold   = batch["ee_triples"][:, :, 1]      # [B,M]
                ee_mask   = batch["ee_mask"]                  # [B,M] (bool)

                if ee_mask.any():
                    ee_true_all.extend(ee_gold[ee_mask].tolist())
                    ee_pred_all.extend(ee_pred[ee_mask].tolist())

                loss.append(out["loss"].item())
                ner_loss += out.get("ner_loss", 0).item()
                et_loss += out.get("et_loss", 0).item()
                ee_loss += out.get("ee_loss", 0).item()
        out['real_cor'] = real_corr/tot
        print("Real correct ET: ", out['real_cor'])
        print("None correct ET: ", real_none/tot)
        metrics = {}
        metrics["ner_f1"]  = seqeval_f1(ner_true_seqs, ner_pred_seqs) if ner_true_seqs else 0.0
        print(seqeval_cr(ner_true_seqs, ner_pred_seqs, digits=4))
        metrics["et_acc"] = (et_correct / et_total) if et_total > 0 else 0.0
        metrics["ee_f1"] = sk_f1(ee_true_all, ee_pred_all, average=ee_average)
        print(sk_cr(ee_true_all, ee_pred_all, target_names=id2label_ee.values(), digits=4))
        metrics["eval_loss"] = sum(loss) / len(loss)
        metrics["ner_loss"] = ner_loss / len(loss)
        metrics["et_loss"] = et_loss / len(loss)
        metrics["ee_loss"] = ee_loss / len(loss)
        return metrics

if __name__=="__main__":
    # model = TIEModel().to("cuda")
    # load = torch.load("D:\\GeoTKG\\results\\tie_model\\tie_model_epoch50.pt")
    # model.load_state_dict(load['model_state_dict'])
    from TIEUtils import TemporalDataset, collator
    from torch.utils.data import DataLoader
    model = TIEModel()
    label2id_ner = LABEL2ID_EVNER
    id2label_ner = ID2LABEL_EVNER
    label2id_ee = LABEL2ID_EE
    id2label_ee = ID2LABEL_EE
    # cleandata_path = "D:\\GeoTKG\\cleandata\\tie\\"
    def collate_fn(examples):
        return collator(examples, label2id_ner=label2id_ner, label2id_ee=label2id_ee)
    # eval = TemporalDataset(cleandata_path + "eval.json")
    # eval_loader = DataLoader(eval, batch_size=2, shuffle=True, collate_fn=collate_fn)
    ex1 = {
        "tokens": [["Alpha", "won", "on", "Friday", "at", "noon","."]],
        "bio_tags": [["O","B-EVENT","O","B-DATE","O","B-TIME","O"]],
        "instances": [
            {"type":"EVENT","sent_id":0,"offset":[1,2],"id":0},           # "won"
            {"type":"DATE","sent_id":0,"offset":[3,4],"id":10},           # "Friday"
            {"type":"DURATION","sent_id":0,"offset":[5,6],"id":11},           # "noon"
            {"type":"EVENT","sent_id":0,"offset":[0,1],"id":1},           # "Alpha" (treat as event for demo)
        ],
        "event_times": [
            {"event":0,"time":10},     # won -> Friday
        ],
        "ee_temprels":[
            {"e1":1,"e2":0,"rel":"BEFORE"}  # Alpha BEFORE won (directional)
        ]
    }
    ex3={"text": [["Israeli", "President", "Moshe", "Katsav", "inked", " ", "a", "decree", "on", "Wednesday", " ", "to", "dissolve", " ", "the", "Knesset", "(", "Parliament", ")", "and", "call", " ", "a", "snap", "election", "."]], 
         "instances": [
             {"offset": [4, 5], "type": "EVENT", "sent_id": 0, "text": "inked", "id": 0}, 
             {"offset": [12, 13], "type": "EVENT", "sent_id": 0, "text": "dissolve", "id": 1}, 
             {"offset": [20, 21], "type": "EVENT", "sent_id": 0, "text": "call", "id": 2}, 
             {"offset": [24, 25], "type": "EVENT", "sent_id": 0, "text": "election", "id": 3}, 
             {"value": "2005-11-23", "type": "TIME", "offset": [1, 0], "id": 0}, 
             {"value": "2005-11-23", "type": "DATE", "sent_id": 0, "offset": [9, 10], "text": "Wednesday", "id": 1}
             ], 
        "event_times": [{"event": 1, "time": 1}], "ee_temprels": [{"e1": 0, "e2": 2, "rel": "AFTER"}, {"e1": 0, "e2": 1, "rel": "AFTER"}], "bio_tags": [["O", "O", "O", "O", "B-EVENT", "O", "O", "O", "O", "B-DATE", "O", "O", "B-EVENT", "O", "O", "O", "O", "O", "O", "O", "B-EVENT", "O", "O", "O", "B-EVENT", "O"]]}
    ex2 = {
        "tokens": [
            ["China","bagged","gold","on","Saturday", "and","Tuesday","."],
            ["Final","starts","Monday","morning","."]
        ],
        "bio_tags": [
            ["O","B-EVENT","B-EVENT","O","B-DATE","O","B-DATE","O"],
            ["O","B-EVENT","B-DATE","O","O"]
        ],
        "instances": [
            {"type":"EVENT","sent_id":0,"offset":[1,2],"id":0},     # bagged
            {"type":"EVENT","sent_id":0,"offset":[2,3],"id":1},     # gold
            {"type":"DATE","sent_id":0,"offset":[4,5],"id":100},    # Saturday
            {"type":"EVENT","sent_id":1,"offset":[1,2],"id":2},     # starts
            {"type":"DATE","sent_id":1,"offset":[2,3],"id":101},     # Monday
            {"type":"DATE","sent_id":0,"offset":[6,7],"id":102}
        ],
        "event_times": [
            {"event":0,"time":100},   # bagged -> Saturday
            {"event":2,"time":101}    # starts -> Monday
        ],
        "ee_temprels":[
            {"e1":0,"e2":1,"rel":"AFTER"},
            {"e1":1,"e2":2,"rel":"BEFORE"}
        ]
    }
    ex4 = {
        "tokens": [
            ["China","bagged","gold","on","Saturday", "and","Tuesday","."],
            ["Final","starts","Monday","morning","."]
        ],
        "bio_tags": [
            ["O","B-EVENT","B-EVENT","O","B-DATE","O","B-DATE","O"],
            ["O","B-EVENT","B-DATE","O","O"]
        ],
        "instances": [
            {"type":"EVENT","sent_id":0,"offset":[1,2],"id":0},     # bagged
            {"type":"EVENT","sent_id":0,"offset":[2,3],"id":1},     # gold
            {"type":"EVENT","sent_id":1,"offset":[1,2],"id":2},     # starts
        ],
        "ee_temprels":[
            {"e1":0,"e2":1,"rel":"AFTER"},
            {"e1":1,"e2":2,"rel":"BEFORE"}
        ]
    }
    batch = collate_fn([ex1,ex2, ex3, ex4])
    print(model.forward(batch["input_ids"], batch["attention_mask"],
                # --- Event and Time Locations
                batch["ev_starts"], batch["ev_ends"], batch["ev_mask"], batch["ti_starts"], batch["ti_ends"], batch["ti_mask"], batch["e_sent_ids"], batch["t_sent_ids"],
                # --- Gold Labels
                batch["ner_labels"], batch["ev_ti_gold"], batch["ee_triples"], batch["ee_mask"]))
    
    #batch_evaluation=model.evaluate_dataloader(batch, id2label_ee=id2label_ee, id2label_ner=id2label_ner)
    # from transformers import AutoTokenizer
    # ex = {"text": [["Eleven", "people", "were", "Eleven", "people", "were", "confirmed", "confirmed", "blast", "blast", "Wednesday", "morning", "Wednesday", "morning", "said", "said"]], 
    #       "instances": [{"offset": [6, 7], "type": "EVENT", "sent_id": 0, "text": "confirmed", "id": 0}, 
    #                     {"offset": [8, 9], "type": "EVENT", "sent_id": 0, "text": "blast", "id": 1}, 
    #                     {"offset": [14, 15], "type": "EVENT", "sent_id": 0, "text": "said", "id": 2}, 
    #                     {"value": "2006-11-29", "type": "TIME", "offset": [1, 0], "id": 0}, 
    #                     {"value": "2006-11-22TMO", "type": "DATE", "sent_id": 0, "offset": [10, 12], "text": "Wednesday morning", "id": 1}], 
    #                     "event_times": [{"event": 1, "time": 1}, 
    #                                     {"event": 0, "time": "NONE"}, 
    #                                     {"event": 2, "time": "NONE"}], 
    #                     "ee_temprels": [{"e1": 0, "e2": 2, "rel": "DURING"}], 
    #                     "bio_tags": [["O", "O", "O", "O", "O", "O", "B-EVENT", "O", "B-EVENT", "O", "B-DATE", "I-DATE", "O", "O", "B-EVENT", "O"]]}
    # ex2 = {
    #     "tokens": [
    #         ["China","bagged","gold","on","Saturday","."],
    #         ["Final","starts","Monday","morning","."]
    #     ],
    #     "bio_tags": [
    #         ["O","B-EVENT","B-EVENT","O","B-DATE","O"],
    #         ["O","B-EVENT","B-DATE","O","O"]
    #     ],
    #     "instances": [
    #         {"type":"EVENT","sent_id":0,"offset":[1,2],"id":0},     # bagged
    #         {"type":"EVENT","sent_id":0,"offset":[2,3],"id":1},     # gold
    #         {"type":"DATE","sent_id":0,"offset":[4,5],"id":100},    # Saturday
    #         {"type":"EVENT","sent_id":1,"offset":[1,2],"id":2},     # starts
    #         {"type":"EVENT","sent_id":1,"offset":[0,1],"id":3},     # final
    #         {"type":"DATE","sent_id":1,"offset":[2,3],"id":101}     # Monday
    #     ],
    #     "event_times": [
    #         {"event":0,"time":100},   # bagged -> Saturday
    #         {"event":1,"time":"NONE"},
    #         {"event":2,"time":101}    # starts -> Monday
    #     ],
    #     "ee_temprels":[
    #         {"e1":0,"e2":1,"rel":"AFTER"},
    #         {"e1":1,"e2":2,"rel":"BEFORE"}
    #     ]
    # }
    # TOKENIZER = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
    # encod = TOKENIZER(ex["text"][0], is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
    # batch = {"input_ids":torch.tensor(encod["input_ids"]), "attention_mask":torch.tensor(encod["attention_mask"])}

    # preds = model.predict(batch)
    # temprels = preds["ee_relations"]
    # ets = preds['event_time_idx']
    # ner = preds["ner_tags"]
    # print(temprels, ets, ner)

    # batch = collator([ex,ex2], label2id_ner=LABEL2ID_EVNER, label2id_ee=LABEL2ID_EE)
    # model.forward(input_ids=batch["input_ids"],
    #             attention_mask=batch["attention_mask"],
    #             ev_starts=batch["ev_starts"], ev_ends=batch["ev_ends"], ev_mask=batch["ev_mask"], e_sent_ids=batch["e_sent_ids"],
    #             ti_starts=batch["ti_starts"], ti_ends=batch["ti_ends"], ti_mask=batch["ti_mask"], t_sent_ids=batch["t_sent_ids"],
    #             ner_gold_labels=batch["ner_labels"],
    #             ev_ti_gold=batch["ev_ti_gold"],
    #             ee_rel_gold=batch["ee_triples"],
    #             ee_mask=batch["ee_mask"])
