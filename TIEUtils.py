from torch.utils.data import Dataset, DataLoader
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import os
import json
from copy import deepcopy

TOKENIZER = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)

TIMEX_TYPES = {"DATE", "TIME", "SET", "DURATION"}

def collator(examples, label2id_ner, label2id_ee):
    """
    examples: list of dicts, each like your JSON:
      - "text" or "tokens": tokens per sentence (list[list[str]]) or flat (list[str])
      - "bio_tags": per-sentence BIO tags (list[list[str]]) or flat (list[str]) [optional]
      - "instances": list of { "type": "EVENT"/("DATE"/"TIME"/"SET"/"DURATION"),
                               "offset": [start, end],   # word indices within that sentence
                               "sent_id": int,           # required if offsets are sentence-local
                               "id": int }               # stable id within the doc (optional but recommended)
      - "event_times": list of { "event": <event_id>, "time": <time_id or "NONE"> }
      - "ee_temprels": list of { "e1": <event_id>, "e2": <event_id>, "rel": <string> }
    """

    B = len(examples)

    # ---------- 1) Flatten tokens & tags to word-level ----------
    flat_tokens, flat_tags, sent_word_base = [], [], []  # per example
    for ex in examples:
        toks = ex.get("tokens") or ex.get("text")
        tags = ex.get("bio_tags")
        # normalise to list[list[str]]
        if toks and isinstance(toks[0], str):
            toks = [toks]
        if tags and isinstance(tags[0], str):
            tags = [tags]

        # compute sentence starting indices (word-level)
        bases = []
        n = 0
        for s in (toks or [[]]):
            bases.append(n)
            n += len(s)
        sent_word_base.append(bases)

        # flatten tokens
        flat_tokens.append([w for s in (toks or [[]]) for w in s])

        # flatten BIO tags (or default to 'O')
        if tags:
            flat_tags.append([t for s in tags for t in s])
        else:
            flat_tags.append(["O"] * len(flat_tokens[-1]))

    # ---------- 2) Tokenize with word alignment ----------
    enc = TOKENIZER(
        flat_tokens,
        is_split_into_words=True,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"]           # [B, L]
    attention_mask = enc["attention_mask"]      # [B, L]
    L = input_ids.size(1)

    word_ids_list = [enc.word_ids(bi) for bi in range(B)]  # lists of length L with word indices or None

    # ---------- 3) Align BIO tags to subwords ----------
    ner_labels = torch.full((B, L), -100, dtype=torch.long)
    for bi in range(B):
        wid_prev = None
        for si, wid in enumerate(word_ids_list[bi]):
            if wid is None:
                continue
            if wid != wid_prev:  # first subword
                tag = flat_tags[bi][wid]
                ner_labels[bi, si] = label2id_ner.get(tag, label2id_ner["O"])
            wid_prev = wid

    # ---------- 4) Build EVENT/TIMEX spans (subword indices) ----------
    ev_starts, ev_ends, ti_starts, ti_ends = [], [], [], []
    ev_lens, ti_lens = [], []
    ev_sent_ids, ti_sent_ids = [], []  # for positional prior
    ev2idx_per_ex, timex2idx_per_ex = [], []

    for bi, ex in enumerate(examples):
        bases = sent_word_base[bi]
        fs, le = _first_last_subtokens(word_ids_list[bi])
        events, times = [], []
        ev2idx, tx2idx = {}, {}
        in_e_si, in_t_si = [], []

        for inst in ex.get("instances", []):
            typ = inst["type"]
            s_id = inst.get("sent_id", -1)
            # sentence-local offsets -> global word indices
            if "sent_id" in inst:
                base = bases[inst["sent_id"]]
            else:
                base = 0  # already global

            wstart, wend = inst["offset"]  # word indices, [start, end)

            gstart, gend = wstart + base, wend + base

            s_sub = fs.get(gstart)
            e_sub_excl = le.get(gend - 1)  # end-exclusive subtoken index

            if s_sub is None or e_sub_excl is None:
                # span fell on a special token / truncated — skip robustly
                continue

            iid = inst.get("id", None)

            if typ == "EVENT":
                ev2idx[iid if iid is not None else len(events)] = len(events)
                events.append((s_sub, e_sub_excl))
                in_e_si.append(s_id if s_id != -1 else -1)
            elif typ in TIMEX_TYPES:
                tx2idx[iid if iid is not None else len(times)] = len(times)
                times.append((s_sub, e_sub_excl))
                in_t_si.append(s_id if s_id != -1 else -1)
                

        ev2idx_per_ex.append(ev2idx)
        timex2idx_per_ex.append(tx2idx)

        if events:
            ev_starts.append(torch.tensor([s for s, _ in events], dtype=torch.long))
            ev_ends.append(torch.tensor([e for _, e in events], dtype=torch.long))
            ev_sent_ids.append(torch.tensor(in_e_si, dtype=torch.long))
            ev_lens.append(len(events))
        else:
            ev_starts.append(torch.tensor([0], dtype=torch.long))
            ev_ends.append(torch.tensor([1], dtype=torch.long))
            ev_sent_ids.append(torch.tensor([-1], dtype=torch.long))
            ev_lens.append(0)

        if times:
            ti_starts.append(torch.tensor([s for s, _ in times], dtype=torch.long))
            ti_ends.append(torch.tensor([e for _, e in times], dtype=torch.long))
            ti_sent_ids.append(torch.tensor(in_t_si, dtype=torch.long))
            ti_lens.append(len(times))
        else:
            ti_starts.append(torch.tensor([0], dtype=torch.long))
            ti_ends.append(torch.tensor([1], dtype=torch.long))
            ti_sent_ids.append(torch.tensor([-1], dtype=torch.long))
            ti_lens.append(0)

    ev_starts = torch.nn.utils.rnn.pad_sequence(ev_starts, batch_first=True, padding_value=-1)
    ev_ends   = torch.nn.utils.rnn.pad_sequence(ev_ends,   batch_first=True, padding_value=-1)
    ti_starts = torch.nn.utils.rnn.pad_sequence(ti_starts, batch_first=True, padding_value=-1)
    ti_ends   = torch.nn.utils.rnn.pad_sequence(ti_ends,   batch_first=True, padding_value=-1)
    ev_sent_ids = torch.nn.utils.rnn.pad_sequence(ev_sent_ids, batch_first=True, padding_value=-1)
    ti_sent_ids = torch.nn.utils.rnn.pad_sequence(ti_sent_ids, batch_first=True, padding_value=-1)

    Ne_max = ev_starts.size(1)
    Nt_max = ti_starts.size(1)

    ev_mask = torch.zeros((B, Ne_max), dtype=torch.bool)
    ti_mask = torch.zeros((B, Nt_max), dtype=torch.bool)
    for bi in range(B):
        ev_mask[bi, :ev_lens[bi]] = True
        ti_mask[bi, :ti_lens[bi]] = True

    # ---------- 5) Build event→time gold pointer indices with per-example NONE ----------
    ev_ti_gold = torch.full((B, Ne_max), -100, dtype=torch.long)
    for bi, ex in enumerate(examples):
        Nt_real = ti_lens[bi]  # NONE index for this example
        if Nt_real < 0:
            Nt_real = 0
        ev2idx = ev2idx_per_ex[bi]
        tx2idx = timex2idx_per_ex[bi]

        for et in ex.get("event_times", []):
            e_id = et["event"]
            t_id = et["time"]
            if e_id not in ev2idx:
                continue
            e_idx = ev2idx[e_id]
            if t_id == "NONE":
                ev_ti_gold[bi, e_idx] = Nt_real
            else:
                if t_id in tx2idx:
                    ev_ti_gold[bi, e_idx] = tx2idx[t_id]
                else:
                    # if time id not present (filtered/truncated), fall back to NONE
                    ev_ti_gold[bi, e_idx] = Nt_real

        # mask out padded events
        ev_ti_gold[bi, ~ev_mask[bi]] = -100

    # ---------- 6) Build EE triples & mask ----------
    # Shape: [B, M, 3] with [e1_idx, rel_id, e2_idx]
    ee_triples, ee_mask = [], []
    M_max = 0
    tmp_triples = []

    for bi, ex in enumerate(examples):
        rows = []
        ev2idx = ev2idx_per_ex[bi]
        for r in ex.get("ee_temprels", []):
            e1_id, e2_id = r["e1"], r["e2"]
            rel_str = r["rel"]
            if e1_id in ev2idx and e2_id in ev2idx and rel_str in label2id_ee:
                rows.append([
                    ev2idx[e1_id],
                    label2id_ee[rel_str],
                    ev2idx[e2_id]
                ])
        tmp_triples.append(rows)
        M_max = max(M_max, len(rows))

    if M_max == 0:
        ee_triples = torch.zeros((B, 1, 3), dtype=torch.long)
        ee_mask    = torch.zeros((B, 1), dtype=torch.bool)
    else:
        ee_triples = torch.full((B, M_max, 3), 0, dtype=torch.long)
        ee_mask    = torch.zeros((B, M_max), dtype=torch.bool)
        for bi, rows in enumerate(tmp_triples):
            if rows:
                t = torch.tensor(rows, dtype=torch.long)
                ee_triples[bi, :t.size(0), :] = t
                ee_mask[bi, :t.size(0)] = True

    batch_out = {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "ner_labels":     ner_labels,                     # [B, L]
        "ev_starts":      ev_starts,  "ev_ends":  ev_ends,  "ev_mask": ev_mask,  "e_sent_ids": ev_sent_ids,  # [B, Ne], [B, Ne], [B, Ne]
        "ti_starts":      ti_starts,  "ti_ends":  ti_ends,  "ti_mask": ti_mask,  "t_sent_ids": ti_sent_ids,  # [B, Nt], [B, Nt], [B, Nt]
        "ev_ti_gold":     ev_ti_gold,                     # [B, Ne], NONE per-example = Nt_real
        "ee_triples":     ee_triples,                     # [B, M, 3] (e1, rel_id, e2)
        "ee_mask":        ee_mask,                        # [B, M]   (True=real)
    }
    return batch_out


# ---------- helpers ----------

def _first_last_subtokens(word_ids):
    """
    Returns:
      first_sub: dict word_index -> first subtoken index
      last_sub_excl: dict word_index -> end-exclusive subtoken index
    """
    first_sub = {}
    last_sub  = {}
    for si, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid not in first_sub:
            first_sub[wid] = si
        last_sub[wid] = si
    last_sub_excl = {w: (idx + 1) for w, idx in last_sub.items()}
    return first_sub, last_sub_excl

class TemporalDataset(Dataset):
    def __init__(self, path):        
        if path.endswith(".json"):
            with open(path, "r") as f:
                self.examples=[json.loads(line) for line in f]
    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return deepcopy(self.examples[i])

if __name__ == "__main__":
    ex1 = {
        "tokens": [["Alpha", "won", "on", "Friday", "at", "noon", "."]],
        "bio_tags": [["O","B-EVENT","O","B-DATE","O","B-TIME","O"]],
        "instances": [
            {"type":"EVENT","sent_id":0,"offset":[1,2],"id":0},           # "won"
            {"type":"DATE","sent_id":0,"offset":[3,4],"id":10},           # "Friday"
            {"type":"DURATION","sent_id":0,"offset":[5,6],"id":11},           # "noon"
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

    ex3 = {"text": [["China", "China", "bagged", "bagged", "tally", "tally", "shooting", "shooting", "Saturday", "Saturday", "won", "won", "decides", " ", "the", "winner", "of", "the", "team", "event", "."], 
                    ["The", "individual", "Trap", "final", "will", "be", "held", "held", "Saturday", "Saturday"]], 
           "instances": [{"offset": [2, 3], "type": "EVENT", "sent_id": 0, "text": "bagged", "id": 0}, 
                         {"offset": [4, 5], "type": "EVENT", "sent_id": 0, "text": "tally", "id": 1}, 
                         {"offset": [6, 7], "type": "EVENT", "sent_id": 0, "text": "shooting", "id": 2}, 
                         {"offset": [10, 11], "type": "EVENT", "sent_id": 0, "text": "won", "id": 3}, 
                         {"offset": [12, 13], "type": "EVENT", "sent_id": 0, "text": "decides", "id": 4}, 
                         {"offset": [6, 7], "type": "EVENT", "sent_id": 1, "text": "held", "id": 5}, 
                         {"value": "2006-12-02", "type": "TIME", "offset": [1, 0], "id": 0}, 
                         {"value": "2006-12-09", "type": "DURATION", "sent_id": 0, "offset": [8, 9], "text": "Saturday", "id": 1}, 
                         {"value": "2006-12-09", "type": "DATE", "sent_id": 1, "offset": [8, 9], "text": "Saturday", "id": 2}], 
           "event_times": [{"event": 5, "time": 2}, 
                           {"event": 2, "time": 1}, 
                           {"event": 0, "time": "NONE"}, 
                           {"event": 1, "time": "NONE"}, 
                           {"event": 3, "time": "NONE"}, 
                           {"event": 4, "time": "NONE"}], 
           "ee_temprels": [{"e1": 3, "e2": 4, "rel": "AFTER"}, {"e1": 0, "e2": 2, "rel": "AFTER"}, {"e1": 0, "e2": 1, "rel": "AFTER"}], 
           "bio_tags": [["O", "O", "B-EVENT", "O", "B-EVENT", "O", "B-EVENT", "O", "B-DURATION", "O", "B-EVENT", "O", "B-EVENT", "O", "O", "O", "O", "O", "O", "O", "O"], ["O", "O", "O", "O", "O", "O", "B-EVENT", "O", "B-DATE", "O"]]}

    from globals import LABEL2ID_EVNER, LABEL2ID_EE

    out=collator([ex3], LABEL2ID_EVNER, LABEL2ID_EE)
    print(out)
    print(out['ner_labels'])
