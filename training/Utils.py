import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def collator(batch, label2id_ner, label2id_ee):
    B = len(batch)
    # 1) Tokenize whole batch -> word to subword
    enc = tokenizer(batch["tokens"], truncation=True, is_split_into_words=True, padding=True)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    word_ids_list = [enc.word_ids(bi) for bi in range(B)]
    L = input_ids.size(1)

    first_sub, last_sub_excl = subtoken_index_map(B, input_ids, word_ids_list)
    ner_labels = BIO_tag_alignment(torch.full((B, L), -100, dtype=torch.long), batch, word_ids_list, label2id_ner)
    ev_starts, ev_ends, ev_mask, ti_starts, ti_ends, ti_mask, ev_ti_gold = build_event_time_span_tensors(batch, first_sub, last_sub_excl)
    gold_ee_trips = build_ee_triples(batch)

    batch_out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "ner_labels": ner_labels,
        "ev_starts": ev_starts, "ev_ends": ev_ends, "ev_mask": ev_mask,
        "ti_starts": ti_starts, "ti_ends": ti_ends, "ti_mask": ti_mask,
        "ev_ti_gold": ev_ti_gold,
        "ee_gold_triples": gold_ee_trips
    }
    return batch_out

def subtoken_index_map(B, input_ids, word_ids_list):
    # Precompute first/last subtoken index for each word
    first_sub = []
    last_sub_excl = []
    for bi, word_ids in enumerate(word_ids_list):
        first = {}
        last = {}
        for si, wid in enumerate(word_ids):
            if wid is None: continue
            if wid not in first: first[wid] = si
            last[wid] = si
        # end-exclusive index for a word = last_subtoken + 1
        max_w = -1 if not last else max(last.keys())
        fs = [first.get(w, None) for w in range(max_w+1)]
        le = [ (last.get(w, None)+1 if w in last else None) for w in range(max_w+1)]
        first_sub.append(fs); last_sub_excl.append(le)
    return first_sub, last_sub_excl

def BIO_tag_alignment(to_fill, batch, word_ids_list, label2id):
    for bi, ex in enumerate(batch):
        tags = ex["ner_tags"]                     # word-level tags
        word_ids = word_ids_list[bi]
        prev_wid = None
        for si, wid in enumerate(word_ids):
            if wid is None: continue
            if wid != prev_wid:                   # first subword gets the tag id
                to_fill[bi, si] = label2id[tags[wid]]
            # else keep -100 for non-first subwords
            prev_wid = wid
    return to_fill

def build_event_time_span_tensors(batch, first_sub, last_sub_excl):
    ev_starts, ev_ends, ev_mask = [], [], []
    ti_starts, ti_ends, ti_mask = [], [], []
    ev_ti_gold_local = []

    for bi, ex in enumerate(batch):
        fs, le = first_sub[bi], last_sub_excl[bi]

        def map_span(ws, we):
            return fs[ws], le[we-1]
        
        events, times = [], []

        for ent in ex["instances"]:
            s, e = map_span(ent["start"], ent["end"])
            if ent["type"] == "EVENT": events.append((s,e))
            elif ent["type"] == "TIMEX": times.append((s,e))

        ptr = []
        if "event_times" in ex:
            for et in ex["event_times"]:
                if et["tid"] == "NONE": ptr.append(len(times)) # None location in training is at the end
                else: ptr.append(et["tid"]) # time index (TO DO in preprocessing)
        
        ev_starts.append([s for s,_ in events] or [0])
        ev_ends.append([e for _,e in events] or [1])
        ev_mask.append([1]*len(events) or [0])

        ti_starts.append([s for s,_ in times] or [0])
        ti_ends.append([e for _,e in times] or [1])
        ti_mask.append([1]*len(times) or [0])

        ev_ti_gold_local.append(ptr or [0])

    ev_starts = torch.nn.utils.rnn.pad_sequence(torch.tensor(ev_starts), batch_first=True, padding_value=0)
    ev_ends = torch.nn.utils.rnn.pad_sequence(torch.tensor(ev_ends), batch_first=True, padding_value=1)
    ev_mask = torch.nn.utils.rnn.pad_sequence(torch.tensor(ev_mask), batch_first=True, padding_value=0).bool()

    ti_starts = torch.nn.utils.rnn.pad_sequence(torch.tensor(ti_starts), batch_first=True, padding_value=0)
    ti_ends = torch.nn.utils.rnn.pad_sequence(torch.tensor(ti_ends), batch_first=True, padding_value=1)
    ti_mask = torch.nn.utils.rnn.pad_sequence(torch.tensor(ti_mask), batch_first=True, padding_value=0).bool()

    Ne = max(len(x) for x in ev_starts)
    Nt = max(len(x) for x in ti_starts)
    gold_ptr = torch.full((len(batch), Ne), -100, dtype=torch.long)
    for bi, ptr in enumerate(ev_ti_gold_local):
        for i, idx in enumerate(ptr):
            gold_ptr[bi, i] = idx if idx < Nt else Nt
        
    return (ev_starts, ev_ends, ev_mask), (ti_starts, ti_ends, ti_mask), gold_ptr

def build_ee_triples(batch):
    triples_list = []
    for ex in batch:
        batch_trips = []
        for temprel in ex["ee_temprels"]:
            batch_trips.append(temprel["e1"], temprel["rel"], temprel["e2"])
        triples_list.append(batch_trips)
    torch.nn.utils.rnn.pad_sequence(torch.tensor(triples_list), batch_first=True, padding_value=0) # Hope this padding will work

def compute_ner_metrics(predictions, labels, id2label):
    # Remove ignored index (special tokens)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "classification_report": classification_report(true_labels, true_predictions),
    }

class TemporalDataset(Dataset):
    def __init__(self, examples):           # list of JSON dicts
        self.examples = examples
    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return self.examples[i]

class NormaliseDataset(Dataset):
    def __init__(self, examples):           # list of JSON dicts
        self.examples = examples
    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return self.examples[i]

class GeoDataset(Dataset):
    def __init__(self, examples):           # list of JSON dicts
        self.examples = examples
    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return self.examples[i]

if __name__ == "__main__":
    train_data = TemporalDataset(train_examples)   # your list of JSON dicts
    dev_data   = TemporalDataset(dev_examples)

    label2id_ner = {"B-EVENT":0,"I-EVENT":1,"B-TIMEX":2,"I-TIMEX":3,"O":4}
    label2id_ee  = {"BEFORE":0,"AFTER":1,"OVERLAP":2,"INCLUDES":3,"IS_INCLUDED":4,"SIMUL":5,"VAGUE":6}

    train_loader = DataLoader(
        train_data, batch_size=8, shuffle=True, num_workers=2, pin_memory=True,
        collate_fn=lambda batch: collator(batch, label2id_ner, label2id_ee)
    )
    dev_loader = DataLoader(
        dev_data, batch_size=8, shuffle=False, num_workers=2, pin_memory=True,
        collate_fn=lambda batch: collator(batch, label2id_ner, label2id_ee)
    )
