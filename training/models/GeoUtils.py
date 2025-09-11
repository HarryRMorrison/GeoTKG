from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import json
from copy import deepcopy
from training.models.globals import LABEL2ID_GEONER, ID2LABEL_GEONER

TOKENIZER = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)

class GeoDataset(Dataset):
    def __init__(self, path):
        if path.endswith(".json"):
            with open(path, "r") as f:
                self.examples=[json.loads(line) for line in f]
    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return deepcopy(self.examples[i])

def collator(examples):
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
                ner_labels[bi, si] = LABEL2ID_GEONER.get(tag, LABEL2ID_GEONER["O"])
            wid_prev = wid
    
    return {"input_ids": input_ids, "attention_mask": attention_mask, "ner_labels": ner_labels}