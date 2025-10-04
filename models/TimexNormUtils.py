from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from copy import deepcopy
from torch.utils.data import Dataset
import json
import isodate
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

def collator(examples, tokenizer=tokenizer, max_tgt_len=128):
    inputs  = [ex["input_text"] for ex in examples]
    targets = [ex["output_text"] for ex in examples]

    enc = tokenizer(
        inputs,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    # New API (avoids deprecated as_target_tokenizer)
    labels = tokenizer(
        text_target=targets,
        padding=True,
        truncation=True,
        max_length=max_tgt_len,
        return_tensors="pt",
    )["input_ids"]

    # mask pad positions in labels with -100 (HF convention)
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,              # keep labels for training AND eval decoding
    }

def gentext_to_iso8601(gentext: str):
    parsers = {
        isodate.parse_date:"DATE",
        isodate.parse_time:"TIME",
        isodate.parse_datetime:"TIME",
        isodate.parse_duration:"DURATION",
        isodate.parse_tzinfo:"SET",
    }

    for parser in parsers:
        try:
            out = parser(gentext)
            type_ = parsers[parser]
            return out, type_
        except Exception:
            continue

    # If none of the parsers worked
    #print(f"UNREC: {gentext}")
    return None

import datetime as _dt
from typing import List, Tuple, Optional

# assumes you already defined this (your cleaner loop-based version)
# from your_module import gentext_to_iso8601

def _cmp_date_components(gold, pred) -> bool:
    """Any of year/month/day matches."""
    g = {"y": getattr(gold, "year", None), "m": getattr(gold, "month", None), "d": getattr(gold, "day", None)}
    p = {"y": getattr(pred, "year", None), "m": getattr(pred, "month", None), "d": getattr(pred, "day", None)}
    return any(g[k] is not None and g[k] == p[k] for k in ("y", "m", "d"))

def _cmp_time_components(gold, pred) -> bool:
    """Any of hour/minute/second matches (ignores microseconds)."""
    g = {"h": getattr(gold, "hour", None), "m": getattr(gold, "minute", None), "s": getattr(gold, "second", None)}
    p = {"h": getattr(pred, "hour", None), "m": getattr(pred, "minute", None), "s": getattr(pred, "second", None)}
    return any(g[k] is not None and g[k] == p[k] for k in ("h", "m", "s"))

def _cmp_datetime_components(gold, pred) -> bool:
    """Any date *or* time component matches."""
    date_ok = _cmp_date_components(gold, pred)
    time_ok = _cmp_time_components(gold, pred)
    return date_ok or time_ok

def _total_seconds(x) -> Optional[float]:
    # isodate durations often become datetime.timedelta
    if isinstance(x, _dt.timedelta):
        return x.total_seconds()
    return None

def _cmp_duration_any_component(gold, pred) -> bool:
    """
    Mark correct if total seconds equal (most practical),
    OR if both encode at least one matching component (days/hours/minutes/seconds) when derivable.
    """
    gs = _total_seconds(gold)
    ps = _total_seconds(pred)
    if gs is not None and ps is not None:
        return abs(gs - ps) < 1e-6

    # Fallback: try to infer rough components if timedelta-like but not precise
    # (Most libraries give timedelta; if not, we can’t safely decompose—return False.)
    return False

def _normalize_set_string(s: str) -> str:
    """
    Very light 'SET' normalization:
    - split common separators, strip whitespace, sort tokens, rejoin.
    Adjust to your dataset’s SET format.
    """
    for sep in [",", ";", "|"]:
        s = s.replace(sep, " ")
    toks = [t for t in s.split() if t]
    toks.sort()
    return " ".join(toks).lower()

def _cmp_set_relaxed(gold_str: str, pred_str: str) -> bool:
    """Any overlap in normalized token sets qualifies as relaxed-correct."""
    g = set(_normalize_set_string(gold_str).split())
    p = set(_normalize_set_string(pred_str).split())
    return len(g & p) > 0

def relaxed_correct_single(gold_text: str, pred_text: str) -> bool:
    """
    Strict equality first; if not equal, apply relaxed rule per ti_type.
    ti_type ∈ {"DATE","TIME","DATETIME","DURATION","SET"} (case-insensitive).
    """
    # Strict exact match first (you can move strict to your main metric if preferred)
    if pred_text == gold_text:
        return True

    g_parsed,t = gentext_to_iso8601(gold_text)
    p_parsed,t = gentext_to_iso8601(pred_text)

    # If parsing fails for either side, fall back to string-based relaxed checks for SET,
    # otherwise we can’t relax-match.
    if g_parsed is None or p_parsed is None:
        if t == "SET":
            return _cmp_set_relaxed(gold_text, pred_text)
        return False

    if t == "DATE":
        # Any of year/month/day matches
        return _cmp_date_components(g_parsed, p_parsed)

    if t == "TIME":
        # Any of hour/minute/second matches
        return _cmp_time_components(g_parsed, p_parsed)

    if t in ("DATETIME", "DATE-TIME", "DATE_TIME"):
        # Any date OR time component matches
        return _cmp_datetime_components(g_parsed, p_parsed)

    if t == "DURATION":
        # Durations equal in total seconds (or fallback logic)
        return _cmp_duration_any_component(g_parsed, p_parsed)

    if t == "SET":
        # Any overlapping value in normalized token sets
        return _cmp_set_relaxed(gold_text, pred_text)

    # Unknown type → no relaxed match
    return False

def relaxed_accuracy(
    decoded_labels: List[str],
    decoded_preds: List[str],
) -> float:
    """
    Vectorized relaxed accuracy:
    - If strict match: correct
    - Else: relaxed_correct_single()
    """
    assert len(decoded_labels) == len(decoded_preds)
    correct = 0
    for y, p in zip(decoded_labels, decoded_preds):
        if p == y or relaxed_correct_single(y, p):
            correct += 1
    return correct / max(1, len(decoded_labels))


def compute_metrics(decoded_labels, decoded_preds, sim_threshold=0.75):
    # strict exact match
    strict_acc = sum(1 for p, y in zip(decoded_preds, decoded_labels) if p == y) / max(1, len(decoded_labels))
    relaxed_acc = relaxed_accuracy(decoded_labels, decoded_preds)

    return {"accuracy strict": strict_acc, "accuracy relaxed": relaxed_acc}

class TemporalDataset(Dataset):
    def __init__(self, path):        
        if path.endswith(".json"):
            with open(path, "r") as f:
                self.examples=[json.loads(line) for line in f]
    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return deepcopy(self.examples[i])
        