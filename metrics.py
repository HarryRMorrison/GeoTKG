from sklearn.metrics import f1_score
import datetime as _dt
from typing import List, Tuple, Optional
from copy import deepcopy
from post_processing import get_ee_temprels

def get_ner_scores(strict, relaxed):
    strict_text_match = [1 if item[0]=="strict" else 0 for item in strict]
    relaxed_text_match = [1 if item[0]=="relaxed" else 0 for item in relaxed]
    type_match = [1 if item[1]==True else 0 for item in relaxed]

    return {"strict_text":f1_score([1]*len(strict_text_match), strict_text_match),
            "relaxed_text":f1_score([1]*len(strict_text_match), relaxed_text_match),
            "type":f1_score([1]*len(strict_text_match), type_match)}

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

def relaxed_correct_single(g: str, p: str) -> bool:
    """
    Strict equality first; if not equal, apply relaxed rule per ti_type.
    ti_type ∈ {"DATE","TIME","DATETIME","DURATION","SET"} (case-insensitive).
    """
    # Strict exact match first (you can move strict to your main metric if preferred)
    if g == p:
        return True
    # If parsing fails for either side, fall back to string-based relaxed checks for SET,
    # otherwise we can’t relax-match.
    if g is None and p is None:
        return True
    elif g is None or p is None:
        return False

    return _cmp_date_components(g, p) or _cmp_time_components(g, p) or _cmp_datetime_components(g, p) or _cmp_duration_any_component(g, p)

def text_match(truth_text, pred_text):
    if pred_text is None and type(truth_text)==str:
        return False

    text_match = False
    if truth_text == pred_text:
        text_match = "strict"
    elif truth_text in pred_text or pred_text in truth_text:
        text_match = "relaxed"
    return text_match

def sample_ner_compare(truths, preds, geo_ner=False):
    preds_copy = deepcopy(preds)
    strict_results = []
    for instance in truths:
        if geo_ner==False and instance['type'] != "EVENT" and instance['id']==0:
            continue
        text_match = False
        type_match = False
        for pred in preds_copy:
            if instance['text'] == pred[0]:
                text_match = "strict"
                type_match = instance['type'] == pred[1]
                preds_copy.remove(pred)
                strict_results.append((text_match, type_match, instance['type']))
                break
        if text_match == False:
            strict_results.append((text_match, type_match, instance['type']))

    preds_copy = deepcopy(preds)
    relaxed_results = []
    for instance in truths:
        if geo_ner==False and instance['type'] != "EVENT" and instance['id']==0:
            continue
        text_match = False
        type_match = False
        for pred in preds_copy:
            if instance['text'] in pred[0] or pred[0] in instance['text']:
                text_match = "relaxed"
                type_match = instance['type'] == pred[1]
                preds_copy.remove(pred)
                relaxed_results.append((text_match, type_match, instance['type']))
                break
        if text_match == False:
            relaxed_results.append((text_match, type_match, instance['type']))
    
    return strict_results, relaxed_results

def sample_quintuple_compare(truths, preds):
    preds_copy = deepcopy(preds)
    strict_results = []
    for truth in truths:
        matched = False
        for pred in preds_copy:
            if truth["event"]==pred["event"] and truth["s_time"]==pred["s_time"] and truth["e_time"]==pred["e_time"]:
                strict_results.append(1)
                preds_copy.remove(pred)
                matched = True
                break
        if matched == False:
            strict_results.append(0)

    preds_copy = deepcopy(preds)
    relaxed_results = []
    for truth in truths:
        matched = False
        for pred in preds_copy:
            if text_match(truth["event"], pred['event'])!=False and relaxed_correct_single(truth["s_time"], pred['s_time']) and relaxed_correct_single(truth["e_time"], pred['e_time']):
                relaxed_results.append(1)
                preds_copy.remove(pred)
                matched = True
                break
        if matched == False:
            relaxed_results.append(0)
    return strict_results, relaxed_results

def sample_triple_compare(truths, preds):
    preds_copy = deepcopy(preds)
    preds_copy = get_ee_temprels(preds_copy, exhaustive=True)
    results = []
    for truth in truths:
        matched = False
        for pred in preds_copy:
            if text_match(truth[0], pred[0])!=False and truth[1]==pred[1] and text_match(truth[2], pred[2])!=False:
                results.append(1)
                preds_copy.remove(pred)
                matched = True
                break
        if matched == False:
            results.append(0)
    return results