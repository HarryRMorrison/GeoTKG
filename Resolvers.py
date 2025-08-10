import spacy
import numpy as np

def CorefResolve(text):
    nlp = spacy.load("en_coreference_web_trf")
    nlp.add_pipe("experimental_span_resolver", after="coref")
    nlp.initialize()
    doc = nlp(text)

        # 1. Build a map: (span_start, span_end) -> representative text
    span_reps = {}
    for key, spans in doc.spans.items():
        if key.startswith("coref_clusters"):
            main = spans[0]  # first mention = representative
            for span in spans:
                span_reps[(span.start, span.end)] = main.text

    # 2. Walk through tokens, replacing spans as a unit
    resolved_tokens = []
    i = 0
    while i < len(doc):
        replaced = False
        for (start, end), main_text in span_reps.items():
            if i == start:
                # emit the full replacement once
                resolved_tokens.append(main_text)
                i = end  # skip past the span
                replaced = True
                break
        if not replaced:
            # no span starts here → keep the original token
            resolved_tokens.append(doc[i].text)
            i += 1

    return " ".join(resolved_tokens)

def span_resolve(original, span2tok, s, e):
    span = []
    for j in range(s, e):
        span_id = span2tok[j]
        span.append(original[span_id])
    return " ".join(span)

def found_span_resolve(i, locs, counts, span2tok, original):   
    # If ner detected span is shorter than actual span word
    if locs[i] - i < counts[span2tok[i]]:
        span = original[span2tok[i]]
        i += counts[span2tok[i]]
    # If ner detected span is larger than actual span word
    elif locs[i] - i > counts[span2tok[i]]:
        span = span_resolve(original, span2tok, i, locs[i])
        i += (locs[i] - i)
    # If ner detected span is equal to actual span word
    else:
        span = original[span2tok[i]]
        i += 1
    return span, i

def reconstruct(tokens, original, geo_time_locs, geo_entity_locs, event_locs, timex_locs):
    span2tok = []
    current_span = 0
    for i, tok in enumerate(tokens[:-1]):
        if "Ġ" == tok[0]:
            current_span += 1
        span2tok.append(current_span)
    
    span2tok = np.array(span2tok)
    unique_elements, counts = np.unique(span2tok, return_counts=True)
    counts = {el:co for el, co in zip(unique_elements, counts)}

    geo_entity_locs = {s:e for s, e in geo_entity_locs}
    starts_geo_ent = list(geo_entity_locs.keys())
    new_geo_ent_locs = []

    geo_time_locs = {s:e for s, e in geo_time_locs}
    starts_geo_time = list(geo_time_locs.keys())
    new_geo_time_locs = []

    event_locs = {s:e for s, e in event_locs}
    starts_events = list(event_locs.keys())
    new_events_locs = []

    timex_locs = {s:e for s, e in timex_locs}
    starts_timexs = list(timex_locs.keys())
    new_timex_locs = []

    out = []
    i = 1

    while i < len(tokens[:-1]):
        # Check if geo ent
        if i in starts_geo_ent:
            starts_geo_ent.remove(i)
            new_geo_ent_locs.append(len(out))
            span, i = found_span_resolve(i, geo_entity_locs, counts, span2tok, original)
        # Check if geo time
        elif i in starts_geo_time:
            starts_geo_time.remove(i)
            new_geo_time_locs.append(len(out))
            span, i = found_span_resolve(i, geo_time_locs, counts, span2tok, original)
        # Check if event
        elif i in starts_events:
            starts_events.remove(i)
            new_events_locs.append(len(out))
            span, i = found_span_resolve(i, event_locs, counts, span2tok, original)
        # Check if timex
        elif i in starts_timexs:
            starts_timexs.remove(i)
            new_timex_locs.append(len(out))
            span, i = found_span_resolve(i, timex_locs, counts, span2tok, original)
        else:
            span = span_resolve(original, span2tok, i, i+1)
            i += counts[span2tok[i]]
        out.append(span)

    out[-1] = out[-1].replace("</s>", "")
    return out, new_geo_ent_locs, new_geo_time_locs, new_events_locs, new_timex_locs