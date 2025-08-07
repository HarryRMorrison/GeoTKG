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

def reconstruct(tokens, original, geo_locations, evtm_locations):
    span2tok = []
    current_span = 0
    for i, tok in enumerate(tokens[:-1]):
        if "Ġ" == tok[0]:
            current_span += 1
        span2tok.append(current_span)
    
    span2tok = np.array(span2tok)
    unique_elements, counts = np.unique(span2tok, return_counts=True)
    counts = {el:co for el, co in zip(unique_elements, counts)}

    locs1 = {s:e for s, e in geo_locations}
    starts1 = list(locs1.keys())
    new_loc1 = []
    locs2 = {s:e for s, e in evtm_locations}
    starts2 = list(locs2.keys())
    new_loc2 = []

    out = []
    i = 0

    while i < len(tokens[:-1]):
        if tokens[i] == "<s>":
            i += 1
            continue
        elif i in starts1:
            starts1.remove(i)
            new_loc1.append(len(out))
            span, i = found_span_resolve(i, locs1, counts, span2tok, original)
        elif i in starts2:
            starts2.remove(i)
            new_loc2.append(len(out))
            span, i = found_span_resolve(i, locs2, counts, span2tok, original)
        else:
            span = span_resolve(original, span2tok, i, i+1)
            i += counts[span2tok[i]]
        out.append(span)
    out[-1] = out[-1].replace("</s>", "")
    return out, new_loc1, new_loc2