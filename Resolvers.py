import spacy

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

def reconstruct(tokens, locations):
    out = []
    multi = False

    if type(locations) == tuple and len(locations)==2:
        locs1 = {s:e for s, e in locations[0]}
        starts1 = list(locs1.keys())
        new_loc1 = []
        locs2 = {s:e for s, e in locations[1]}
        starts2 = list(locs2.keys())
        new_loc2 = []
        multi = True
    else:
        locs1 = {s:e for s, e in locations}
        starts1 = list(locs1.keys())
        new_loc1 = []

    i = 0
    while i < len(tokens):
        if tokens[i] in ["<s>", "</s>"]:
            i += 1
            continue
        elif i in starts1:
            starts1.remove(i)
            new_loc1.append(len(out))
            out.append("".join([tokens[j] for j in range(i, locs1[i]+1)]))
        elif multi and i in starts2:
            starts2.remove(i)
            new_loc2.append(len(out))
            out.append("".join([tokens[j] for j in range(i, locs2[i]+1)]))
        else:
            out.append(tokens[i])
        i += 1
    if multi:
        return out, new_loc1, new_loc2
    else:
        return out, new_loc1