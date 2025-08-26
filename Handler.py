import spacy
import numpy as np
from pyrolite.util.time import Timescale
from spacy.matcher import Matcher
from spacy.util import filter_spans
from spacy.tokens import Span, Doc
from neo4j.graph import Node, Relationship
from globals import ID2LABEL_GEONER

AGENT_PREPS = {"by"}                    # passive agent
INSTRUMENT_PREPS = {"with", "using", "via", "by"}  # 'by' sometimes means instrument
LOCATION_PREPS = {"in", "at", "on", "into", "onto", "from", "to", "inside", "outside", "near"}

class Handler:
    def __init__(self):
        self.nlp = spacy.load("en_coreference_web_trf")
        self.nlp.add_pipe("experimental_span_resolver", after="coref")
        self.nlp.initialize()
        self.matcher = Matcher(self.nlp.vocab)
        pattern_no_space = [
            {"TEXT": {"REGEX": r"^~?\d+(\.\d+)?ma$"}}
        ]
        # with-space: “~1000 ma” or “1000.00 ma”
        pattern_with_space = [
            {"TEXT": {"REGEX": r"^~?\d+(\.\d+)?$"}},
            {"LOWER": "ma"}
        ]
        self.matcher.add("GEO_DATE", [pattern_no_space, pattern_with_space])

    def CorefResolve(self, text):
        doc = self.nlp(text)

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
    
    def geo_timescale_norm(self, text, geo_time_idxs):
        ts = Timescale()
        geo_times = []
        for loc in geo_time_idxs:
            min, max = ts.text2age(self.tokens[loc])
            geo_times.append([loc, (min, max)])

        doc = self.nlp(text)

        for _, start, end in self.matcher(doc):
            date = int(doc[start:end].text.lower().strip("~ma"))
            geo_times.append([start, (date, None)])
        
        return geo_times

    def get_locations(predictions, bi_map, id2lab=None):
        locations = []
        token_types = []
        Bs = bi_map.keys()
        i = 0
        while i < len(predictions[0]):
            if predictions[0][i].item() in Bs:
                start = i
                ent_type = predictions[0][i].item()
                i += 1
                while i < len(predictions[0]) and predictions[0][i] == bi_map[ent_type]:
                    i += 1
                locations.append([start, i])  # [start, end) format
                token_types.append(id2lab[ent_type]) if id2lab is not None else token_types.append(None)
            else:
                i += 1
        return locations, token_types if id2lab is None else locations

    @staticmethod
    def decode(text, enc):
        encodings = enc(text, padding=True, truncation=True, return_tensors="pt")
        decodings = enc.convert_ids_to_tokens(encodings["input_ids"][0])
        out = "".join(decodings)
        out = out.split("Ġ")
        return decodings, out
    
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
            span = Handler.span_resolve(original, span2tok, i, locs[i])
            i += (locs[i] - i)
        # If ner detected span is equal to actual span word
        else:
            span = original[span2tok[i]]
            i += 1
        return span, i

    def reconstruct(tokens, original, geo_time_locs, geo_entity_locs, event_locs):
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

        out = []
        i = 1

        while i < len(tokens[:-1]):
            # Check if geo ent
            if i in starts_geo_ent:
                starts_geo_ent.remove(i)
                new_geo_ent_locs.append(len(out))
                span, i = Handler.found_span_resolve(i, geo_entity_locs, counts, span2tok, original)
            # Check if geo time
            elif i in starts_geo_time:
                starts_geo_time.remove(i)
                new_geo_time_locs.append(len(out))
                span, i = Handler.found_span_resolve(i, geo_time_locs, counts, span2tok, original)
            # Check if event
            elif i in starts_events:
                starts_events.remove(i)
                new_events_locs.append(len(out))
                span, i = Handler.found_span_resolve(i, event_locs, counts, span2tok, original)
            else:
                span = Handler.span_resolve(original, span2tok, i, i+1)
                i += counts[span2tok[i]]
            out.append(span)

        out[-1] = out[-1].replace("</s>", "")
        return out, new_geo_ent_locs, new_geo_time_locs, new_events_locs

    def expand_span(token):
        """Return a nice argument span (full NP if available)."""
        # prefer the noun chunk containing token; otherwise minimal span
        for chunk in token.doc.noun_chunks:
            if token.i >= chunk.start and token.i < chunk.end:
                return chunk
        return token.subtree

    def get_event_anchor(event_span):
        """Pick a token to anchor the event: verb > event root > head."""
        root = event_span.root
        if root.pos_ == "VERB":
            return root
        # If the event is a noun (nominalized), try to find a related verb or use the head
        # e.g., "the explosion of the tank" -> root is NOUN; we still anchor on it
        return root

    def extract_roles_for_event(event_span):
        t = Handler.get_event_anchor(event_span)
        roles = {"agent": None, "patient": None, "instrument": None, "location": None}

        # --- Active voice agent/patient ---
        for child in t.children:
            if child.dep_ in ("nsubj", "nsubj:outer"):
                roles["agent"] = Handler.expand_span(child)
            if child.dep_ in ("dobj", "obj", "attr", "xcomp", "ccomp"):
                roles["patient"] = Handler.expand_span(child)

        # --- Passive voice agent/patient ---
        # nsubjpass (patient), agent via 'by'-phrase
        for child in t.children:
            if child.dep_ in ("nsubjpass",):
                roles["patient"] = roles["patient"] or Handler.expand_span(child)
            if child.dep_ == "agent" and any(gc.text.lower() in AGENT_PREPS for gc in child.children if gc.dep_ == "case"):
                # agent subtree: "by X"
                pobj = next((gc for gc in child.children if gc.dep_ == "pobj"), None)
                roles["agent"] = roles["agent"] or Handler.expand_span(pobj or child)

        # --- Prepositional arguments for instrument/location ---
        for prep in (c for c in t.children if c.dep_ == "prep"):
            head_lemma = prep.lemma_.lower()
            pobj = next((gc for gc in prep.children if gc.dep_ in ("pobj", "pcomp")), None)
            if not pobj:
                continue
            if head_lemma in INSTRUMENT_PREPS:
                roles["instrument"] = roles["instrument"] or Handler.expand_span(pobj)
            if head_lemma in LOCATION_PREPS:
                roles["location"] = roles["location"] or Handler.expand_span(pobj)

        # --- Nominalized patterns: agent via 'by', patient via 'of' ---
        if t.pos_ == "NOUN":
            for prep in (c for c in t.children if c.dep_ == "prep"):
                head_lemma = prep.lemma_.lower()
                pobj = next((gc for gc in prep.children if gc.dep_ in ("pobj", "pcomp")), None)
                if not pobj:
                    continue
                if head_lemma == "of":
                    roles["patient"] = roles["patient"] or Handler.expand_span(pobj)
                if head_lemma == "by":
                    roles["agent"] = roles["agent"] or Handler.expand_span(pobj)
                if head_lemma in LOCATION_PREPS:
                    roles["location"] = roles["location"] or Handler.expand_span(pobj)
                if head_lemma in INSTRUMENT_PREPS:
                    roles["instrument"] = roles["instrument"] or Handler.expand_span(pobj)

        # Return nice strings (full spans)
        out = {}
        for k, v in roles.items():
            if v is None:
                out[k] = None
            else:
                if hasattr(v, "__iter__") and not isinstance(v, str):  # generator from token.subtree
                    v = list(v)
                    out[k] = v[0].doc[v[0].i : v[-1].i + 1].text
                else:
                    out[k] = v.text
        return out

    def extract_srl(doc):
        """Find EVENT entities and extract roles for each."""
        results = []
        for ev in doc.ents:
            if ev.label_ == "EVENT":
                roles = Handler.extract_roles_for_event(ev)
                results.append({
                    "event_text": ev.text,
                    "event_anchor": Handler.get_event_anchor(ev).lemma_,
                    "roles": roles
                })
        return results

    def create_nodes_extract(self, text, geo_ner_output, event_locs, enc):
        decodings, out = Handler.decode(text, enc)
        ent_locs, types = Handler.get_locations(geo_ner_output, bi_map={0:6, 1:7, 2:8, 3:9, 4:10}, id2lab=ID2LABEL_GEONER)
        geotime_locs = Handler.get_locations(geo_ner_output, bi_map={5:11})
        event_locs = Handler.get_locations(event_locs, bi_map={2:6})
        recon, ent_locs, geotime_locs, event_locs = Handler.reconstruct(decodings, out, geotime_locs, ent_locs, event_locs)
        
        doc = Doc(self.nlp.vocab, recon)
        spans = []
        for start, end, type_ in zip(ent_locs, types):
            span = Span(doc, start, end, label=type_)
            spans.append(span)
        for start, end in geotime_locs:
            span = Span(doc, start, end, label="TIMESCALE")
            spans.append(span)
        for start, end in event_locs:
            span = Span(doc, start, end, label="EVENT")
            spans.append(span)

        
        doc.ents = filter_spans(list(doc.ents) + spans)
        roles = Handler.extract_srl(doc)

        norm_geotimes = self.geo_timescale_norm(text, geotime_locs)
        # Link geo times to entites some how -> do it after nodes are made and consolidated so that geotime nodes are normalised

        nodes = []

        for instance in roles:
            print(instance)
            nodes.append(Node("Event", 
                              event=instance['event_text'],
                              event_anchor=instance['event_anchor'],
                              agent=instance['roles']['agent'], 
                              patient=instance['roles']['patient'],
                              location=instance['roles']['location'],
                              instrument=instance['roles']['instrument'],
                              ))
        return nodes, norm_geotimes