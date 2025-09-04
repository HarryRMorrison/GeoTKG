import spacy
import numpy as np
from pyrolite.util.time import Timescale
from spacy.matcher import Matcher
from spacy.util import filter_spans
from spacy.tokens import Span, Doc, Token
from neo4j.graph import Node, Relationship
from transformers import AutoTokenizer
from globals import ID2LABEL_GEONER

AGENT_PREPS = {"by"}                    # passive agent
INSTRUMENT_PREPS = {"with", "using", "via", "by"}  # 'by' sometimes means instrument
LOCATION_PREPS = {"in", "at", "on", "into", "onto", "from", "to", "inside", "outside", "near"}
TOKENIZER = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)

class CorefResolver:
    def __init__(self):
        self.nlp = spacy.load("en_coreference_web_trf")
        self.nlp.add_pipe("experimental_span_resolver", after="coref")
        self.nlp.initialize()

    def __call__(self, text):
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

class SRL:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_trf")
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

    @staticmethod
    def decode(text, enc=TOKENIZER):
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
            span = SRL.span_resolve(original, span2tok, i, locs[i])
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

        geo_entity_types = {s:t for s, e, t in geo_entity_locs}
        geo_entity_locs = {s:e for s, e, t in geo_entity_locs}
        starts_geo_ent = list(geo_entity_locs.keys())
        new_geo_ent_locs = []
        new_geo_ent_types = []

        geo_time_locs = {s:e for s, e, t in geo_time_locs}
        starts_geo_time = list(geo_time_locs.keys())
        new_geo_time_locs = []

        event_locs = {s:e for s, e, t in event_locs}
        starts_events = list(event_locs.keys())
        new_events_locs = []

        out = []
        i = 1

        while i < len(tokens[:-1]):
            # Check if geo ent
            if i in starts_geo_ent:
                starts_geo_ent.remove(i)
                new_geo_ent_types.append(geo_entity_types[i])
                new_geo_ent_locs.append(len(out))
                span, i = SRL.found_span_resolve(i, geo_entity_locs, counts, span2tok, original)
            # Check if geo time
            elif i in starts_geo_time:
                starts_geo_time.remove(i)
                new_geo_time_locs.append(len(out))
                span, i = SRL.found_span_resolve(i, geo_time_locs, counts, span2tok, original)
            # Check if event
            elif i in starts_events:
                starts_events.remove(i)
                new_events_locs.append(len(out))
                span, i = SRL.found_span_resolve(i, event_locs, counts, span2tok, original)
            else:
                span = SRL.span_resolve(original, span2tok, i, i+1)
                i += counts[span2tok[i]]
            out.append(span)

        out[-1] = out[-1].replace("</s>", "")
        return out, list(zip(new_geo_ent_locs,new_geo_ent_types)), list(zip(new_geo_time_locs,["TIMESCALE"]*len(new_geo_time_locs))), list(zip(new_events_locs, ["EVENT"]*len(new_events_locs)))

    def char_locs(recon, ent_locs, geotime_locs, event_locs):
        text = ""
        locs = {loc:type_ for loc, type_ in ent_locs + geotime_locs + event_locs}
        out = []
        for idx, span in enumerate(recon):
            if idx in locs:
                s = len(text)
                text += f" {span}"
                out.append((s, len(text), locs[idx]))
            else:
                if idx==0:
                    text += span
                else:
                    text += f" {span}"
        return out, text

    def noun_phrase(tok: Token) -> str:
        """Return a readable NP for a token (prefer its noun_chunk; fallback to subtree span)."""
        doc = tok.doc
        for nc in doc.noun_chunks:
            if nc.start <= tok.i < nc.end:
                return nc.text
        span = doc[tok.left_edge.i : tok.right_edge.i + 1]
        return span.text

    def extract_subject_object(event_node: Token):
        """Return (subjects, objects) lists for a verb/event root token."""
        subjects, objects = [], []

        # voice detection
        is_passive = any(c.dep_ in ("auxpass", "nsubjpass", "csubjpass") for c in event_node.children)

        # ---- subjects ----
        if not is_passive:
            for c in event_node.children:
                if c.dep_ in ("nsubj", "csubj"):
                    subjects.append(SRL.noun_phrase(c))
        else:
            # semantic subject via passive agent: agent -> pobj
            for ag in (c for c in event_node.children if c.dep_ == "agent"):
                pobj = next((gc for gc in ag.children if gc.dep_ == "pobj"), None)
                if pobj:
                    subjects.append(SRL.noun_phrase(pobj))

        # backoff: coordinated verbs often share subject with head
        if not subjects and event_node.dep_ == "conj":
            for c in event_node.head.children:
                if c.dep_ in ("nsubj", "csubj"):
                    subjects.append(SRL.noun_phrase(c))

        # ---- objects & complements ----
        for c in event_node.children:
            if c.dep_ in ("dobj", "obj", "attr", "oprd"):
                objects.append(SRL.noun_phrase(c))
            elif c.dep_ in ("ccomp", "xcomp"):
                # You can use the whole clause; tweak as needed
                objects.append(c.subtree.text)  # or c.text
            elif c.dep_ == "dative":  # indirect object
                objects.append(SRL.noun_phrase(c))
            elif c.dep_ == "prep":
                pobj = next((gc for gc in c.children if gc.dep_ == "pobj"), None)
                if pobj:
                    objects.append(SRL.noun_phrase(pobj))

        # In passives, the semantic "object" is often the nsubjpass
        if is_passive:
            for c in event_node.children:
                if c.dep_ in ("nsubjpass", "csubjpass"):
                    objects.append(SRL.noun_phrase(c))

        # de-dup while preserving order
        seen = set()
        subjects = [s for s in subjects if not (s in seen or seen.add(s))]
        seen.clear()
        objects = [o for o in objects if not (o in seen or seen.add(o))]

        return subjects, objects

    # Single example at a time
    def __call__(self, text, geo_ents, geo_times, events):
        decodings, out = SRL.decode(text)
        recon, ent_locs, geotime_locs, event_locs = SRL.reconstruct(decodings, out, geo_times, geo_ents, events)
        event_char_spans, text = SRL.char_locs(recon, ent_locs, geotime_locs, event_locs)

        # 1) Tokenize only (no pipeline yet)
        doc = self.nlp.make_doc(text)

        # 2) Create spans & merge them into single tokens
        spans = []
        for s, e, label in event_char_spans:
            span = doc.char_span(s, e, label=label, alignment_mode="expand")
            if span is not None:
                spans.append(span)

        spans = filter_spans(spans)  # avoid overlaps
        # Keep the entity labels and collapse to single tokens
        with doc.retokenize() as retok:
            for sp in spans:
                retok.merge(sp, attrs={"ENT_TYPE": sp.label_})

        doc = self.nlp(doc)

        # Collect merged EVENT tokens (now single tokens with ENT_TYPE='EVENT')
        event_tokens = [t for t in doc if t.ent_type_ == "EVENT"]

        event_subject_object = []
        for event in event_tokens:
            root = event.root if isinstance(event, Span) else event
            subj, obj = SRL.extract_subject_object(root)
            event_subject_object.append({
                "event": event.text,
                "subject": subj if subj else None,
                "object": obj if obj else None
            })

        return event_subject_object
    
class KGConstructor:
    def __init__(self):
        pass

    def __call__(self, roles, temprels, normtimes):
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
        return nodes