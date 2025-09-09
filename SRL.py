import spacy
from pyrolite.util.time import Timescale
from spacy.matcher import Matcher
from spacy.util import filter_spans
from spacy.tokens import Span, Doc, Token

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
    def __call__(self, text, geo_ents, geo_times, events, timex_locs):
        decodings, out = SRL.decode(text)
        recon, ent_locs, geotime_locs, event_locs, timex_locs = SRL.reconstruct(decodings, out, geo_times, geo_ents, events, timex_locs)
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

        return event_subject_object, recon, timex_locs
    