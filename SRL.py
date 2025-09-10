import spacy
from pyrolite.util.time import Timescale
from spacy.matcher import Matcher
from spacy.util import filter_spans
from spacy.tokens import Span, Doc, Token

class SRL:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_trf")
        self.matcher = Matcher(self.nlp.vocab)
        # 1) No-space case: "1000ma", "~1,234.5MA", "7Ma", etc.  (single token)
        pattern_no_space = [
            {"TEXT": {"REGEX": r'(?i)^[~∼≈]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?ma$'}}
        ]

        # 2) Space case: "1000 ma", "~1,234.5 Ma"  (two tokens)
        pattern_with_space = [
            {"TEXT": {"REGEX": r'^[~∼≈]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?$'}},
            {"LOWER": "ma"}  # case-insensitive match for the unit token
        ]

        self.matcher.add("TIMESCALE", [pattern_no_space, pattern_with_space])
    
    def geo_timescale_norm(text):
        ts = Timescale()
        try:
            date = int(text.lower().strip("~ma"))
            return (date, None)
        except:
            min, max = ts.text2age(text)
            return (min, max)

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
    
    # 1) Return a Span instead of plain text
    def noun_phrase_span(tok: Token) -> Span:
        """Return a Span NP for a token (prefer its noun_chunk; fallback to subtree span)."""
        doc = tok.doc
        for nc in doc.noun_chunks:
            if nc.start <= tok.i < nc.end:
                return nc
        return doc[tok.left_edge.i : tok.right_edge.i + 1]

    def span_check_convert_timescale(sp: Span) -> bool:
        """True if any token or entity inside the span is labeled TIMESCALE."""
        norm_geo_times = []
        # (a) token-level ent_type_
        for t in sp:
            if t.ent_type_ == "TIMESCALE":
                norm_geo_times.append((t,SRL.geo_timescale_norm(t.text)))
        return norm_geo_times

    def extract_subject_object(event_node: Token):
        subjects_sp, objects_sp = [], []

        # voice detection
        is_passive = any(c.dep_ in ("auxpass", "nsubjpass", "csubjpass") for c in event_node.children)

        # ---- subjects ----
        if not is_passive:
            for c in event_node.children:
                if c.dep_ in ("nsubj", "csubj"):
                    subjects_sp.append(SRL.noun_phrase_span(c))
        else:
            for ag in (c for c in event_node.children if c.dep_ == "agent"):
                pobj = next((gc for gc in ag.children if gc.dep_ == "pobj"), None)
                if pobj:
                    subjects_sp.append(SRL.noun_phrase_span(pobj))

        if not subjects_sp and event_node.dep_ == "conj":
            for c in event_node.head.children:
                if c.dep_ in ("nsubj", "csubj"):
                    subjects_sp.append(SRL.noun_phrase_span(c))

        # ---- objects & complements ----
        for c in event_node.children:
            if c.dep_ in ("dobj", "obj", "attr", "oprd", "dative"):
                objects_sp.append(SRL.noun_phrase_span(c))
            elif c.dep_ == "prep":
                pobj = next((gc for gc in c.children if gc.dep_ == "pobj"), None)
                if pobj:
                    objects_sp.append(SRL.noun_phrase_span(pobj))

        if is_passive:
            for c in event_node.children:
                if c.dep_ in ("nsubjpass", "csubjpass"):
                    objects_sp.append(SRL.noun_phrase_span(c))

        # de-dup by (start,end)
        def dedup_spans(spans):
            seen = set()
            out = []
            for sp in spans:
                key = (sp.start, sp.end)
                if key not in seen:
                    seen.add(key)
                    out.append(sp)
            return out

        subjects_sp = dedup_spans(subjects_sp)
        objects_sp  = dedup_spans(objects_sp)

        # Return structured info (text + TIMESCALE flag)
        subjects = [{"text": sp.text, "timescale": SRL.span_check_convert_timescale(sp)} for sp in subjects_sp]
        objects  = [{"text": sp.text, "timescale": SRL.span_check_convert_timescale(sp)} for sp in objects_sp]

        return subjects, objects


    # Single example at a time
    def __call__(self, recon, ent_locs, geotime_locs, event_locs, timex_locs):
        char_spans, text = SRL.char_locs(recon, ent_locs, geotime_locs, event_locs)

        # 1) Tokenize only (no pipeline yet)
        doc = self.nlp.make_doc(text)

        # 2) Create spans & merge them into single tokens
        spans = []
        for s, e, label in char_spans:
            span = doc.char_span(s, e, label=label, alignment_mode="expand")
            if span is not None:
                spans.append(span)
        
        matches = self.matcher(doc)
        label_id = doc.vocab.strings["TIMESCALE"]  # integer hash for the label
        for _, s, e in matches:
            spans.append(Span(doc, s, e, label=label_id))

        spans = filter_spans(spans)  # avoid overlaps
        # Keep the entity labels and collapse to single tokens
        with doc.retokenize() as retok:
            for sp in spans:
                retok.merge(sp, attrs={"ENT_TYPE": sp.label_})
        
        with self.nlp.select_pipes(disable=["ner"]):
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
    