import spacy
from typing import List
from spacy.tokens import Doc, Span, Token
from spacy.matcher import Matcher
from spacy.util import filter_spans
from pyrolite.util.time import Timescale
from copy import deepcopy
import regex as re
from spacy.util import filter_spans

class DependencyParser:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_trf")
        coref_src = spacy.load("en_coreference_web_trf")

        self.nlp.add_pipe("coref", source=coref_src)

        resolver_name = "span_resolver" if "span_resolver" in coref_src.pipe_names else "experimental_span_resolver"
        if resolver_name not in self.nlp.pipe_names:
            self.nlp.add_pipe(resolver_name, source=coref_src, after="coref")

        if "span_cleaner" in coref_src.pipe_names and "span_cleaner" not in self.nlp.pipe_names:
            self.nlp.add_pipe("span_cleaner", source=coref_src, after=resolver_name)

        self.timescale_matcher = Matcher(self.nlp.vocab)
        pattern_no_space = [
            {"TEXT": {"REGEX": r"(?i)^[~∼≈]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?ma$"}}
        ]
        pattern_with_space = [
            {"TEXT": {"REGEX": r"^[~∼≈]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?$"}},
            {"LOWER": "ma"},
        ]
        self.timescale_matcher.add("TIMESCALE", [pattern_no_space, pattern_with_space])
        self.geotime_coverter = Timescale()

        self.L_TIMESCALE = self.nlp.vocab.strings["TIMESCALE"]

    # --------------------------
    # Utility helpers
    # --------------------------
    @staticmethod
    def clean_unicode(s: str):
        # remove all Unicode punctuation
        s = re.sub(r'[^\p{L}\p{M}\s]+', ' ', s)
        return re.sub(r'\s+', ' ', s).strip()

    @staticmethod
    def geo_timescale_norm(ts: Timescale, text: str):
        s = text.strip().lower().replace(" ", "")
        s = s.replace("∼", "~").replace("≈", "~")
        if s.endswith("ma"):
            try:
                # tolerate leading "~"
                s = s.lstrip("~").rstrip("ma").replace(",", "")
                val = float(re.sub(r'[^0-9.,]+', '', s))
                return (val, None)
            except Exception:
                pass
        try:
            mn, mx = ts.text2age(DependencyParser.clean_unicode(text))
            return (mn, mx)
        except Exception:
            return (None, None)

    @staticmethod
    def noun_phrase_span(tok: Token) -> Span:
        """Return a Span NP for a token (prefer its noun_chunk; fallback to subtree span)."""
        doc = tok.doc
        for nc in doc.noun_chunks:
            if nc.start <= tok.i < nc.end:
                return nc
        return doc[tok.left_edge.i : tok.right_edge.i + 1]

    def _span_timescales_with_norms(self, sp: Span):
        # collect entity spans labelled TIMESCALE that fall inside sp
        candidates = [e for e in sp.doc.ents if e.label_ == "TIMESCALE" and sp.start <= e.start < sp.end]
        # keep longest / earliest
        clean = filter_spans(candidates)
        out = []
        for e in clean:
            mn, mx = self.geo_timescale_norm(self.geotime_coverter, e.text)
            out.append({"text": e.text, "norm_min": mn, "norm_max": mx})
        return out

    @staticmethod
    def _dedup_spans(spans: List[Span]):
        seen = set()
        out = []
        for sp in spans:
            key = (sp.start, sp.end)
            if key not in seen:
                seen.add(key)
                out.append(sp)
        return out
    
    @staticmethod
    def _extract_subject_object(event_node: Token):
        subjects_sp: List[Span] = []
        objects_sp: List[Span] = []

        # passive voice?
        is_passive = any(c.dep_ in ("auxpass", "nsubjpass", "csubjpass") for c in event_node.children)

        # subjects
        if not is_passive:
            for c in event_node.children:
                if c.dep_ in ("nsubj", "csubj"):
                    subjects_sp.append(DependencyParser.noun_phrase_span(c))
        else:
            # BY-phrase agent in passive
            for ag in (c for c in event_node.children if c.dep_ == "agent"):
                pobj = next((gc for gc in ag.children if gc.dep_ == "pobj"), None)
                if pobj:
                    subjects_sp.append(DependencyParser.noun_phrase_span(pobj))

        # inherit subject from coordinated head (X conj event_node)
        if not subjects_sp and event_node.dep_ == "conj":
            for c in event_node.head.children:
                if c.dep_ in ("nsubj", "csubj"):
                    subjects_sp.append(DependencyParser.noun_phrase_span(c))

        # objects & complements
        for c in event_node.children:
            if c.dep_ in ("dobj", "obj", "attr", "oprd", "dative"):
                objects_sp.append(DependencyParser.noun_phrase_span(c))
            elif c.dep_ == "prep":
                pobj = next((gc for gc in c.children if gc.dep_ == "pobj"), None)
                if pobj:
                    objects_sp.append(DependencyParser.noun_phrase_span(pobj))

        if is_passive:
            for c in event_node.children:
                if c.dep_ in ("nsubjpass", "csubjpass"):
                    objects_sp.append(DependencyParser.noun_phrase_span(c))

        return DependencyParser._dedup_spans(subjects_sp), DependencyParser._dedup_spans(objects_sp)
    
    # --------------------------
    # Text + indices construction
    # --------------------------
    @staticmethod
    def get_char_locs(input_tokens, input_word_ids, entity_locs):
        """
        entity_locs rows are [span_text, start_tok, end_tok_exclusive, label]
        tokens are RoBERTa subwords (specials already stripped)
        """
        tokens = list(input_tokens)
        word_ids = list(input_word_ids)
        assert len(tokens) == len(word_ids), "tokens and word_ids must be same length"

        # If any None remain, drop them and remap indices
        idx_map = {}  # old_idx -> new_idx
        kept_tokens, kept_wids = [], []
        for old_i, (tok, wid) in enumerate(zip(tokens, word_ids)):
            if wid is None:
                continue
            idx_map[old_i] = len(kept_tokens)
            kept_tokens.append(tok)
            kept_wids.append(wid)

        tokens, word_ids = kept_tokens, kept_wids

        # Build text with spaces only when word_id changes
        reconstruction = []
        token_char_locs = {}
        prev_wid = None
        pos = 0

        for i, (tok, wid) in enumerate(zip(tokens, word_ids)):
            # space before a *new word* (wid change), not after
            if prev_wid is not None and wid != prev_wid:
                reconstruction.append(" ")
                pos += 1
            start = pos
            reconstruction.append(tok)
            pos += len(tok)
            end = pos
            token_char_locs[i] = (start, end)
            prev_wid = wid

        text = "".join(reconstruction)

        # Remap entity token indices if we dropped any None positions above
        def remap_index(old_i):
            return idx_map.get(old_i, old_i)

        # Convert token spans to char spans (end is exclusive)
        char_locs = []
        for _, s_tok_old, e_tok_old, label in entity_locs:
            s_tok = remap_index(s_tok_old)
            e_tok = remap_index(e_tok_old - 1)  # last token in span after remap
            if s_tok not in token_char_locs or e_tok not in token_char_locs or s_tok > e_tok:
                continue  # skip malformed
            s_char = token_char_locs[s_tok][0]
            e_char = token_char_locs[e_tok][1]
            char_locs.append((s_char, e_char, label))

        return char_locs, text
    
    def __call__(self, tokens, word_ids, geo_ents, geo_times, events):
        # --- RETENTION: build event-only char targets (id -> (s_char, e_char)) ---
        events_copy = deepcopy(events)
        event_rows = [[span, s, e, "EVENT"] for idx, (span, (s, e, _)) in enumerate(events_copy)]
        event_char_spans, _ = DependencyParser.get_char_locs(tokens, word_ids, event_rows)
        event_targets = [(s, e) for (s, e, _) in event_char_spans]  # list aligned to input events

        unpacked = [[span, s, e, ty] for entity_set in [events, geo_ents, geo_times] for span, (s, e, ty) in entity_set]
        # 1) Build text and character spans from token indices
        char_spans, text = DependencyParser.get_char_locs(tokens, word_ids, unpacked)

        # 2) Make a bare Doc (keep our entities intact; we won't run NER)
        doc = self.nlp.make_doc(text)

        # 3) Convert to spaCy spans and also auto-detect TIMESCALE mentions with the matcher
        spans: List[Span] = []
        for s, e, label in char_spans:
            sp = doc.char_span(s, e, label=label, alignment_mode="expand")
            if sp is not None:
                spans.append(sp)

        matches = self.timescale_matcher(doc)
        for _, s, e in matches:
            spans.append(Span(doc, s, e, label=self.L_TIMESCALE))

        # De-overlap & assign as entities
        spans = filter_spans(spans)
        doc.set_ents(spans, default="unmodified")

        # 4) Run the rest of the pipeline (disable NER so ents are preserved)
        with self.nlp.select_pipes(disable=["ner"]):
            doc = self.nlp(doc)

         # 5) Propagate entity label from cluster's main mention to other mentions
        new_ents: List[Span] = list(doc.ents)

        # Normalize clusters to a list[SpanGroup]
        clusters = []
        if "coref_clusters" in doc.spans:
            val = doc.spans["coref_clusters"]
            clusters = val if isinstance(val, list) else [val]
        else:
            for k, v in doc.spans.items():
                if k.startswith("coref_clusters"):
                    clusters.append(v)

        for cluster in clusters:
            cluster_spans = list(cluster)
            if not cluster_spans:
                continue
            main = cluster_spans[0]

            # Find label from overlapping entity on the main mention
            main_label_id = 0
            if not main_label_id:
                continue  # don't create label=0 ents

            for e in doc.ents:
                if e.start <= main.start < e.end:
                    main_label_id = e.label
                    break

            for m in cluster_spans[1:]:
                overlap_existing = any(e.start <= m.start < e.end for e in doc.ents)
                if not overlap_existing and any(tok.is_alpha for tok in m):
                    new_ents.append(Span(doc, m.start, m.end, label=main_label_id))

        new_ents = filter_spans(new_ents)
        doc.set_ents(new_ents, default="unmodified")

        # Check final event retention
        retention = [-100]*len(event_rows)

        # 6) For each EVENT entity, extract subject(s) and object(s) using dependency parse
        results = []
        so_index = 0
        all_evs = [e for e in doc.ents if e.label_ == "EVENT"]
        for ev in all_evs:
            head: Token = ev.root  # head token of the event span
            subs, objs = self._extract_subject_object(head)
            if (ev.start_char, ev.end_char) in event_targets:
                ev_indx = event_targets.index((ev.start_char, ev.end_char))
                retention[ev_indx] = so_index

            subs_json = [
                {"text": sp.text, "timescale": self._span_timescales_with_norms(sp)} for sp in subs
            ] or None
            objs_json = [
                {"text": sp.text, "timescale": self._span_timescales_with_norms(sp)} for sp in objs
            ] or None

            results.append(
                {
                    "event": ev.text,
                    "span": (ev.start_char, ev.end_char),
                    "subject": subs_json,
                    "object": objs_json,
                }
            )
            so_index+=1

        return results, retention
            