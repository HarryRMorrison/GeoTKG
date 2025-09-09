import spacy

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