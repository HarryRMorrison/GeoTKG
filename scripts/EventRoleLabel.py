import spacy
from spacy.tokens import Doc
import neo4j

class EventRoleLabel:
    def __init__(self, resolved_text):
        self.resolved_text = resolved_text

    def object_subject_extract(self, pretokenized):
        nlp = spacy.load("en_core_web_trf")
        doc = Doc(nlp.vocab, pretokenized)
        toks = []
        for i, sent in enumerate(doc):
            print(i)
            for token in sent:
                if token.dep_ in ["nsubj"]:
                    print(f"SUBJECT: {token.text}")
                elif token.dep_ in ["dobj", "pobj"]:
                    print(f"OBJECT: {token.text}")
                toks.append(token)
        

    
    
    @staticmethod
    def resolve_coref(text):
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

if __name__=="__main__":
    text = "The Henry River Project began on the south-western limb of the Wanna Syncline in 2004. NeFou drilled at the location in 2005. The discoverd quartz veins then gold."
    resolved_text = EventRoleLabel.resolve_coref(text)
    idk = EventRoleLabel(resolved_text)
    idk.object_subject_extract()
