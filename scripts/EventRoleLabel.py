import spacy
from spacy.tokens import Doc
from neo4j.graph import Node

# pretokenized is list of list of strings which are each sentences
class EventRoleLabel:
    def __init__(self, pretokenized):
        self.pretokenized = pretokenized
    
    def reconstruct(self, geo_entity_locs, event_time_locs):
        i = 0
        original = self.pretokenized

        event_starts = [s for s, e in event_time_locs]
        event_ends = [e for s, e in event_time_locs]
        entit_starts = [s for s, e in geo_entity_locs]
        entit_ends = [e for s, e in geo_entity_locs]

        token_len = len([token for sent in self.pretokenized for token in sent])
        out = []
        event_idxs = []
        entit_idxs = []

        while i < token_len:
            if original[i] == "<s>":
                sent = []
                event_idx = []
                entit_idx = []
            elif original[i] == "</s>":
                out.append(sent)
                event_idxs.append(event_idx)
                entit_idxs.append(entit_idx)
            elif i in event_starts:
                ind = event_starts.index(i)
                event_idx.append(len(sent))
                sent.append(original[event_starts[ind]:event_ends[ind]])
            elif i in entit_starts:
                ind = entit_starts.index(i)
                entit_idx.append(len(sent))
                sent.append(original[entit_starts[ind]:entit_ends[ind]])
            else:
                sent.append(original[i])
        
        return out, event_idxs, entit_idxs


    # Assume all information in sentence (post coref resolve)
    def object_subject_extract(self, geo_entity_locs, event_time_locs):
        nlp = spacy.load("en_core_web_trf")
        recontructed, event_idxs, entity_idxs = self.reconstruct(geo_entity_locs, event_time_locs)
        nodes = []

        for sent, sent_event_idxs in zip(recontructed, event_idxs):
            doc = Doc(nlp.vocab, sent)

            for event_i in sent_event_idxs:

                for child in doc[event_i].children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        print(f"SUBJECT: {child.text}")
                        ev_subject = child.text
                    elif child.dep_ in ["dobj", "pobj"]:
                        print(f"OBJECT: {child.text}")
                        ev_object = child.text
                if ev_subject and ev_object:
                    nodes.append(Node("Event", event=doc[event_i].text, subject=ev_subject, object=ev_object))
                
                
        

    
    
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
