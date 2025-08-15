import spacy
from spacy.tokens import Doc
from neo4j.graph import Node

# pretokenized is list of strings which are each sentences
class EventRoleLabel:
    def __init__(self, pretokenized):
        self.pretokenized = pretokenized

    def span_resolver(self):
        return
    
    def reconstruct(self, geo_entity_locs, event_time_locs):
        nlp = spacy.load("en_core_web_trf")
        doc = Doc(nlp.vocab, self.pretokenized)
        
        event_starts = [s for s, e in event_time_locs]
        event_ends = [e for s, e in event_time_locs]
        geo_starts = [s for s, e in geo_entity_locs]
        geo_ends = [e for s, e in geo_entity_locs]

        token_len = len(self.pretokenized)
        i = 0
        out = []
        event_idxs = []
        geo_idxs = []

        while i < token_len:
            if doc[i].text in ["<s>", "</s>"]:
                i += 1
                continue
            elif i in event_starts:
                ind = event_starts.index(i)
                event_idxs.append(len(out))
                out.extend(doc[event_starts[ind]:event_ends[ind]+1])
            elif i in geo_starts:
                ind = geo_starts.index(i)
                geo_idxs.append(len(out))
                out.append("".join([doc[j].text for j in range(geo_starts[ind], geo_ends[ind]+1)]))
            else:
                out.append(doc[i])
            i += 1
        
        return out, event_idxs, geo_idxs


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
            
    

if __name__=="__main__":
    text = "The Henry River Project began on the south-western limb of the Wanna Syncline in 2004. They discoverd quartz veins then gold."
    resolved_text = EventRoleLabel.resolve_coref(text)
    idk = EventRoleLabel(resolved_text)
    idk.object_subject_extract()
