import networkx as nx
import matplotlib.pyplot as plt
from globals import ID2LABEL_GEONER, ID2LABEL_EE

class KGConstructor:
    def __init__(self):
        pass

    def __call__(self, roles, temprels, normtimes=None):
        '''
        Args:
            roles:
            temprels:
            normtimes: {'event_idx', 'sTime', 'eTime'}
        '''
        G = nx.DiGraph()

        for i, ev in enumerate(roles):
            event_id = f"event_{i}"
            G.add_node(event_id, label="Event", display=ev["event"])
            subj = " ".join([subjt['text'] for subjt in ev["subject"]]) if ev["subject"] is not None else None
            obj = " ".join([objt['text'] for objt in ev["object"]]) if ev["object"] is not None else None
            if subj is not None:
                G.add_node(subj, label="Entity", display=subj)
                G.add_edge(event_id, subj, rel="HAS_SUBJECT")
            if obj is not None:
                G.add_node(obj, label="Entity", display=obj)
                G.add_edge(event_id, obj, rel="HAS_OBJECT")

            # if ev["subject"] is not None:
            #     for subj in ev["subject"]:
            #         text = subj['text']
            #         G.add_node(text, label="Entity", display=text)
            #         G.add_edge(ev["event"], text, rel="HAS_SUBJECT")
            # if ev["object"] is not None:
            #     for obj in ev["object"]:
            #         text = obj['text']
            #         G.add_node(text, label="Entity", display=text)
            #         G.add_edge(ev["event"], text, rel="HAS_OBJECT")


            # for time in normtimes[i]:
            #     G.add_node(time.strftime("%Y-%m-%d"), label="Time")
            #     G.add_edge(ev["event"], time.strftime("%Y-%m-%d"), rel="HAS_TIME")
        
        for e1, e2, r in temprels:
            ev1 = f"event_{e1}"
            ev2 = f"event_{e2}"
            G.add_edge(ev1, ev2, rel=ID2LABEL_EE[r.item()])

        # Layout and drawing
        pos = nx.spring_layout(G)
        edge_labels = nx.get_edge_attributes(G, 'rel')

        # Extract labels from the "display" attribute
        node_labels = nx.get_node_attributes(G, 'display')

        nx.draw(G, pos, node_size=2000, node_color="lightblue", font_size=8, with_labels=False)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

        plt.show()

        return