import networkx as nx
import matplotlib.pyplot as plt
from globals import ID2LABEL_GEONER, ID2LABEL_EE

class KGConstructor:
    def __init__(self):
        pass

    def __call__(self, roles, temprels, normtimes):
        '''
        Args:
            roles:
            temprels:
            normtimes: {'event_idx', 'sTime', 'eTime'}
        '''
        G = nx.DiGraph()
        labels = {}
        print(normtimes)
        for i, ev in enumerate(roles):
            sub = ev["subject"][0] if ev["subject"] is not None else "None"
            obj = ev["object"][0] if ev["object"] is not None else "None"
            labels[ev["event"]] = f"{ev['event'].upper()}\nsubj:{sub}\nobj:{obj}"
            G.add_node(ev["event"], label="Event", subject=sub, object=obj)
            for time in normtimes[i]:
                G.add_node(time.strftime("%Y-%m-%d"), label="Time")
                G.add_edge(ev["event"], time.strftime("%Y-%m-%d"), rel="HAS_TIME")
        
        for e1, e2, r in temprels:
            ev1 = roles[e1]["event"]
            ev2 = roles[e2]["event"]
            G.add_edge(ev1, ev2, rel=ID2LABEL_EE[r.item()])

        pos = nx.spring_layout(G)  # layout algorithm
        edge_labels = nx.get_edge_attributes(G, 'rel')

        nx.draw(G, pos, labels=labels,
                with_labels=True, node_size=2000, node_color="lightblue", font_size=8)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)



        plt.show()

        return