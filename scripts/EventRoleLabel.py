from fastcoref import FCoref

class EventRoleLabel:
    def __init__(self, resolved_text):
        self.resolved_text = resolved_text
    
    
    
    def resolve_coref(text):
        model = FCoref(device="cpu")
        preds = model.predict(texts=[text])
        result = preds[0]

        # 3) get char-level spans for each cluster
        clusters = result.get_clusters(as_strings=False)

        # 4) build a list of all replacements: (start, end, rep_text)
        replacements = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            # the “antecedent” is the first span in the cluster
            rep_start, rep_end = cluster[0]
            rep_text = text[rep_start:rep_end]

            # replace every other span with rep_text
            for mention_start, mention_end in cluster[1:]:
                replacements.append((mention_start, mention_end, rep_text))

        # 5) apply replacements from end→start so earlier edits don’t shift later spans
        replacements.sort(key=lambda x: x[0], reverse=True)
        resolved = text
        for start, end, rep_text in replacements:
            resolved = resolved[:start] + rep_text + resolved[end:]

        return resolved

if __name__=="__main__":
    text = "Alice picked up her book because she wanted to read it."
    resolved_text = EventRoleLabel.resolve_coref(text)
    print(resolved_text)
