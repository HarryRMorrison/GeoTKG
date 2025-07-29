from sklearn.metrics.pairwise import cosine_similarity
import torch
import spacy

class EventEntityLinker:
    def __init__(self, text):
        self.resolved_text = EventEntityLinker.resolve_coref(text)
    
    def resolve_coref(text):
        nlp = spacy.load("en_coreference_web_trf")
        doc = nlp(text)
        clusters = [
            span_group
            for key, span_group in doc.spans.items()
            if key.startswith("coref_clusters")
        ]
        
        # 4) build a map: for each non‑main mention span, remember its start→(end, main_text)
        replace_map = {}
        for cluster in clusters:
            spans = list(cluster)          # ← convert to a list
            if len(spans) < 2:
                continue
            main = spans[0]                # first mention is the “main”
            for mention in spans[1:]:      # now slicing works
                replace_map[mention.start] = (mention.end, main.text)
        
        # 5) walk the token sequence, performing span‑level replacement
        resolved_tokens = []
        i = 0
        while i < len(doc):
            if i in replace_map:
                span_end, main_text = replace_map[i]
                resolved_tokens.append(main_text)
                i = span_end
            else:
                resolved_tokens.append(doc[i].text)
                i += 1
        
        # 6) join and return
        return " ".join(resolved_tokens)
    
    def get_average_embedding(outputs, token_indices):
        # Remove batch dim → [seq_len, hidden_dim]
        hidden_states = outputs.squeeze(0)
        if not token_indices:
            return None
        # Gather the rows, average over the token dimension
        span_embeddings = hidden_states[token_indices, :]       # [n_tokens, hidden_dim]
        avg_embedding   = span_embeddings.mean(dim=0)           # [hidden_dim]
        return avg_embedding.cpu().numpy()
    
    @staticmethod
    def embeddings_cosine_similarity(event_embeddings, entity_embeddings):
        # Match entities to events using cosine similarity
        similars = {}
        for i, event_emb in enumerate(event_embeddings):
            print(f"\nEvent {i}:")
            similars[i] = {}
            for j, entity_emb in enumerate(entity_embeddings):
                sim = cosine_similarity([event_emb], [entity_emb])[0][0]
                print(f"  → Entity {j}: {sim:.3f}")
                if sim > 0.6:
                    similars[i][j] = sim
        return sim

    def embed_resolved_text(self, tok, model):
        model.eval()
        inputs = tok(
            self.resolved_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        for k,v in inputs.items():
            inputs[k] = v.to(device)

        with torch.no_grad():
            outputs = model(**inputs,output_hidden_states=True)
        # Extract and return the last hidden state
        # (shape: [1, seq_len, hidden_dim])
        return outputs.last_hidden_state


    def get_cosine_similarity(self, event_token_indices, entity_token_indices, GeoNER_outputs, EvTimeNER_outputs):
        event_embeddings = [EventEntityLinker.get_average_embedding(EvTimeNER_outputs, indices) for indices in event_token_indices]
        entity_embeddings = [EventEntityLinker.get_average_embedding(GeoNER_outputs, indices) for indices in entity_token_indices]
        return EventEntityLinker.embeddings_cosine_similarity(event_embeddings, entity_embeddings)


if __name__=="__main__":
    1
