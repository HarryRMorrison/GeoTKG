import requests
import torch
from sklearn.metrics.pairwise import cosine_similarity
from allennlp.predictors.predictor import Predictor
import allennlp_models.coref

# predictor = Predictor.from_path(
#     "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
# )

class EventEntityLinker:
    def __init__(self, text):
        self.resolved_text = EventEntityLinker.resolve_coref_allennlp(text)
    
    def resolve_coref_allennlp(text):
        
        result = predictor.predict(document="John went to the store. He bought milk.")

        tokens = result["document"]
        clusters = result["clusters"]

        # Build resolved text using clusters
        resolved_tokens = tokens[:]
        for cluster in clusters:
            main_mention = cluster[0]
            for mention in cluster[1:]:
                start, end = mention
                replacement = tokens[main_mention[0]:main_mention[1] + 1]
                resolved_tokens[start] = " ".join(replacement)
                for i in range(start + 1, end + 1):
                    resolved_tokens[i] = ""

        resolved_text = " ".join(t for t in resolved_tokens if t != "")
        return resolved_text
    
    def get_average_embedding(outputs, token_indices):
        hidden_states = outputs.last_hidden_state.squeeze(0)  # shape: [seq_len, hidden_dim]
        if not token_indices:
            return None
        # Average embeddings of all relevant tokens
        selected = hidden_states[token_indices]  # shape: [n_tokens, hidden_dim]
        return selected.mean(dim=0).detach().numpy()
    
    @staticmethod
    def cosine_similarity(event_embeddings, entity_embeddings):
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

    def embed_resolved_text(self, model, tokenizer):
        inputs = tokenizer(self.resolved_text, return_tensors="pt", return_offsets_mapping=True)
        return model(**inputs)

    def get_cosine_similarity(self, event_token_indices, entity_token_indices, GeoNER_outputs, EvTimeNER_outputs):
        event_embeddings = [EventEntityLinker.get_average_embedding(EvTimeNER_outputs, indices) for indices in event_token_indices]
        entity_embeddings = [EventEntityLinker.get_average_embedding(GeoNER_outputs, indices) for indices in entity_token_indices]
        return cosine_similarity(event_embeddings, entity_embeddings)


if __name__=="__main__":
    from transformers import AutoTokenizer, AutoModel
    AutoTokenizer.from_pretrained("SpanBERT/spanbert-large-cased")
    AutoModel.from_pretrained("SpanBERT/spanbert-large-cased")
