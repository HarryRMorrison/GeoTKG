import spacy
import coreferee
import torch
from sklearn.metrics.pairwise import cosine_similarity

class EventEntityLinker:
    def __init__(self, text):
        self.nlp = spacy.load("en_core_web_trf")
        self.nlp.add_pipe("coreferee")
        self.resolved_text = self.get_coref_resolved_text(text)
    
    def get_coref_resolved_text(self, text):
        doc = self.nlp(text)
        return doc._.coref_resolved if doc._.has_coref else text
    
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


