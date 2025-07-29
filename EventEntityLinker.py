from sklearn.metrics.pairwise import cosine_similarity
import torch
import spacy

class EventEntityLinker:
    def __init__(self, text):
        self.resolved_text = EventEntityLinker.resolve_coref(text)
    
    def resolve_coref(text):
        return
    
    def get_average_embedding(outputs, token_indices):
        return
    
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
        return


    def get_cosine_similarity(self, event_token_indices, entity_token_indices, GeoNER_outputs, EvTimeNER_outputs):
        event_embeddings = [EventEntityLinker.get_average_embedding(EvTimeNER_outputs, indices) for indices in event_token_indices]
        entity_embeddings = [EventEntityLinker.get_average_embedding(GeoNER_outputs, indices) for indices in entity_token_indices]
        return EventEntityLinker.embeddings_cosine_similarity(event_embeddings, entity_embeddings)


if __name__=="__main__":
    1
