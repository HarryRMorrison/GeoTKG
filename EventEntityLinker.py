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

def get_geo_entity_locations(predictions):
    locations = []
    bi_map = {0:6, 1:7, 2:8, 3:9, 4:10}
    i = 0
    while i < len(predictions[0]):
        if predictions[0][i].item() >= 0 and predictions[0][i].item() <= 4:  # 'B-LOCATION': 0, 'B-MINERAL': 1, 'B-ORE_DEPOSIT': 2, 'B-ROCK': 3, 'B-STRAT': 4, 'B-TIMESCALE': 5
            start = i
            i += 1
            ent_type = predictions[0][i].item()
            while i < len(predictions) and predictions[i] == bi_map[ent_type]:  # 'I-LOCATION': 6, 'I-MINERAL': 7, 'I-ORE_DEPOSIT': 8, 'I-ROCK': 9, 'I-STRAT': 10, 'I-TIMESCALE': 11
                i += 1
            locations.append([start, i])  # [start, end) format
        else:
            i += 1
    return locations

def get_event_locations(predictions):
    locations = []
    i = 0
    while i < len(predictions[0]):
        if predictions[0][i].item() == 2:  # B-Event
            print(event_entity_linker.resolved_text[i])
            start = i
            i += 1
            while i < len(predictions) and predictions[i] == 7:  # I-Event
                print(event_entity_linker.resolved_text[i])
                i += 1
            locations.append([start, i])  # [start, end) format
        else:
            i += 1
    return locations
