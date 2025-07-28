from EventEntityLinker import EventEntityLinker
import torch
from transformers import RobertaForTokenClassification, RobertaTokenizerFast

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
            start = i
            i += 1
            while i < len(predictions) and predictions[i] == 7:  # I-Event
                i += 1
            locations.append([start, i])  # [start, end) format
        else:
            i += 1
    return locations

#example = "The flu season is winding down, and it has killed 105 children so far."
example = "The Henry River Project is located on the south-western limb of the Wanna Syncline. The Syncline has a central northwest - southeast axial plane located 20-25km north-east of the project. The Project is underlain by Paleoproterozoic - Mesoproterozoic sediments and volcanics of the Edmund Group which with the overlying Collier Group comprises the Bangemall Supergroup. The deposition of the Edmund Group occurred between 1620 Ma and 1465 Ma."

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device('cuda')
print("Current Device:", torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))

# ------------------------------------------- Geo NER -------------------------------------------
GeoNER = RobertaForTokenClassification.from_pretrained("./results/Geo-NER/final_model/")
GeoNER_tokenizer = RobertaTokenizerFast.from_pretrained("./results/Geo-NER/final_model/")

encodings=GeoNER_tokenizer(example, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    outputs = GeoNER(**encodings)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

geo_entity_locations = get_geo_entity_locations(predictions)
print(geo_entity_locations)
# ----------------------------------------- Event Time NER -----------------------------------------
EventTimexNER = RobertaForTokenClassification.from_pretrained("./results/EventTimex-NER/final_model/")
EventTimexNER_tokenizer = RobertaTokenizerFast.from_pretrained("./results/EventTimex-NER/final_model/")
# Bert got a date seq2seq model hugging face, DateBERT
encodings=EventTimexNER_tokenizer(example, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    outputs = EventTimexNER(**encodings)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

event_locations = get_event_locations(predictions)

# ---------------------------------------- Event Entity Linking ----------------------------------------
# Need to update to recognise people and organisations. Can use: SpaCy, EventStoryLine Corpus, RicherEvent Description
event_entity_linker = EventEntityLinker(example)
Geo_resolved_text = event_entity_linker.embed_resolved_text(GeoNER, GeoNER_tokenizer)
EventTimex_resolved_text = event_entity_linker.embed_resolved_text(EventTimexNER, EventTimexNER_tokenizer)
EventEntityPairs = event_entity_linker.get_cosine_similarity(event_locations, geo_entity_locations, Geo_resolved_text, EventTimex_resolved_text)

# ---------------------------------------- Timex Normalisation ----------------------------------------
# Need to do geo timescale normalisation
# Models: SUTime, ARTime, Masked Language model, QA?, Generative, BART, T5, Llama, Elmo, Mistral

# ---------------------------------------- Event to Event Temporal Relations ----------------------------------------
# Here I will make triples of <event1, temprel, event2>

# ---------------------------------------- Event to Timex Temporal Relations ----------------------------------------
# Here I make triples of <event, temprel, time> to see how an event relates to a time

# ---------------------------------------- Resolution ----------------------------------------
# Link pairs and triples by event





