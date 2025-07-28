from EventEntityLinker import EventEntityLinker
import torch
from transformers import RobertaForTokenClassification, RobertaTokenizerFast

def get_geo_entity_locations(predictions):
    return

def get_event_locations(predictions):
    return

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

# ----------------------------------------- Event Time NER -----------------------------------------
EventTimexNER = RobertaForTokenClassification.from_pretrained("./results/EvTimex-NER/final_model/")
EventTimexNER_tokenizer = RobertaTokenizerFast.from_pretrained("./results/EvTimex-NER/final_model/")

encodings=EventTimexNER_tokenizer(example, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    outputs = GeoNER(**encodings)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

event_locations = get_event_locations(predictions)

# ---------------------------------------- Event Entity Linking ----------------------------------------
event_entity_linker = EventEntityLinker(example)
Geo_resolved_text = event_entity_linker.embed_resolved_text(GeoNER, GeoNER_tokenizer)
EventTimex_resolved_text = event_entity_linker.embed_resolved_text(EventTimexNER, EventTimexNER_tokenizer)
EventEntityPairs = event_entity_linker.get_cosine_similarity(event_locations, geo_entity_locations, Geo_resolved_text, EventTimex_resolved_text)

# ---------------------------------------- Timex Normalisation ----------------------------------------
# Need to do geo timescale normalisation

# ---------------------------------------- Event to Event Temporal Relations ----------------------------------------
# Here I will make triples of <event1, temprel, event2>

# ---------------------------------------- Event to Timex Temporal Relations ----------------------------------------
# Here I make triples of <event, temprel, time> to see how an event relates to a time

# ---------------------------------------- Resolution ----------------------------------------
# Link pairs and triples by event





