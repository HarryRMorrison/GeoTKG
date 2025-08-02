from EventRoleLabel import EventRoleLabel
import torch
from Model import NERModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current Device:", torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))

#example = "The flu season is winding down, and it has killed 105 children so far."
text = "The Henry River Project began on the south-western limb of the Wanna Syncline in 2004."
resolved_text = EventRoleLabel.resolve_coref(text)

# ------------------------------------------- Geo NER -------------------------------------------

GeoNER = NERModel("results/Geo-NER/model")
geo_predictions, geo_entity_locs = GeoNER.predict(resolved_text, return_locations=True)

# ----------------------------------------- Event Time NER -----------------------------------------

EventTimexNER = NERModel("results/EventTimex-NER/final_model")
event_time_predictions, event_time_locs = EventTimexNER.predict(resolved_text, return_locations=True)

# ---------------------------------------- Event Entity Linking ----------------------------------------
# Need to update to recognise people and organisations.


# ---------------------------------------- Timex Normalisation ----------------------------------------
# Need to do geo timescale normalisation
# Models: SUTime, ARTime, Masked Language model, QA?, Generative, BART, T5, Llama, Elmo, Mistral

# ---------------------------------------- Event to Event Temporal Relations ----------------------------------------
# Here I will make triples of <event1, temprel, event2>

# ---------------------------------------- Event to Timex Temporal Relations ----------------------------------------
# Here I make triples of <event, temprel, time> to see how an event relates to a time

# ---------------------------------------- Resolution ----------------------------------------
# Link pairs and triples by event





