from scripts.EventRoleLabel import EventRoleLabel
import torch
from Model import NERModel
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current Device:", torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))

#example = "The flu season is winding down, and it has killed 105 children so far."
text = "The Henry River Project began on the south-western limb of the Wanna Syncline in 2004."
resolved_text = EventRoleLabel.resolve_coref(text)

# ------------------------------------------- Geo NER -------------------------------------------

GeoNER = NERModel("scripts\\results\\Geo-NER\\checkpoint-1000")
geo_predictions, geo_entity_locs = GeoNER.predict(resolved_text, return_locations=True)

# ----------------------------------------- Event Time NER -----------------------------------------

# EventTimexNER = NERModel("results/EventTimex-NER/final_model")
# event_time_predictions, event_time_locs = EventTimexNER.predict(resolved_text, return_locations=True)

# ---------------------------------------- Event Entity Linking ----------------------------------------
# Need to update to recognise people and organisations.
# Make Event ->subject, object, timeS, timeE nodes


# ---------------------------------------- Timex Normalisation ----------------------------------------
# Need to do geo timescale normalisation
# Models: SUTime, ARTime, Masked Language model, QA?, Generative, BART, T5, Llama, Elmo, Mistral





