from scripts.EventRoleLabel import EventRoleLabel
import torch
from Model import NERModel
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current Device:", torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))

#example = "The flu season is winding down, and it has killed 105 children so far."
text = "The Henry River Project began on the south-western limb of the Wanna Syncline in 2004. NeFou drilled at the location in 2005. The discoverd quartz veins then gold."
resolved_text = EventRoleLabel.resolve_coref(text)

# ------------------------------------------- Geo NER -------------------------------------------

GeoNER = NERModel("scripts\\results\\Geo-NER")
geo_predictions, geo_entity_locs = GeoNER.predict(resolved_text, return_locations=True)

# ----------------------------------------- Event Time NER -----------------------------------------

EventTimexNER = NERModel("scripts\\results\\EventTimex-NER")
event_time_predictions, event_time_locs, tokens = EventTimexNER.predict(resolved_text, return_locations=True, return_decoded_tokens=True)

# ---------------------------------------- Event Entity Linking ----------------------------------------
# Need to update to recognise people and organisations.
# Make Event ->subject, object, timeS, timeE nodes

linker = EventRoleLabel(resolved_text)
linker.object_subject_extract(tokens)


# ---------------------------------------- Timex Normalisation ----------------------------------------
# Need to do geo timescale normalisation
# Models: SUTime, ARTime, Masked Language model, QA?, Generative, BART, T5, Llama, Elmo, Mistral





