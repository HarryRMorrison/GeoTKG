from scripts.EventRoleLabel import EventRoleLabel, resolve_coref
import torch
from Model import NERModel, TimexNormModel
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current Device:", torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))

#example = "The flu season is winding down, and it has killed 105 children so far."
text = "The Henry River Project began on the south-western limb of Perth in 2004. At the location, they discoverd quartz veins then gold. The formation was dated to the Archean."
resolved_text = resolve_coref(text)

# ------------------------------------------- Geo NER -------------------------------------------

GeoNER = NERModel("scripts\\results\\Geo-NER")
geo_predictions = GeoNER.predict(resolved_text)
geo_entity_locs = GeoNER.get_geo_entity_locations(geo_predictions)

# ----------------------------------------- Event Time NER -----------------------------------------

EventTimexNER = NERModel("scripts\\results\\EventTimex-NER")
event_time_predictions = EventTimexNER.predict(resolved_text)
event_time_locs = EventTimexNER.get_event_locations(event_time_predictions)
tokens = EventTimexNER.decode(resolved_text)

# ----------------------------------------- Create Linker -------------------------------------------

linker = EventRoleLabel(tokens)

# ---------------------------------------- Timex Normalisation ----------------------------------------

geo_time_locs = NERModel.get_geo_entity_locations(geo_predictions, bi_map={5:11})
timex_locs, timex_types = NERModel.get_event_locations(event_time_predictions, bi_map={0:5, 1:6, 3:8, 4:9}, return_types=True)

#TimexNorm = TimexNormModel("")
recon = linker.reconstruct(geo_time_locs, timex_locs)
#TimexNorm.preprocessing(recon, timex_types)


# ---------------------------------------- Event Entity Linking ----------------------------------------
# Need to update to recognise people and organisations.
# Make Event ->subject, object, timeS, timeE nodes

#linker.object_subject_extract(tokens)





