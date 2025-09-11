from SRL import SRL
from TIEModel import TIEModel
from GeoEntityModel import GeoEntityModel
from TimexNormModel import TimexNormModel
from KGConstructor import KGConstructor
from CorefResolver import CorefResolver
import torch
import numpy as np
from transformers import AutoTokenizer
import warnings
from transformers import logging as hf_logging

# Suppress FutureWarnings
warnings.filterwarnings("ignore", message=".*resume_download.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*autocast.*", category=FutureWarning)

# Suppress Hugging Face model init warnings
hf_logging.set_verbosity_error()

TOKENIZER = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)

class GeoTKGPipeline:
    def __init__(self, DCTs):
        self.DCT = DCTs

        self.GeoNER = GeoEntityModel()
        load = torch.load("results\\geo_model\\geo_model_test.pt")
        self.GeoNER.load_state_dict(load['model_state_dict'])

        self.TIEModel = TIEModel()
        load = torch.load("results\\tie_model_hT\\tie_model_epoch15.pt")
        self.TIEModel.load_state_dict(load['model_state_dict'])

        self.NormModel = TimexNormModel("results/norm_model/time_norm_epoch15.pt")

        self.corefresolver = CorefResolver()
        self.slr = SRL()
        self.kgconstructor = KGConstructor()

    def pred(self, batch_unresolved_text):
        # 1) Coreference Resolution
        batched_resolved_text = [self.corefresolver(text) for text in batch_unresolved_text]

        # 2a) Temporal Information Extraction
        tokens, events, times, et_preds, ee_triples, ee_mask = self.TIEModel.predict(batched_resolved_text)
        # 2b) Geological Entity Extraction
        geo_times, geo_ents = self.GeoNER.predict(batched_resolved_text) 
        #print(geo_times)
        roles = []
        normalised_times=[]
        for i, resolved_text in enumerate(batched_resolved_text):
            decodings, out = GeoTKGPipeline.decode(resolved_text)
            recon, recon_ent_locs, recon_geotime_locs, recon_event_locs, recon_timex_locs = GeoTKGPipeline.reconstruct(decodings, out, geo_times[i], geo_ents[i], events[i], times[i])
            # 3) Semantic Role Labelling
            eso, recon, timex_locs = self.slr(recon, recon_ent_locs, recon_geotime_locs, recon_event_locs, recon_timex_locs)
            roles.append(eso)
            # 4) Time Normalisation
            normalised_times.append(self.NormModel.predict(recon, timex_locs, self.DCT))
        
        print(normalised_times)
        print(roles)
        # 5) KG Construction
        kg = self.kgconstructor(roles[0], ee_triples[0], normalised_times[0], et_preds[0])
        return
    
    @staticmethod
    def decode(text, enc=TOKENIZER):
        encodings = enc(text, padding=True, truncation=True, return_tensors="pt")
        decodings = enc.convert_ids_to_tokens(encodings["input_ids"][0])
        out = "".join(decodings)
        out = out.split("Ġ")
        return decodings, out
    
    def span_resolve(original, span2tok, s, e):
        span = []
        for j in range(s, e):
            span_id = span2tok[j]
            span.append(original[span_id])
        return " ".join(span)

    def found_span_resolve(i, locs, counts, span2tok, original):   
        # If ner detected span is shorter than actual span word
        if locs[i] - i < counts[span2tok[i]]:
            span = original[span2tok[i]]
            i += counts[span2tok[i]]
        # If ner detected span is larger than actual span word
        elif locs[i] - i > counts[span2tok[i]]:
            span = GeoTKGPipeline.span_resolve(original, span2tok, i, locs[i])
            i += (locs[i] - i)
        # If ner detected span is equal to actual span word
        else:
            span = original[span2tok[i]]
            i += 1
        return span, i

    def reconstruct(tokens, original, geo_time_locs, geo_entity_locs, event_locs, timex_locs):
        span2tok = []
        current_span = 0
        for i, tok in enumerate(tokens[:-1]):
            if "Ġ" == tok[0]:
                current_span += 1
            span2tok.append(current_span)
        
        span2tok = np.array(span2tok)
        unique_elements, counts = np.unique(span2tok, return_counts=True)
        counts = {el:co for el, co in zip(unique_elements, counts)}

        geo_entity_types = {s:t for s, e, t in geo_entity_locs}
        geo_entity_locs = {s:e for s, e, t in geo_entity_locs}
        starts_geo_ent = list(geo_entity_locs.keys())
        new_geo_ent_locs = []
        new_geo_ent_types = []

        geo_time_locs = {s:e for s, e, t in geo_time_locs}
        starts_geo_time = list(geo_time_locs.keys())
        new_geo_time_locs = []

        event_locs = {s:e for s, e, t in event_locs}
        starts_events = list(event_locs.keys())
        new_events_locs = []

        timex_types = {s:t for s, e, t in timex_locs}
        timex_locs = {s:e for s, e, t in timex_locs}
        starts_timex = list(timex_locs.keys())
        new_timex_locs = []
        new_timex_types = []

        out = []
        i = 1

        while i < len(tokens[:-1]):
            if i == 66:
                print("here")
            # Check if geo ent
            if i in starts_geo_ent:
                starts_geo_ent.remove(i)
                new_geo_ent_types.append(geo_entity_types[i])
                new_geo_ent_locs.append(len(out))
                span, i = GeoTKGPipeline.found_span_resolve(i, geo_entity_locs, counts, span2tok, original)
            # Check if geo time
            elif i in starts_geo_time:
                starts_geo_time.remove(i)
                new_geo_time_locs.append(len(out))
                span, i = GeoTKGPipeline.found_span_resolve(i, geo_time_locs, counts, span2tok, original)
            # Check if event
            elif i in starts_events:
                starts_events.remove(i)
                new_events_locs.append(len(out))
                span, i = GeoTKGPipeline.found_span_resolve(i, event_locs, counts, span2tok, original)
            elif i in starts_timex:
                starts_timex.remove(i)
                new_timex_types.append(timex_types[i])
                new_timex_locs.append(len(out))
                span, i = GeoTKGPipeline.found_span_resolve(i, timex_locs, counts, span2tok, original)
            else:
                span = GeoTKGPipeline.span_resolve(original, span2tok, i, i+1)
                i += counts[span2tok[i]]
            out.append(span)

        out[-1] = out[-1].replace("</s>", "")
        return out, list(zip(new_geo_ent_locs,new_geo_ent_types)), list(zip(new_geo_time_locs,["TIMESCALE"]*len(new_geo_time_locs))), list(zip(new_events_locs, ["EVENT"]*len(new_events_locs))), list(zip(new_timex_locs,new_timex_types))

if __name__=="__main__":
    model = GeoTKGPipeline(DCTs = "2018-12-17")
    #text1 = "The Henry River Project began on the south-western limb of Perth in 2004. A year later, they discoverd a quartz vein formation, which the team dated to the Jurassic or ~1000ma."
    #text2 = "The mineralisation was characterised by traces of disseminated pyrite with zones of trace pyrrhotite and chalcopyrite in felsic schist."
    #text = "The Henry River Project began on the south-western limb of Perth in 2004. A year after the project started, they discoverd a quartz vein formation."
    #text = "1000ma 1000 ma 1000 Ma 1000.102 ma 1000.102ma ~1000ma ~1000 ma ~1000.22ma ~1000.22 ma"
    text1 = "In 2019, BHP found a rock formation characterised by traces of pyrite. BHP transported the rock to Perth in 2020."
    #extract = "The Lyons project area consists of rocks that are constituents of the Meso-Proterozoic Edmund Basin, which is a component of the West Australian cratonic complex. This is an assemblage of the Pilbara and Yigarn cratons and the Glenburgh Terrane. The Pilbra Craton and Glenburgh Terrane amalgamated first, and during the process developed the overlying Hamersley and Ashburton Basins. A magmatic package along the southern margin of the Glengurgh terrain (Dalgaringa Arc) has been interpreted to have chemical signatures to that of continental margin arcs (e.g. Sth America), suggesting that the Yigarn Craton was closing northwards and subducting oceanic crust beneath the combined Pilbra Craton and Glenburgh terrane, eventually culminating in the amalgamation of the Yigarn Craton (Glenburgh Orogeny 2005Ma to 1950Ma). The Mangaroon Orogeny - ~1650Ma, although the driver of orogenesis is currently unknown, with its high-temperature — low-pressure metamorphic conditions, and short duration of metamorphism, magmatism and sedimentation imply an extension dominated orogeny. This event re-activated the terrane bounding trans-crustal faults (e.g. Lyons River, Talga, Cardilya Faults) and initiated the development of the Edmund Basin (unconformable on the Glenburgh Terrain and Ashburton Basin)."
    extract = "Multiple drill lines along 20km of the Prairie Downs Fault (PDF) were completed in the 2017-2018 exploration season. A total of 6276.6m was drilled for 54 drill holes. The aim of the program was to test 20km of the PDF for base metal mineralisation in tenements E52 and E52. Numerous drill holes intersected significant base metal, vanadium and gold mineralisation including 19m @ 5.9% Pb, 0.1% Zn, 0.1% Cu and 40 g/t Ag from 87m in hole PDP456, at the Husky South prospect. Down hole total electro magnetics was completed on the two diamond drill holes PDD504 and PDD506 at Husky South. No significant off hole responses were detected. "
    model.pred([text1])

