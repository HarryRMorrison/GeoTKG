from Handler import SRL, CorefResolver, KGConstructor, TOKENIZER
from TIEModel import TIEModel, ID2LABEL_EE
from GeoEntityModel import GeoEntityModel
import torch

class GeoTKGPipeline:
    def __init__(self, DCTs):
        self.DCT = DCTs

        self.GeoNER = GeoEntityModel()
        load = torch.load("results\\geo_model\\geo_model.pt")
        self.GeoNER.load_state_dict(load['model_state_dict'])

        self.TIEModel = TIEModel()
        load = torch.load("results\\tie_model\\tie_model_epoch15.pt")
        self.TIEModel.load_state_dict(load['model_state_dict'])

        #self.NormModel = time_norm
        self.corefresolver = CorefResolver()
        self.slr = SRL()
        #self.kgconstructor = KGConstructor()

    def pred(self, batch_unresolved_text):
        # 1) Coreference Resolution
        batched_resolved_text = [self.corefresolver(text) for text in batch_unresolved_text]

        # 2a) Temporal Information Extraction
        tokens, events, times, et_preds, ee_triples, ee_mask = self.TIEModel.predict(batched_resolved_text)

        # 2b) Geological Entity Extraction
        geo_times, geo_ents = self.GeoNER.predict(batched_resolved_text)

        # for bi, text in enumerate(batched_resolved_text):
        #     geotime_locs = geo_times[bi]
        #     ent_locs = geo_ents[bi]
        #     event_locs = events[bi]
        #     decodings, out = SRL.decode(text)
        #     recon, ent_locs, geotime_locs, event_locs, geoent_types = SRL.reconstruct(decodings, out, geotime_locs, ent_locs, event_locs)
        #     print(f"---------- {bi} ----------")
        #     print(recon)
        #     print(ent_locs)
        #     print(geotime_locs)
        #     print(event_locs)
        #     print(geoent_types)
        # 3a) Temporal Transitivity Event Filtering
        # for bi in [0,1]:
        #     print(f"--------------- TEXT {bi+1} ---------------")
        #     eventsbi = []
        #     timesbi = []
        #     geoentsbi = []
        #     geotimesbi = []
        #     words = TOKENIZER.convert_ids_to_tokens(tokens['input_ids'][bi])

        #     for s, e, t in events[bi]:
        #         try:
        #             eventsbi.append(words[s:e][0].strip("Ġ"))
        #         except IndexError:
        #             continue
        #     print("EVENTS: ", eventsbi)

        #     for s, e, t in times[bi]:
        #         timesbi.append((words[s:e][0].strip("Ġ"), t))
        #     print("TIMES: ", timesbi)

        #     for s, e, t in geo_times[bi]:
        #         geotimesbi.append((words[s:e][0].strip("Ġ"), t))
        #     print("GEO TIMES: ", geotimesbi)

        #     for s, e, t in geo_ents[bi]:
        #         geoentsbi.append((words[s:e][0].strip("Ġ"), t))
        #     print("GEO ENTS: ", geoentsbi)

        #     for trip in ee_triples[bi][ee_mask[bi]]:
        #         print(f"<{eventsbi[trip[0].item()]} --> {ID2LABEL_EE[trip[2].item()]} --> {eventsbi[trip[1].item()]}>")

        #     for ei, et in enumerate(et_preds[bi]):
        #         for ti, rel in enumerate(et):
        #             if rel.item()==1:
        #                 print(f"<{eventsbi[ei]}, {timesbi[ti]}>")

        # 3b) Semantic Role Labelling
        roles = [self.slr(resolved_text, geo_ents[i], geo_times[i], events[i]) for i, resolved_text in enumerate(batched_resolved_text)]

        # 4) Time Normalisation -> Use DCTs, ETs, and NormModel


        # 5) KG Construction
        #kg = self.kgconstructor(roles, temprels, normtimes)
        return

if __name__=="__main__":
    model = GeoTKGPipeline(DCTs = "2024-03-22")
    text1 = "The Henry River Project began on the south-western limb of Perth in 2004. A year later, they discoverd a quartz vein formation. The formation was dated to the Archean. Other projects have found gold dated to the Jurassic or ~1000ma."
    text2 = "The mineralisation was characterised by traces of disseminated pyrite with zones of trace pyrrhotite and chalcopyrite in felsic schist."
    #text = "The Henry River Project began on the south-western limb of Perth in 2004. A year after the project started, they discoverd a quartz vein formation."
    #text = "1000ma 1000 ma 1000 Ma 1000.102 ma 1000.102ma ~1000ma ~1000 ma ~1000.22ma ~1000.22 ma"
    #text = "In 2019, BHP found a rock formation which they dated to the Archean."
    model.pred([text1, text2])

