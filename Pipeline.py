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
        self.kgconstructor = KGConstructor()

    def pred(self, batch_unresolved_text):
        # 1) Coreference Resolution
        batched_resolved_text = [self.corefresolver(text) for text in batch_unresolved_text]

        # 2a) Temporal Information Extraction
        tokens, events, times, et_preds, ee_triples, ee_mask = self.TIEModel.predict(batched_resolved_text)

        # 2b) Geological Entity Extraction
        geo_times, geo_ents = self.GeoNER.predict(batched_resolved_text)

        # 3) Semantic Role Labelling
        roles = [self.slr(resolved_text, geo_ents[i], geo_times[i], events[i]) for i, resolved_text in enumerate(batched_resolved_text)]

        # 4) Time Normalisation -> Use DCTs, ETs, and NormModel


        # 5) KG Construction
        kg = self.kgconstructor(roles[0], ee_triples[0])
        return

if __name__=="__main__":
    model = GeoTKGPipeline(DCTs = "2024-03-22")
    text1 = "The Henry River Project began on the south-western limb of Perth in 2004. A year later, they discoverd a quartz vein formation. The formation was dated to the Archean. Other projects have found gold dated to the Jurassic or ~1000ma."
    text2 = "The mineralisation was characterised by traces of disseminated pyrite with zones of trace pyrrhotite and chalcopyrite in felsic schist."
    #text = "The Henry River Project began on the south-western limb of Perth in 2004. A year after the project started, they discoverd a quartz vein formation."
    #text = "1000ma 1000 ma 1000 Ma 1000.102 ma 1000.102ma ~1000ma ~1000 ma ~1000.22ma ~1000.22 ma"
    #text = "In 2019, BHP found a rock formation which they dated to the Archean."
    model.pred([text1, text2])

