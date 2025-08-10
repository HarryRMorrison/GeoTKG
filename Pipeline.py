import spacy
from Resolvers import reconstruct

class GeoTKGPipeline:
    def __init__(self, DCT, coref_resolver, geo_ner, evt_ner, time_norm, temp_rel):
        self.DCT = DCT
        self.CorefResolver = coref_resolver
        self.GeoNER = geo_ner
        self.EventTimeNER = evt_ner
        self.TimeNorm = time_norm
        self.TempRel = temp_rel

    @staticmethod
    def spacify(resolved_text):
        nlp = spacy.load("en_core_web_trf")
        doc = nlp(resolved_text)
        sentences = []
        for sent in doc.sents:
            sentences.append(sent.text)
        return sentences

    def fit(self, unresolved_text):
        resolved_text = self.CorefResolver(unresolved_text)
        sentences = GeoTKGPipeline.spacify(resolved_text)
        for sent in sentences:
            # 1. Predict BIO sequences
            #   1.1 Geo Entities and Timescales
            geo_preds = self.GeoNER.predict(sent)
            geo_entity_locs = self.GeoNER.get_geo_entity_locations(geo_preds)
            geo_time_locs = self.GeoNER.get_geo_entity_locations(geo_preds, bi_map={5:11})
            
            #   1.2 Events and Time Expressions
            event_time_preds = self.EventTimeNER.predict(sent)
            event_locs = self.EventTimeNER.get_event_locations(event_time_preds)
            timex_locs, timex_types = self.EventTimeNER.get_event_locations(event_time_preds, bi_map={0:5, 1:6, 3:8, 4:9}, return_types=True)

            #   1.3 Reconstructing original text with BIO tags
            tokens, original = self.GeoNER.decode(sent)
            text, geo_entity_locs, geo_time_locs, event_locs, timex_locs = reconstruct(tokens, original, geo_time_locs, geo_entity_locs, event_locs, timex_locs)

            # 2. Normalise Times
            #   2.1 Preprocessing to create inputs
            self.TimeNorm.preprocessing(text, timex_locs, geo_time_locs, timex_types, self.DCT)

            #   2.2 Predicting normalised times
            cal_times, geo_times = self.TimeNorm.predict()

            # 3. Predict E-T and E-E Temporal Relations
            #   3.1 Preprocessing to create inputs
            self.TempRel.preprocessing(text, cal_times, timex_types, geo_times, event_locs, self.DCT)

            #   3.2 
            ET_preds, EE_preds = self.TempRel.predict()

            print(text)
            id2label=self.TempRel.model.config.id2label

            print("-----ET-----")

            for (t,e),rel in ET_preds:
                print(text[t],id2label[rel],text[e])
            
            print("-----EE-----")

            for (e1,e2),rel in EE_preds:
                print(text[e1],id2label[rel],text[e2])

        # print(text)
        # print(timex_types)
        # for loc in [cal_times, geo_times]:
        #     print(loc)

        # for loc in [geo_ent_locs, geo_time_locs, events_locs, timex_locs]:
        #     for i in loc:
        #         print(text[i])
        #     print("------------")

        # print(timex_types, cal_times, geo_times)

        return

if __name__=="__main__":
    from Model import NERModel, TimexNormModel, TempRelModel
    from Resolvers import CorefResolve
    model = GeoTKGPipeline(
        DCT = "2024-03-22",
        coref_resolver=CorefResolve,
        geo_ner = NERModel("scripts\\results\\Geo-NER"),
        evt_ner = NERModel("scripts\\results\\EventTimex-NER"),
        time_norm = TimexNormModel("scripts\\results\\TimeNormBart"),
        temp_rel = TempRelModel("scripts\\results\\TempRel\\checkpoint-10000")
    )
    #text = "The Henry River Project began on the south-western limb of Perth in 2004. A year later, they discoverd a quartz vein formation. The formation was dated to the Archean. Other projects have found gold dated to the Jurassic or ~1000ma."
    #text = "The Henry River Project began on the south-western limb of Perth in 2004. A year after the project started, they discoverd a quartz vein formation."
    #text = "1000ma 1000 ma 1000 Ma 1000.102 ma 1000.102ma ~1000ma ~1000 ma ~1000.22ma ~1000.22 ma"
    text = "In 2019, BHP found a rock formation which they dated to the Archean."
    model.fit(text)


