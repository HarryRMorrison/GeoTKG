import spacy
from Resolvers import reconstruct

class GeoTKGPipeline:
    def __init__(self, DCT, coref_resolver, geo_ner, evt_ner, time_norm):
        self.DCT = DCT
        self.CorefResolver = coref_resolver
        self.GeoNER = geo_ner
        self.EventTimeNER = evt_ner
        self.TimeNorm = time_norm

    def spacify(self):
        return   

    def fit(self, unresolved_text):
        resolved_text = self.CorefResolver(unresolved_text)

        # 1. Predict BIO sequences
        #   1.1 Geo Entities and Timescales
        geo_preds = self.GeoNER.predict(resolved_text)
        geo_entity_locs = self.GeoNER.get_geo_entity_locations(geo_preds)
        geo_time_locs = self.GeoNER.get_geo_entity_locations(geo_preds, bi_map={5:11})
        
        #   1.2 Events and Time Expressions
        event_time_preds = self.EventTimeNER.predict(resolved_text)
        event_locs = self.EventTimeNER.get_event_locations(event_time_preds)
        timex_locs, timex_types = self.EventTimeNER.get_event_locations(event_time_preds, bi_map={0:5, 1:6, 3:8, 4:9}, return_types=True)

        # 2. Normalise Times
        #   2.1 Preprocessing to create inputs
        tokens, original = self.GeoNER.decode(resolved_text)
        text, geo_t, timex = reconstruct(tokens, original, geo_time_locs, timex_locs)
        self.TimeNorm.preprocessing(text, timex, geo_t, timex_types, self.DCT)

        #   2.2 Predicting normalised times
        cal_times, geo_times = self.TimeNorm.predict()

        # 3. Predict E-T and E-E Temporal Relations
        #   3.1 Preprocessing to create inputs

        #   3.2 

        print(tokens)
        print(original)
        print(timex_types)
        for loc in [cal_times, geo_times]:
            print(loc)

        # print(timex_types, cal_times, geo_times)

        return

if __name__=="__main__":
    from Model import NERModel, TimexNormModel
    from Resolvers import CorefResolve
    model = GeoTKGPipeline(
        DCT = "2024-03-22",
        coref_resolver=CorefResolve,
        geo_ner = NERModel("scripts\\results\\Geo-NER"),
        evt_ner = NERModel("scripts\\results\\EventTimex-NER"),
        time_norm = TimexNormModel("scripts\\results\\TimeNormBart")
    )
    #text = "The Henry River Project began on the south-western limb of Perth in 2004. A year later, they discoverd a quartz vein formation. The formation was dated to the Archean. Other projects have found gold dated to the Jurassic or ~1000ma."
    text = "The Henry River Project began on the south-western limb of Perth in 2004. A year after the project began, they discoverd a quartz vein formation."
    #text = "1000ma 1000 ma 1000 Ma 1000.102 ma 1000.102ma ~1000ma ~1000 ma ~1000.22ma ~1000.22 ma"
    #text = "The formation was dated to the Archean."
    model.fit(text)


