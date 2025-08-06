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

        geo_preds = self.GeoNER.predict(resolved_text)
        geo_entity_locs = self.GeoNER.get_geo_entity_locations(geo_preds)
        geo_time_locs = self.GeoNER.get_geo_entity_locations(geo_preds, bi_map={5:11})

        event_time_preds = self.EventTimeNER.predict(resolved_text)
        event_locs = self.EventTimeNER.get_event_locations(event_time_preds)
        timex_locs, timex_types = self.EventTimeNER.get_event_locations(event_time_preds, bi_map={0:5, 1:6, 3:8, 4:9}, return_types=True)

        tokens = self.EventTimeNER.decode(resolved_text)
        
        self.TimeNorm.preprocessing(tokens, timex_locs, geo_time_locs, timex_types, self.DCT)
        cal_times, geo_times = self.TimeNorm.predict()

        for loc in [geo_entity_locs, geo_time_locs, event_locs, timex_locs]:
            for s,e in loc:
                print(tokens[s:e])
            print("-------------------------")

        print(timex_types, cal_times, geo_times)

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
    text = "The Henry River Project began on the south-western limb of Perth in 2004. At the location, they discoverd quartz veins then gold. The formation was dated to the Archean."
    model.fit(text)


