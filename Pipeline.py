from Handler import Handler

class GeoTKGPipeline:
    def __init__(self, DCT, geo_ner, evt_ner, time_norm):#, temp_rel):
        self.DCT = DCT
        self.GeoNER = geo_ner
        self.TIEModel = evt_ner
        self.NormModel = time_norm
        self.Handler = Handler()

    def pred(self, unresolved_text):
        # 1) Coreference Resolution
        resolved_text = self.Handler.CorefResolve(unresolved_text)
        print(resolved_text)

        # 2a) Temporal Information Extraction
        self.TIEModel()

        # 2b) Temporal Transitivity Event Filtering


        # 2c) Geological Entity Extraction


        # 3a) Semantic Role Labelling and Node Extraction


        # 3b) Time Normalisation


        # 4) KG Node Relationship Insertion

        return

if __name__=="__main__":
    from TimexNormModel import NERModel, TimexNormModel, TempRelModel
    from Resolvers import CorefResolve
    model = GeoTKGPipeline(
        DCT = "2024-03-22",
        coref_resolver=CorefResolve,
        geo_ner = NERModel("scripts\\results\\Geo-NER"),
        evt_ner = NERModel("scripts\\results\\EventTimex-NER"),
        time_norm = TimexNormModel("scripts\\results\\TimeNormBart"),
        #temp_rel = TempRelModel("scripts\\results\\TempRel\\checkpoint-10000")
    )
    text = "The Henry River Project began on the south-western limb of Perth in 2004. A year later, they discoverd a quartz vein formation. The formation was dated to the Archean. Other projects have found gold dated to the Jurassic or ~1000ma."
    text = "The mineralisation was characterised by traces of disseminated pyrite with zones of trace pyrrhotite and chalcopyrite in felsic schist."
    #text = "The Henry River Project began on the south-western limb of Perth in 2004. A year after the project started, they discoverd a quartz vein formation."
    #text = "1000ma 1000 ma 1000 Ma 1000.102 ma 1000.102ma ~1000ma ~1000 ma ~1000.22ma ~1000.22 ma"
    #text = "In 2019, BHP found a rock formation which they dated to the Archean."
    model.pred(text)

'''
The Henry River Project began on the south - western limb of Perth in 2004 . A year later , The Henry River Project discoverd a quartz vein formation . a quartz vein formation was dated to the Archean . Other projects have found gold dated to the Jurassic or ~1000ma .
------- Reconstructed Text --------
['The', 'Henry', 'River', 'Project', 'began', 'on', 'the', 'south', '-', 'western', 'limb', 'of', 'Perth', 'in', '2004', '.', 'A year', 'later', ',', 'The', 'Henry', 'River', 'Project', 'discoverd', 'a', 'quartz vein', 'formation', '.', 'a', 'quartz vein', 'formation', 'was', 'dated', 'to', 'the', 'Archean', '.', 'Other', 'projects', 'have', 'found', 'gold', 'dated', 'to', 'the', 'Jurassic', 'or', '~1000ma', '.']
----- Detected Calender Times -----
LOC: 14, TEXT: 2004, VAL: 2004, TYPE: DATE
LOC: 16, TEXT: A year, VAL: P1Y, TYPE: DURATION
------- Detected Geo Times --------
LOC: 35, TEXT: Archean, VAL: (4600.0, 2500.0)
LOC: 45, TEXT: Jurassic, VAL: (201.4, 145.0)
LOC: 47, TEXT: ~1000ma, VAL: (1000, None)
------ Detected Geo Entities ------
LOC: 12, TEXT: Perth
LOC: 25, TEXT: quartz vein
LOC: 29, TEXT: quartz vein
LOC: 41, TEXT: gold
--------- Detected Events ---------
LOC: 4, TEXT: began
LOC: 23, TEXT: discoverd
LOC: 32, TEXT: dated
LOC: 40, TEXT: found
LOC: 42, TEXT: dated
'''

'''
The mineralisation was characterised by traces of disseminated pyrite with zones of trace pyrrhotite and chalcopyrite in felsic schist .
------- Reconstructed Text --------
['The', 'mineralisation', 'was', 'characterised', 'by', 'traces', 'of', 'disseminated', 'pyrite', 'with', 'zones', 'of', 'trace', 'pyrrhotite', 'and', 'chalcopyrite', 'in', 'felsic', 'schist', '.']
----- Detected Calender Times -----
------- Detected Geo Times --------
------ Detected Geo Entities ------
LOC: 8, TEXT: pyrite
LOC: 13, TEXT: pyrrhotite
LOC: 15, TEXT: chalcopyrite
LOC: 17, TEXT: felsic
--------- Detected Events ---------
LOC: 3, TEXT: characterised
'''


