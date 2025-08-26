import torch
from transformers import BartTokenizer, BartForConditionalGeneration, RobertaForSequenceClassification
from pyrolite.util.time import Timescale
from transformers import pipeline
from spacy.matcher import Matcher
import spacy
from spacy.tokens import Doc

class TimexNormModel():
    def __init__(self, model_path):
        self.normalizer = pipeline(
            "text2text-generation",
            model=model_path,
            tokenizer=model_path,
            device=0  # or -1 for CPU
        )

    def preprocessing(self, tokens, timex_locs, geo_time_locs, timex_types, DCT):
        input_text = []
        for i, loc in enumerate(timex_locs):
            sample = tokens.copy()
            time = sample.pop(loc)
            sample.insert(loc, f"<timex type={timex_types[i]}>{time}</timex>")
            sample.insert(0, f"normalise time <sep>{DCT}<sep> text:")
            input_text.append(" ".join(sample))

        self.tokens = tokens
        self.input_text = input_text
        self.geo_time_idxs = geo_time_locs
        self.time_idxs = timex_locs

    def predict(self):
        geo_times = self.geo_timescale()

        results = self.normalizer(
            self.input_text,
            max_length=30,
            num_beams=5,
        )

        cal_times = [[index,result["generated_text"]] for index, result in zip(self.time_idxs, results)]

        return cal_times, geo_times
    
    # Need to add text matching for "ma"
    def geo_timescale(self):
        ts = Timescale()
        geo_times = []
        for loc in self.geo_time_idxs:
            min, max = ts.text2age(self.tokens[loc])
            geo_times.append([loc, (min, max)])

        nlp = spacy.blank("en")
        matcher = Matcher(nlp.vocab)

        pattern_no_space = [
            {"TEXT": {"REGEX": r"^~?\d+(\.\d+)?ma$"}}
        ]
        # with-space: “~1000 ma” or “1000.00 ma”
        pattern_with_space = [
            {"TEXT": {"REGEX": r"^~?\d+(\.\d+)?$"}},
            {"LOWER": "ma"}
        ]
        matcher.add("GEO_DATE", [pattern_no_space, pattern_with_space])

        doc = Doc(nlp.vocab, self.tokens)

        for _, start, end in matcher(doc):
            date = int(doc[start:end].text.lower().strip("~ma"))
            geo_times.append([start, (date, None)])
        
        return geo_times


        


    