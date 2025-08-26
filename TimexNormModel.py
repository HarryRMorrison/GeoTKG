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

    def preprocessing(self, tokens, timex_locs, timex_types, DCT):
        input_text = []
        for i, loc in enumerate(timex_locs):
            sample = tokens.copy()
            time = sample.pop(loc)
            sample.insert(loc, f"<timex type={timex_types[i]}>{time}</timex>")
            sample.insert(0, f"normalise time <sep>{DCT}<sep> text:")
            input_text.append(" ".join(sample))

        self.tokens = tokens
        self.input_text = input_text
        self.time_idxs = timex_locs

    def predict(self):
        results = self.normalizer(
            self.input_text,
            max_length=30,
            num_beams=5,
        )
        cal_times = [[index,result["generated_text"]] for index, result in zip(self.time_idxs, results)]
        return cal_times


        


    