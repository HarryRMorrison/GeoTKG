import torch
from transformers import RobertaForTokenClassification, RobertaTokenizerFast, BartTokenizer, BartForConditionalGeneration, RobertaForSequenceClassification
from pyrolite.util.time import Timescale
from transformers import pipeline
from spacy.matcher import Matcher
import spacy

class Model:
    def __init__(self, model_path):
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
        self.model = RobertaForTokenClassification.from_pretrained(model_path)
        self.task = model_path.split('\\')[-1]

    def get_prediction(self, text, dim, is_split_into_words=False):
        encodings=self.tokenizer(text, padding=True, truncation=True, return_tensors="pt", is_split_into_words=is_split_into_words)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=dim)
        return predictions

class NERModel(Model):
    def __init__(self, model_path):
        super().__init__(model_path)

    def predict(self, text):
        return self.get_prediction(text, dim=-1)
    
    def decode(self, text):
        encodings = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        decodings = self.tokenizer.convert_ids_to_tokens(encodings["input_ids"][0])
        out = "".join(decodings)
        out = out.split("Ġ")
        return decodings, out
    
    # 'B-LOCATION': 0, 'B-MINERAL': 1, 'B-ORE_DEPOSIT': 2, 'B-ROCK': 3, 'B-STRAT': 4, 'B-TIMESCALE': 5
    # 'I-LOCATION': 6, 'I-MINERAL': 7, 'I-ORE_DEPOSIT': 8, 'I-ROCK': 9, 'I-STRAT': 10, 'I-TIMESCALE': 11
    @staticmethod
    def get_geo_entity_locations(predictions, bi_map={0:6, 1:7, 2:8, 3:9, 4:10}):
        locations = []
        Bs = bi_map.keys()
        i = 0
        while i < len(predictions[0]):
            if predictions[0][i].item() in Bs:
                start = i
                ent_type = predictions[0][i].item()
                i += 1
                while i < len(predictions[0]) and predictions[0][i] == bi_map[ent_type]:
                    i += 1
                locations.append([start, i])  # [start, end) format
            else:
                i += 1
        return locations

    @staticmethod
    def get_event_locations(predictions, bi_map={2:7}, return_types = False):
        predictions = predictions[0]
        labels = ["B-DATE", "B-DURATION", "B-EVENT", "B-SET", "B-TIME", "I-DATE", "I-DURATION", "I-EVENT", "I-SET", "I-TIME"]
        locations = []
        types = []
        Bs = list(bi_map.keys())
        i = 0
        while i < len(predictions):
            if predictions[i].item() in Bs:  # B-Event: 2
                start = i
                i += 1
                ent_type = predictions[start].item()
                types.append(labels[ent_type][2:])
                while i < len(predictions) and predictions[i] == bi_map[ent_type]:  # I-Event: 7
                    i += 1
                locations.append([start, i])  # [start, end) format
            else:
                i += 1

        if return_types:
            return locations, types
        else:
            return locations
    
class TempRelModel(Model):
    def __init__(self, model_path):
        self.tokenizer = RobertaTokenizerFast.from_pretrained("scripts\\results\\TempRel")
        self.model = RobertaForSequenceClassification.from_pretrained("scripts\\results\\TempRel\\checkpoint-10000")
        self.task = model_path.split('\\')[-1]

    def preprocessing(self, tokens, timex_locs, timex_types, geo_locs, event_locs, DCT):
        ET_input_text, ET_order = [], []
        EE_input_text, EE_order = [], []
    
        for i, (loc, value) in enumerate(timex_locs):
            sample = tokens.copy()
            time = sample.pop(loc)
            sample.insert(loc, f"<timex VALUE={value} TYPE={timex_types[i]}>")
            sample.insert(loc+1, time)
            sample.insert(loc+2, "</timex>")
            for e_loc in event_locs:
                example = sample.copy()
                offset = 2 if e_loc>loc else 0
                event = example.pop(e_loc+offset)
                example.insert(e_loc+offset, "<e>")
                example.insert(e_loc+1+offset, event)
                example.insert(e_loc+2+offset, "</e>")
                example.insert(0, f"Document creation date is {DCT}")
                ET_input_text.append(" ".join(example).split(" "))
                print(ET_input_text[-1])
                ET_order.append((loc, e_loc))
        
        for loc, (s_val, e_val) in geo_locs:
            sample = tokens.copy()
            time = sample.pop(loc)
            sample.insert(loc, f"<timex VALUE={-1*((s_val+e_val)/2):.2f} TYPE=UNKOWN>")
            sample.insert(loc+1, time)
            sample.insert(loc+2, "</timex>")
            for e_loc in event_locs:
                example = sample.copy()
                offset = 2 if e_loc>loc else 0
                event = example.pop(e_loc+offset)
                example.insert(e_loc+offset, "<e>")
                example.insert(e_loc+1+offset, event)
                example.insert(e_loc+2+offset, "</e>")
                example.insert(0, f"Document creation date is {DCT}.")
                ET_input_text.append(" ".join(example).split(" "))
                print(ET_input_text[-1])
                ET_order.append((loc, e_loc))

        for e_loc1 in event_locs:
            sample = tokens.copy()
            event1 = sample.pop(e_loc1)
            sample.insert(e_loc1, "<e1>")
            sample.insert(e_loc1+1, event1)
            sample.insert(e_loc1+2, "</e1>")
            for e_loc2 in event_locs:
                if e_loc1==e_loc2:
                    continue
                example = sample.copy()
                offset = 2 if e_loc2>e_loc1 else 0
                event2 = example.pop(e_loc2+offset)
                example.insert(e_loc2+offset, "<e1>")
                example.insert(e_loc2+1+offset, event2)
                example.insert(e_loc2+2+offset, "</e1>")
                EE_input_text.append(" ".join(example).split(" "))
                print(EE_input_text[-1])
                EE_order.append((e_loc1, e_loc2))
        
        self.ET_input = ET_input_text
        self.EE_input = EE_input_text
        self.ET_order = ET_order
        self.EE_order = EE_order

    def predict(self):
        ET_preds = []
        EE_preds = []

        for ET_in, ET_locs in zip(self.ET_input, self.ET_order):
            ET_preds.append((ET_locs, self.get_prediction(ET_in, -1, is_split_into_words=True).item()))

        for EE_in, EE_locs in zip(self.EE_input, self.EE_order):
            EE_preds.append((EE_locs, self.get_prediction(EE_in, -1, is_split_into_words=True).item()))

        return ET_preds, EE_preds

# Assume only geo_time_locs
class TimexNormModel(Model):
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

        doc = nlp(" ".join(self.tokens))

        for _, start, end in matcher(doc):
            date = int(doc[start:end].text.lower().strip("~ma"))
            geo_times.append([start, (date, None)])
        
        return geo_times


        


    