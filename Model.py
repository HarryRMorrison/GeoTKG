import torch
from transformers import RobertaForTokenClassification, RobertaTokenizerFast, BartTokenizer, BartForConditionalGeneration

class Model:
    def __init__(self, model_path):
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
        self.model = RobertaForTokenClassification.from_pretrained(model_path)
        self.task = model_path.split('\\')[-1]

    def get_prediction(self, text, dim):
        encodings=self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=dim)
        return predictions

class NERModel(Model):
    def __init__(self, model_path):
        super().__init__(model_path)

    def predict(self, text, return_locations=False, return_decoded_tokens=False):
        predictions = self.get_prediction(text, dim=-1)
        return_pack = [predictions]

        if return_locations:
            if self.task == "Geo-NER":
                locations = NERModel.get_geo_entity_locations(predictions)
            else:
                locations = NERModel.get_event_locations(predictions)
            return_pack.append(locations)

        if return_decoded_tokens:
            encodings = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            decodings = self.tokenizer.convert_ids_to_tokens(encodings["input_ids"][0])
            out = []
            for token in decodings:
                out.append(token.strip("Ġ"))
            return_pack.append(out)
        return return_pack
    
    @staticmethod
    def get_geo_entity_locations(predictions, bi_map={0:6, 1:7, 2:8, 3:9, 4:10}):
        locations = []
        Bs = bi_map.keys()
        i = 0
        while i < len(predictions[0]):
            if predictions[0][i].item() in Bs:  # 'B-LOCATION': 0, 'B-MINERAL': 1, 'B-ORE_DEPOSIT': 2, 'B-ROCK': 3, 'B-STRAT': 4, 'B-TIMESCALE': 5
                start = i
                i += 1
                ent_type = predictions[0][i].item()
                while i < len(predictions) and predictions[i] == bi_map[ent_type]:  # 'I-LOCATION': 6, 'I-MINERAL': 7, 'I-ORE_DEPOSIT': 8, 'I-ROCK': 9, 'I-STRAT': 10, 'I-TIMESCALE': 11
                    i += 1
                locations.append([start, i])  # [start, end) format
            else:
                i += 1
        return locations

    @staticmethod
    def get_event_locations(predictions, bi_map={2:7}, return_types = False):
        labels = ["B-DATE", "B-DURATION", "B-EVENT", "B-SET", "B-TIME", "I-DATE", "I-DURATION", "I-EVENT", "I-SET", "I-TIME"]
        locations = []
        types = []
        Bs = bi_map.keys()
        i = 0
        while i < len(predictions[0]):
            if predictions[0][i].item() in Bs:  # B-Event: 2
                start = i
                i += 1
                ent_type = predictions[0][i].item()
                types.append(labels[ent_type])
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
        super().__init__(model_path)

    def predict(self, text):
        return self.get_prediction(text, 1)

# Assume only geo_time_locs
class TimexNormModel(Model):
    def __init__(self, model_path):
        self.tokenizer = BartTokenizer.from_pretrained(model_path)
        self.model = BartForConditionalGeneration.from_pretrained(model_path)

    def preprocessing(self, reconstruction, timex_types, DCT):
        text, timex_idxs, geo_time_idxs = reconstruction
        input_text = []
        timex_counter = 0

        for sent, timex_sent_ids in zip(text, timex_idxs):
            for timex_id in timex_sent_ids:
                sample = sent.copy()
                sample.insert(timex_id+1, "</timex>")
                sample.insert(timex_id, f"<timex type={timex_types[timex_counter]}>")
                timex_counter += 1
                sample.insert(0, f"normalise time <sep>{DCT}<sep> text:")
                input_text.append(sample)

        self.input_text = input_text
        self.geo_time_idxs = geo_time_idxs


        


    