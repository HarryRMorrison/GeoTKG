import torch
from transformers import RobertaForTokenClassification, RobertaTokenizerFast

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
                # if token == "<s>":
                #     sent = ["<s>"]
                # elif token == "</s>":
                #     sent.append("</s>")
                #     out.append(sent)
                # else:
                #     sent.append(token.strip("Ġ"))
                out.append(token.strip("Ġ"))
            return_pack.append(out)
        return return_pack
    
    @staticmethod
    def get_geo_entity_locations(predictions):
        locations = []
        bi_map = {0:6, 1:7, 2:8, 3:9, 4:10}
        i = 0
        while i < len(predictions[0]):
            if predictions[0][i].item() >= 0 and predictions[0][i].item() <= 4:  # 'B-LOCATION': 0, 'B-MINERAL': 1, 'B-ORE_DEPOSIT': 2, 'B-ROCK': 3, 'B-STRAT': 4, 'B-TIMESCALE': 5
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
    def get_event_locations(predictions):
        locations = []
        i = 0
        while i < len(predictions[0]):
            if predictions[0][i].item() == 2:  # B-Event
                start = i
                i += 1
                while i < len(predictions) and predictions[i] == 7:  # I-Event
                    i += 1
                locations.append([start, i])  # [start, end) format
            else:
                i += 1
        return locations
    
class TempRelModel(Model):
    def __init__(self, model_path):
        super().__init__(model_path)

    def predict(self, text):
        return self.get_prediction(text, 1)

    