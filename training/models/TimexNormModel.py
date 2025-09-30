import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import isodate
from datetime import timedelta, datetime, date

class TimexNormModel():
    def __init__(self, path):
        # Recreate model & tokenizer the same way as during training
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        tokenizer.add_special_tokens({"additional_special_tokens": ["DCT:", "TYPE:", "TEXT:", "SPAN:"]})
        self.tokenizer = tokenizer

        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
        model.resize_token_embeddings(len(tokenizer))

        # Load weights
        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        self.model = model
        self.model.eval()

    def collator(self, inputs):
        enc = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    def predict(self, raw_inputs, dcts):
        time_decodings = []
        for raw_input, dct in zip(raw_inputs, dcts):
            inputs = self.collator(raw_input)
            gen_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=64,
                num_beams=6,
                early_stopping=True
            )
            decoded_preds = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            time_decodings.append([TimexNormModel.time_decode(gen_time, dct) for gen_time in decoded_preds])
        return time_decodings

    @staticmethod
    def gentext_to_iso8601(gentext: str):
        parsers = {
            isodate.parse_date,
            isodate.parse_datetime,
            isodate.parse_time,
            isodate.parse_duration,
        }
        for parser in parsers:
            try:
                output = parser(gentext)
                if output is not None:
                    return output
            except Exception:
                continue
            return None
    
    def time_decode(gentext: str, dct):
        if gentext[-3:] == "REF":
            gentext = dct
        parsed = TimexNormModel.gentext_to_iso8601(gentext)
        return parsed

        



        


    