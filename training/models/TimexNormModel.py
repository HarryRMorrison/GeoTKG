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

    def collator(self, text, times_locs, dct, roberta_tok):
        input_ids = roberta_tok(text, add_special_tokens=True, truncation=True, return_tensors="pt")['input_ids'][0]
        inputs = []
        for s, e, ty in times_locs:
            out_text = roberta_tok.decode(input_ids[max(0, s-100):min(len(input_ids), e+100)], skip_special_tokens=True)
            out_text = out_text
            input_text = f'DCT: {dct} \nTYPE: {ty} \nTEXT: {out_text} \nSPAN: \"{roberta_tok.decode(input_ids[s:e], skip_special_tokens=True)[1:]}\"'
            inputs.append(input_text)
        enc = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    def predict(self, text, time_locs, dct, roberta_tokenizer):
        inputs = self.collator(text, time_locs, dct, roberta_tokenizer)
        gen_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=64,
            num_beams=6,
            early_stopping=True
        )
        decoded_preds = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        print(decoded_preds)
        time_decode = [TimexNormModel.time_decode(time, dct) for time in decoded_preds]
        return time_decode

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
    
    def time_decode(gentext: str, dct: str):
        if gentext[-3:] == "REF":
            gentext = dct
        parsed = TimexNormModel.gentext_to_iso8601(gentext)
        if type(parsed) == isodate.duration.Duration:
            parsed = TimexNormModel.gentext_to_iso8601(dct) + parsed
        return parsed
    
    def to_datetime_abs(time, dct):
        # if type(time) == 
        return

        



        


    