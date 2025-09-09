import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import isodate

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

    def collator(self, text, times_locs, dct):
        inputs = []
        for s, t in times_locs:
            out_text = text[max(0, s-100):min(len(text), s+100)]
            out_text = " ".join(out_text)
            input_text = f'DCT: {dct} \nTYPE: {t} \nTEXT: {text} \nSPAN: \"{text[s]}\"'
            inputs.append(input_text)
        enc = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    def predict(self, text, time_locs, dct):
        inputs = self.collator(text, time_locs, dct)
        gen_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=64,
            num_beams=4,
            early_stopping=True
        )
        decoded_preds = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        time_decode = [TimexNormModel.time_decode(time, dct) for time in decoded_preds]
        return time_decode

    @staticmethod
    def gentext_to_iso8601(gentext: str):
        parsers = {
            isodate.parse_date,
            isodate.parse_datetime,
            isodate.parse_time,
            isodate.parse_duration,
            isodate.parse_tzinfo,
        }
        for parser in parsers:
            try:
                return parser(gentext)
            except Exception:
                continue
            return None
    
    def time_decode(gentext: str, dct: str):
        if gentext[-3:] == "REF":
            gentext = dct
        return TimexNormModel.gentext_to_iso8601(gentext)
    
    def to_datetime_abs(time, dct):
        # if type(time) == 
        return
    
    def get_start_and_end_times(all_gentext, dct):
        '''
            all_gentext: [B, Ne, Nt] but only linked Nts
        '''
        time_trips = []
        for batch in all_gentext:
            batch_times = []
            batch_types = []
            for event in batch:
                for time in event:
                    time_norm, time_type = TimexNormModel.time_decode(time, dct)

        return time_trips

        



        


    