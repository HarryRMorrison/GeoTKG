from SRL import SRL
from training.models.TIEModel import TIEModel
from training.models.GeoEntityModel import GeoEntityModel
from training.models.TimexNormModel import TimexNormModel
from KGConstructor import KGConstructor
from CorefResolver import CorefResolver
import torch
import numpy as np
from transformers import AutoTokenizer
import warnings
from transformers import logging as hf_logging
from datetime import timedelta, datetime, date
from training.models.globals import ID2LABEL_EE

# Suppress FutureWarnings
warnings.filterwarnings("ignore", message=".*resume_download.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*autocast.*", category=FutureWarning)

# Suppress Hugging Face model init warnings
hf_logging.set_verbosity_error()

TOKENIZER = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)

class GeoTKGPipeline:
    def __init__(self, DCTs):
        self.TIEModel = TIEModel()
        load = torch.load("training\\results\\tie_model\\tie_model_epoch15.pt")
        self.TIEModel.load_state_dict(load['model_state_dict'])

        self.NormModel = TimexNormModel("training\\results\\norm_model\\time_norm_epoch15.pt")
        self.DCT = TimexNormModel.gentext_to_iso8601(DCTs)

    def pred(self, text):

        # Temporal Information Extraction
        tokens, events, times, et_preds, ee_triples, ee_mask = self.TIEModel.predict(text)
        # Time Normalisation
        normalised_times = self.NormModel.predict(text[0], times[0], self.DCT, TOKENIZER)

        event_times = [[[normalised_times[i],times[0][i][2]] for i in range(len(bina)) if bina[i].item()==1] for bina in et_preds[0]]
        quintuples = []
        for i, (s, e, ty) in enumerate(events[0]):
            st, et = GeoTKGPipeline.get_start_end_times(event_times[i], self.DCT)
            decoded_str = TOKENIZER.decode(tokens['input_ids'][0][s:e], skip_special_tokens=True)[1:]
            quintuples.append([None, decoded_str, None, st, et])

        event_triples = []
        for i, (e1, e2, rel) in enumerate(ee_triples[0]):
            e1, e2, rel = e1.item(), e2.item(), rel.item()
            event_triples.append([quintuples[e1][1], ID2LABEL_EE[rel], quintuples[e2][1]])
        
        return quintuples, event_triples
    
    def get_start_end_times(event_times: list, dct):
        if len(event_times) == 0:
            return None, None
        dates = [time[0] for time in event_times if time[1] == "DATE"]
        times = [time[0] for time in event_times if time[1] == "TIME"]
        durs = [time[0] for time in event_times if time[1] == "DURATION"]

        if len(dates)>0:
            s_time = min(dates)
            e_time = max(dates)
        elif len(times)>0:
            s_time = min(times)
            e_time = max(times)
        else:
            s_time = dct
            e_time = dct
        
        for time in times:
            value = time
            if type(s_time) == date:
                value = time.date()
            if s_time >= value:
                s_time = time
            elif e_time <= value:
                e_time = time
        
        s_time = datetime.combine(s_time, datetime.min.time()) if type(s_time)==date else s_time
        e_time = datetime.combine(e_time, datetime.min.time()) if type(e_time)==date else e_time

        for duration in durs:
            if s_time + duration > e_time:
                e_time = s_time + duration

        return s_time, e_time

if __name__=="__main__":
    import json
    with open("D:\\GeoTKG\\cleandata\\tie\\test.json", "r") as f:
        examples=[json.loads(line) for line in f]
    preds = []
    model = GeoTKGPipeline(DCTs = "2018-12-17")
    for i, sample in enumerate(examples):
        text = " ".join([wrd for sent in sample['text'] for wrd in sent])
        preds.append([model.pred([text])])
        print(i)
    
    with open("GeoTKG-TIE-test-preds.json", 'w') as json_file:
        for sample in preds:
            json_file.write(json.dumps(sample)+"\n")

