import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import pipeline
import isodate


class TimexNormModel():
    def __init__(self, path):
        super().__init__()

    @staticmethod
    def gentext_to_iso8601(gentext: str):
        parsers = {
            isodate.parse_date:"DATE",
            isodate.parse_datetime:"TIME",
            isodate.parse_time:"TIME",
            isodate.parse_duration:"DURATION",
            isodate.parse_tzinfo:"SET",
        }

        for parser in parsers:
            try:
                out = parser(gentext)
                type_ = parsers[parser]
                return out, type_
            except Exception:
                continue

        # If none of the parsers worked
        #print(f"UNREC: {gentext}")
        return None
    
    def time_decode(gentext: str, dct: str):
        if gentext[-3:] == "REF":
            gentext = dct
        time, time_type = TimexNormModel.gentext_to_iso8601(gentext)
        return time, time_type
    
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

        



        


    