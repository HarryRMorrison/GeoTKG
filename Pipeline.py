from DependencyParsing import DependencyParser
import torch
import numpy as np
from transformers import AutoTokenizer
import warnings
from transformers import logging as hf_logging
from datetime import timedelta, datetime, date
from isodate.duration import Duration
from models.globals import ID2LABEL_EE
import re
from copy import deepcopy
from models.TimexNormModel import TimexNormModel

# Suppress FutureWarnings
warnings.filterwarnings("ignore", message=".*resume_download.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*autocast.*", category=FutureWarning)

# Suppress Hugging Face model init warnings
hf_logging.set_verbosity_error()

TOKENIZER = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)


class GeoTKGPipeline:
    def __init__(self, model_type, uses_ca=True):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if model_type == "pipeline":
            self.is_pipe = True
            from models.TIEModel import TIEModel
            from models.GeoEntityModel import GeoEntityModel

            self.GeoNER = GeoEntityModel(base="roberta-base").to(device=self.device)
            load = torch.load("results\\geo_model\\geo_model.pt")
            self.GeoNER.load_state_dict(load['model_state_dict'])

            self.NormModel = TimexNormModel("results\\norm_model\\time_norm_epoch15.pt")
            if uses_ca:
                print("Loading Pipeline with CA :", uses_ca)
                tie_path = "results\\tie_model\\tie_model_epoch20.pt"
            else:
                print("Loading Pipeline without CA :", uses_ca)
                tie_path = "results\\tie_model_no_ca\\tie_model_NO_CA_epoch20.pt"
            self.TIEModel = TIEModel(use_ca=uses_ca).to(device=self.device)
            load = torch.load(tie_path)
            self.TIEModel.load_state_dict(load['model_state_dict'])
        elif model_type == "multi-task":
            if uses_ca:
                print("Loading Multi-task with CA :", uses_ca)
                path = "results\\geotkg\\geotkg_model_epoch20.pt"
            else:
                print("Loading Multi-task without CA :", uses_ca)
                path = "results\\geotkg_no_ca\\geotkg_model_epoch20.pt"
            from models.GeoTKG import GeoTKG
            self.GeoTKG = GeoTKG(use_ca=uses_ca).to(device=self.device)
            load = torch.load(path)
            self.GeoTKG.load_state_dict(load['model_state_dict'])
            self.is_pipe = False

        self.depparse = DependencyParser()

    def pred(self, batch_text, DCTs, mode):
        
        dcts = [TimexNormModel.gentext_to_iso8601(doctime) for doctime in DCTs]

        if self.is_pipe:
            # Temporal Information Extraction
            enc, events, timexs, et_preds, ee_triples, ee_mask = self.TIEModel.predict(batch_text)
            # Geological NER
            geo_times, geo_entities = self.GeoNER.predict(batch_text)

            word_maps = self.get_word_mappings(enc)
            event_spans_and_locs = self.get_spans(word_maps, events)
            timex_spans_and_locs = self.get_spans(word_maps, timexs)
            geotime_spans_and_locs = self.get_spans(word_maps, geo_times)
            geoent_spans_and_locs = self.get_spans(word_maps, geo_entities)

            norm_inputs = self.get_bart_normalisation_input(dcts, enc['input_ids'], timex_spans_and_locs)            
            normalised_times = self.NormModel.predict(norm_inputs, dcts)
        else:
            (enc, 
            word_maps,
            event_spans_and_locs, 
            timex_spans_and_locs, 
            normalised_times,
            et_preds, ee_triples, ee_mask,
            geoent_spans_and_locs, 
            geotime_spans_and_locs) = self.GeoTKG.predict(batch_text, dcts)
        
        events_backup = deepcopy(event_spans_and_locs)
        events_copy = deepcopy(event_spans_and_locs)

        triples = self.form_triples(event_spans_and_locs, ee_triples, ee_mask)

        if mode=="tie_isolation_testing": 
            timex_spans = [[[span, s, e, ty] for span, (s, e, ty) in list(timexs)] for timexs in timex_spans_and_locs]
            event_spans = [[[span, s, e, ty] for span, (s, e, ty) in list(events)] for events in events_backup]
            output = [{'times':btimes, 'events':bevents} for (btimes, bevents) in zip(timex_spans, event_spans)]
        elif mode=="geo_isolation_testing":
            all_geos = []
            for bi in range(len(batch_text)):
                temp = []
                for span, (s, e, ty) in list(geotime_spans_and_locs[bi]):
                    temp.append([span, s, e, ty])
                for span, (s, e, ty) in list(geoent_spans_and_locs[bi]):
                    temp.append([span, s, e, ty])
                temp = sorted(temp, key=lambda x : x[1])
                all_geos.append(temp)
            output = [{"geo":geo_ents} for geo_ents in all_geos]
        else:
            batch_quintuples = []
            batch_event_retention = []
            for bi in range(len(batch_text)):
                stripped_tokens = [TOKENIZER.decode(id_tok, skip_special_tokens=True).strip() for id_tok in enc['input_ids'][bi]]
                dep_out, retention = self.depparse(stripped_tokens, word_maps[1][bi], geoent_spans_and_locs[bi], geotime_spans_and_locs[bi], events_backup[bi])
                batch_quintuples.append(dep_out)
                batch_event_retention.append(retention)
            quintuples, timescale_ents = self.form_quintuples(events_copy, et_preds, normalised_times, batch_quintuples, dcts, batch_event_retention)

            output = [{"quintuples":quins, "triples":trips, "timescales":timescale_ent} for (quins, trips, timescale_ent) in zip(quintuples, triples, timescale_ents)]
        return output
        
    def get_word_mappings(self, enc):
        input_ids = enc["input_ids"]
        B, L = input_ids.shape
        word_ids_list = [enc.word_ids(bi) for bi in range(B)]
        word_maps = []
        for bi, (wids, ids) in enumerate(zip(word_ids_list, input_ids)):
            first_last = {}
            for ti, wid in enumerate(wids):
                if wid is None:
                    continue
                if wid not in first_last:
                    first_last[wid] = [ti, ti]
                else:
                    first_last[wid][1] = ti
            word_maps.append(first_last)
        return (word_maps, word_ids_list, input_ids)
    
    def get_spans(self, word_maps, bio_ranges):
        word_maps, word_ids_list, input_ids = word_maps
        locations = []
        for bi, first_last in enumerate(word_maps):
            wids = word_ids_list[bi]
            ids = input_ids[bi]
            new_locations = []
            token_slices = []
            for (s, e, _t) in bio_ranges[bi]:
                if s>=len(wids) or e>=len(wids) or s is None or e is None:
                    start = 1
                    end = 0
                else:
                    try:
                        s_word, e_word = wids[int(s)], wids[int(e)]
                    except:
                        print(wids, ids, s, e)
                    if s_word is None or e_word is None:
                        start = 1
                        end = 0
                    else:
                        start = first_last[s_word][0]
                        end   = first_last[e_word][1]
                token_slices.append(ids[start:end].tolist())
                new_locations.append([start,end, _t])
            decoded = TOKENIZER.batch_decode(token_slices, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            sample_spans = [re.sub(r"[.!?,<>\[\]}{;:\"']", "", raw_span.strip()) for raw_span in decoded]
            locations.append(zip(sample_spans, new_locations))

        return locations
    
    def get_bart_normalisation_input(self, dcts, input_ids, time_spans):
        inputs = []
        for dct, ids, times in zip(dcts, input_ids, time_spans):
            batch = []
            for span, (s, e, ty) in times:
                text_window = ids[max(0, s-100):min(len(ids), e+100)]
                mask = text_window==1
                out_text = TOKENIZER.decode(text_window[mask], skip_special_tokens=True)
                input_text = f'DCT: {dct} \nTYPE: {ty} \nTEXT: {out_text} \nSPAN: \"{span}\"'
                batch.append(input_text)
            inputs.append(batch)
        return inputs
    
    def get_start_end_times(self, event_times: list, dct):
        if len(event_times) == 0:
            return None, None
        dates = [time for time in event_times if type(time) == date]
        times = [time for time in event_times if type(time) == datetime]
        durs = [time for time in event_times if type(time) == Duration]

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
        if s_time==e_time:
            e_time = None
        return s_time, e_time
    
    def form_tie_quintuples(self, batch_events, batch_event_times, batch_normalised_times, dcts):
        quintuples = []
        for events, event_times, norm_times, dct in zip(batch_events, batch_event_times, batch_normalised_times, dcts):
            batch_quintuples = []
            for span, et_list in zip(events, event_times):
                if span == '':
                    continue
                event_time = [norm_times[i] for i, binary in enumerate(et_list) if binary.item()==1]
                s_time, e_time = self.get_start_end_times(event_time, dct)
                batch_quintuples.append({
                    'subject':None,
                    'event':span,
                    'object':None,
                    's_time':s_time,
                    'e_time':e_time
                })
            quintuples.append(batch_quintuples)
        return quintuples
    
    def form_quintuples(self, batch_events, batch_event_times, batch_normalised_times, batch_so, dcts, batch_event_retention):
        quintuples = []
        timescale_ents = []
        got_so = []
        for events, event_times, norm_times, so, dct, retained in zip(batch_events, batch_event_times, batch_normalised_times, batch_so, dcts, batch_event_retention): 
            batch_quintuples = []
            bacth_timscale_ents = []
            events = list(events)
            for (span,(s,e,ty)), et_list, so_index in zip(events, event_times, retained):
                timescales_nums = [[],[]]
                if span == '' or so_index==-100:
                    continue
                got_so.append(so)
                event_time = [norm_times[i] for i, binary in enumerate(et_list) if binary.item()==1]
                s_time, e_time = self.get_start_end_times(event_time, dct)
                subject = so[so_index]['subject']
                object = so[so_index]['object']

                if subject is not None:
                    ev_subs = []
                    for entity in subject:
                        if entity['timescale']!=[]:
                            for timescale in entity['timescale']:
                                bacth_timscale_ents.append([entity['text'], timescale['norm_min'], timescale['norm_max']])
                                timescales_nums[0].append(timescale['norm_min'])
                                timescales_nums[1].append(timescale['norm_max'])
                        ev_subs.append(entity["text"])

                if object is not None:
                    ev_objs = []
                    for entity in object:
                        if entity['timescale']!=[]:
                            for timescale in entity['timescale']:
                                bacth_timscale_ents.append([entity['text'], timescale['norm_min'], timescale['norm_max']])
                                timescales_nums[0].append(timescale['norm_min'])
                                timescales_nums[1].append(timescale['norm_max'])
                        ev_objs.append(entity["text"])

                if s_time is None and e_time is None:
                    if timescales_nums[0] != []:
                        s_time = max(timescales_nums[0])
                    if timescales_nums[1] != []:
                        e_time = min(timescales_nums[1])

                batch_quintuples.append({
                    'subject':ev_subs if subject is not None else None,
                    'event':span,
                    'object':ev_objs if object is not None else None,
                    's_time':s_time,
                    'e_time':e_time
                })
            quintuples.append(batch_quintuples)
            timescale_ents.append(bacth_timscale_ents)
        return quintuples, timescale_ents
                
    def form_triples(self, batch_events, batch_temprels, ee_mask):
        triples = []
        for bi, (ee_temprels, mask) in enumerate(zip(batch_temprels, ee_mask)):
            batch_triples = []
            events = list(batch_events[bi])
            for e1, e2, rel in ee_temprels[mask]:
                e1, e2, rel = e1.item(), e2.item(), rel.item()
                if events[e1][0] == '' or events[e2][0] == '':
                    continue
                batch_triples.append((events[e1][0], ID2LABEL_EE[rel], events[e2][0]))
            triples.append(batch_triples)
        return triples

    def put_into_neo4j_graph(quintuples, triples, timescales):

        return
if __name__=="__main__":
    import json
    model_paths = {
        "geo_tkg": "results\\geotkg\\geotkg_model_epoch20.pt"
    }
    model = GeoTKGPipeline(model_paths=model_paths, uses_ca=True)
    with open('cleandata\geological_tkg_test.json', 'r') as file:
        test_data = json.load(file)
    out_preds = []
    for sample in test_data:
        dcts = sample['dct']
        text = sample['text']
        output = model.pred([text], [dcts], mode="default")
        out_preds.extend(output)
        print(f"Processed {len(out_preds)}/{len(test_data)}")
        print(output[0]['quintuples'])

    

