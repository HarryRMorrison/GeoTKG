from SRL import SRL
from geotkg.models.TIEModel import TIEModel
from geotkg.models.GeoEntityModel import GeoEntityModel
from geotkg.models.TimexNormModel import TimexNormModel
from CorefResolver import CorefResolver
import torch
import numpy as np
from transformers import AutoTokenizer
import warnings
from transformers import logging as hf_logging
from datetime import timedelta, datetime, date
from isodate.duration import Duration
from geotkg.models.globals import ID2LABEL_EE
import re

# Suppress FutureWarnings
warnings.filterwarnings("ignore", message=".*resume_download.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*autocast.*", category=FutureWarning)

# Suppress Hugging Face model init warnings
hf_logging.set_verbosity_error()

TOKENIZER = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)

class GeoTKGPipeline:
    def __init__(self):
        #self.GeoNER = GeoEntityModel()
        #load = torch.load("results\\geo_model\\geo_model_test.pt")
        #self.GeoNER.load_state_dict(load['model_state_dict'])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.TIEModel = TIEModel().to(device=self.device)
        load = torch.load("geotkg\\results\\tie_model\\tie_model_epoch20.pt")
        self.TIEModel.load_state_dict(load['model_state_dict'])

        self.NormModel = TimexNormModel("geotkg\\results\\norm_model\\time_norm_epoch15.pt")

        #self.corefresolver = CorefResolver()
        #self.slr = SRL()
        #self.kgconstructor = KGConstructor()

    def pred(self, batch_text, DCTs, return_ner_results=False):
        
        #batched_resolved_text = [self.corefresolver(text) for text in batch_unresolved_text]
        dcts = [TimexNormModel.gentext_to_iso8601(doctime) for doctime in DCTs]

        # Temporal Information Extraction
        enc, events, times, et_preds, ee_triples, ee_mask = self.TIEModel.predict(batch_text)

        word_maps = self.get_word_mappings(enc)
        event_spans = self.get_spans(word_maps, events)
        time_spans_and_locs = self.get_spans(word_maps, times, return_token_slices=True)

        norm_inputs = self.get_bart_normalisation_input(dcts, enc['input_ids'], time_spans_and_locs)            
        normalised_times = self.NormModel.predict(norm_inputs, dcts)

        quintuples = self.form_quintuples(event_spans, et_preds, normalised_times, dcts)
        triples = self.form_triples(event_spans, ee_triples, ee_mask)
        
        if return_ner_results:
            time_spans = [[[span, ty] for span, (s, e, ty) in times] for times in self.get_spans(word_maps, times, return_token_slices=True)]
            event_spans = [[[span, ty] for span, (s, e, ty) in events] for events in self.get_spans(word_maps, events, return_token_slices=True)]
            output = [{"quintuples":quins, "triples":trips, 'times':btimes, 'events':bevents} for (quins, trips, btimes, bevents) in zip(quintuples, triples, time_spans, event_spans)]
        else:
            output = [{"quintuples":quins, "triples":trips} for (quins, trips) in zip(quintuples, triples)]

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
    
    def get_spans(self, word_maps, bio_ranges, return_token_slices = False):
        word_maps, word_ids_list, input_ids = word_maps
        spans = []
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
                token_slices.append(ids[start:end+1].tolist())
                new_locations.append((start,end+1, _t))
            decoded = TOKENIZER.batch_decode(token_slices, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            sample_spans = [re.sub(r"[.!?,<>\[\]}{;:\"']", "", raw_span.strip()) for raw_span in decoded]
            spans.append(sample_spans)
            locations.append(zip(sample_spans, new_locations))

        if return_token_slices:
            return locations
        else:
            return spans
    
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
    
    def form_quintuples(self, batch_events, batch_event_times, batch_normalised_times, dcts):
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
    
    def form_triples(self, batch_events, batch_temprels, ee_mask):
        triples = []
        for events, ee_temprels, mask in zip(batch_events, batch_temprels, ee_mask):
            batch_triples = []
            for e1, e2, rel in ee_temprels[mask]:
                e1, e2, rel = e1.item(), e2.item(), rel.item()
                if events[e1] == '' or events[e2] == '':
                    continue
                batch_triples.append((events[e1], ID2LABEL_EE[rel], events[e2]))
            triples.append(batch_triples)
        return triples

if __name__=="__main__":
    # model = GeoTKGPipeline()
    # text = "The head of the UN nuclear watchdog agency Mohamed ElBaradei Saturday received the 2005 Nobel Peace Prize and called for a world free of atomic weapons , saying   existing nuclear states should lead   by example . \"If we hope   to escape   self - destruction , then nuclear weapons should have   no place in our collective conscience , and no role in our security , \" ElBaradei said   in his acceptance speech   at a ceremony   in Oslo 's City Hall . \"We must ensure , absolutely , that no more countries acquire   nuclear weapons : that nuclear weapon states take   concrete steps towards nuclear disarmament ; and we must put   in place a security system that does not rely   on nuclear deterrence , \" he added .ElBaradei and the International Atomic Energy Agency ( IAEA ) , represented   by the chairman of its board of governors , Yukiya Amano , were jointly honored   on Saturday   for \" their efforts to prevent   nuclear energy from being used   for military purposes\" . They received   their distinction from the chairman of the Nobel Committee Ole Mjoes 60 years   after the United States dropped   two atomic bombs on Hiroshima and Nagasaki in Japan on August 6 and 9 , 1945 , the world 's only nuclear attacks .\"At a time when the threat of nuclear arms is again increasing , the Norwegian Nobel Committee wishes   to underline   that this threat must be met   through the broadest possible international cooperation said .In his acceptance speech , ElBaradei emphasized   that the threat of nuclear proliferation was closely linked   to inequalities in the world . \"In regions where conflicts have been left   to fester   for decades , countries continue   to look   for ways to offset   their insecurities or project   their power ... They may be tempted   to seek   their own weapons of mass destruction , like others who have preceded   them , \" he said . Fifteen years   after the Cold War came   to an end , the IAEA chief lamented   that \" we may have torn   down the walls between East and West , but we have yet to build   the bridges between North and South , the rich and the poor . \"To rid   the world of the threat of nuclear weapons , \" a good start would be if the nuclear weapons states reduced   the strategic role given to these weapons , \" he said . \" Today , eight or nine countries possess   nuclear weapons . Today we still have   27,000 warheads in existence . To me , this is 27,000 too many , \" he added .The IAEA and its chief have most recently been instrumental   in thorny nuclear negotiations   with Iran , threatening   to take   the country before the UN Security Council for violating   nuclear non - proliferation rules . Iran has insisted   that its nuclear program is merely designed   to meet   domestic energy needs , while the United States , Israel and others have charged   it is a cover for a programme   to develop   an atom bomb . On Saturday , ElBaradei said   that to avoid   such ambiguity , he planned   to set   up a \" reserve fuel bank \" under IAEA control . \"This assurance of supply will remove   the incentive , and the justification , for each country to develop   its own fuel cycle , \" he said . Ending   his speech   on an upbeat note , ElBaradei asked   the audience to \" imagine   what would happen   if the nations of the world spent   as much on development as on the machines of war . \"Imagine that the only nuclear weapons remaining are the relics in our museums . Imagine   the legacy we could leave   to our children . Imagine that such a world is actually within our grasp , \" he concluded .The agency and its director received   their award , consisting split   between them , in a brightly decorated City Hall , decked ceremony   on Saturday , the anniversary of the death   of prize founder Alfred Nobel , the winners of this year 's literature , medicine , physics , chemistry and economics prizes received   their awards from King Carl XVI Gustaf in Stockholm 's Concert Hall . That ceremony   was to be followed   by a gala banquet at Stockholm 's City Hall for 1,300 guests ."
    # test = "Back to the RBA's statement and taking a look at the all-important final paragraphs, which indicate it's taking a wait-and-see approach as the three interest rate cuts so far this year filter through the economy."
    # dct = "2005-12-10"
    # quins, trips = model.pred([text, test], [dct, "2025-09-09"])

    # for q in quins:
    #     print(q)

    # for t in trips:
    #     print(t)

    import json
    from Pipeline import GeoTKGPipeline

    with open("D:\\GeoTKG\\cleandata\\tie\\test.json", "r") as f:
        examples=[json.loads(line) for line in f]
    preds = []
    model = GeoTKGPipeline()
    for i in range(0, len(examples), 8):
        samples = examples[i:i+8]
        dcts = [inst['value'] for sample in samples for inst in sample['instances'] if inst['type'] != "EVENT" and inst['id'] == 0]
        text = [" ".join([wrd for sent in sample['text'] for wrd in sent]) for sample in samples]
        output = model.pred(text, dcts)
        preds.extend(output)
        print(i)
    

