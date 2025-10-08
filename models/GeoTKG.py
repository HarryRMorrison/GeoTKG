import torch
import torch.nn as nn
from models.TIEModel import TIEModel
from models.GeoEntityModel import GeoEntityModel
from models.TimexNormUtils import compute_metrics as comput_norm_metrics
from transformers import AutoModelForSeq2SeqLM, BartTokenizer, AutoTokenizer
import re
import isodate
from datetime import timedelta, datetime, date
from copy import deepcopy

TOKENIZER = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)

class BartSeq2SeqFineTuner(nn.Module):
    def __init__(self, model_name: str = "facebook/bart-base", label_smoothing: float = 0.0):
        super().__init__()
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        tokenizer.add_special_tokens({"additional_special_tokens": ["DCT:", "TYPE:", "TEXT:", "SPAN:"]})
        self.tokenizer = tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.label_smoothing = label_smoothing  # passed to loss via config if desired
        self.model.resize_token_embeddings(len(tokenizer))

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        **gen_kwargs,  # for generate()
    ):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,  # triggers seq2seq CE loss (label -100 ignored)
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )
        return out  # has .loss (if labels) and .logits (decoder vocab logits)
    
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
        parsed = BartSeq2SeqFineTuner.gentext_to_iso8601(gentext)
        return parsed

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, **kwargs):
        """Convenience passthrough for inference."""
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    
    @torch.no_grad()
    def collator(self, inputs):
        enc = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        return enc
    
    @torch.no_grad()
    def predict(self, raw_inputs, dcts):
        model_device = next(self.parameters()).device
        time_decodings = []
        for raw_input, dct in zip(raw_inputs, dcts):
            if raw_input == []:
                time_decodings.append([])
                continue
            encodings = self.collator(raw_input).to(model_device)
            gen_ids = self.model.generate(
                input_ids=encodings["input_ids"],
                attention_mask=encodings["attention_mask"],
                max_new_tokens=64,
                num_beams=6,
                early_stopping=True
            )
            decoded_preds = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            time_decodings.append([BartSeq2SeqFineTuner.time_decode(gen_time, dct) for gen_time in decoded_preds])
        return time_decodings

class GeoTKG(nn.Module):
    def __init__(self, use_ca=True):
        super().__init__()
        self.tie_model = TIEModel(use_ca=use_ca)
        self.geo_model = GeoEntityModel()
        self.norm_model = BartSeq2SeqFineTuner()

    def save(self, save_path):
        torch.save({'model_state_dict':self.state_dict()}, save_path)

    def forward(self, tie_batch, geo_batch, norm_batch):
        tie_out = self.tie_model(
                input_ids=tie_batch["input_ids"],
                attention_mask=tie_batch["attention_mask"],
                ev_starts=tie_batch["ev_starts"], ev_ends=tie_batch["ev_ends"], ev_mask=tie_batch["ev_mask"], e_sent_ids=tie_batch["e_sent_ids"],
                ti_starts=tie_batch["ti_starts"], ti_ends=tie_batch["ti_ends"], ti_mask=tie_batch["ti_mask"], t_sent_ids=tie_batch["t_sent_ids"],
                ner_gold_labels=tie_batch["ner_labels"],
                ev_ti_gold=tie_batch["ev_ti_gold"],
                ee_rel_gold=tie_batch["ee_triples"],
                ee_mask=tie_batch["ee_mask"],
            )
        geo_out = self.geo_model(
                input_ids=geo_batch["input_ids"],
                attention_mask=geo_batch["attention_mask"],
                labels=geo_batch["ner_labels"],
            )
        norm_out = self.norm_model(**norm_batch)

        norm_loss = norm_out.loss
        tie_loss = tie_out['loss']
        geo_loss = geo_out['loss']

        return {'tie_loss':tie_loss, 'geo_loss':geo_loss, 'norm_loss':norm_loss}
    
    def evaluate_dataloaders(self, tie_loader, geo_loader, norm_loader):
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            tie_metrics = self.tie_model.evaluate_dataloader(tie_loader)
            geo_metrics = self.geo_model.evaluate_dataloader(geo_loader)

        norm_eval_loss = 0.0
        decoded_preds, decoded_labels = [], []

        with torch.no_grad():
            for batch in norm_loader:
                # keep a copy of labels for loss and for decoding ground truth
                labels_copy = batch["labels"].clone()
                batch = {k: v.to(device) for k, v in batch.items()}

                # loss
                out = self.norm_model(**batch)
                norm_eval_loss += out.loss.item()

                # predictions
                gen_ids = self.norm_model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_new_tokens=64,
                    num_beams=4,
                    early_stopping=True
                )
                decoded_preds.extend(self.norm_model.tokenizer.batch_decode(gen_ids, skip_special_tokens=True))
                labels_copy[labels_copy == -100] = self.norm_model.tokenizer.pad_token_id
                decoded_labels.extend(self.norm_model.tokenizer.batch_decode(labels_copy, skip_special_tokens=True))

        avg_norm_eval_loss = norm_eval_loss / max(1, len(norm_loader))
        norm_metrics = comput_norm_metrics(decoded_labels, decoded_preds)

        out = {}
        out['eval_loss'] = tie_metrics['eval_loss'] + geo_metrics['eval_loss'] + avg_norm_eval_loss
        out['eval_tie_loss'] = tie_metrics['eval_loss']
        out['eval_geo_loss'] = geo_metrics['eval_loss']
        out['eval_norm_loss'] = avg_norm_eval_loss
        out['et_ner_f1'] = tie_metrics['ner_f1']
        out['geo_ner_f1'] = geo_metrics['f1']
        out['et_f1'] = tie_metrics['et_f1']
        out['ee_f1'] = tie_metrics['ee_f1']
        out['norm_strict'] = norm_metrics["accuracy strict"]
        out['norm_relaxed'] = norm_metrics["accuracy relaxed"]
        return out
    
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
    
    def predict(self, batch_text, dcts):
        self.eval()
        # Temporal Information Extraction
        enc, events, timexs, et_preds, ee_triples, ee_mask = self.tie_model.predict(batch_text)
        geo_times, geo_entities = self.geo_model.predict(batch_text)

        word_maps = self.get_word_mappings(enc)
        event_spans_and_locs = self.get_spans(word_maps, events)
        events_backup = deepcopy(event_spans_and_locs)
        timex_spans_and_locs = self.get_spans(word_maps, timexs)
        timex_backup = deepcopy(timex_spans_and_locs)
        geotime_spans_and_locs = self.get_spans(word_maps, geo_times)
        geoent_spans_and_locs = self.get_spans(word_maps, geo_entities)

        norm_inputs = self.get_bart_normalisation_input(dcts, enc['input_ids'], timex_spans_and_locs)            
        normalised_times = self.norm_model.predict(norm_inputs, dcts)

        return (enc, 
                word_maps,
                events_backup, 
                timex_backup, 
                normalised_times,
                et_preds, ee_triples, ee_mask,
                geoent_spans_and_locs, 
                geotime_spans_and_locs)



