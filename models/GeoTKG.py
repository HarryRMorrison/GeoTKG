import torch
import torch.nn as nn
from models.TIEModel import TIEModel
from models.GeoEntityModel import GeoEntityModel
from models.TimexNormUtils import compute_metrics as comput_norm_metrics
from transformers import AutoModelForSeq2SeqLM

class BartSeq2SeqFineTuner(nn.Module):
    def __init__(self, model_name: str = "facebook/bart-base", label_smoothing: float = 0.0):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.label_smoothing = label_smoothing  # passed to loss via config if desired

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

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, **kwargs):
        """Convenience passthrough for inference."""
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

class GeoTKG(nn.Module):
    def __init__(self):
        super().__init__()
        self.tie_model = TIEModel()
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

        loss = tie_loss + geo_loss + norm_loss

        return {'loss':loss, 'tie_loss':tie_loss, 'geo_loss':geo_loss, 'norm_loss':norm_loss}
    
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
