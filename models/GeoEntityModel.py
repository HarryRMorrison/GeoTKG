import torch
import torch.nn as nn
from transformers import AutoModel
from seqeval.metrics import f1_score as seqeval_f1, classification_report as seqeval_cr
from models.globals import LABEL2ID_GEONER, ID2LABEL_GEONER, GEOENT_BI, GEOTIME_BI
from transformers import AutoTokenizer
TOKENIZER = AutoTokenizer.from_pretrained("roberta-large", add_prefix_space=True)
    
class GeoEntityModel(nn.Module):
    def __init__(self, base: str = "roberta-large", num_ner: int = len(LABEL2ID_GEONER), dropout: float = 0.1):
        super().__init__()
        # Transformer Encoder
        self.enc = AutoModel.from_pretrained(base)
        d = self.enc.config.hidden_size

        # NER Head
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(d, num_ner)
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=-100)

    def save(self, save_path):
        torch.save({"model_state_dict": self.state_dict()}, save_path)

    def forward(self, input_ids, attention_mask, labels=None):
        # 1) Encode
        H = self.enc(input_ids, attention_mask=attention_mask).last_hidden_state

        # 2) Decode
        logits = self.classifier(self.drop(H))  # [B, L, C]
        if labels is not None:
            loss = self.loss_ce(logits.view(-1, logits.size(-1)), labels.view(-1))
        else:
            loss = 0

        return {"loss": loss, "logits": logits}
    
    def evaluate_dataloader(self, dev_loader, average="micro", id2label_ner=ID2LABEL_GEONER):
        self.eval()
        device = next(self.parameters()).device
        ner_true_seqs, ner_pred_seqs = [], []
        losses = []

        with torch.no_grad():
            for batch in dev_loader:
                # move to device
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

                out = self.forward(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["ner_labels"]
                )

                # -------- NER F1 (seqeval) --------
                ner_logits   = out["logits"]              # [B,L,C]
                ner_pred_ids = ner_logits.argmax(-1)          # [B,L]
                ner_gold_ids = batch["ner_labels"]            # [B,L]
                losses.append(out["loss"].item())
                B, L = ner_gold_ids.shape

                for i in range(B):
                    ti = ner_gold_ids[i].tolist()
                    pi = ner_pred_ids[i].tolist()
                    true_seq, pred_seq = [], []
                    for t_id, p_id in zip(ti, pi):
                        if t_id == -100:
                            continue
                        true_seq.append(id2label_ner[t_id])
                        pred_seq.append(id2label_ner[p_id])
                    ner_true_seqs.append(true_seq)
                    ner_pred_seqs.append(pred_seq)
        metrics = {}
        metrics["f1"]  = seqeval_f1(ner_true_seqs, ner_pred_seqs, average=average)
        print(seqeval_cr(ner_true_seqs, ner_pred_seqs, digits=4))
        metrics["eval_loss"] = sum(losses) / len(losses)
        return metrics
    
    @torch.no_grad()
    def decode(self, logits):
        decoded = logits.argmax(-1)
        bi_map = {**GEOENT_BI, **GEOTIME_BI}
        Bs = bi_map.keys()
        times = []
        entities = []
        for example in decoded:
            i = 0
            temp_ti, temp_en = [], []
            while i < example.shape[0]:
                if example[i].item() in Bs:
                    start = i
                    ent_type = example[i].item()
                    is_timex = True if ent_type in GEOTIME_BI else False
                    i += 1
                    while i < len(example) and example[i] == bi_map[ent_type]:
                        i += 1
                    if is_timex: 
                        temp_ti.append((start, i, ID2LABEL_GEONER[ent_type]))
                    else: 
                        temp_en.append((start, i, ID2LABEL_GEONER[ent_type]))
                else:
                    i += 1
            times.append(temp_ti)
            entities.append(temp_en)

        return times, entities

    def predict(self, text_batch, return_tokens=False):
        self.eval()
        model_device = next(self.parameters()).device
        tokens = TOKENIZER(text_batch, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt")
        ids = tokens['input_ids'].to(model_device)
        attention_mask = tokens['attention_mask'].to(model_device)
        with torch.no_grad():
            out = self.forward(input_ids=ids, attention_mask=attention_mask, labels=None)
        if return_tokens:
            return self.decode(out['logits']), tokens
        return self.decode(out['logits'])