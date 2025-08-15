import torch
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
#from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class Utils:
    def __init__(self, tokenizer, label2id, id2label):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.id2label = id2label
    
    def compute_metrics(self, eval_prediction):
        pass
    
    def tokenize_datasets(self, datasets):
        pass
        
class NER_Utils(Utils):
    def __init__(self, tokenizer, label2id, id2label):
        super().__init__(tokenizer, label2id, id2label)

    def BIO_tokenize_and_align_labels(self, samples):
        tokenized_inputs = self.tokenizer(
            samples["tokens"], truncation=True, is_split_into_words=True, padding=True
        )
        all_label_ids = []
        for i, label_seq in enumerate(samples["label"]):
            label_seq = [
                self.label2id[l] if isinstance(l, str) else int(l)
                for l in label_seq
            ]
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label_seq[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            all_label_ids.append(label_ids)
        tokenized_inputs["labels"] = all_label_ids
        return tokenized_inputs

    def data_collator(self, data):
        input_ids = [torch.tensor(item["input_ids"]) for item in data]
        attention_mask = [torch.tensor(item["attention_mask"]) for item in data]
        labels = [torch.tensor(item["labels"]) for item in data]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def tokenize_datasets(self, datasets):
        datasets = datasets.map(self.BIO_tokenize_and_align_labels, batched=True)
        if "label" in datasets["train"].column_names:
            datasets = datasets.remove_columns("label")
        if "tokens" in datasets["train"].column_names:
            datasets = datasets.remove_columns("tokens")
        return datasets
    
    def compute_metrics(self, eval_prediction):
        predictions, labels = eval_prediction
        predictions = np.argmax(predictions, axis=2)
        # Remove ignored index (special tokens)
        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
            "classification_report": classification_report(true_labels, true_predictions),
        }
        
class TimexNorm_Utils(Utils):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize_datasets(self, datasets):
        return datasets.map(self.TimexNorm_tokenize, batched=True, remove_columns=["input_text","target_text"])
    
    def TimexNorm_tokenize(self, batch):
        model_inputs = self.tokenizer(batch["input_text"], truncation=True, padding="longest", max_length=512)
        labels       = self.tokenizer(batch["target_text"], truncation=True, padding="longest", max_length=32)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def extract_norm_values(output_text):
        return output_text
    
    def compute_metrics(self, eval_prediction):
        preds, labels = eval_prediction  # both are np.arrays of shape [batch_size, seq_len]

        # 1) Decode predictions
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        # 2) Prepare & decode labels (replace -100 with pad_token_id so they decode cleanly)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        relaxed_preds = self.tfidf_relaxed(decoded_labels, decoded_preds)

        # 4) Flatten to single lists
        strict_acc  = sum(1 for pred, lab in zip(decoded_preds, decoded_labels) if pred==lab)/len(decoded_labels)
        relaxed_acc = sum(1 for pred in relaxed_preds if pred==1)/len(decoded_labels)

        # 5) Compute your (relaxed) metrics
        return {
            "accuracy strict": strict_acc,
            "accuracy relaxed": relaxed_acc
        }
    
    def tfidf_relaxed(self, decoded_labels, decoded_preds):
        # 1) build a char-ngram TF-IDF vectorizer
        vec = TfidfVectorizer(analyzer='char', ngram_range=(3,5))
        all_strings = decoded_preds + decoded_labels
        X = vec.fit_transform(all_strings)

        # 2) split back into preds / labels
        P = X[:len(decoded_preds)]
        L = X[len(decoded_preds):]

        # 3) get pairwise cosines
        sim_scores = cosine_similarity(P, L).diagonal()
        relaxed_preds = []
        for i, sim in enumerate(sim_scores):
            if decoded_labels[i] == decoded_preds[i]:
                relaxed_preds.append(1)
            else:
                relaxed_preds.append(1 if sim > 0.75 else 0)

        return relaxed_preds
        


            


# def classification_report(self, test_set):
#     encodings=self.tokenizer(test_set["tokens"], padding=True, truncation=True, return_tensors="pt", is_split_into_words=True)
#     self.model.eval()
#     with torch.no_grad():
#         outputs = self.model(**encodings)
#         logits = outputs.logits
#         predictions = torch.argmax(logits, dim=-1)

#     preds = [self.id2label[pred.item()] for pred in list(predictions)]
#     ids = [self.id2label[lab] for lab in test_set["label"]]

#     print(classification_report(ids, preds))