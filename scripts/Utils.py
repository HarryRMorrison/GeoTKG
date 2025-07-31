import torch
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
import numpy as np

class Utils:
    def __init__(self, tokenizer, label_list):
        self.tokenizer = tokenizer
        self.label_list = label_list
    
    def compute_metrics(self, true_predictions, true_labels, average):
        return {
            "precision": precision_score(true_labels, true_predictions, average=average),
            "recall": recall_score(true_labels, true_predictions, average=average),
            "f1": f1_score(true_labels, true_predictions, average=average),
        }
    
    def tokenize_datasets(self, datasets):
        pass

    def classification_rep(self, predictions, labels, average, encodings):
        true_labels = []
        predicted_labels = []
        encodings.to("cpu")
        for i in range(len(predictions)):
            pred_ids = predictions[i].cpu().numpy()
            label_ids = labels[i]

            # Match only actual tokens (ignore padding)
            word_ids = encodings.word_ids(batch_index=i)
            aligned_preds = []
            aligned_labels = []

            previous_word_idx = None
            for j, word_idx in enumerate(word_ids):
                if word_idx is None or word_idx == previous_word_idx:
                    continue  # skip subwords and special tokens
                aligned_preds.append(self.label_list[pred_ids[j]])
                aligned_labels.append(self.label_list[label_ids[word_idx]])
                previous_word_idx = word_idx

            predicted_labels.append(aligned_preds)
            true_labels.append(aligned_labels)
        print(classification_report(true_labels, predicted_labels, average=average))
        

class NER_Utils(Utils):
    def __init__(self, tokenizer, label_list):
        super().__init__(tokenizer, label_list)

    def BIO_tokenize_and_align_labels(self, samples):
        tokenized_inputs = self.tokenizer(
            samples["tokens"], truncation=True, is_split_into_words=True, padding=True
        )
        labels = []
        for i, label in enumerate(samples["label"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
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
        return datasets.map(self.BIO_tokenize_and_align_labels, batched=True)
    
    def compute_metrics(self, eval_prediction):
        predictions, labels = eval_prediction
        predictions = np.argmax(predictions, axis=2)
        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        return super().compute_metrics(true_predictions, true_labels, "micro")
    
class TempRel_Utils(Utils):
    def __init__(self, tokenizer, label_list):
        super().__init__(tokenizer, label_list)

    def tokenize_datasets(self, datasets):
        return datasets.map(self.TempRel_tokenize, batched=True)
    
    def TempRel_tokenize(self, samples):
        return self.tokenizer(samples["tokens"], truncation=True, is_split_into_words=True, padding=True, return_tensors="pt")
    
    def compute_metrics(self, eval_prediction):
        predictions, labels = eval_prediction
        predictions = np.argmax(predictions, axis=1)
        true_labels = [[self.label_list[label[0]]] for label in labels]
        true_preds = [[self.label_list[pred]] for pred in predictions]
        return super().compute_metrics(true_preds, true_labels, "micro")
    
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

        # 4) Flatten to single lists
        preds  = [[item] for item in decoded_preds]
        labels = [[item] for item in decoded_labels]

        # 5) Compute your (relaxed) metrics
        return super().compute_metrics(preds, labels, "micro")
    
    def relaxed_score(self, predictions, truth_values):
        # 1) Decode predictions
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        # 2) Prepare & decode labels (replace -100 with pad_token_id so they decode cleanly)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 4) Flatten to single lists
        preds  = [[item] for item in decoded_preds]
        labels = [[item] for item in decoded_labels]

        


