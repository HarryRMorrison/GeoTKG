import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from transformers import BartTokenizer

TOKENIZER = BartTokenizer.from_pretrained("facebook/bart-base")

def tokenize_datasets(datasets):
    return datasets.map(TimexNorm_tokenize, batched=True, remove_columns=["input_text","target_text"])

def TimexNorm_tokenize(batch):
    model_inputs = TOKENIZER(batch["input_text"], truncation=True, padding="longest", max_length=512)
    labels       = TOKENIZER(batch["target_text"], truncation=True, padding="longest", max_length=32)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def extract_norm_values(output_text):
    return output_text
    
def compute_metrics(eval_prediction):
    preds, labels = eval_prediction  # both are np.arrays of shape [batch_size, seq_len]

    # 1) Decode predictions
    decoded_preds = TOKENIZER.batch_decode(preds, skip_special_tokens=True)

    # 2) Prepare & decode labels (replace -100 with pad_token_id so they decode cleanly)
    labels = np.where(labels != -100, labels, TOKENIZER.pad_token_id)
    decoded_labels = TOKENIZER.batch_decode(labels, skip_special_tokens=True)

    relaxed_preds = tfidf_relaxed(decoded_labels, decoded_preds)

    # 4) Flatten to single lists
    strict_acc  = sum(1 for pred, lab in zip(decoded_preds, decoded_labels) if pred==lab)/len(decoded_labels)
    relaxed_acc = sum(relaxed_preds)/len(decoded_labels)

    # 5) Compute your (relaxed) metrics
    return {
        "accuracy strict": strict_acc,
        "accuracy relaxed": relaxed_acc
    }

def tfidf_relaxed(decoded_labels, decoded_preds):
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
        