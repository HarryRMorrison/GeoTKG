from scripts.model_utils import Utils
import torch
from transformers import RobertaForTokenClassification, RobertaTokenizerFast, TrainingArguments, Trainer
from scripts.Reader import obtain_dataset, obtain_label_list

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device('cuda')
print("Current Device:", torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))

label_list, label2id, id2label = obtain_label_list("TempEval3")

# Load the model and tokenizer from the folder
model = RobertaForTokenClassification.from_pretrained("./results/Timex3-NER/final_model/")
tokenizer = RobertaTokenizerFast.from_pretrained("./results/Timex3-NER/final_model/")

utils = Utils(tokenizer, label_list)
tokenized_datasets = utils.tokenize_datasets(obtain_dataset("TempEval3"))

encodings=tokenizer(list(tokenized_datasets['test']['tokens']), padding=True, truncation=True, return_tensors="pt", is_split_into_words=True)

with torch.no_grad():
    outputs = model(**encodings)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

conf = [[0 for _ in range(13)] for _ in range(13)]
for preds, actua in zip(predictions,tokenized_datasets['test']['ner_tags']):
    preds = preds[:len(actua)]
    for j, predicted_label in enumerate(preds):
        conf[actua[j]][predicted_label] += 1  

def metr(conf, index):
    TP = conf[index][index]
    FN = sum(conf[index])-TP
    FP = sum(uh[index] for uh in conf)-TP
    try:
        recall = TP/(TP+FN)
    except ZeroDivisionError:
        recall = 0
    try:
        prec = TP/(TP+FP)
    except ZeroDivisionError:
        prec=0
    try:
        f1 = (2*recall*prec)/(recall+prec)
    except ZeroDivisionError:
        f1=0
    return recall, prec, f1

for i in range(len(label_list)):
    r,p,f = metr(conf,i)
    print(round(r,4),round(p,4),round(f,4))
