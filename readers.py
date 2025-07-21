from datasets import Dataset, DatasetDict
import xml.etree.ElementTree as ET
import os

def read_ozrock(auto_filepath, eval_filepath):
    for filepath in [auto_filepath, eval_filepath]:
        with open(filepath, 'r') as file:
            lines = file.readlines()
            data, sentence = [], {"tokens": [], "ner_tags": []}
            for line in lines[1:]:
                try:
                    word, tag = line.strip("\n").split(" ")
                except ValueError:
                    data.append(sentence)
                    sentence = {"tokens": [], "ner_tags": []}
                if word == "" or tag == "":
                    data.append(sentence)
                    sentence = {"tokens": [], "ner_tags": []}
                else:
                    sentence["tokens"].append(word)
                    sentence["ner_tags"].append(tag)
            
            if filepath == auto_filepath:
                trian = data
            else:
                test = data
    label_list, label2id, id2label = get_label_list(trian, label2id=True, id2label=True)
    datasets = DatasetDict({
        "train": convert_to_dataset(trian, label2id),
        "eval": convert_to_dataset(test, label2id)
    })
    return datasets, label_list, label2id, id2label

def read_timeml(directory_path):
    for folder in os.listdir(directory_path):
        folderpath = os.path.join(directory_path, folder)
        if os.path.isdir(folderpath):
            for filename in os.listdir(folderpath):
                if filename.endswith('.tml'):
                    filepath = os.path.join(folderpath, filename)
                    if os.path.isfile(filepath):
                        print(f"Processing file: {filepath}")
    # tree = ET.parse(filepath)

    return

def convert_to_dataset(data, label_map):
    formatted_data = {"tokens":[], "ner_tags":[]}
    for sentence in data:
        ner_tags = [label_map[tag] for tag in sentence["ner_tags"]]
        formatted_data["tokens"].append(sentence["tokens"])
        formatted_data["ner_tags"].append(ner_tags)
    return Dataset.from_dict(formatted_data)

def get_label_list(data, label2id=True, id2label=True):
    label_list = sorted(list(set([tag for sentence in data for tag in sentence['ner_tags']])))
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    return label_list, label2id if label2id else None, id2label if id2label else None