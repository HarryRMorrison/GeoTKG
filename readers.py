from datasets import Dataset, DatasetDict
import xml.etree.ElementTree as ET
import os
import spacy
import json

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

def read_timeml(folderpath, intended_path):
    data = []
    if os.path.isdir(folderpath):
        for filename in os.listdir(folderpath):
            if filename.endswith('.tml'):
                filepath = os.path.join(folderpath, filename)
                if os.path.isfile(filepath):
                    print(f"Processing file: {filename}")
                    data.extend(timex3_bio_tagger(filepath))
    label_list, label2id, id2label = get_label_list(data, label2id=True, id2label=True)

    datasets = convert_to_dataset(data, label2id)

    datasets.to_json(intended_path)

    return datasets, label_list, label2id, id2label

# Have not included the DCT in the data structure, but it can be added if needed.
def timex3_bio_tagger(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    text = root.find('TEXT')

    DCT = root.find('DCT').find('TIMEX3').get('value')

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("sentencizer")

    data, sentence = [], {"tokens": [], "ner_tags": []}

    for node in text.iter():
        text_tokens = nlp(node.text.strip("\n"))
        if isinstance(node.tag, str):
            if node.tag == 'TIMEX3':
                for i in range(len(text_tokens)):
                    sentence["ner_tags"].append(f"B-"+node.attrib["type"] if i == 0 else "I-"+node.attrib["type"])
            else:
                sentence["ner_tags"].extend(["O"] * len(text_tokens))
            sentence["tokens"].extend(text_tokens)
            
        if node.tail:
            tail_tokens = nlp(node.tail.replace("\n\n"," ").lstrip())

            if len(list(tail_tokens.sents)) > 1:
                sents = list(tail_tokens.sents)
                
                sentence["tokens"].extend(sents[0])
                sentence["ner_tags"].extend(["O"] * len(sents[0]))

                data.append(sentence)
                sentence = {"tokens": [], "ner_tags": []}

                tail_tokens = sents[1]

            sentence["tokens"].extend(tail_tokens)
            sentence["ner_tags"].extend(["O"] * len(tail_tokens))
    data.append(sentence)
    return data

def convert_to_dataset(data, label_map):
    formatted_data = {"tokens":[], "ner_tags":[]}
    for sentence in data:
        formatted_data["tokens"].append([str(token) for token in sentence["tokens"]])
        formatted_data["ner_tags"].append([label_map[tag] for tag in sentence["ner_tags"]])
    return Dataset.from_dict(formatted_data)

def get_label_list(data, label2id=True, id2label=True):
    label_list = sorted(list(set([tag for sentence in data for tag in sentence['ner_tags']])))
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    return label_list, label2id if label2id else None, id2label if id2label else None