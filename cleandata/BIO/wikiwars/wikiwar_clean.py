import pandas as pd
import json
from datasets import load_dataset, Dataset
from scripts.Reader import Reader

labtoid={
    "B-DATE": 0,
    "B-DURATION": 1,
    "B-SET": 2,
    "B-TIME": 3,
    "I-DATE": 4,
    "I-DURATION": 5,
    "I-SET": 6,
    "I-TIME": 7,
    "O": 8
}

with open("dev.txt", "r", encoding='utf-8') as file:
    lines = file.readlines()

    data = []
    sentence = {"tokens":[], "ner_tags":[]}

    for line in lines:
        contents = line.strip("\n").split(" ")
        if contents[0] == "#":
            continue
        elif contents == ['']:
            data.append(sentence)
            sentence = {"tokens":[], "ner_tags":[]}
        else:
            sentence['tokens'].append(contents[0])
            sentence['ner_tags'].append(contents[1])

dataset = Reader.convert_to_dataset(data, labtoid)

Reader.to_json(dataset, "eval_wars.json")
