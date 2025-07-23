from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
import xml.etree.ElementTree as ET
import os
import spacy
import json

class Reader:
    def __init__(self, path : str):
        self.path = path
        self.file_paths_to_read = self.get_file_paths()

    def get_file_paths(self):
        filepaths = []
        if os.path.isdir(self.path):
            for filename in os.listdir(self.path):
                if os.path.isdir(os.path.join(self.path, filename)):
                    for subfilename in os.listdir(os.path.join(self.path, filename)):
                        subfilepath = os.path.join(self.path, filename, subfilename)
                        if os.path.isfile(subfilepath):
                            filepaths.append(subfilepath)
                else:
                    filepath = os.path.join(self.path, filename)
                    if os.path.isfile(filepath):
                        filepaths.append(filepath)
        elif os.path.isfile(self.path):
            filepaths.append(self.path)
        else:
            raise ValueError(f"Path {self.path} is neither a file nor a directory.")
        return filepaths
    
    def to_json(data, intended_path : str):
        if type(data) == Dataset:
            with open(intended_path, 'a') as f:
                for row in data:
                    f.write(json.dumps(row) + '\n')
        else:
            with open(intended_path, 'w') as f:
                json.dump(data, f, indent=4)

    def convert_to_dataset(data, label_map):
        formatted_data = {"tokens":[], "ner_tags":[]}
        for sentence in data:
            formatted_data["tokens"].append([str(token) for token in sentence["tokens"]])
            formatted_data["ner_tags"].append([label_map[tag] for tag in sentence["ner_tags"]])
        return Dataset.from_dict(formatted_data)

    def get_label_list(data, label2id=True, id2label=True):
        label_list = sorted(list(set([tag for sentence in data for tag in sentence['ner_tags']])))
        label2id = {label: int(i) for i, label in enumerate(label_list)}
        id2label = {int(i): label for label, i in label2id.items()}
        return label_list, label2id, id2label
 

class TimeMLReader(Reader):
    def __init__(self, path: str):
        super().__init__(path)

    def read(self, method : str, json_path: str = None):

        if method == "timex3_bio_tagger":
            extractor = TimeMLReader.TIMEX3_BIO_tagger
        else:
            raise ValueError(f"Method {method} is not supported.")
        
        if not json_path.endswith('.json'):
            raise ValueError("JSON path must end with .json")
        
        if os.path.exists(json_path):
            os.remove(json_path)

        data = []
        indicator = 0
        num_file = len(self.file_paths_to_read)
        label_list, label2id, id2label = obtain_label_list("TempEval3")

        for filepath in self.file_paths_to_read:
            if filepath.endswith('.tml'):
                print(f"Processing file {indicator+1}/{num_file}")
                data.extend(extractor(filepath = filepath))
                indicator += 1
            if indicator % 200 == 0:
                datasets = TimeMLReader.convert_to_dataset(data, label2id)
                TimeMLReader.to_json(datasets, json_path)
                data = []
                datasets = None
        if len(data)!=0:
            datasets = TimeMLReader.convert_to_dataset(data, label2id)
            TimeMLReader.to_json(datasets, json_path)
        return

    @staticmethod
    def TIMEX3_BIO_tagger(filepath: str):
        tree = ET.parse(filepath)
        root = tree.getroot()
        text = root.find('TEXT')

        DCT = root.find('DCT').find('TIMEX3').get('value')

        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("sentencizer")

        data, sentence = [], {"tokens": [], "ner_tags": []}

        for node in text.iter():
            text_tokens = nlp(node.text.strip("\n"))

            # Checks if the node is a TAG
            if isinstance(node.tag, str):
                # Check if the node is a TIMEX3
                if node.tag == 'TIMEX3':
                    # Create BIO tags for TIMEX3
                    for i in range(len(text_tokens)):
                        sentence["ner_tags"].append(f"B-"+node.attrib["type"] if i == 0 else "I-"+node.attrib["type"])
                # Must be other node
                else:
                    # Creates O tag for other nodes
                    sentence["ner_tags"].extend(["O"] * len(text_tokens))
                sentence["tokens"].extend(text_tokens)
            
            # Checks if the node has tail text
            if node.tail:

                # Strips tail text of newlines and leading spaces
                tail_tokens = nlp(node.tail.replace("\n\n"," ").lstrip())

                # Checks if the sentence has ended
                if len(list(tail_tokens.sents)) > 1:
                    sents = list(tail_tokens.sents)

                    # Due to poor sentence segmentation, we need to handle the first sentence separately
                    # Check if the first sentence ends with a sentence ender
                    if str(sents[0][-1]) in [".", "!", "?"]:
                        sentence["tokens"].extend(sents[0])
                        sentence["ner_tags"].extend(["O"] * len(sents[0]))
                        if sentence["ner_tags"].count("O") != len(sentence["ner_tags"]):
                            data.append(sentence)
                        sentence = {"tokens": [], "ner_tags": []}
                        tail_tokens = sents[1]

                sentence["tokens"].extend(tail_tokens)
                sentence["ner_tags"].extend(["O"] * len(tail_tokens))
        if sentence["ner_tags"].count("O") != len(sentence["ner_tags"]):
            data.append(sentence)
        return data
    
    @staticmethod
    def TLINK_seqencer(filepath : str):
        tree = ET.parse(filepath)
        root = tree.getroot()
        text = root.find('TEXT')

        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("sentencizer")

        data, sentence = [], {"tokens": [], "ner_tags": []}

        for node in text.iter():
            continue
        return
    
    @staticmethod
    def map_entity_to_event(text, events):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)

        for token in doc:
            print(token)
    
class OzRockReader(Reader):
    def __init__(self, path: str):
        super().__init__(path)

    def read(self, train_json: str = None, test_json: str = None):
        if train_json and not train_json.endswith('.json'):
            raise ValueError("Train JSON path must end with .json")
        if test_json and not test_json.endswith('.json'):
            raise ValueError("Test JSON path must end with .json")
        if (test_json and not train_json) or (train_json and not test_json):
            raise ValueError("Both train and test JSON paths must be provided or neither.")
        
        if os.path.exists(train_json):
            os.remove(train_json)

        if os.path.exists(test_json):
            os.remove(test_json)

        for filepath in self.file_paths_to_read:
            if filepath == self.file_paths_to_read[1]:
                print("Processing file 1/2")
            with open(filepath, 'r') as file:
                lines = file.readlines()
                data, sentence = [], {"tokens": [], "ner_tags": []}
                for line in lines[1:]:
                    try:
                        word, tag = line.strip("\n").split(" ")
                    except ValueError:
                        data.append(sentence)
                        sentence = {"tokens": [], "ner_tags": []}
                        continue
                    if word == "" or tag == "":
                        data.append(sentence)
                        sentence = {"tokens": [], "ner_tags": []}
                    else:
                        sentence["tokens"].append(word)
                        sentence["ner_tags"].append(tag)
                
                if filepath == self.file_paths_to_read[0]:
                    label_list, label2id, id2label = OzRockReader.get_label_list(data, label2id=True, id2label=True)
                    train = OzRockReader.convert_to_dataset(data, label2id)
                else:
                    test = OzRockReader.convert_to_dataset(data, label2id)

        OzRockReader.to_json(label_list, "cleandata/OzRock/label_list.json")
        OzRockReader.to_json(label2id, "cleandata/OzRock/label2id.json")
        OzRockReader.to_json(id2label, "cleandata/OzRock/id2label.json")
        OzRockReader.to_json(data=train, intended_path=train_json)
        OzRockReader.to_json(data=test, intended_path=test_json)
        return

def splitter(data, test_size):
    data = data.train_test_split(test_size=test_size, seed=42)
    return data["train"], data["test"]

# Could change later to make exact train, test, eval json files
def obtain_dataset(dataset_name):
    if dataset_name == "TempEval3":
        data = concatenate_datasets([load_dataset("json", data_files = f"cleandata/{dataset_name}/silver-O-less.json")["train"], load_dataset("json", data_files = f"cleandata/{dataset_name}/gold-O-less.json")["train"]])
        data = data.shuffle(seed=42)
        train_val = data.train_test_split(test_size=0.1, seed=42)
        return DatasetDict({
            "train": train_val["train"],
            "test": load_dataset("json", data_files = f"cleandata/{dataset_name}/platinum-O-less.json")["train"],
            "eval": train_val["test"]
        })
    else:
        return DatasetDict({
            "train": load_dataset("json", data_files = f"cleandata/OzRock/train.json")["train"],
            "eval": load_dataset("json", data_files = f"cleandata/OzRock/eval.json")["train"]
        })
    
    
def obtain_label_list(dataset_name):
    if dataset_name not in ["OzRock", "TempEval3"]:
        raise ValueError("Dataset name is not valid.")
    with open(f"cleandata/{dataset_name}/label_list.json", 'r') as f:
        label_list = json.load(f)
    with open(f"cleandata/{dataset_name}/label2id.json", 'r') as f:
        label2id = json.load(f)
    with open(f"cleandata/{dataset_name}/id2label.json", 'r') as f:
        id2label = json.load(f)
    return label_list, label2id, id2label

if __name__ == "__main__":
    tree = ET.parse("rawdata\\TempEval3\\Evaluation\\te3-platinum\\AP_20130322.tml")
    root = tree.getroot()
    text = root.find('TEXT')
    events = text.findall('EVENT')
    nlp = spacy.load("en_core_web_sm")

    article = []    
    token_hot_encode = []

    for node in text.iter():
        if isinstance(node.tag, str):
            if node.tag == "EVENT":
                token_hot_encode.append(len(article))
            article.extend(nlp(node.text.replace("\n\n"," ").lstrip()))
        if node.tail:
            article.extend(nlp(node.tail.replace("\n\n"," ").lstrip()))

    article = [str(token) for token in article]
    doc = nlp(" ".join(article))
    
    for i, token in enumerate(doc):
        if i in token_hot_encode:
            for child in token.children:
                if child.ent_type_ not in ["DATE", "TIME"]:
                    print(f"Entity associated with {token}: {child.text}")

    for index in token_hot_encode:
        event = doc[index]
        closest_ent = min(doc.ents, key=lambda ent: abs(ent.start - index))
        print(f"Closest entity to {event}: {closest_ent}")





    #TimeMLReader.map_entity_to_event(pp, events)