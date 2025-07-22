from datasets import Dataset, DatasetDict
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
                filepath = os.path.join(self.path, filename)
                if os.path.isfile(filepath):
                    filepaths.append(filepath)
        elif os.path.isfile(self.path):
            filepaths.append(self.path)
        else:
            raise ValueError(f"Path {self.path} is neither a file nor a directory.")
        return filepaths
    
    def to_json(self, data : Dataset, intended_path : str):
        data.to_json(intended_path)

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
        return label_list, label2id, id2label
 

class TimeMLReader(Reader):
    def __init__(self, path: str):
        super().__init__(path)

    def read(self, method : str, json_path: str = None, return_data: bool = False):

        if method == "timex3_bio_tagger":
            extractor = TimeMLReader.TIMEX3_BIO_tagger
        else:
            raise ValueError(f"Method {method} is not supported.")
        
        if json_path and not json_path.endswith('.json'):
            raise ValueError("JSON path must end with .json")

        data = []
        indicator = 1
        num_file = len(self.file_paths_to_read)
        for filepath in self.file_paths_to_read:
            if filepath.endswith('.tml'):
                print(f"Processing file {indicator}/{num_file}")
                data.extend(extractor(filepath = filepath))
                indicator += 1


        label_list, label2id, id2label = TimeMLReader.get_label_list(data)
        datasets = TimeMLReader.convert_to_dataset(data, label2id)

        if json_path:
            self.to_json(datasets, json_path)

        if return_data:
            return datasets, label_list, label2id, id2label
        else:
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
                        data.append(sentence)
                        sentence = {"tokens": [], "ner_tags": []}
                        tail_tokens = sents[1]

                sentence["tokens"].extend(tail_tokens)
                sentence["ner_tags"].extend(["O"] * len(tail_tokens))
        data.append(sentence)
        return data
    
class OzRockReader(Reader):
    def __init__(self, path: str):
        super().__init__(path)

    def read(self, train_json: str = None, test_json: str = None, return_data: bool = False):
        if train_json and not train_json.endswith('.json'):
            raise ValueError("Train JSON path must end with .json")
        if test_json and not test_json.endswith('.json'):
            raise ValueError("Test JSON path must end with .json")
        if (test_json and not train_json) or (train_json and not test_json):
            raise ValueError("Both train and test JSON paths must be provided or neither.")
        

        for filepath in self.file_paths_to_read:
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
                
                if filepath == self.file_paths_to_read[0]:
                    label_list, label2id, id2label = OzRockReader.get_label_list(data, label2id=True, id2label=True)
                    train = OzRockReader.convert_to_dataset(data, label2id)
                else:
                    test = OzRockReader.convert_to_dataset(test, label2id)

        if train_json:
            OzRockReader.to_json(train, train_json)
            OzRockReader.to_json(test, test_json)
        if return_data:
            datasets = DatasetDict({
                "train": OzRockReader.convert_to_dataset(train, label2id),
                "eval": OzRockReader.convert_to_dataset(test, label2id)
            })
            return datasets, label_list, label2id, id2label
        else:
            return

    

if __name__ == "__main__":
    s0 = TimeMLReader("rawdata\\TempEval3\\Training\\TE3-Silver-data-0-copy")
    s0.read(method="timex3_bio_tagger", json_path="data.json")