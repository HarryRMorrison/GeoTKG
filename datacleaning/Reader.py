from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
import xml.etree.ElementTree as ET
import os
import spacy
from spacy.symbols import ORTH
from spacy.lang.en import English
import json
import numpy as np
from copy import deepcopy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAWDATA_PATH = os.path.join(BASE_DIR, "rawdata")
CLEANDATA_PATH = os.path.join(BASE_DIR, "cleandata")

# Map labels (from TE3, MAVEN, MATRES, TBDense) -> (allen_label, flip_args?)
ALLEN_MAP = {
    # ---- Ordering / adjacency
    # BEFORE side
    "BEFORE":   ("BEFORE", False),
    "IBEFORE":  ("BEFORE", False),  # immediate before still counts as BEFORE
    
    # AFTER side (flip to BEFORE if you want consistent arg order)
    "AFTER":    ("AFTER", False),
    "IAFTER":   ("AFTER", False),    # immediate after still counts as AFTER

    # DURING = A is inside B  (A during B) TE3 has bad definitions
    "IS_INCLUDED":  ("DURING", False),     # TE3, TBDense

    # ---- Overlap
    "OVERLAP":      ("OVERLAPS", False), # MAVEN
    "DURING_INV":   ("OVERLAPS", True),      # TE3
    "DURING":       ("OVERLAPS", False), # TE3

    # CONTAINS = A contains B  (inverse of DURING)
    "INCLUDES":     ("CONTAINS", False),   # TE3
    "CONTAINS":     ("CONTAINS", False),   # MAVEN

    # ---- Equality
    "SIMULTANEOUS": ("EQUALS", False), # TE3, MAVEN, TBDense
    "EQUAL":        ("EQUALS", False), # MATRES

    # ---- Starts / Finishes (boundary matches)
    "BEGINS":   ("STARTS", False),     # TE3
    "BEGUN_BY": ("STARTS", True),      # TE3 inverse
    "BEGINS-ON":("STARTS", False),     # MAVEN

    "ENDS":     ("FINISHES", False),   # TE3
    "ENDED_BY": ("FINISHES", True),    # TE3 inverse
    "ENDS-ON":  ("FINISHES", False),   # MAVEN

    # IDENTITY
    "IDENTITY":     ("IDENTITY", False), # TE3, MAVEN comentions
}

class Reader:
    def __init__(self, path : str):
        self.path = path
        self.file_paths_to_read = self.get_file_paths()

    def recursive_dir_search(folder_path):
        file_list = []
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                file_list.append(os.path.join(dirpath, filename))
        return file_list

    def get_file_paths(self):
        filepaths = []
        if os.path.isdir(self.path):
            for filename in Reader.recursive_dir_search(self.path):
                if os.path.isfile(filename):
                    filepaths.append(filename)
        else:
            raise ValueError(f"Path {self.path} is not a directory.")
        return filepaths

    def get_label_list(labels):
        print(type(labels[0]))
        if type(labels[0]) == list:
            label_list = sorted(list(set([tag for sentence in labels for tag in sentence])))
        else:
            label_list = sorted(list(set([tag for tag in labels])))
        label2id = {label: int(i) for i, label in enumerate(label_list)}
        id2label = {int(i): label for label, i in label2id.items()}
        return label_list, label2id, id2label
    
    # Redo the mappings
    def to_allen(label, m0, m1):
        lab = (label or "").upper().strip()
        if lab not in ALLEN_MAP or ALLEN_MAP[lab] is None:
            raise ValueError(f"Unknown or unmapped label: {label}")
        allen, flip = ALLEN_MAP[lab]
        if flip:
            m0, m1 = m1, m0
        return allen, m0, m1 

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
        
        train_json = os.path.join(CLEANDATA_PATH, "BIO", "OzRock", train_json)
        test_json = os.path.join(CLEANDATA_PATH, "BIO", "OzRock", test_json)

        if os.path.exists(train_json):
            os.remove(train_json)
        if os.path.exists(test_json):
            os.remove(test_json)

        for filepath in self.file_paths_to_read:
            if filepath == self.file_paths_to_read[1]:
                print("Processing file 1/2")
            with open(filepath, 'r') as file:
                lines = file.readlines()
                data, sentence = [], {"tokens": [], "label": []}
                for line in lines[1:]:
                    try:
                        word, tag = line.strip("\n").split(" ")
                    except ValueError:
                        data.append(sentence)
                        sentence = {"tokens": [], "label": []}
                        continue
                    if word == "" or tag == "":
                        data.append(sentence)
                        sentence = {"tokens": [], "label": []}
                    else:
                        sentence["tokens"].append(word)
                        sentence["label"].append(tag)
                
                if filepath == self.file_paths_to_read[0]:
                    train = OzRockReader.convert_to_dataset(data)
                else:
                    test = OzRockReader.convert_to_dataset(data)

        OzRockReader.to_json(data=train, intended_path=train_json)
        OzRockReader.to_json(data=test, intended_path=test_json)
        return

class MAVENReader(Reader):
    def __init__(self, path: str):
        super().__init__(path)

    def read(self):
        json_path = os.path.join(CLEANDATA_PATH, "MAVEN", "data")

        for file in self.file_paths_to_read:
            info = []
            name = os.path.basename(file).split(".")[0]
            print(f"Reading {name}")
            with open(file, 'r') as f:
                data = [json.loads(line) for line in f]
                for i,line in enumerate(data):
                    print(f"Processing file {i}")
                    info.append(MAVENReader.get_doc_info(line))
            data2upload = Dataset.from_list(info)
            data2upload.to_json(os.path.join(json_path, name+".json"))
        return
    
    def get_doc_info(data):
        events = {instance["id"]:instance["mention"] for instance in data["events"]}
        timexs = {}
        for instance in data["TIMEX"]:
            if instance["type"] == "PREPOSTEXP":
                continue
            elif instance['type'] == "QUANTIFIER":
                instance['type'] = "SET"
            timexs[instance["id"]] = {
                    'id': instance['id'],
                    'value': instance['value'],
                    'type': instance['type'],
                    'sent_id':instance["sent_id"],
                    'offset': instance["offset"],
                    'text': instance["mention"]
                }
            
        relations = data["temporal_relations"]
        sents = data["tokens"]

        out = {"text":None, "instances":[], "event_times":[], "ee_temprels":[], "bio_tags":None}

        for rel_type in relations:
            for relation in relations[rel_type]:
                pair = (relation[0].partition("_")[0], relation[1].partition("_")[0])
                match pair:
                    case ("EVENT", "EVENT"):
                        p0 = events[relation[0]]
                        p1 = events[relation[1]]
                        for m0 in p0:
                            for m1 in p1:
                                rel_type, m0, m1 = Reader.to_allen(rel_type, m0, m1)
                                out["ee_temprels"].append({"e1":m0['id'], "e2":m1['id'], "rel":rel_type})
                    case ("EVENT", "TIME"):
                        if rel_type is not "SIMULTANEOUS":
                            continue
                        p0 = events[relation[0]]
                        p1 = timexs[relation[1]]
                        out["event_times"].extend([{"event":m0['id'], "time":p1['id']} for m0 in p0])
                    case ("TIME", "EVENT"):
                        if rel_type is not "SIMULTANEOUS":
                            continue
                        p0 = timexs[relation[0]]
                        p1 = events[relation[1]]
                        out["event_times"].extend([{"event":m1['id'], "time":p0['id']} for m1 in p1])
                    case _:
                        continue
        
        for event in events:
            for m in events[event]:
                out["instances"].append({
                    "id": instance["id"],
                    'type': "EVENT",
                    'sent_id':instance["sent_id"],
                    'offset': instance["offset"],
                    'text': instance["trigger_word"]
                })
                for n in events[event]:
                    if m == n:
                        continue
                    else:
                        out["ee_temprels"].append({"e1":m['id'], "e2":n['id'], "rel":"IDENTITY"})
        
        out["instances"].extend(list(timexs.values()))
        out["bio_tags"] = MAVENReader.get_bio(sents, out["instances"])
        return out
    
    @staticmethod
    def get_bio(text, instances):
        labels = [["O" for token in sent] for sent in text]
        for inst in instances:
            replace = f"B-{inst['type']}"
            for i in range(inst["offset"][0], inst["offset"][1]):
                labels[inst["sent_id"]][i] = replace
                replace = f"I-{inst['type']}"
        return labels

class TimeMLReader(Reader):
    
    @staticmethod
    def get_doc_and_loc(root, eid2eiid):
        nlp = English()
        nlp.add_pipe("sentencizer")
        doc = root.find('TEXT')
        dct = root.find('DCT').find('TIMEX3')
        text, events, timexs, sentence, sentid = [], {}, {}, [], 0
        timexs['t0'] = {'value':dct.attrib.get('value'), 'type':dct.attrib.get('type'), 'offset':(1, len(text))}
        
        for elem in doc.iter():
            start = len(sentence)
            sentence.append(elem.text.replace("\n",""))
            if elem.tag == 'EVENT':
                try:
                    events[eid2eiid[elem.attrib.get('eid')]] = {
                        'offset': (start, len(sentence)),
                        'type':"EVENT",
                        'sent_id':sentid,
                        'text': elem.text
                    }
                except KeyError:
                    continue
            elif elem.tag == 'TIMEX3':
                timexs[elem.attrib.get('tid')] = {
                    'value': elem.attrib.get('value'),
                    'type': elem.attrib.get('type'),
                    'sent_id':sentid,
                    'offset': (start, len(sentence)),
                    'text': elem.text
                }
            else:
                tail = elem.tail.replace("\n","")
                tail_doc = nlp(tail)
                tail = list(tail_doc.sents)[0]
                if len(list(tail_doc.sents)) > 1:
                    text.append(sentence+list(tail_doc.sents)[0])
                    sentence = list(tail_doc.sents)[1]
                    tail = tail[1]
                    sentid += 1
                sentence.extend(tail)
        return text, events, timexs
    
    @staticmethod
    def check_id(id, instances):
        try:
            m0 = instances[id]
            return id
        except KeyError:
            return 0

    @staticmethod
    def tlink_to_input(links, events, timexs=None):
        tlink_type = "relatedToEventInstance" if timexs is None else "relatedToTime"
        out = []
        n_events = {}
        n_timexs = {}

        for link in links:
            if link["relType"] == "NONE":
                continue

            id0 = TimeMLReader.check_id(link['eventInstanceID'], events)
            if tlink_type == "relatedToEventInstance":
                id1 = TimeMLReader.check_id(link[tlink_type], events)
            else:
                id1 = TimeMLReader.check_id(timexs[link[tlink_type]], timexs)
            if id0==0 or id1==0:
                continue
            
            rel_type, id0, id1 = Reader.to_allen(link["relType"], id0, id1)
            if tlink_type == "relatedToEventInstance":
                out.append({"e1":id0, "e2":id1, "rel":rel_type})
                n_events[id0] = events[id0]
                n_events[id1] = events[id1]
            elif rel_type is "EQUALS":
                out.append({"event":id0, "time":id1})
                n_events[id0] = events[id0]
                n_timexs[id1] = timexs[id1]

        if tlink_type == "relatedToEventInstance":
            return out, n_events
        else:
            return out, n_events, n_timexs

    @staticmethod
    def get_doc_info(file):
        tree = ET.parse(file)
        root = tree.getroot()
        out = {"text":None, "instances":None, "event_times":None, "ee_temprels":None}
        eid2eiid = {mi.attrib.get('eventID'):mi.attrib.get('eiid') for mi in root.findall('MAKEINSTANCE')}
        EElinks = [{"eventInstanceID":link.attrib.get("eventInstanceID"), 
                    "relatedToEventInstance":link.attrib.get("relatedToEventInstance"), 
                    "relType":link.attrib.get("relType")} 
                    for link in root.findall('TLINK[@relatedToEventInstance][@eventInstanceID]')]
        ETlinks = [{"eventInstanceID":link.attrib.get("eventInstanceID"), 
                    "relatedToTime":link.attrib.get("relatedToTime"), 
                    "relType":link.attrib.get("relType")} 
                    for link in root.findall('TLINK[@relatedToTime][@eventInstanceID]')]
        text, events, timexs = TimeMLReader.get_doc_and_loc(root, eid2eiid)
        ee_temprels, events = TimeMLReader.tlink_to_input(EElinks, events)
        event_times, events, timexs = TimeMLReader.tlink_to_input(ETlinks, events, timexs)
        joint, join_list = events | timexs, []
        for inst in joint:
            info = joint[inst]
            info["id"] = inst
            join_list.append(info)
        out["ee_temprels"] = ee_temprels
        out["event_times"] = event_times
        out["instances"] = join_list
        out["bio_tags"] = TimeMLReader.get_bio(text, out["instances"])
        out['text'] = text
        return out

    @staticmethod
    def get_bio(text, instances):
        bio_tags = [["O" for token in sent] for sent in text]

        for inst in instances:
            replace = f"B-{inst['type']}"
            for i in range(inst["offset"][0], inst["offset"][1]):
                bio_tags[inst["sent_id"]][i] = replace
                replace = f"I-{inst['type']}"

        return bio_tags

    @staticmethod
    def get_timex_values(doc):
        found_index = next((index for index, item in enumerate(doc["instances"]) if item.get("id") == "t0"), None)
        dct = doc["instances"][found_index]['value']
        found_indexes = []
        for index, item_dict in enumerate(doc["instances"]):
            if "value" in item_dict:
                found_indexes.append(index)

        data = []
        task = f"Document creation time is {dct}<sep> normalise time text:"

        for time_loc in found_indexes:
            info = doc["instances"][time_loc]
            text = doc["text"].copy()

            text[info["sent_id"]].insert(info["offset"][0], f"<timex type={info["type"]}>")
            text[info["sent_id"]].insert(info["offset"][1]+1, "</timex>")

            text = text[max(0,info["sent_id"]-1):max(len(text),info["sent_id"]+1)]
            data.append({"input_text": task + " ".join([str(token) for sent in text for token in sent ]), "target_text": info["value"]})
    
        return data       
    
    @staticmethod
    def get_quintuples(filepath):
        return
            
class TempEval3Reader(TimeMLReader):
    def __init__(self, path):
        super().__init__(path)

    def read(self, dataset_name="TempEval3"):
        json_path = os.path.join(CLEANDATA_PATH, dataset_name, "data")
        json_path_norm = os.path.join(CLEANDATA_PATH, dataset_name, "normalise")
        
        data, norm_data = [], []
        current_folder = self.file_paths_to_read[0].split('\\')[2]
        num_f = len(self.file_paths_to_read)
        print(f"Starting {current_folder}")
        print(self.file_paths_to_read)

        for i, filepath in enumerate(self.file_paths_to_read):
            if filepath.split('\\')[2] != current_folder:
                data2upload = Dataset.from_list(data)
                data2upload.to_json(os.path.join(json_path, current_folder+".json"))

                data2upload = Dataset.from_list(norm_data)
                data2upload.to_json(os.path.join(json_path_norm, current_folder+".json"))

                data2upload = None
                current_folder = filepath.split('\\')[2]
            print(f"Processing file {i}/{num_f}")
            info = TempEval3Reader.get_doc_info(filepath)
            norms = TempEval3Reader.get_timex_values(info)
            data.append(info)
            norm_data.append(norms)


        data2upload = Dataset.from_list(data)
        data2upload.to_json(os.path.join(json_path, current_folder+".json"))
        data2upload = Dataset.from_list(norm_data)
        data2upload.to_json(os.path.join(json_path_norm, current_folder+".json"))

class TBDenseReader(TempEval3Reader):
    def __init__(self, path):
        super().__init__(path)
    
    def read(self):
        super().read(dataset_name="TBDense")
        
def id_token_labels(dataset, label2id):
    def change_id(row):
        row["label"] = [label2id[tag] for tag in row["label"]]
        return row
    return dataset.map(change_id) 
   
def obtain_combined_dataset(dataset_names, method):
    data = []
    for dataset_name in dataset_names:
        for json_name in ["train.json","test.json","eval.json"]:
            if dataset_name in ["OzRock", "MAVEN"] and json_name == "test.json":
                continue
            data.append(load_dataset("json", data_files = os.path.join(CLEANDATA_PATH, method, dataset_name, json_name))["train"])
    data = concatenate_datasets(data)
    label_list, label2id, id2label = Reader.get_label_list(data["label"])
    print(label_list, label2id, id2label)
    if method == "TempRel":
        ids = [label2id[lab] for lab in data["label"]]
        data = data.add_column("labels", ids).class_encode_column("labels")
        data = data.train_test_split(test_size=0.2, shuffle=True, seed=42, stratify_by_column="labels")
        train = data["train"]
        test = data["test"].remove_columns("label")
        data = None
        train = train.train_test_split(test_size=0.1, shuffle=True, seed=42, stratify_by_column="labels")
        val = train["test"].remove_columns("label")
        train = train["train"].remove_columns("label")
    else:
        data = data.train_test_split(test_size=0.2, shuffle=True, seed=42)
        train = data["train"]
        test = data["test"]
        data = None
        train = train.train_test_split(test_size=0.1, shuffle=True, seed=42)
        val = train["test"]
        train = train["train"]
    return DatasetDict({"test": test, "train":train, "eval": val}), label_list, label2id, id2label

if __name__ == "__main__":
    # te = TBDenseReader("rawdata\\TBDense")
    # te.read()

    # te = TempEval3Reader("rawdata\\TempEval3")
    # te.read()

    te = MAVENReader("rawdata\\MAVEN_ERE")
    te.read()
