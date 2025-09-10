from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
import xml.etree.ElementTree as ET
import os
#import spacy
#from spacy.symbols import ORTH
from spacy.lang.en import English
import json
#import numpy as np
from copy import deepcopy
import re

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
    "BEGINS":   ("DURING", False),     # TE3
    "BEGUN_BY": ("DURING", True),      # TE3 inverse
    "BEGINS-ON":("DURING", False),     # MAVEN

    "ENDS":     ("DURING", False),   # TE3
    "ENDED_BY": ("DURING", True),    # TE3 inverse
    "ENDS-ON":  ("DURING", False),   # MAVEN

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

    def read(self):     
        out = {}
        for filepath in self.file_paths_to_read:
            with open(filepath, 'r') as file:
                lines = file.readlines()
                data, sentence = [], {"tokens": [], "bio_tags": []}
                for line in lines[1:]:
                    try:
                        word, tag = line.strip("\n").split(" ")
                    except ValueError:
                        data.append(sentence)
                        sentence = {"tokens": [], "bio_tags": []}
                        continue
                    if word == "" or tag == "":
                        data.append(sentence)
                        sentence = {"tokens": [], "bio_tags": []}
                    else:
                        sentence["tokens"].append(word)
                        sentence["bio_tags"].append(tag)
                
            out[os.path.basename(filepath).split(".")[0]] = data
        return out

class MAVENReader(Reader):
    def __init__(self, path: str):
        super().__init__(path)

    def read(self):
        out = {}
        for file in self.file_paths_to_read:
            info = []
            mention_counter = 0
            name = os.path.basename(file).split(".")[0]
            print(f"Reading {name}")
            with open(file, 'r') as f:
                data = [json.loads(line) for line in f]
                for i,line in enumerate(data):
                    print(f"Processing file {i}")
                    if name == "test":
                        info.append(MAVENReader.test_data_info(line))
                    else:
                        info.append(MAVENReader.get_doc_info(line))
                    if "events" in line:
                        mention_counter += sum([len(event["mention"]) for event in line["events"]])
            print(mention_counter)
            out[name] = info
        return out

    def test_data_info(data):
        events = []
        timexs = []
        for ev in data["event_mentions"]:
            events.append({
                "id": ev["id"],
                'type': "EVENT",
                'sent_id':ev["sent_id"],
                'offset': ev["offset"],
                'text': ev["trigger_word"]
            })
        for instance in data["TIMEX"]:
            if instance["type"] == "PREPOSTEXP":
                continue
            elif instance['type'] == "QUANTIFIER":
                instance['type'] = "SET"
            timexs.append({
                    'id': instance['id'],
                    'type': instance['type'],
                    'sent_id':instance["sent_id"],
                    'offset': instance["offset"],
                    'text': instance["mention"]
                })
        out = {"text":data['tokens'], "instances":events+timexs, "bio_tags":None}
        out["bio_tags"] = MAVENReader.get_bio(data["tokens"], out["instances"])
        return out

    
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
                    'type': instance['type'],
                    'sent_id':instance["sent_id"],
                    'offset': instance["offset"],
                    'text': instance["mention"]
                }
            
        relations = data["temporal_relations"]
        sents = data["tokens"]

        out = {"text":data['tokens'], "instances":[], "event_times":[], "ee_temprels":[], "bio_tags":None}
        ev_times_found = {}

        for rel_type in relations:
            for relation in relations[rel_type]:
                pair = (relation[0].partition("_")[0], relation[1].partition("_")[0])
                match pair:
                    case ("EVENT", "EVENT"):
                        p0 = events[relation[0]]
                        p1 = events[relation[1]]
                        for m0 in p0:
                            for m1 in p1:
                                altered_rel_type, m0, m1 = Reader.to_allen(rel_type, m0, m1)
                                out["ee_temprels"].append({"e1":m0['id'], "e2":m1['id'], "rel":altered_rel_type})
                    case ("EVENT", "TIME"):
                        p0 = events[relation[0]]
                        try:
                            p1 = timexs[relation[1]]
                        except KeyError:
                            continue

                        for m0 in p0:
                            altered_rel_type, e, t = Reader.to_allen(rel_type, m0['id'], p1['id'])
                            if e in ev_times_found:
                                if ev_times_found[e][0] == "EQUALS":
                                    continue
                                else:
                                    ev_times_found[e] = (altered_rel_type, {"event":e, "time":t})
                            elif altered_rel_type in ["DURING","EQUALS","CONTAINS","IDENTITY"]:
                                ev_times_found[e] = (altered_rel_type, {"event":e, "time":t})
                    case ("TIME", "EVENT"):
                        try:
                            p0 = timexs[relation[0]] # Time
                        except KeyError:
                            continue
                        p1 = events[relation[1]] # Event

                        for m0 in p1:
                            altered_rel_type, e, t = Reader.to_allen(rel_type, m0['id'], p0['id'])
                            if e in ev_times_found:
                                if ev_times_found[e][0] == "EQUALS":
                                    continue
                                else:
                                    ev_times_found[e] = (altered_rel_type, {"event":e, "time":t})
                            elif altered_rel_type in ["DURING","EQUALS","CONTAINS","IDENTITY"]:
                                ev_times_found[e] = (altered_rel_type, {"event":e, "time":t})
                    case _:
                        continue
        
        out["event_times"] = [et[1] for et in ev_times_found.values()]
        
        for event in events:
            for m in events[event]:
                out["instances"].append({
                    "id": m["id"],
                    'type': "EVENT",
                    'sent_id':m["sent_id"],
                    'offset': m["offset"],
                    'text': m["trigger_word"]
                })
                for n in events[event]:
                    if m["id"] == n["id"]:
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
        tokenizer = nlp.tokenizer
        doc = root.find('TEXT')
        dct = root.find('DCT').find('TIMEX3')
        text, events, timexs, sentence, sentid = [], {}, {}, [], 0
        timexs['t0'] = {'value':dct.attrib.get('value'), 'type':dct.attrib.get('type'), 'offset':(1, len(text))}
        
        for elem in doc.iter():
            start = len(sentence)
            sentence.extend([bit.text for bit in tokenizer(elem.text.replace("\n",""))])
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
            tail = [bit.text for bit in tokenizer(elem.tail.replace("\n",""))]
            tail_doc = nlp(elem.tail.replace("\n",""))
            if len(list(tail_doc.sents)) > 1:
                text.append(sentence+[str(tok) for tok in list(tail_doc.sents)[0]])
                sentence = []
                tail = [str(tok) for tok in list(tail_doc.sents)[1]]
                sentid += 1
                sentence.extend(tail)
            else:
                sentence.extend(tail)
        text.append(sentence)
        return text, events, timexs
    
    @staticmethod
    def check_id(id, instances):
        try:
            m0 = instances[id]
            return id
        except KeyError:
            return 0

    @staticmethod
    def ee_link_to_input(links, events):
        out = []

        for link in links:
            id0 = TimeMLReader.check_id(link['eventInstanceID'], events)
            if id0==0: continue

            id1 = TimeMLReader.check_id(link["relatedToEventInstance"], events)
            if id1 == 0: continue

            if link["relType"] == "NONE": continue

            rel_type, id0, id1 = Reader.to_allen(link["relType"], id0, id1)
            out.append({"e1":id0, "e2":id1, "rel":rel_type})
        return out
    
    @staticmethod
    def et_link_to_input(links, events, timexs):
        #out = {}
        out = []

        for link in links:
            id0 = TimeMLReader.check_id(link['eventInstanceID'], events)
            if id0==0: continue
            
            id1 = TimeMLReader.check_id(link["relatedToTime"], timexs)
            if id1 == 0: continue

            if link["relType"] == "NONE": continue

            rel_type, id0, id1 = Reader.to_allen(link["relType"], id0, id1)
            if id1[0]=="e":
                id1, id0 = id0, id1  # swap if time is first

            # if id0 in out:
            #     print(f'Dupe found: {out[id0][0]} and {rel_type}')
            #     if out[id0][0] == "EQUALS":
            #         continue
            #     elif rel_type == "EQUALS":
            #         out[id0] = (rel_type, {"event":id0, "time":id1})

            # if rel_type in ["DURING","EQUALS","CONTAINS","IDENTITY"]:
            #    out[id0] = (rel_type, {"event":id0, "time":id1})
            if rel_type in ["DURING","EQUALS","CONTAINS","IDENTITY"]:
                out.append({"event":id0, "time":id1})
        #return [et[1] for et in out.values()]
        return out

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
        if len(EElinks) == 0:
            print(file)
        ETlinks = [{"eventInstanceID":link.attrib.get("eventInstanceID"), 
                    "relatedToTime":link.attrib.get("relatedToTime"), 
                    "relType":link.attrib.get("relType")} 
                    for link in root.findall('TLINK[@relatedToTime][@eventInstanceID]')]
        ETlinks.extend([{"eventInstanceID":link.attrib.get("relatedToEventInstance"), 
                         "relatedToTime":link.attrib.get("timeID"), 
                         "relType":link.attrib.get("relType")} 
                         for link in root.findall('TLINK[@timeID][@relatedToEventInstance]')])

        text, events, timexs = TimeMLReader.get_doc_and_loc(root, eid2eiid)
        ee_temprels = TimeMLReader.ee_link_to_input(EElinks, events)
        # if len(EElinks) != len(ee_temprels):
        #     print(f"Warning: {len(EElinks)} EE links found, but {len(ee_temprels)} temporal relations created.")
        #     print(file)
        #     print(eid2eiid)
        event_times = TimeMLReader.et_link_to_input(ETlinks, events, timexs)
        joint, join_list = events | timexs, []
        for inst in joint:
            info = joint[inst]
            info["id"] = inst
            join_list.append(info)
        out["ee_temprels"] = ee_temprels
        out["event_times"] = event_times
        out["instances"] = join_list
        out['text'] = text
        out["bio_tags"] = TimeMLReader.get_bio(text, out["instances"])
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
    def get_quintuples(filepath):
        return
            
class TempEval3Reader(TimeMLReader):
    def __init__(self, path):
        super().__init__(path)

    def read(self, dataset_name="TempEval3"):
        data, out = [], {}
        split_num = 4
        current_folder = self.file_paths_to_read[0].split('\\')[split_num]
        num_f = len(self.file_paths_to_read)
        print(f"Starting {current_folder}")

        for i, filepath in enumerate(self.file_paths_to_read):
            if filepath.split('\\')[split_num] != current_folder:
                out[str(current_folder)] = deepcopy(data)
                current_folder = filepath.split('\\')[split_num]
                data = []
                print(f"Starting {current_folder}")
            print(f"Processing file {i+1}/{num_f}")
            info = TempEval3Reader.get_doc_info(filepath)
            data.append(info)
        out[str(current_folder)] = data.copy()
        return out

class TBDenseReader(TempEval3Reader):
    def __init__(self, path):
        super().__init__(path)
    
    def read(self):
        return super().read(dataset_name="TBDense")
        
class TweetsReader(TimeMLReader):
    def __init__(self, path):
        super().__init__(path)

    def read(self):
        data = []
        for file in self.file_paths_to_read:
            if not file.endswith(".tml"):
                continue
            text, timexs = TweetsReader.get_doc_and_timex(ET.parse(file).getroot())
            data.append({"text":text, "instances":list(timexs.values())})
        return data
    
    @staticmethod
    def get_doc_and_timex(root):
        nlp = English()
        nlp.add_pipe("sentencizer")
        tokenizer = nlp.tokenizer
        doc = root.find('TEXT')
        dct = root.find('DCT').find('TIMEX3')
        text, timexs, sentence, sentid = [], {}, [], 0
        timexs['t0'] = {'id':'t0', 'value':dct.attrib.get('value'), 'type':dct.attrib.get('type'), 'offset':(1, len(text))}
        
        for elem in doc.iter():
            start = len(sentence)
            sentence.extend([bit.text for bit in tokenizer(elem.text.replace("\n",""))])
            if elem.tag == 'TIMEX3':
                timexs[elem.attrib.get('tid')] = {
                    'id': elem.attrib.get('tid'),
                    'value': elem.attrib.get('value'),
                    'type': elem.attrib.get('type'),
                    'sent_id':sentid,
                    'offset': (start, len(sentence)),
                    'text': elem.text
                }
            tail = [bit.text for bit in tokenizer(elem.tail.replace("\n",""))]
            tail_doc = nlp(elem.tail.replace("\n",""))
            if len(list(tail_doc.sents)) > 1:
                text.append(sentence+[str(tok) for tok in list(tail_doc.sents)[0]])
                sentence = []
                tail = [str(tok) for tok in list(tail_doc.sents)[1]]
                sentid += 1
                sentence.extend(tail)
            else:
                sentence.extend(tail)
        text.append(sentence)
        return text, timexs

    @staticmethod
    def get_timex_values(text, timexs):
        dct = timexs["t0"]['value']

        data = []
        task = f"Document creation time is {dct}<sep> normalise time text:"

        for time_ in timexs:
            if time_=="t0":
                continue
            info = timexs[time_]
            sample = deepcopy(text)

            sample[info["sent_id"]].insert(info["offset"][0], f"<timex type={info['type']}>")
            sample[info["sent_id"]].insert(info["offset"][1]+1, "</timex>")

            sample = [wrd for inner in sample[info["sent_id"]-1:info["sent_id"]+1] for wrd in inner]
            data.append({"input_text": task + " ".join(sample), "target_text": info["value"]})
    
        return data  

class WikiWarsReader(TweetsReader):
    def __init__(self, path):
        super().__init__(path)

    def read(self):
        data = []
        for file in self.file_paths_to_read:
            if not file.endswith(".tml"):
                continue
            text, timexs = WikiWarsReader.get_doc_and_timex(ET.parse(file).getroot())
            data.append({"text":text, "instances":list(timexs.values())})
        return data



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
    out = TBDenseReader("D:\\GeoTKG\\rawdata\\TBDense").read()
