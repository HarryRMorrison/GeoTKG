from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
import xml.etree.ElementTree as ET
import os
import spacy
from spacy.symbols import ORTH
import json
import numpy as np
from copy import deepcopy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAWDATA_PATH = os.path.join(BASE_DIR, "rawdata")
CLEANDATA_PATH = os.path.join(BASE_DIR, "cleandata")

E_E_SEPS = ["<e1>", "</e1>", "<e2>", "</e2>"]
E_T_SEPS = ["<e>", "</e>", "<timex", "</timex>", "TIMEVAL=", "TYPE="]
# Map labels (from TE3, MAVEN, MATRES, TBDense) -> (allen_label, flip_args?)
ALLEN_MAP = {
    # ---- Ordering / adjacency
    # BEFORE side
    "BEFORE":   ("BEFORE", False),
    "IBEFORE":  ("BEFORE", False),  # immediate before still counts as BEFORE
    
    # AFTER side (flip to BEFORE if you want consistent arg order)
    "AFTER":    ("AFTER", False),
    "IAFTER":   ("AFTER", False),    # immediate after still counts as AFTER

    # ---- Equality
    "SIMULTANEOUS": ("EQUALS", False), # TE3, MAVEN, TBDense
    "IDENTITY":     ("EQUALS", False), # TE3
    "EQUAL":        ("EQUALS", False), # MATRES

    # ---- Overlap
    "OVERLAP":      ("OVERLAPS", False), # MAVEN

    # ---- Starts / Finishes (boundary matches)
    "BEGINS":   ("STARTS", False),     # TE3
    "BEGUN_BY": ("STARTS", True),      # TE3 inverse
    "BEGINS-ON":("STARTS", False),     # MAVEN

    "ENDS":     ("FINISHES", False),   # TE3
    "ENDED_BY": ("FINISHES", True),    # TE3 inverse
    "ENDS-ON":  ("FINISHES", False),   # MAVEN

    # ---- Containment (kept separate)
    # DURING = A is inside B  (A during B)
    "DURING":       ("DURING", False),     # TE3
    "IS_INCLUDED":  ("DURING", False),     # TE3, TBDense

    # DURING_INV flips to DURING by swapping
    "DURING_INV":   ("DURING", True),      # TE3

    # CONTAINS = A contains B  (inverse of DURING)
    "INCLUDES":     ("CONTAINS", False),   # TE3
    "CONTAINS":     ("CONTAINS", False),   # MAVEN
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
    
    # Allen’s 7 forward relations output:
    # PRECEDES, MEETS, OVERLAPS, STARTS, DURING, FINISHES, EQUALS
    def to_allen(label, m0, m1):
        lab = (label or "").upper().strip()
        if lab not in ALLEN_MAP or ALLEN_MAP[lab] is None:
            raise ValueError(f"Unknown or unmapped label: {label}")
        allen, flip = ALLEN_MAP[lab]
        if flip:
            m0, m1 = m1, m0
        return allen, m0, m1

    def read(self, method : str, dataset : str, json_name: str):

        if method == "bio_tagger":
            extractor = TimeMLReader.BIO_tagger
            json_name = os.path.join(CLEANDATA_PATH, "BIO", dataset, json_name)
        elif method == "tlink_event_time":
            extractor = TimeMLReader.TLINK_ET_seqencer
            json_name = os.path.join(CLEANDATA_PATH, "E-T", dataset, json_name)
        elif method == 'tlink_event_event':
            extractor = TimeMLReader.TLINK_EE_sequencer
            json_name = os.path.join(CLEANDATA_PATH, "E-E", dataset, json_name)
        elif method == "timex_value":
            extractor = TimeMLReader.TIMEX_value_gen
            json_name = os.path.join(CLEANDATA_PATH, "normalised", dataset, json_name)
        else:
            raise ValueError(f"Method {method} is not supported.")
        
        if not json_name.endswith('.json'):
            raise ValueError("JSON path must end with .json")
        
        if os.path.exists(json_name):
            os.remove(json_name)

        data = []
        indicator = 0
        num_file = len(self.file_paths_to_read)

        for filepath in self.file_paths_to_read:

            if filepath.endswith('.tml'):
                print(f"Processing file {indicator+1}/{num_file}")
                part = extractor(filepath)
                if part == []:
                    print("resolved ", filepath)
                elif not all([len(sent["input_text"].split("</timex"))-1 == 1 for sent in part]):
                    print(filepath)
                    print([len(sent["input_text"].split("</timex"))-1 == 1 for sent in part])
                data.extend(part)
                indicator += 1

            if len(data)>200:
                datasets = TimeMLReader.convert_to_dataset(data, method)
                TimeMLReader.to_json(datasets, json_name)
                data = []
                datasets = None

        if len(data)!=0:
            datasets = TimeMLReader.convert_to_dataset(data, method)
            TimeMLReader.to_json(datasets, json_name)
        return   

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

    def read(self, method):
        if method == "BIO":
            json_path = os.path.join(CLEANDATA_PATH, "BIO", "MAVEN")
            extractor = MAVENReader.get_bio
        if method == "TempRel":
            json_path = os.path.join(CLEANDATA_PATH, "TempRel", "MAVEN")
            self.file_paths_to_read = self.file_paths_to_read[1:]
            extractor = MAVENReader.get_temprel

        for file in self.file_paths_to_read:
            ins, labs = [], []
            name = os.path.basename(file).split(".")[0]
            print(f"Reading {name}")
            with open(file, 'r') as f:
                data = [json.loads(line) for line in f]
                for i,line in enumerate(data):
                    print(f"Processing file {i}")
                    input, lab = extractor(line)
                    ins.extend(input)
                    labs.extend(lab)
            data2upload = Dataset.from_dict({"tokens":ins, "label":labs})
            data2upload.to_json(os.path.join(json_path, name+".json"))
   
        return
    
    @staticmethod
    def get_temprel(line):
        events = {instance["id"]:instance["mention"] for instance in line["events"]}
        timexs = {instance["id"]:instance for instance in line["TIMEX"]}
        relations = line["temporal_relations"]
        sents = line["tokens"]

        tokens, labels = [], []

        for rel_type in relations:
            orig_rel = rel_type
            for relation in relations[rel_type]:
                if relation[0].partition("_")[0] == "EVENT":
                    p0 = events[relation[0]]
                    parts = "E"
                else:
                    p0 = [timexs[relation[0]]]
                    parts = "T"
                if relation[1].partition("_")[0] == "EVENT":
                    p1 = events[relation[1]]
                    parts += "E"
                else:
                    p1 = [timexs[relation[1]]]
                    parts += "T"
                seps = E_E_SEPS if parts=="EE" else E_T_SEPS
                for m0 in p0:
                    for m1 in p1:
                        #rel_type, m0, m1 = Reader.to_allen(orig_rel, m0, m1)
                        text = deepcopy(sents)
                        if parts[0] == "E":
                            sep0_s, sep0_e = seps[0], seps[1]
                        else:
                            sep0_s, sep0_e = f"{seps[2]} {seps[4]}UNKNOWN {seps[5]}{m0['type']}>", seps[3]
                        if parts[1] == "E":
                            sep1_s, sep1_e = seps[2], seps[3]
                        else:
                            sep1_s, sep1_e = f"{seps[2]} {seps[4]}UNKNOWN {seps[5]}{m1['type']}>", seps[3]
          
                        text[m0["sent_id"]].insert(m0["offset"][1], sep0_e)
                        text[m0["sent_id"]].insert(m0["offset"][0], sep0_s)

                        if m0["sent_id"] == m1["sent_id"] and m0["offset"][0] < m1["offset"][0]:
                            offset = 2
                        else:
                            offset = 0

                        text[m1["sent_id"]].insert(m1["offset"][1] + offset, sep1_e)
                        text[m1["sent_id"]].insert(m1["offset"][0] + offset, sep1_s)

                        #text = text[ min(m0["sent_id"], m1["sent_id"]) : max(m0["sent_id"], m1["sent_id"])+1 ]
                        if m0["sent_id"] < m1["sent_id"]:
                            text = text[m0["sent_id"]] + ['<sep>'] + text[m1["sent_id"]]
                        elif m0["sent_id"] == m1["sent_id"]:
                            text = text[m0["sent_id"]]
                        else:
                            text = text[m1["sent_id"]] + ['<sep>'] + text[m0["sent_id"]]
                        
                        tokens.append(text)
                        labels.append(rel_type)

        return tokens, labels
    
    @staticmethod
    def get_bio(line):
        try:
            events = [instance for mention in line["events"] for instance in mention["mention"] ]
        except KeyError:
            events = line["event_mentions"]
        timexs = line["TIMEX"]
        sents = line["tokens"]
        labels = [["O" for token in sent] for sent in sents]

        for time in timexs:
            if time['type'] == "PREPOSTEXP":
                continue
            if time['type'] == "QUANTIFIER":
                time['type'] = "SET"
            replace = f"B-{time['type']}"
            for i in range(time["offset"][0], time["offset"][1]):
                labels[time["sent_id"]][i] = replace
                replace = f"I-{time['type']}"

        for event in events:
            replace = "B-EVENT"
            for i in range(event["offset"][0], event["offset"][1]):
                labels[event["sent_id"]][i] = replace
                replace = "I-EVENT"
        return sents, labels

class TimeMLReader(Reader):
    
    @staticmethod
    def get_doc_and_loc(root, eid2eiid):
        doc = root.find('TEXT')
        dct = root.find('DCT').find('TIMEX3')
        text, events, timexs, sents, sent_num = [], {}, {}, {}, 0
        text.extend(["Document creation date is ", dct.attrib.get('value')])
        timexs['t0'] = {'value':dct.attrib.get('value'), 'type':dct.attrib.get('type'), 'offset':(1, len(text))}
        text.append(".")
        for elem in doc.iter():
            start = len(text)
            text.append(elem.text.replace("\n",""))
            if elem.tag == 'EVENT':
                try:
                    events[eid2eiid[elem.attrib.get('eid')]] = {
                        'offset': (start, len(text))
                    }
                except KeyError:
                    continue
            elif elem.tag == 'TIMEX3':
                timexs[elem.attrib.get('tid')] = {
                    'value': elem.attrib.get('value'),
                    'type': elem.attrib.get('type'),
                    'offset': (start, len(text))
                }
            # elif elem.tail.count("\n") >= 2:
            #     sent_end = elem.tail.split("\n\n")
            #     text.append(sent_end[0])
            #     sents[sent_num] = len(text)
            #     text.append(sent_end[1:])
            # else:
            text.extend(elem.tail.split("\n"))
        return text, events, timexs, sents
    
    @staticmethod
    def get_time_seps(value, type):
        return E_T_SEPS[0], E_T_SEPS[1], f"{E_T_SEPS[2]} {E_T_SEPS[4]}{value} {E_T_SEPS[5]}{type}>", E_T_SEPS[3]
    
    @staticmethod
    def get_event_seps(value, type):
        return E_E_SEPS[0], E_E_SEPS[1], E_E_SEPS[2], E_E_SEPS[3]
    
    @staticmethod
    def get_event(eid, events):
        try:
            m0 = events[eid]
        except KeyError:
            try:
                m0 = events["ei10000"+eid[-2:]]
            except KeyError:
                try:
                    m0 = events["ei"+eid[-2:]]
                except KeyError:
                    try:
                        m0 = events["ei100000"+eid[-1:]]
                    except KeyError:
                        return 0
        return m0

    @staticmethod
    def link_to_input(sep_maker, text, links, events, timexs=None):
        out, labels = [], []
        if timexs is None:
            tlink_type = "relatedToEventInstance"
        else:
            tlink_type = "relatedToTime"

        for link in links:
            # if link["relType"] in ["NONE","VAGUE"]:
            #     continue

            m0 = TimeMLReader.get_event(link['eventInstanceID'], events)
            if m0==0:
                continue

            if tlink_type == "relatedToTime":
                try:
                    m1 = timexs[link[tlink_type]]
                except KeyError:
                    print(f"Key Error {m1}")
            else:
                m1 = TimeMLReader.get_event(link[tlink_type], events)
                if m1==0:
                    continue
            
            #rel_type, m0, m1 = Reader.to_allen(link["relType"], m0, m1)
            rel_type = link["relType"]

            sample = deepcopy(text)
            try:
                sep0s, sep0e, sep1s, sep1e = sep_maker(1 if timexs is None else m1["value"], 1 if timexs is None else m1["type"])
            except KeyError:
                sep1s, sep1e, sep0s, sep0e = sep_maker(1 if timexs is None else m0["value"], 1 if timexs is None else m0["type"])

            sample.insert(m0["offset"][1], sep0e)
            sample.insert(m0["offset"][0], sep0s)

            offset = 2 if m0["offset"][0] < m1["offset"][0] else 0

            sample.insert(m1["offset"][1] + offset, sep1e)
            sample.insert(m1["offset"][0] + offset, sep1s)

            # for sent_start in sents:
            #     if m0["offset"][0] <= sents[sent_start]:
            #         m0_sents_lower = max(0, sents[sent_start-1])
            #         m0_sents_upper = min(len(sample), sents[sent_start+1])
            #     if m1["offset"][0] <= sents[sent_start]:
            #         m1_sents_lower = max(0, sents[sent_start-1])
            #         m1_sents_upper = min(len(sample), sents[sent_start+1])

            m0_start = 0 if tlink_type=="relatedToTime" and link["relatedToTime"]=="t0" else max(0, m0['offset'][0]-20)
            m0_end = min(len(sample), m0['offset'][1]+20)
            m1_start = 0 if tlink_type=="relatedToTime" and link["relatedToTime"]=="t0" else max(0, m1['offset'][0]-20)
            m1_end = min(len(sample), m1['offset'][1]+20)

            if m0_start <= m1_end and m1_start <= m0_end:
                sample = sample[min(m0_start, m1_start):max(m0_end, m1_end)]
            elif m0["offset"][0] < m1["offset"][0]:
                sample = sample[m0_start: m0_end] + ['<sep>'] + sample[m1_start: m1_end]
            else:
                sample = sample[m1_start: m1_end] + ['<sep>'] + sample[m0_start: m0_end]   
            out.append(" ".join(sample).split(" "))
            labels.append(rel_type)
        return out, labels

    @staticmethod
    def get_temprel(file, pre_eiids=[]):
        tree = ET.parse(file)
        root = tree.getroot()
        out, labels = [], []
        eid2eiid = {mi.attrib.get('eventID'):mi.attrib.get('eiid') for mi in root.findall('MAKEINSTANCE')}
        if len(pre_eiids) > 0:
            EElinks = [{"eventInstanceID":"ei"+eiid1, 
                        "relatedToEventInstance":"ei"+eiid2, 
                        "relType":link} 
                        for eiid1, eiid2, link in pre_eiids]
            text, events, timexs, sents = TimeMLReader.get_doc_and_loc(root, eid2eiid)
            ee_inputs, ee_labels = TimeMLReader.link_to_input(TimeMLReader.get_event_seps, text, EElinks, events)
            out.extend(ee_inputs)
            labels.extend(ee_labels)
        else:
            EElinks = [{"eventInstanceID":link.attrib.get("eventInstanceID"), 
                        "relatedToEventInstance":link.attrib.get("relatedToEventInstance"), 
                        "relType":link.attrib.get("relType")} 
                        for link in root.findall('TLINK[@relatedToEventInstance][@eventInstanceID]')]
            ETlinks = [{"eventInstanceID":link.attrib.get("eventInstanceID"), 
                        "relatedToTime":link.attrib.get("relatedToTime"), 
                        "relType":link.attrib.get("relType")} 
                        for link in root.findall('TLINK[@relatedToTime][@eventInstanceID]')]
            text, events, timexs, sents = TimeMLReader.get_doc_and_loc(root, eid2eiid)
            ee_inputs, ee_labels = TimeMLReader.link_to_input(TimeMLReader.get_event_seps, text, EElinks, events)
            et_inputs, et_labels = TimeMLReader.link_to_input(TimeMLReader.get_time_seps, text, ETlinks, events, timexs)
            out.extend(ee_inputs)
            out.extend(et_inputs)
            labels.extend(ee_labels)
            labels.extend(et_labels)

        return out, labels

    @staticmethod
    def get_bio(file):
        tree = ET.parse(file)
        root = tree.getroot()
        text = root.find('TEXT')
        
        DCT = root.find('DCT').find('TIMEX3').get('value')

        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("sentencizer")

        data, sentence = [], {"tokens": [], "label": []}

        for node in text.iter():
            text_tokens = nlp(node.text.strip("\n"))

            # Checks if the node is a TAG
            if isinstance(node.tag, str):
                # Check if the node is a TIMEX3 or EVENT
                if node.tag == 'TIMEX3':
                    # Create BIO tags for TIMEX
                    for i in range(len(text_tokens)):
                        sentence["label"].append(f"B-"+node.attrib["type"] if i == 0 else "I-"+node.attrib["type"])
                # Check if the node is a EVENT
                elif node.tag =="EVENT":
                    # Create BIO tags for EVENT
                    for i in range(len(text_tokens)):
                        sentence["label"].append(f"B-EVENT" if i == 0 else "I-EVENT")
                # Must be another node
                else:
                    # Creates O tag for other nodes
                    sentence["label"].extend(["O"] * len(text_tokens))
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
                        sentence["label"].extend(["O"] * len(sents[0]))
                        data.append(sentence)
                        sentence = {"tokens": [], "label": []}
                        tail_tokens = sents[1]

                sentence["tokens"].extend(tail_tokens)
                sentence["label"].extend(["O"] * len(tail_tokens))

        data.append(sentence)
        return data

    @staticmethod
    def get_timex_values(filepath):
        tree = ET.parse(filepath)
        root = tree.getroot()
        text = root.find('TEXT')
        dct = root.find('DCT').find('TIMEX3').attrib["value"]
        sep = "<sep>"

        nlp = spacy.load("en_core_web_sm")
        seps = ["<timex","type=DATE>","type=TIME>","type=DURATION>","type=SET>","</timex>"]
        [nlp.tokenizer.add_special_case(thing, [{ORTH: thing}]) for thing in seps]
        nlp.add_pipe("sentencizer")

        locs = {}
        dist = {}
        types = {}
        values = {}
        article=[]

        for node in text.iter():
            tokens = nlp(node.text.replace("\n\n"," ").lstrip())
            if node.tag == "TIMEX3":
                try:
                    if node.attrib["value"].lower() != "null":
                        values[node.attrib["tid"]] = node.attrib["value"]
                    else:
                        continue
                except KeyError:
                    continue
                locs[node.attrib["tid"]] = len(article)
                dist[node.attrib["tid"]] = len(tokens)
                types[node.attrib["tid"]] = node.attrib["type"]
            article.extend(tokens)
            if node.tail:
                article.extend(nlp(node.tail.replace("\n\n"," ").lstrip()))

        article = [str(token) for token in article]
        data = []

        task = f"normalise time {sep}{dct}{sep} text:"

        for tid in locs:
            para = article.copy()
            para.insert(locs[tid], f"<timex type={types[tid]}>")
            para.insert(locs[tid]+dist[tid]+1, "</timex>")

            found = False
            trimmed = []
            prev_sent = []

            for sent in nlp(" ".join(para)).sents:
                if "</timex>" in [str(toke) for toke in sent]:
                    found = True
                    trimmed.extend(prev_sent)
                    trimmed.extend(sent)
                elif found:
                    trimmed.extend(sent)
                    break
                prev_sent = sent

            data.append({"input_text": task + " ".join([str(token) for token in trimmed]), "target_text": values[tid]})
    
        return data       
    
    @staticmethod
    def get_quintuples(filepath):
        tree = ET.parse(filepath)
        root = tree.getroot()
        out = []
        eiid2eid = {mi.attrib.get('eiid'):mi.attrib.get('eventID') for mi in root.findall('MAKEINSTANCE')}
        etlinks = {}
        for link in root.findall('TLINK[@relatedToTime][@eventInstanceID]'):
            eid = eiid2eid[link.attrib.get("eventInstanceID")]
            if eid in etlinks:
                etlinks[eid].append({"rel":link.attrib.get("relType"), "tid":link.attrib.get("relatedToTime")})
            else:
                etlinks[eid] = [{"rel":link.attrib.get("relType"), "tid":link.attrib.get("relatedToTime")}]
        eelinks = [{"eid1": eiid2eid[link.attrib.get("eventInstanceID")], "rel":link.attrib.get("relType"), "eid2":eiid2eid[link.attrib.get("relatedToEventInstance")]} for link in root.findall('TLINK[@relatedToEventInstance][@eventInstanceID]')]
        tids = {elem.attrib.get("tid"):elem.text for elem in root.findall('TIMEX')}
        eids = {elem.attrib.get("eid"):elem.text for elem in root.findall('EVENT')}

        for inst in eelinks:
            e1 = inst["eid"]
            e2 = inst["eid2"]

            label, e1, e2 = Reader.to_allen(inst["rel"], e1, e2)

            # redo to make temporal transistivity hold
            t1 = etlinks[e1] if e1 in etlinks else None
            t2 = etlinks[e2] if e2 in etlinks else None

            if t1 is None and t2 is None:
                continue

            out.append((eids[e1], label, eids[e2], t1, t2))

        return out
            
class TempEval3Reader(TimeMLReader):
    def __init__(self, path):
        super().__init__(path)

    def read(self, method, dataset_name="TempEval3"):
        if method == "TempRel":
            extractor = TimeMLReader.get_temprel
            json_path = os.path.join(CLEANDATA_PATH, "TempRel", dataset_name)
        elif method == "BIO":
            extractor = TimeMLReader.get_bio
            json_path = os.path.join(CLEANDATA_PATH, "BIO", dataset_name)
        elif method == "Normalise":
            extractor = TimeMLReader.get_bio
            json_path = os.path.join(CLEANDATA_PATH, "normalised", dataset_name)
        else:
            raise ValueError("Choose either TempRel, BIO, or Normalise as a method")
        
        ins, labs = [], []
        current_folder = self.file_paths_to_read[0].split('\\')[2]
        num_f = len(self.file_paths_to_read)
        print(f"Starting {current_folder}")
        print(self.file_paths_to_read)
        for i, filepath in enumerate(self.file_paths_to_read):
            if filepath.split('\\')[2] != current_folder:
                data2upload = Dataset.from_dict({"tokens":ins, "label":labs})
                data2upload.to_json(os.path.join(json_path, current_folder+".json"))
                data2upload = None
                current_folder = filepath.split('\\')[2]
            print(f"Processing file {i}/{num_f}")
            input, lab = extractor(filepath)
            ins.extend(input)
            labs.extend(lab)
        data2upload = Dataset.from_dict({"tokens":ins, "label":labs})
        data2upload.to_json(os.path.join(json_path, current_folder+".json"))

class MATRESReader(TimeMLReader):
    def __init__(self, path):
        super().__init__(path)

    def read(self):
        json_path = os.path.join(CLEANDATA_PATH, "TempRel", "MATRES")

        name_conv = {"aquaint.txt":"eval.json", "platinum.txt":"test.json", "timebank.txt":"train.json"}
      
        tempeval_files = np.array(Reader(os.path.join(RAWDATA_PATH, "TempEval3")).file_paths_to_read)

        ins, labs = [], []

        for file in self.file_paths_to_read:
            print(file)
            
            info = np.loadtxt(file, dtype=str)
            
            unique_files, indices = np.unique(info[:, 0], return_inverse=True)
            num_files = len(unique_files)
            
            for i, timeml_file in enumerate(unique_files):
                print(f"Processing file {i+1}/{num_files}")
                
                path_mask = np.char.find(tempeval_files, timeml_file) != -1
                
                path = tempeval_files[path_mask]
                eiids = info[info[:,0]==timeml_file, -3:]

                input, lab = TimeMLReader.get_temprel(path[0], eiids)
                ins.extend(input)
                labs.extend(lab)
                data = Dataset.from_dict({"tokens":ins, "label":labs})        
            data.to_json(os.path.join(json_path, name_conv[file.split("\\")[-1]]))
        return

class TBDenseReader(TempEval3Reader):
    def __init__(self, path):
        super().__init__(path)
    
    def read(self):
        super().read(method="TempRel", dataset_name="TBDense")
        
def id_token_labels(dataset, label2id):
    def change_id(row):
        row["label"] = [label2id[tag] for tag in row["label"]]
        return row
    return dataset.map(change_id) 
   
def obtain_combined_dataset(dataset_names, method):
    data = []
    for dataset_name in dataset_names:
        for json_name in ["train.json","test.json","eval.json"]:
            if dataset_name == "MAVEN" and json_name == "test.json":
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
    # te.read("TempRel")
    # te.read("BIO")
    # te.read("normalised")

    # te = MATRESReader("rawdata\\MATRES")
    # te.read()

    te = MAVENReader("rawdata\\MAVEN_ERE")
    te.read("TempRel")
    # te.read("BIO")
