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
    
    def to_json(data, intended_path : str):
        return

    def convert_to_dataset(data, method):
        return

    def get_label_list(data, label2id=True, id2label=True):
        label_list = sorted(list(set([tag for sentence in data for tag in sentence['label']])))
        label2id = {label: int(i) for i, label in enumerate(label_list)}
        id2label = {int(i): label for label, i in label2id.items()}
        return label_list, label2id, id2label

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

class MATRESReader(Reader):
    def __init__(self, path):
        super().__init__(path)

    def read(self):
        json_path = os.path.join(CLEANDATA_PATH, "E-E", "MATRES")

        tempeval_files = []

        for quality in ['Gold',"Training","Evaluation\\te3-platinum"]:
            tempeval_files.extend(Reader(os.path.join(RAWDATA_PATH, "TempEval3", quality)).file_paths_to_read)

        tempeval_files = np.array(tempeval_files)

        data = []

        for file in self.file_paths_to_read:
            print(file)
            info = np.loadtxt(file, dtype=str)
            unique_files, indices = np.unique(info[:, 0], return_inverse=True)
            num_files = len(unique_files)
            indicator = 0
            for timeml_file in unique_files:
                indicator += 1
                print(f"Processing file {indicator}/{num_files}")
                path_mask = np.char.find(tempeval_files, timeml_file) != -1
                path = tempeval_files[path_mask]
                eiids = info[info[:,0]==timeml_file, -3:]
                data.extend(MATRESReader.TLINK_event_event_finder(path[0], eiids))
        
        data = TimeMLReader.convert_to_dataset(data).shuffle(seed=42).train_test_split(test_size=0.2, seed=42)
        test = data["test"]
        train = data["train"].train_test_split(test_size=0.1, seed=42)
        data=None
        val = train["test"]
        train = train["train"]
        
        test.to_json(os.path.join(json_path, "test.json"))
        train.to_json(os.path.join(json_path, "train.json"))
        val.to_json(os.path.join(json_path, "eval.json"))


        
        return

    @staticmethod
    def TLINK_event_event_finder(path, eiids):
        tree = ET.parse(path)
        root = tree.getroot()
        text = root.find('TEXT')
        dct = root.find('DCT').find('TIMEX3')

        nlp = spacy.load("en_core_web_sm")
        seps = ["[E1S]","[E1E]","[E2S]","[E2E]"]
        [nlp.tokenizer.add_special_case(sep, [{ORTH: sep}]) for sep in seps]
        nlp.add_pipe("sentencizer")

        article = []
        locs = {}
        dist = {}
        order = []

        for node in text.iter():
            tokens = nlp(node.text.replace("\n\n"," ").lstrip())
            if node.tag == "EVENT":
                locs[node.attrib["eid"]] = len(article)
                dist[node.attrib["eid"]] = len(tokens)
                order.append(node.attrib["eid"])
            article.extend(tokens)
            if node.tail:
                article.extend(nlp(node.tail.replace("\n\n"," ").lstrip()))

        article = [str(token) for token in article]
        data = []

        for eiid1, eiid2, relation in eiids:
            para = article.copy()
            try:
                eiid1 = root.find(f'MAKEINSTANCE[@eiid="ei{str(eiid1)}"]').attrib["eventID"]
                eiid2 = root.find(f'MAKEINSTANCE[@eiid="ei{str(eiid2)}"]').attrib["eventID"]
            except AttributeError:
                continue

            try:
                ordering = order.index(eiid1) < order.index(eiid2)
            # Labeling error 'e1000036' in file 5 and so on
            except ValueError:
                continue

            try:
                if ordering:
                    para.insert(locs[eiid1], seps[0])
                    para.insert(locs[eiid1]+dist[eiid1]+1, seps[1])
                    para.insert(locs[eiid2]+2, seps[2])
                    para.insert(locs[eiid2]+dist[eiid2]+3, seps[3])
                else:
                    para.insert(locs[eiid2], seps[2])
                    para.insert(locs[eiid2]+dist[eiid2]+1, seps[3])
                    para.insert(locs[eiid1]+2, seps[0])
                    para.insert(locs[eiid1]+dist[eiid1]+3, seps[1])                 
            except KeyError:
                continue
            para = nlp(" ".join(para))
            sep_found = 0
            trimmed = []

            for sent in list(para.sents):
                sent = [str(token) for token in sent]
                if any(sep in sent for sep in seps):
                    trimmed.extend(sent)
                    sep_found += sum(sent.count(sep) for sep in seps)
                    if sep_found == 4:
                        break
                elif sep_found > 0:
                    trimmed.extend(sent)

            data.append({'tokens':trimmed, 'label':[relation]})
            
        return data

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

                        text = text[ min(m0["sent_id"], m1["sent_id"]) : max(m0["sent_id"], m1["sent_id"])+1 ]
                        
                        tokens.append(sum(text, []))
                        labels.append([rel_type])

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
    def get_temprel(file):
        tree = ET.parse(file)
        root = tree.getroot()
        doc = root.find('TEXT')
        dct = root.find('DCT').find('TIMEX3')
        eid2eiid = {mi.attrib.get('eventID'):mi.attrib.get('eiid') for mi in root.findall('MAKEINSTANCE')}
        EElinks = root.findall('TLINK[@relatedToEventInstance][@eventInstanceID]')
        ETlinks = root.findall('TLINK[@relatedToTime][@eventInstanceID]')
        
        text, events, timexs, out, labels = [], {}, {}, [], []
        text.extend(["Document creation date is ", dct.attrib.get('value')])
        timexs['t0'] = {'value':dct.attrib.get('value'), 'type':dct.attrib.get('type'), 'offset':(1, len(text))}

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
            text.extend(elem.tail.split("\n"))

        for EElink in EElinks:
            try:
                m0 = events[EElink.attrib.get("eventInstanceID")]
                m1 = events[EElink.attrib.get("relatedToEventInstance")]
            except KeyError:
                continue

            sample = deepcopy(text)

            sample.insert(m0["offset"][1], E_E_SEPS[1])
            sample.insert(m0["offset"][0], E_E_SEPS[0])

            offset = 2 if m0["offset"][0] < m1["offset"][0] else 0

            sample.insert(m1["offset"][1] + offset, E_E_SEPS[3])
            sample.insert(m1["offset"][0] + offset, E_E_SEPS[2])

            sample = sample[max(0, min(m0["offset"][0], m1["offset"][0])-10) : min(len(sample), max(m0["offset"][1], m1["offset"][1])+10)]
            out.append(" ".join(sample).split(" "))
            labels.append([EElink.attrib.get("relType")])

        for ETlink in ETlinks:
            try:
                m0 = events[ETlink.attrib.get("eventInstanceID")]
                m1 = timexs[ETlink.attrib.get("relatedToTime")]
            except KeyError:
                continue

            sample = deepcopy(text)

            sample.insert(m0["offset"][1], E_T_SEPS[1])
            sample.insert(m0["offset"][0], E_T_SEPS[0])

            offset = 2 if m0["offset"][0] < m1["offset"][0] else 0

            sample.insert(m1["offset"][1] + offset, E_T_SEPS[3])
            sample.insert(m1["offset"][0] + offset, f"{E_T_SEPS[2]} {E_T_SEPS[4]}{m1['value']} {E_T_SEPS[5]}{m1['type']}>")

            cut_s = 0 if ETlink.attrib.get("relatedToTime")=="t0" else max(0, min(m0["offset"][0], m1["offset"][0])-10)
            sample = sample[cut_s : min(len(sample), max(m0["offset"][1], m1["offset"][1])+10)]
            out.append(" ".join(sample).split(" "))
            labels.append([ETlink.attrib.get("relType")])
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
    
class TempEval3Reader(TimeMLReader):
    def __init__(self, path):
        super().__init__(path)

    def read(self, method):
        if method == "TempRel":
            extractor = TimeMLReader.get_temprel
            json_path = os.path.join(CLEANDATA_PATH, "TempRel", "TempEval3")
        elif method == "BIO":
            extractor = TimeMLReader.get_bio
            json_path = os.path.join(CLEANDATA_PATH, "BIO", "TempEval3")
        elif method == "Normalise":
            extractor = TimeMLReader.get_bio
            json_path = os.path.join(CLEANDATA_PATH, "normalised", "TempEval3")
        else:
            raise ValueError("Choose either TempRel, BIO, or Normalise as a method")
        
        ins, labs = [], []
        current_folder = self.file_paths_to_read[0].split('\\')[2]
        num_f = len(self.file_paths_to_read)
        print(f"Starting {current_folder}")

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


def id_token_labels(dataset, label2id):
    def change_id(row):
        row["label"] = [label2id[tag] for tag in row["label"]]
        return row
    return dataset.map(change_id) 

# Could change later to make exact train, test, eval json files
def obtain_dataset(dataset_name, method):
    train = load_dataset("json", data_files = os.path.join(CLEANDATA_PATH, method, dataset_name, "train.json"))["train"]
    val = load_dataset("json", data_files = os.path.join(CLEANDATA_PATH, method, dataset_name, "eval.json"))["train"]
    if method == "normalised":   
        test = load_dataset("json", data_files = os.path.join(CLEANDATA_PATH, method, dataset_name, "test.json"))["train"]
        return DatasetDict({"test": test, "train":train, "eval": val})
    else:
        label_list, label2id, id2label = obtain_label_list(train)
        dataset = {"train": id_token_labels(train, label2id),"eval": id_token_labels(val, label2id)}
        train, val = None, None
        if dataset_name != "OzRock":
            dataset["test"] = id_token_labels(load_dataset("json", data_files = os.path.join(CLEANDATA_PATH, method, dataset_name, "test.json"))["train"], label2id)
        return DatasetDict(dataset), label_list, label2id, id2label
    
def obtain_label_list(dataset):
    return Reader.get_label_list(dataset)

def obtain_combined_dataset(dataset_names, method):
    train = []
    val = []
    test = []
    for dataset_name in dataset_names:
        train.append(load_dataset("json", data_files = os.path.join(CLEANDATA_PATH, method, dataset_name, "train.json"))["train"])
        test.append(load_dataset("json", data_files = os.path.join(CLEANDATA_PATH, method, dataset_name, "test.json"))["train"])
        try:
            val.append(load_dataset("json", data_files = os.path.join(CLEANDATA_PATH, method, dataset_name, "eval.json"))["train"])
        except:
            continue
    train = concatenate_datasets(train).shuffle(seed=42)
    val = val[0]
    test = concatenate_datasets(test).shuffle(seed=42)
    return DatasetDict({"test": test, "train":train, "eval": val})

if __name__ == "__main__":
    te = TempEval3Reader("rawdata\\TempEval3")
    te.read("TempRel")

    # te = MAVENReader("rawdata\\MAVEN_ERE")
    # te.read("TempRel")
