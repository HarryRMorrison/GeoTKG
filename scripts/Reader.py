from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
import xml.etree.ElementTree as ET
import os
import spacy
from spacy.symbols import ORTH
import json
import numpy as np

RAWDATA_PATH = os.path.join("rawdata")
CLEANDATA_PATH = os.path.join("cleandata")

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

    def convert_to_dataset(data):
        if data[0].keys() == 'tokens':
            formatted_data = {"tokens":[], "label":[]}
            for sentence in data:
                formatted_data["tokens"].append([str(token) for token in sentence["tokens"]])
                formatted_data["label"].append([tag for tag in sentence["label"]])
            return Dataset.from_dict(formatted_data)
        else:
            return Dataset.from_list(data)

    def get_label_list(data, label2id=True, id2label=True):
        label_list = sorted(list(set([tag for sentence in data for tag in sentence['label']])))
        label2id = {label: int(i) for i, label in enumerate(label_list)}
        id2label = {int(i): label for label, i in label2id.items()}
        return label_list, label2id, id2label

class TimeMLReader(Reader):
    def __init__(self, path: str):
        super().__init__(path)

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
                elif len(part[0]["input_text"].split("</timex"))-1 != len(part[0]["target_text"].split("<sep>")):
                    print(filepath)
                    print([len(part[0]["input_text"].split("</timex"))-1, len(part[0]["target_text"].split("<sep>"))])
                data.extend(part)
                indicator += 1

            if indicator % 200 == 0:
                datasets = TimeMLReader.convert_to_dataset(data)
                TimeMLReader.to_json(datasets, json_name)
                data = []
                datasets = None

        if len(data)!=0:
            datasets = TimeMLReader.convert_to_dataset(data)
            TimeMLReader.to_json(datasets, json_name)
        return

    @staticmethod
    def BIO_tagger(filepath: str):
        tree = ET.parse(filepath)
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
    def TLINK_EE_sequencer(filepath: str):
        tree = ET.parse(filepath)
        root = tree.getroot()
        text = root.find('TEXT')
        tlinks = root.findall('TLINK[@relatedToEventInstance][@eventInstanceID]')

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

        for tlink in tlinks:
            eiid1 = tlink.attrib["eventInstanceID"]
            eiid2 = tlink.attrib["relatedToEventInstance"]

            para = article.copy()
            try:
                eid1 = root.find(f'MAKEINSTANCE[@eiid="{str(eiid1)}"]').attrib["eventID"]
                eid2 = root.find(f'MAKEINSTANCE[@eiid="{str(eiid2)}"]').attrib["eventID"]
            except AttributeError:
                continue

            try:
                ordering = order.index(eid1) < order.index(eid2)
            # Labeling error 'e1000036' in file 5 and so on
            except ValueError:
                continue

            try:
                if ordering:
                    para.insert(locs[eid1], seps[0])
                    para.insert(locs[eid1]+dist[eid1]+1, seps[1])
                    para.insert(locs[eid2]+2, seps[2])
                    para.insert(locs[eid2]+dist[eid2]+3, seps[3])
                else:
                    para.insert(locs[eid2], seps[2])
                    para.insert(locs[eid2]+dist[eid2]+1, seps[3])
                    para.insert(locs[eid1]+2, seps[0])
                    para.insert(locs[eid1]+dist[eid1]+3, seps[1])                 
            except KeyError:
                continue
            para = nlp(" ".join(para).strip("\n"))
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

            data.append({'tokens':trimmed, 'label':[tlink.attrib["relType"]]})
            
        return data

    @staticmethod
    def TLINK_ET_seqencer(filepath : str):
        tree = ET.parse(filepath)
        root = tree.getroot()
        text = root.find('TEXT')
        dct = root.find('DCT').find('TIMEX3')
        tlinks = root.findall('TLINK[@relatedToTime][@eventInstanceID]')

        nlp = spacy.load("en_core_web_sm")
        seps = ["[ES]","[EE]","[TS]","[TE]"]
        [nlp.tokenizer.add_special_case(sep, [{ORTH: sep}]) for sep in seps]
        nlp.add_pipe("sentencizer")

        article = ["Document", "creation", "date", "is"]
        locs = {}
        dist = {}
        order = []

        #article.extend(nlp(title.text.replace("\n\n"," ").lstrip()))

        dct_tokens = nlp(dct.text.replace("\n\n"," ").lstrip())
        locs[dct.attrib["tid"]] = len(article)
        dist[dct.attrib["tid"]] = len(dct_tokens)
        article.extend(dct_tokens)
        article.append(".")
        order.append(dct.attrib["tid"])


        for node in text.iter():
            tokens = nlp(node.text.replace("\n\n"," ").lstrip())
            if node.tag == "EVENT":
                locs[node.attrib["eid"]] = len(article)
                dist[node.attrib["eid"]] = len(tokens)
                order.append(node.attrib["eid"])
            elif node.tag == "TIMEX3":
                locs[node.attrib["tid"]] = len(article)
                dist[node.attrib["tid"]] = len(tokens)
                order.append(node.attrib["tid"])
            article.extend(tokens)
            if node.tail:
                article.extend(nlp(node.tail.replace("\n\n"," ").lstrip()))

        article = [str(token) for token in article]
        data = []

        for tlink in tlinks:
            t_id = tlink.attrib["relatedToTime"]
            e_id = tlink.attrib["eventInstanceID"]
            e_id = root.find(f'MAKEINSTANCE[@eiid="{str(e_id)}"]').attrib["eventID"]

            relType = tlink.attrib["relType"]

            para = article.copy()
            try:
                ordering = order.index(t_id) > order.index(e_id)
            # Labeling error 'e1000036' in file 5 and so on
            # APW19980308.0201 t69 doesnt exist
            except ValueError:
                print(filepath, t_id, e_id)
                continue

            try:
                if ordering:
                    para.insert(locs[e_id], seps[0])
                    para.insert(locs[e_id]+dist[e_id]+1, seps[1])
                    para.insert(locs[t_id]+2, seps[2])
                    para.insert(locs[t_id]+dist[t_id]+3, seps[3])
                else:
                    para.insert(locs[t_id], seps[2])
                    para.insert(locs[t_id]+dist[t_id]+1, seps[3])
                    para.insert(locs[e_id]+2, seps[0])
                    para.insert(locs[e_id]+dist[e_id]+3, seps[1])                 
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

            data.append({'tokens':trimmed, 'label':[relType]})
                            
        return data
    
    @staticmethod
    def TIMEX_value_gen(filepath : str):
        tree = ET.parse(filepath)
        root = tree.getroot()
        text = root.find('TEXT')
        dct = root.find('DCT').find('TIMEX3').attrib["value"]

        nlp = spacy.load("en_core_web_sm")
        sep = "<sep>"
        nlp.tokenizer.add_special_case(sep, [{ORTH: sep}])
        nlp.add_pipe("sentencizer")

        article = []
        sentence = []
        values = []
        timex_value_in_sentence = False
        timex_count = 0

        for node in text.iter():
            tokens = nlp(node.text.replace("\n\n"," ").lstrip())
            if node.tag == "TIMEX3":
                values.append(node.attrib["value"])
                sentence.extend([f"<timex type={node.attrib["type"]}>{" ".join([str(tok) for tok in tokens])}</timex>"])
                timex_value_in_sentence = True
                timex_count += 1
            else:
                sentence.extend(tokens)
            if node.tail:
                node_tail_stripped = node.tail.replace("\n\n"," ").lstrip()
                tail_tokens = nlp(node_tail_stripped)
                sents = list(tail_tokens.sents)

                if node_tail_stripped != "" and (len(sents) > 1 or str(sents[0]).rstrip()[-1] in [".", "!", "?", '"', "'"]) or node.tail == "\n\n" or node.tail[-2:] == "\n\n":
                    
                    if node.tail == "\n\n":
                        to_add = []
                        tail_tokens == []
                    elif len(sents) > 1:
                        tail_tokens = sents[1]
                        to_add = sents[0]
                    else:
                        to_add = tail_tokens


                    if timex_value_in_sentence:
                        sentence.extend(to_add)
                        article.extend(sentence)
                        timex_value_in_sentence = False

                    sentence = []
                sentence.extend(tail_tokens)
        
        task = f"normalise time {sep}{dct}{sep} text:"
        values = sep.join(values)


        return [{"input_text":task + " ".join([str(token) for token in article]), "target_text":values}] if timex_count > 0 else []

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

# def obtain_combined_dataset(dataset_names, method):
#     for dataset_name in dataset_names:


if __name__ == "__main__":
    article = TimeMLReader.TIMEX_value_gen("rawdata\TempEval3\Training\TE3-Silver-data-0\AFP_ENG_19970402.0207.tml")
    print(article)
    print("--------------------------------------------------------------------------------------")
    for time in article[0]["input_text"].split("</timex>"):
        print(time)
        print("--------------------------------------------------------------------------------------")
    for val in article[0]["target_text"].split("<sep>"):
        print(val)

    print(len(article[0]["input_text"].split("</timex>")))
    print(len(article[0]["target_text"].split("<sep>")))