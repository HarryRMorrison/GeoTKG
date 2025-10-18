from Reader import TBDenseReader, TempEval3Reader, MAVENReader, OzRockReader, TweetsReader, WikiWarsReader
from copy import deepcopy
from sklearn.model_selection import train_test_split
import json

def get_dataset(reader):
    with open("datacleaning\\test_set.json", "r") as f:
        examples= [json.loads(line) for line in f][0]
    blacklist = examples['blacklist']
    if type(reader) in [TBDenseReader, TempEval3Reader]:
        data = reader.read(blacklist=blacklist)
    else:
        data = reader.read()
    if type(reader) is MAVENReader:
        data.pop("test")
    for split in data:
        for sample in data[split]:
            sample = reindex(sample)
    return data

def get_sample_temprel_ratio_class(sample):
    total = {"BIG": 0, "SMALL": 0}
    quantiles = [0.04398368926670814, 0.10526315789473684, 0.17049914241693068, 0.2608695652173913]
    for temprel in sample["ee_temprels"]:
        if temprel["rel"] in ["BEFORE", "AFTER", "CONTAINS"]:
            total["BIG"] += 1
        else:
            total["SMALL"] += 1

    if total["SMALL"] + total["BIG"] == 0:
        return 0
    
    ratio = total["SMALL"]/(total["SMALL"]+total["BIG"])

    # Need to find ratio thresholds -> precompute it with whole dataset
    if ratio < quantiles[0]: return "TINY"
    elif ratio < quantiles[1]: return "SMALL"
    elif ratio < quantiles[2]: return "MEDIUM"
    elif ratio < quantiles[3]: return "BIG"
    else: return "HUGE"

def data_temprel_select(data):
    counts = {"BEFORE": 0, "AFTER": 0, "DURING": 0, "CONTAINS": 0, "OVERLAPS": 0, "EQUALS": 0, "IDENTITY": 0}
    target = 75855
    balanced = {}

    for name in data:
        balanced[name] = {}
        for split in data[name]:
            balanced[name][split] = []
            for sample in data[name][split]:
                new_sample = deepcopy(sample)
                new_sample["ee_temprels"] = []
                for temprel in sample["ee_temprels"]:

                    if name in ["TempEval3", "TBDense"] and split == "train" and temprel['rel'] == "BEFORE":
                        continue

                    elif temprel['rel'] == "BEFORE" and counts["BEFORE"] == target and counts["AFTER"] < target:
                        counts["AFTER"] += 1
                        new_sample["ee_temprels"].append({"rel": "AFTER", "e1": temprel["e2"], "e2": temprel["e1"]})

                    elif counts[temprel['rel']] < target:
                            counts[temprel['rel']] += 1
                            new_sample["ee_temprels"].append(temprel)

                if len(new_sample["ee_temprels"])==0 and any(True if re['rel'] != "BEFORE" and re['rel'] != "AFTER" else False for re in sample["ee_temprels"]):
                    print(counts)
                    print(name, split, [re['rel'] for re in sample["ee_temprels"] if re['rel'] != "BEFORE" or re['rel'] != "AFTER"])
                balanced[name][split].append(new_sample)

    return balanced, counts

def reindex(data):
    eid2index = {}
    tid2index = {}
    for inst in data["instances"]:
        if inst["type"] == "EVENT":
            eid2index[inst["id"]] = len(eid2index)
            inst['id'] = eid2index[inst['id']]
        else:
            tid2index[inst["id"]] = len(tid2index)
            inst['id'] = tid2index[inst['id']]

    for temprel in data["ee_temprels"]:
        temprel['e1'] = eid2index[temprel['e1']]
        temprel['e2'] = eid2index[temprel['e2']]

    for eventtimes in data["event_times"]:
        if eventtimes['time'][0] == "e":
            time = eventtimes['event']
            event = eventtimes['time']
            eventtimes['event'] = event
            eventtimes['time'] = time

        eventtimes['event'] = eid2index[eventtimes['event']]
        eventtimes['time'] = tid2index[eventtimes['time']]

    return data

def save_data(data, path):
    import json
    if 'test' not in data:
        cycle = [("train.json", data['train']), ("eval.json", data['eval'])]
    else:
        cycle = [("train.json", data['train']), ("eval.json", data['eval']), ("test.json", data['test'])]
    for name, set in cycle:
        with open(path+name, 'w') as json_file:
            for sample in set:
                json_file.write(json.dumps(sample)+"\n")

def get_quintuples_test_data(data, y):
    import numpy as np
    norms, norms_y = [], []
    defaults, defaults_y = [], []
    for i, sample in enumerate(data):
        add_sample = False
        for instance in sample['instances']:
            if instance['type'] != 'EVENT' and 'value' in instance:
                if instance['value'] is not None and instance['value'] != "null":
                    add_sample = True
                    break
        if add_sample:
            norms.append(sample)
            norms_y.append(y[i])
        else:
            defaults.append(sample)
            defaults_y.append(y[i])

    values, counts = np.unique(y, return_counts=True)
    props = {values[i]:{"current":0, "target":np.ceil(600*(counts[i]/len(y)))} for i in range(len(values))}
    test_set = []
    new_data = []
    new_y = []
    paired_list = list(zip(norms, norms_y))
    import random
    random.seed(42)
    random.shuffle(paired_list)
    norms, norms_y = zip(*paired_list)
    norms, norms_y = list(norms), list(norms_y)
    for i, sample in enumerate(norms):
        if props[norms_y[i]]["current"] < props[norms_y[i]]["target"]:
            test_set.append(sample)
            props[norms_y[i]]["current"] += 1
        else:
            new_data.append(sample)
            new_y.append(norms_y[i])

    new_data.extend(defaults)
    new_y.extend(defaults_y)

    return test_set, new_data, new_y


def combine_and_stratify(data):
    all_data = []
    y = []
    for name in data:
        for split in data[name]:
            for sample in data[name][split]:
                class_out = get_sample_temprel_ratio_class(sample)
                if class_out == 0:
                    continue
                else:
                    all_data.append(sample)
                    y.append(class_out)

    #test_set, new_data, new_y = get_quintuples_test_data(all_data, y)

    X_train, X_val, y_train2, y_val = train_test_split(all_data, y, test_size=0.1, random_state=42, stratify=y, shuffle=True)
    return {"train": X_train, "eval": X_val}#, "test": test_set}

def obtain_tie_data(path = "D:\\GeoTKG\\rawdata\\"):
    data = {}
    for name, reader in [("TempEval3", TempEval3Reader),("TBDense", TBDenseReader),('MAVEN_ERE', MAVENReader)]:
        extracted = get_dataset(reader(path + name))
        data[name] = extracted
    bal, cnts = data_temprel_select(data)
    bal = combine_and_stratify(bal)
    save_data(bal, path = "D:\\GeoTKG\\cleandata\\tie\\")

def obtain_geo_data(path = "D:\\GeoTKG\\rawdata\\"):
    reader = OzRockReader(path + "OzRock")
    out = reader.read()
    save_data(out, path = "D:\\GeoTKG\\cleandata\\geo\\")

# Change to filter samples that actually have timexs
def retrieve_norm_text(sample):
    dct = [inst for inst in sample["instances"] if inst['id'] == "t0" and inst['type'] != 'EVENT'][0]['value']
    out = []
    types = []
    for instance in sample["instances"]:
        if instance['type'] != 'EVENT' and instance['id'] != "t0":
            if instance['value'] == None or instance['value']=="null":
                continue
            text = deepcopy(sample['text'])
            sent_id = instance["sent_id"]
            mention = " ".join(text[sent_id][instance['offset'][0]:instance['offset'][1]])
            type_ = instance['type']

            out_text = [word for sent in text for word in sent]
            lens = [len(sent) for sent in text[:sent_id]]
            flat_loc_s = sum(lens)+instance['offset'][0]
            flat_loc_e = sum(lens)+instance['offset'][0]

            text = out_text[max(0, flat_loc_s-100):min(len(out_text), flat_loc_e+100)]
            #text = text[max(0, sent_id-1):min(len(text), sent_id+1)]

            text = " ".join([word for word in text])
            input_text = f'DCT: {dct} \nTYPE: {type_} \nTEXT: {text} \nSPAN: \"{mention}\"'
            out.append({'input_text':input_text, 'output_text':instance['value']})
            types.append(type_)
    return out, types
                      
def obtain_norm_data(path = "D:\\GeoTKG\\rawdata\\"):
    data = []
    types = []
    with open("datacleaning\\test_set.json", "r") as f:
        examples= [json.loads(line) for line in f][0]
    blacklist = examples['blacklist']
    for reader in [WikiWarsReader]:
        ds = reader(path + reader.__name__.replace("Reader","")).read()
        for sample in ds:
            out, t_types = retrieve_norm_text(sample)
            data.extend(out)
            types.extend(t_types)
    for reader in [TempEval3Reader, TBDenseReader]:
        ds = reader(path + reader.__name__.replace("Reader","")).read(blacklist=blacklist)
        for split in ds:
            for sample in ds[split]:
                out, t_types = retrieve_norm_text(sample)
                data.extend(out)
                types.extend(t_types)

    X_train, X_val, y_train, y_val = train_test_split(data, types, test_size=0.1, random_state=42, stratify=types, shuffle=True)
    save_data({"train": X_train, "eval": X_val}, path = "D:\\GeoTKG\\cleandata\\normalise\\")
                
    
if __name__ == "__main__":
    obtain_tie_data()