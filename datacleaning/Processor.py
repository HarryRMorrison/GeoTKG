from Reader import TBDenseReader, TempEval3Reader, MAVENReader, OzRockReader, TweetsReader, WikiWarsReader
from copy import deepcopy
from sklearn.model_selection import train_test_split

def get_dataset(reader):
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

    X_train, X_test, y_train, y_test = train_test_split(all_data, y, test_size=0.1, random_state=42, stratify=y, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train, shuffle=True)
    return {"train": X_train, "eval": X_val, "test": X_test}

def obtain_tie_data(path = "D:\\GeoTKG\\rawdata\\"):
    data = {}
    for name, reader in [("TempEval3", TempEval3Reader),("TBDense", TBDenseReader),('MAVEN_ERE', MAVENReader)]:
        data[name] = get_dataset(reader(path + name))
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
            text = text[max(0, sent_id-1):min(len(text), sent_id+1)]
            text = " ".join([word for sent in text for word in sent])
            input_text = f'DCT: {dct} \nTYPE: {type_} \nTEXT: {text} \nSPAN: \"{mention}\"'
            out.append({'input_text':input_text, 'output_text':instance['value']})
            types.append(type_)
    return out, types
                      
def obtain_norm_data(path = "D:\\GeoTKG\\rawdata\\"):
    data = []
    types = []
    for reader in [TweetsReader, WikiWarsReader]:
        ds = reader(path + reader.__name__.replace("Reader","")).read()
        for sample in ds:
            out, t_types = retrieve_norm_text(sample)
            data.extend(out)
            types.extend(t_types)
    for reader in [TempEval3Reader, TBDenseReader]:
        ds = reader(path + reader.__name__.replace("Reader","")).read()
        for split in ds:
            for sample in ds[split]:
                out, t_types = retrieve_norm_text(sample)
                data.extend(out)
                types.extend(t_types)

    X_train, X_test, y_train, y_test = train_test_split(data, types, test_size=0.1, random_state=42, stratify=types, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train, shuffle=True)
    save_data({"train": X_train, "eval": X_val, "test": X_test}, path = "D:\\GeoTKG\\cleandata\\normalise\\")
                
    
if __name__ == "__main__":
    obtain_norm_data()