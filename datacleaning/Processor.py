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

def get_et_pairs(dset):
    all_pairs = {"train":[], "eval":[], "test":[]}
    for split in dset:
        for sample in dset[split]:
            ev_ti_pairs = {}
            for et in sample["event_times"]:
                if et["event"] in ev_ti_pairs:
                    print(f"Duplicate event found: {et['event']}")
                ev_ti_pairs[et["event"]] = et["time"]
            all_pairs[split].append(ev_ti_pairs)
    return all_pairs

def apply_none_event_times(bal_data):
    bal_cpy = deepcopy(bal_data)
    
    for dset in bal_cpy:
        all_pairs = get_et_pairs(bal_cpy[dset])
        for split in bal_cpy[dset]:
            for i, sample in enumerate(bal_cpy[dset][split]):
                pairs = list(all_pairs[split][i].keys())
                new_event_times = deepcopy(sample['event_times'])

                for event in [inst for inst in sample['instances'] if inst['type']=='EVENT']:
                    
                    if event["id"] not in pairs:
                        new_event_times.append({"event": event["id"], "time": "NONE"})

                sample['event_times'] = new_event_times

                if len([inst for inst in sample['instances'] if inst['type']=="EVENT"]) != len(sample['event_times']):
                    print(f"Mismatch in event times for sample {bal_cpy[dset][split].index(sample)} in {dset} {split}: {len([inst for inst in sample['instances'] if inst['type']=='EVENT'])} vs {len(sample['event_times'])}")
    return bal_cpy

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
    bal = apply_none_event_times(bal)
    bal = combine_and_stratify(bal)
    save_data(bal, path = "D:\\GeoTKG\\cleandata\\tie\\")

def obtain_geo_data(path = "D:\\GeoTKG\\rawdata\\"):
    reader = OzRockReader(path + "OzRock")
    out = reader.read()
    save_data(out, path = "D:\\GeoTKG\\cleandata\\geo\\")

if __name__ == "__main__":
    obtain_geo_data()