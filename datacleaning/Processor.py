from Reader import MAVENReader
#from transformers import AutoTokenizer
import numpy as np

# Create index per sample for events and times (eid and tid)
# Split tokens up -> either by white space or roberta tokenizer ---> Maybe do this in Reader instead actually
# Select only 120 000 temprels per label
# collapse START and END temprels to contains or during?

def get_norm_inputs(doc):
    found_index = next((index for index, item in enumerate(doc["instances"]) if item.get("id") == "t0"), None)
    if found_index is None:
        return []
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
        if info['id']=="t0":
            continue

        text[info["sent_id"]].insert(info["offset"][0], f"<timex type={info['type']}>")
        text[info["sent_id"]].insert(info["offset"][1]+1, "</timex>")

        text = text[max(0,info["sent_id"]-1):max(len(text),info["sent_id"]+1)]
        text = [wrd for inner in text[info["sent_id"]-1:info["sent_id"]+1] for wrd in inner]
        data.append({"input_text": task + " ".join(text), "target_text": info["value"]})

    return data

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
        eventtimes['event'] = eid2index[eventtimes['event']]
        eventtimes['time'] = tid2index[eventtimes['time']]
    return data

def get_temprel_counts(data):
    total = {}
    for split in data:
        split_counts = []
        for sample in data[split]:
            counts = {}
            for temprel in sample["ee_temprels"]:
                counts[temprel["rel"]] = counts.get(temprel["rel"], 0) + 1
            split_counts.append(counts)
        total[split] = split_counts
    return total

def get_dataset(reader):
    data = reader.read()
    for split in data:
        if type(reader) is MAVENReader and split == "test":
            continue
        for sample in data[split]:
            sample = reindex(sample)
    return data

# counts = {}
# data = []
# for name, reader in [('maven', MAVENReader("rawdata\\MAVEN_ERE"))]:#, ("tbdense", TBDenseReader("rawdata\\TBDense")), ("te3", TempEval3Reader("rawdata\\TempEval3"))]:
#     out = get_dataset(reader)
    
# print(counts)
