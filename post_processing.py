import json
import isodate
from datetime import datetime, date
from copy import deepcopy
from geotkg.models.globals import LABEL2ID_EVNER

def get_data(path, preprocessor=None):
    with open(path, "r") as f:
        examples=[json.loads(line) for line in f]
    if preprocessor is not None:
        examples = [preprocessor(example) for example in examples]
    return examples

def gentext_to_iso8601(gentext: str):
    parsers = {
        isodate.parse_date:"DATE",
        isodate.parse_time:"TIME",
        isodate.parse_datetime:"TIME",
        isodate.parse_duration:"DURATION",
        isodate.parse_tzinfo:"SET",
    }

    for parser in parsers:
        try:
            out = parser(gentext)
            type_ = parsers[parser]
            return out, type_
        except Exception:
            continue
    return None

def ner_formating(ner_preds):
    poster = []

    for fn, pred in enumerate(ner_preds):
        stripped = pred['pred'].replace('"entity surface text": ', '').replace('"type": ', '').replace("}",']').replace('{"entity": ',"[").replace("{","[").replace('"surface text":', "").replace('"surface_text":', "").replace('"event": ', "").replace('"entity": ', "").replace('"date": ', "").replace('"time": ', "").replace('"duration": ', "").replace('"set": ', "").replace('"text": ', "")
        texty = stripped.partition('[')[-1].rpartition(']')[0]
        pred_json = jsonify("["+texty+"]")

        if pred_json is None:
            pred_json = jsonify("["+texty+"]]")
        
        if pred_json is None:
            pred_json = jsonify("["+texty.replace("(","[").replace(")","]")+"]")

        if pred_json is None:
            print(fn)
            pred_json = texty
            
        poster.append({'text':pred['text'], 'pred':pred_json})
    return poster

def jsonify(output):
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return None
    
INVERSE = {
    "AFTER": "BEFORE",
    "BEFORE": "AFTER",
    "CONTAINS": "DURING",
    "DURING": "CONTAINS",
    "EQUALS": "EQUALS",
    "OVERLAPS": "OVERLAPS",       # add if you use these
    "IDENTITY": "IDENTITY"
    # extend as needed
}

def get_ee_temprels(temprel_list, exhaustive=False, only_flipped=False):
    if exhaustive and only_flipped:
        raise TypeError("exhaustive and only flipped and mutually exclusive.")
    all_pairs = []
    for triple in temprel_list:
        try:
            e1, rel, e2 = triple["e1"], triple['rel'], triple['e2']
        except:
            e1, rel, e2 = triple[0], triple[1], triple[2]

        all_pairs.append([e1, rel, e2])
    if only_flipped:
        for (e1, rel, e2) in all_pairs:
            c = all_pairs.count([e2, INVERSE[rel], e1])
            if c == 0:
                all_pairs.append([e2, INVERSE[rel], e1])
    elif exhaustive:
        for (e1, rel, e2) in all_pairs:
            c = all_pairs.count([e2, INVERSE[rel], e1])
            if c == 0:
                all_pairs.append([e2, INVERSE[rel], e1])
        for trip1 in all_pairs:
            for trip2 in all_pairs:
                if trip1 == trip2 or trip1[1] != trip2[1] or trip1[2] != trip2[0]:
                    continue
                if trip1[1] == "BEFORE" and trip2[1] == "BEFORE":
                    if all_pairs.count([trip1[0], "BEFORE", trip2[2]]) == 0:
                        all_pairs.append([trip1[0], "BEFORE", trip2[2]])
                if trip1[1] == "AFTER" and trip2[1] == "AFTER":
                    if all_pairs.count([trip1[0], "AFTER", trip2[2]]) == 0:
                        all_pairs.append([trip1[0], "AFTER", trip2[2]])
                if trip1[1] == "CONTAINS" and trip2[1] == "CONTAINS":
                    if all_pairs.count([trip1[0], "CONTAINS", trip2[2]]) == 0:
                        all_pairs.append([trip1[0], "CONTAINS", trip2[2]])
                if trip1[1] == "DURING" and trip2[1] == "DURING":
                    if all_pairs.count([trip1[0], "DURING", trip2[2]]) == 0:
                        all_pairs.append([trip1[0], "DURING", trip2[2]])
                if trip1[1] == "EQUALS" and trip2[1] == "EQUALS":
                    if all_pairs.count([trip1[0], "EQUALS", trip2[2]]) == 0:
                        all_pairs.append([trip1[0], "EQUALS", trip2[2]])
    return all_pairs

def get_start_end_times(event_times: list, dct):
    all_times = [gentext_to_iso8601(time) for time in event_times]
    if len(all_times) == 0:
        return None, None
    dates = [time[0] for time in all_times if time[1] == "DATE"]
    times = [time[0] for time in all_times if time[1] == "TIME"]
    durs = [time[0] for time in all_times if time[1] == "DURATION"]

    if len(dates)>0:
        s_time = min(dates)
        e_time = max(dates)
    elif len(times)>0:
        s_time = min(times)
        e_time = max(times)
    else:
        s_time = dct
        e_time = dct
    
    for time in times:
        value = time
        if type(s_time) == date:
            value = time.date()
        if s_time >= value:
            s_time = time
        elif e_time <= value:
            e_time = time
    
    s_time = datetime.combine(s_time, datetime.min.time()) if type(s_time)==date else s_time
    e_time = datetime.combine(e_time, datetime.min.time()) if type(e_time)==date else e_time

    for duration in durs:
        if s_time + duration > e_time:
            e_time = s_time + duration

    return s_time, e_time

def truth_quintuples_and_triples_formating(example):
    events = {}
    times = {}
    event_quins = {}
    ets = {}
    for instance in example['instances']:
        instance_id = instance["id"]
        if instance["type"] == "EVENT":
            events[instance_id] = instance
        else:
            times[instance_id] = instance

    ee_trips = get_ee_temprels(example['ee_temprels'])
    for trip in ee_trips:
        trip[0] = events[trip[0]]['text']
        trip[2] = events[trip[2]]['text']

    for et in example["event_times"]:
        evid = et["event"] 
        if 'value' in times[et["time"]]:
            value = times[et["time"]]['value']
        else:
            value = None
        if evid not in ets:
            ets[evid] = [value]
        else:
            ets[evid].append(value)

    dct = gentext_to_iso8601(times[0]['value'])[0]

    for eid, event in events.items():
        s_time, e_time = get_start_end_times(ets.get(eid, []), dct)
        quint = {
            "event": event["text"],
            "subject": None,
            "object": None,
            "s_time": s_time,
            "e_time": e_time
        }
        event_quins[eid] = quint

    return {'times':list(times.values()), 'quintuples':event_quins, 'triples':list(ee_trips)}

def pred_quintuple_formating(preds):
    preds = preds['pred']
    times = {}
    for entry in preds['times']:
        value = gentext_to_iso8601(entry[2])
        if value is not None:
            time_val = value[0]
            if value[1] == "DATE":
                time_val = datetime.combine(time_val, datetime.min.time())
        try:
            times[entry[0]] = (entry[1], time_val, entry[3])
        except IndexError:
            times[entry[0]] = (entry[1], time_val, entry[-1])
        
    quintuples = {}
    for event in preds['quintuples']:
        if len(event)==5:
            stime = times[event[-1]][1] if event[-1] in times else None
            etime = None
        elif len(event) == 6:
            stime = times[event[-2]][1] if event[-2] in times else None
            etime = times[event[-1]][1] if event[-1] in times else None
        else:
            stime = None
            etime = None
        quintuples[event[0]] = {'subject':event[1], 'event':event[2], 'object':event[3], 's_time':stime, 'e_time':etime}

    trips = []
    for trip in preds['triples']:
        try:
            trips.append([quintuples[trip[0]]['event'], trip[1], quintuples[trip[2]]['event']])
        except:
            try:
                trips.append([quintuples["E"+trip[0][1]]['event'], trip[1], quintuples["E"+trip[2][1]]['event']])
            except:
                continue
    all_trips = get_ee_temprels(trips, exhaustive=True)
    return {"times":times, "quintuples":quintuples, "triples":all_trips}

def geo_eval_formating(example):
    instances = []
    i = 0
    while i < len(example["bio_tags"]):
        if example["bio_tags"][i] != "O":
            text = example["tokens"][i]
            tag = example["bio_tags"][i][2:]
            i += 1
            while i < len(example["bio_tags"]) and example["bio_tags"][i] == "I" + example["bio_tags"][i][1:]:
                text += example["tokens"][i] + " "
                i += 1
            instances.append({"type": tag, "text":text.strip()})
        i += 1
    return instances