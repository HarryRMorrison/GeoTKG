from Reader import TimeMLReader, OzRockReader
import os.path as path

####################### TempEval3 #######################
rawdata_path = path.join("rawdata", "TempEval3")

# te3_train_bio = TimeMLReader(path.join(rawdata_path,"Training"))
# te3_train_bio.read(method="timex3_bio_tagger", json_name="train.json")

# te3_test_bio = TimeMLReader(path.join(rawdata_path,"Evaluation","te3-platinum-normalized"))
# te3_test_bio.read(method="timex3_bio_tagger", json_name="test.json")

# te3_eval_bio = TimeMLReader(path.join(rawdata_path,"Gold"))
# te3_eval_bio.read(method="timex3_bio_tagger", json_name="eval.json")

# te3_train_tlink = TimeMLReader(path.join(rawdata_path,"Training"))
# te3_train_tlink.read(method="tlink_event_time", json_name="train.json")

# te3_test_tlink = TimeMLReader(path.join(rawdata_path,"Evaluation","te3-platinum-normalized"))
# te3_test_tlink.read(method="tlink_event_time", json_name="test.json")

te3_eval_tlink = TimeMLReader(path.join(rawdata_path,"Gold"))
te3_eval_tlink.read(method="tlink_event_time", json_name="eval.json")

######################### OzRock #########################
rawdata_path = path.join("rawdata", "OzRock")

ozrock_train = OzRockReader(rawdata_path)
ozrock_train.read("train.json", "eval.json")

