from Reader import TimeMLReader, OzRockReader, MATRESReader
import os.path as path

####################### TempEval3 #######################
rawdata_path = path.join("rawdata", "TempEval3")

# te3_train_bio = TimeMLReader(path.join(rawdata_path,"Training"))
# te3_train_bio.read(method="bio_tagger", json_name="train.json")

# te3_test_bio = TimeMLReader(path.join(rawdata_path,"Evaluation","te3-platinum-normalized"))
# te3_test_bio.read(method="bio_tagger", json_name="test.json")

# te3_eval_bio = TimeMLReader(path.join(rawdata_path,"Gold"))
# te3_eval_bio.read(method="bio_tagger", json_name="eval.json")

# te3_train_tlink = TimeMLReader(path.join(rawdata_path,"Training"))
# te3_train_tlink.read(method="tlink_event_time", json_name="train.json")

# te3_test_tlink = TimeMLReader(path.join(rawdata_path,"Evaluation","te3-platinum-normalized"))
# te3_test_tlink.read(method="tlink_event_time", json_name="test.json")

# te3_eval_tlink = TimeMLReader(path.join(rawdata_path,"Gold"))
# te3_eval_tlink.read(method="tlink_event_time", json_name="eval.json")

te3_train_value = TimeMLReader(path.join(rawdata_path,"Training"))
te3_train_value.read(method="timex_value", dataset="TempEval3", json_name="train.json")

te3_test_value = TimeMLReader(path.join(rawdata_path,"Evaluation","te3-platinum-normalized"))
te3_test_value.read(method="timex_value", dataset="TempEval3", json_name="test.json")

te3_eval_value = TimeMLReader(path.join(rawdata_path,"Gold"))
te3_eval_value.read(method="timex_value", dataset="TempEval3", json_name="eval.json")

######################### OzRock #########################
# rawdata_path = path.join("rawdata", "OzRock")

# ozrock_train = OzRockReader(rawdata_path)
# ozrock_train.read("train.json", "eval.json")

######################### MATRES #########################
# rawdata_path = path.join("rawdata", "MATRES")

# matres = MATRESReader(path.join(rawdata_path))
# matres.read()

######################### TBDense #########################
# rawdata_path = path.join("rawdata", "TBDense")

# tbdense_et_dev = TimeMLReader(path.join(rawdata_path, "dev"))
# tbdense_et_dev.read('tlink_event_time', 'TBDense', "eval.json")

# tbdense_et_test = TimeMLReader(path.join(rawdata_path, "test"))
# tbdense_et_test.read('tlink_event_event', 'TBDense', "test.json")

# tbdense_et_train = TimeMLReader(path.join(rawdata_path, "train"))
# tbdense_et_train.read('tlink_event_event', 'TBDense', "train.json")



