from Reader import TimeMLReader, OzRockReader, MATRESReader
import os.path as path

jsons = ["train.json", "eval.json", "test.json"]
rawdata_path = path.join("rawdata")

####################### TempEval3 #######################
methods = ["timex_value"]# "tlink_event_time"
folder = ["Training", "Gold", "Evaluation\\te3-platinum-normalised"]

for method in methods:
    for folder, json_name in zip(folders, jsons):
        te3_read = TimeMLReader(path.join(rawdata_path, "TempEval3", folder))
        te3_read.read(method, "TempEval3", json_name)

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



