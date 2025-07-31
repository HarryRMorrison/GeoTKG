from Reader import TimeMLReader, OzRockReader, MATRESReader
import os.path as path

jsons = ["train.json","eval.json","test.json"]
rawdata_path = path.join("rawdata")

####################### TempEval3 #######################
# methods = ["timex_value"]# "tlink_event_time"
# folders = ["Training", "Gold", "Evaluation\\te3-platinum-normalised"]

# for method in methods:
#     for folder, json_name in zip(folders, jsons):
#         te3_read = TimeMLReader(path.join(rawdata_path, "TempEval3", folder))
#         te3_read.read(method, "TempEval3", json_name)

######################### OzRock #########################
# ozrock_train = OzRockReader(path.join(rawdata_path,"OzRock"))
# ozrock_train.read("train.json", "eval.json")

######################### MATRES #########################
# matres = MATRESReader(path.join(rawdata_path, "MATRES"))
# matres.read()

######################### TBDense #########################
# folders = ["train", "dev", "test"]

# for folder, json_name in zip(folders, jsons):
#     tbd_read = TimeMLReader(path.join(rawdata_path, "TBDense", folder))
#     tbd_read.read('tlink_event_time', "TBDense", json_name)

######################### WikiWars #########################
# folders = ["trainingset", "wikiwars_test_with_newline"]

# for folder, json_name in zip(folders, [jsons[0],jsons[2]]):
#     te3_read = TimeMLReader(path.join(rawdata_path, "wikiwars", folder))
#     te3_read.read("timex_value", "wikiwars", json_name)

folders = ["trainingset", "tweets_test_with_newline"]

for folder, json_name in zip(folders, [jsons[0],jsons[2]]):
    te3_read = TimeMLReader(path.join(rawdata_path, "tweets", folder))
    te3_read.read("timex_value", "tweets", json_name)

