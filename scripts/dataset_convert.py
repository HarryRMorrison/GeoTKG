from Reader import TimeMLReader, OzRockReader
import os.path as path

rawdata_path = path.join("rawdata", "TempEval3")
cleandata_path = path.join("cleandata", "TempEval3")

silver = TimeMLReader(path.join(rawdata_path,"Training"))
silver.read(method="timex3_bio_tagger", json_path=path.join(cleandata_path,"silver-O-less.json"))

platinum = TimeMLReader(path.join(rawdata_path,"Evaluation","te3-platinum-normalized"))
platinum.read(method="timex3_bio_tagger", json_path=path.join(cleandata_path,"platinum-O-less.json"))

gold = TimeMLReader(path.join(rawdata_path,"Gold"))
gold.read(method="timex3_bio_tagger", json_path=path.join(cleandata_path,"gold-O-less.json"))

# rawdata_path = path.join("rawdata", "OzRock")
# cleandata_path = path.join("cleandata", "OzRock")

# ozrock_train = OzRockReader(rawdata_path)
# ozrock_train.read(path.join(cleandata_path, "train.json"), path.join(cleandata_path, "eval.json"))

