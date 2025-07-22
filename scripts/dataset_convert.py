from Reader import TimeMLReader, OzRockReader

rawdata_path = "rawdata\\TempEval3\\"
cleandata_path = "cleandata\\TempEval3\\"

# s1 = TimeMLReader(rawdata_path+"Training\\TE3-Silver-data-1")
# s1.read(method="timex3_bio_tagger", json_path=cleandata_path+"silver-1.json", return_data=False)

# pl = TimeMLReader(rawdata_path+"Evaluation\\te3-platinum")
# pl.read(method="timex3_bio_tagger", json_path=cleandata_path+"platinum.json", return_data=False)

s0 = TimeMLReader(rawdata_path+"Training\\TE3-Silver-data-0")
s0.read(method="timex3_bio_tagger", json_path=cleandata_path+"silver-0.json", return_data=False)

pln = TimeMLReader(rawdata_path+"Evaluation\\te3-platinum-normalized")
pln.read(method="timex3_bio_tagger", json_path=cleandata_path+"platinum-normalized.json", return_data=False)

