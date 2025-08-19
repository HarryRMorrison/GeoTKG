from Reader import TBDenseReader, TempEval3Reader, MAVENReader, OzRockReader, TweetsReader, WikiWarsReader

# Create index per sample for events and times (eid and tid)
# Split tokens up -> either by white space or roberta tokenizer ---> Maybe do this in Reader instead actually
# Select only 120 000 temprels per label
# collapse START and END temprels to contains or during?