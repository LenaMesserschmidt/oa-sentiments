import spacy
import json
from spacy_sentiws import spaCySentiWS

# Setup SentiWS
nlp = spacy.load("de")
sentiws = spaCySentiWS(sentiws_path='data/sentiws/')
nlp.add_pipe(sentiws)

# open input data
f = open('data/json/data.json')
data = json.load(f)

item_counter = 0

# calculate sentiment for each item
for (k, v) in data.items():
    for article in v:
        doc = article["content"]
        res = nlp(doc)
        n = 0
        sum = 0
        magnitude = 0
        for token in res:
            if token._.sentiws is not None:
                sum += token._.sentiws
                n+= 1
                if abs(token._.sentiws) > magnitude:
                    magnitude = abs(token._.sentiws)
        article["sentiment_score"] = sum/n
        article["magnitude_score"] = magnitude
        del article["content"]
        item_counter += 1
f.close()

# write data to output file
with open("data/json/res_file.json", "w") as write_file:
    json.dump(data, write_file)

print("Successfully processed", item_counter, "articles.")
