from nltk.tokenize import (sent_tokenize,
                           word_tokenize)

from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

glovefile = "../glove.6B/glove.6B.100d.txt"
textfile = "../text/mobydick.txt"
queryfile = "../text/mobydick_queries.txt"

UNK = "<UNK>"

def loadGlove(glovefile):
  with open(glovefile, "r") as f:
    content = f.readlines()

  model = {}
  model[UNK] = {"id": 0, "emb": [0.0]*100}
  for i, line in enumerate(content):
    splitline = line.split()
    word = splitline[0]
    embedding = [float(val) for val in splitline[1:]]
    model[word] = {"id": i+1, "emb": embedding}

  return model

if textfile:
  with open(textfile, "r") as f:
    s = f.read().replace('\n', '')
else:
  s = "...text goes here..."

s = s.lower()
sents = sent_tokenize(s)
tokens = []
for sent in sents:
  tokens.extend(word_tokenize(sent))

model = loadGlove(glovefile)

ids = []
with open(queryfile, "w") as f:
  for token in tqdm(tokens):
    if token not in model:
      token = UNK
    ids.append(model[token]["id"])
    f.write(str(model[token]["id"]) + '\n')
    #print(model[token]["id"])


## plot histogram
#c = Counter(ids)

#labels, values = zip(*c.items())
#indexes = np.arange(len(labels))
#width = 1

#plt.bar(indexes, values, width)
#plt.xticks(indexes + width * 0.5, labels)
#plt.show()
