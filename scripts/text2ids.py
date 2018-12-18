from nltk.tokenize import word_tokenize
from collections import Counter

s = "...text goes here..."

s = s.lower()
tokens = word_tokenize(s)
c = Counter(tokens)
token2id = dict([(word, id) for id,(word,count) in enumerate(c.most_common())])
text_ids = [token2id[w] for w in tokens]

for text_id in text_ids:
  print(text_id)
