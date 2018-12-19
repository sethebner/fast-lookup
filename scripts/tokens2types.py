# Removes redundant token ids with a list of positions at which they occur

from collections import defaultdict

queryfile = "../text/mobydick_queries.txt"
outfile = "../text/mobydick_queries_unique.txt"

with open(queryfile, "r") as f:
  queries = f.readlines()

queries = [q.strip() for q in queries]
#unique_queries = set(queries)
positions = defaultdict(list)
for i,q in enumerate(queries):
  positions[q].append(i)

with open(outfile, "w") as f:
  for uq in positions:
    s = " ".join([str(uq)] + list(map(str,positions[uq])))
    f.write(s + "\n")
