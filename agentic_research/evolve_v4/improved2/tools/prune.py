"""Analysis only: hindsight-pruned certificate size of a policy's reads (uses full labels offline)."""
import sys
import numpy as np
from tr import load, run
from agentic_research.evolve_v4.colab_task import load_cases
from agentic_research.evolve_v4.core import possible_starts, certificates
m = load(sys.argv[2]); cs = load_cases()
tot = [0, 0]
for i, c in enumerate(cs):
    qs = [(a, b) for a, b, _, _ in run(c, m)]
    if not qs or c['expected']: continue
    labels, lower, upper = c['labels'], c['lower'], c['upper']
    keep = list(qs)
    for q in reversed(qs):
        trial = [x for x in keep if x != q]
        P = np.full(labels.shape, -1, dtype=np.int8)
        for a, b in trial: P[a, :, b] = labels[a, :, b]
        if certificates(possible_starts(P), lower, upper)[0] == c['expected']: keep = trial
    tot[0] += len(qs); tot[1] += len(keep)
    print(i, c['group'].split('|')[0] + c['group'][-3:], c['epsilon'], len(qs), len(keep))
print('incompat total', tot)
