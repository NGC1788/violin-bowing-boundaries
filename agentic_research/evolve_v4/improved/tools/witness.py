"""For a pruned read set, classify how each rule is refuted."""
import sys
import numpy as np
from trace import *

def classify(case, reads):
    labels, lower, upper = case["labels"], case["lower"], case["upper"]
    partial = np.full(labels.shape, -1, dtype=np.int8)
    for c, r in reads:
        partial[c, :, r] = labels[c, :, r]
    poss = possible_starts(partial)
    valid = np.isfinite(lower)
    cand = poss & valid
    C, R, N = labels.shape
    out = []
    for r in range(R):
        has = cand[:, r].any(-1)
        if not has.all():
            out.append(f"r{r}:dead{list(np.flatnonzero(~has))}")
            continue
        lo = np.where(cand[:, r], lower[:, 0], np.inf).min(-1)
        hi = np.where(cand[:, r], upper[:, 0], -np.inf).max(-1)
        a, b = int(lo.argmax()), int(hi.argmin())
        p = int(np.flatnonzero(cand[a, r])[0]); q = int(np.flatnonzero(cand[b, r])[-1])
        # how is b's q bound achieved: forced pair at q or later zeros
        forced = (partial[b, r, :-1] == 1) & (partial[b, r, 1:] == 1)
        how = 'pair' if forced.any() else 'zeros'
        out.append(f"r{r}:A{a}@{p} B{b}@{q}{how}")
    return out

if __name__ == "__main__":
    m = load(sys.argv[1])
    cases = load_cases()
    for i in [int(x) for x in sys.argv[2].split(',')]:
        case = cases[i]
        qs = run(case, m.choose); pr = prune(case, qs)
        print(i, case['group'].split('|')[0], case['group'][-3:], case['epsilon'], len(qs), len(pr), 'C', case['labels'].shape[0])
        print('  reads', sorted(pr))
        print('  ', ' | '.join(classify(case, pr)))
        lb = case['public']['log_beta']
        print('   beta order (low->high):', list(np.argsort(lb)))
