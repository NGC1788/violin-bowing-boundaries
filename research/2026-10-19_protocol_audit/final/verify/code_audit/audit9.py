"""code_audit: (i) upper 'hit' credited when a D model predicts no Helmholtz in a right-censored column; (ii) pre-registered hit shares from the JSON."""
import sys, json, numpy as np
sys.argv = [sys.argv[0]]
sys.path.insert(0, "/Users/jeongjihoon/violin-rne/final")
import compare as C
from data import load, TRAIN, TEST
T = 0.2
dat = {t: load(t) for t in TRAIN + TEST}
rows = {t: C.intervals(dat[t], T) for t in TEST}
dm = C.data_models([dat[t] for t in TRAIN])
for n, f in dm.items():
    tot = 0
    for t in TEST:
        for r in rows[t]:
            p = C.band_from_prob(f, r, T)
            if not np.isfinite(p[0]) and np.isfinite(r["upper"][0]) and np.isinf(r["upper"][1]):
                tot += 1
    print(n, "columns where 'no Helmholtz predicted' is scored as an upper-boundary hit although H reaches the top force:", tot)
R = json.load(open("/Users/jeongjihoon/violin-rne/final/compare_t0.2.json"))["results"]
for m in ["P-Sch", "P-Beigi", "P-free", "D-logit", "D-gbm"]:
    print(m, "zero-shot hit share lower %.2f upper %.2f" % (np.mean([R[t][m]["zero_shot"]["lower_hit"] for t in TEST]), np.mean([R[t][m]["zero_shot"]["upper_hit"] for t in TEST])))
