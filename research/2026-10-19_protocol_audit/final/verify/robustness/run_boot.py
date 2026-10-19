"""Paired bootstrap over beta columns for model differences (reviewer: robustness). Uses main_store.pkl from run_main.py.
Statistic = mean over test specimens of the per-specimen median |log err| (the report's statistic).
Resampling unit = beta index (all 3 speeds of a beta index go together), independently within each specimen; the same
resampled indices are used for every model (paired). Also a two-level version that first resamples strings
(B_T2 rep_1+rep_2 treated as one string cluster)."""
import pickle
from pathlib import Path

import numpy as np

st = pickle.load(open(Path(__file__).parent / "main_store.pkl", "rb"))
rng = np.random.default_rng(12345)
NB = 4000
PAIRS = [("D-gbm", "P-free"), ("D-gbm", "P-Beigi"), ("P-free", "P-Beigi"), ("D-logit", "P-free"), ("P-Sch", "P-free"),
         ("P-Sch", "P-Beigi"), ("P-Sch", "D-gbm"), ("Triv-5", "P-free"), ("Triv-5", "D-gbm"), ("Triv-nn", "D-gbm"), ("Triv-5", "Triv-nn")]


def stat(res, tags, idx_by_tag, m, key):
    vals = []
    for g, ii in zip(tags, idx_by_tag):
        e = res[g][m][key]; col = res[g][m]["col"]
        # map each beta index to the column rows having it
        sel = np.concatenate([np.flatnonzero(col == c) for c in ii])
        vals.append(np.nanmedian(e[sel]))
    return np.mean(vals)


for (K, t) in ((2, 0.2), (2, 0.0), (2, 0.3), (3, 0.2)):
    res = st[(K, t, "main")]
    tags = list(res)
    print(f"\n=== K={K} t={t}  (B={NB}; columns resampled by beta index within specimen)")
    for key, lab in (("eL", "lower"), ("eU", "upper")):
        models = list(res[tags[0]])
        point = {m: stat(res, tags, [np.arange(40)] * len(tags), m, key) for m in models}
        boots = {m: [] for m in models}
        boots2 = {m: [] for m in models}
        clusters = [["B1"], ["C1"], ["D1"], ["B_T2r1", "B_T2r2"]]
        for b in range(NB):
            ii = [rng.integers(0, 40, 40) for _ in tags]
            for m in models:
                boots[m].append(stat(res, tags, ii, m, key))
            # two-level: resample string clusters, then beta columns
            cl = [clusters[j] for j in rng.integers(0, 4, 4)]
            tg2 = sum(cl, [])
            ii2 = [rng.integers(0, 40, 40) for _ in tg2]
            for m in models:
                boots2[m].append(stat(res, tg2, ii2, m, key))
        boots = {m: np.array(v) for m, v in boots.items()}
        boots2 = {m: np.array(v) for m, v in boots2.items()}
        print(f"  [{lab}] point: " + ", ".join(f"{m} {point[m]:.3f} [{np.percentile(boots[m], 2.5):.3f},{np.percentile(boots[m], 97.5):.3f}]" for m in models))
        for a, b2 in PAIRS:
            if a not in boots or b2 not in boots:
                continue
            d = boots[a] - boots[b2]; d2 = boots2[a] - boots2[b2]
            print(f"   {a:8s} - {b2:8s}: {point[a] - point[b2]:+.3f}  CI95 cols [{np.percentile(d, 2.5):+.3f},{np.percentile(d, 97.5):+.3f}] P(<0) {np.mean(d < 0):.3f}"
                  f" | two-level CI [{np.percentile(d2, 2.5):+.3f},{np.percentile(d2, 97.5):+.3f}] P(<0) {np.mean(d2 < 0):.3f}", flush=True)
