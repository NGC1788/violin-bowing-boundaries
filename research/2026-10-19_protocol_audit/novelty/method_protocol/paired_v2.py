"""Paired per-diagram comparisons (post hoc) on all-column metrics for the round-2 candidates."""
import json
import numpy as np
R = json.load(open("validation_v2_results.json"))["results"]
TAGS = ["A1", "A2", "B1", "C1", "D1", "A_T2", "B_T2s1", "B_T2r1", "B_T2r2", "C_T2", "D_T2"]
CAND = "protocol|m1|clip1|quad_s"
BASES = ["lookup|clip1", "lookup", "mean|m1|clip1|quad_s", "props|m1|clip1|quad_s", "full|m1|clip1|quad_s",
         "mean|m1|clip0", "props|m1|clip0", "protocol|m1|clip0"]
out = {}
for split, res in R.items():
    out[split] = {}
    for b in BASES:
        rec = {}
        for key, sign in (("dist", 1), ("logs", -1), ("col_median", 1)):
            d = np.array([sign * (res[CAND]["per_diagram"][t][key] - res[b]["per_diagram"][t][key]) for t in TAGS])
            rng = np.random.default_rng(20261019)
            bs = [d[rng.integers(0, 11, 11)].mean() for _ in range(4000)]
            rec[key] = dict(mean=float(d.mean()), cand_better=int((d < 0).sum()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
        out[split][b] = rec
        print(split, f"{CAND} vs {b:24s}", "  ".join(f"{k}: {v['mean']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] better {v['cand_better']}/11" for k, v in rec.items()))
json.dump(dict(candidate=CAND, note="negative = candidate better (log score sign flipped); diagram bootstrap, 11 diagrams, exploratory", results=out),
          open("paired_v2.json", "w"), indent=1)
