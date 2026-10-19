"""Reproduce the primary table, add trivial baselines, per-speed breakdown, K=3, repeat baselines (reviewer: robustness)."""
import json
import pickle
from pathlib import Path

import numpy as np

import rb_core as R

OUT = Path(__file__).parent
dat = R.get_data()
z_of = {t: float(dat[t]["Z"][0]) for t in R.ALL}
print("speeds per specimen:", {t: sorted(set(np.round(dat[t]["speed"], 3))) for t in R.ALL})
TRAIN = ("A1", "A2")
TEST = ("B1", "C1", "D1", "B_T2r1", "B_T2r2")
store = {}
for K in (2, 3):
    for t in (0.0, 0.2, 0.3):
        rows = R.make_rows(dat, t, K)
        extra = {"Triv-nn": R.trivial_predictor(rows, TRAIN, 0), "Triv-5": R.trivial_predictor(rows, TRAIN, 2)}
        res, phys = R.evaluate(dat, rows, TRAIN, TEST, t, z_of, extra)
        s = R.summary(res)
        R.print_summary(f"K={K} t={t} train A1+A2", s)
        store[(K, t, "main")] = res
        if K == 2:
            # per-speed breakdown
            print("  per-speed lower median |err| (mean over 5 tests) and share of left-censored/finite columns")
            for m in res["B1"]:
                line = []
                for sp in (0.05, 0.1, 0.2):
                    vals = [np.nanmedian(res[g][m]["eL"][res[g][m]["speed"] == sp]) for g in TEST]
                    line.append(f"v{sp}: {np.mean(vals):.3f} [{' '.join(f'{x:.2f}' for x in vals)}]")
                print(f"   {m:10s} " + " | ".join(line))
            for m in ("D-gbm",):
                line = []
                for sp in (0.05, 0.1, 0.2):
                    vals = [np.nanmedian(res[g][m]["eU"][res[g][m]["speed"] == sp]) for g in TEST]
                    cens = [np.mean(res[g][m]["cU"][res[g][m]["speed"] == sp] == "right") for g in TEST]
                    nfin = [np.sum(res[g][m]["cU"][res[g][m]["speed"] == sp] == "finite") for g in TEST]
                    line.append(f"v{sp}: U {np.mean(vals):.3f} right-cens {np.mean(cens):.2f} nfin {nfin}")
                print(f"   upper {m}: " + " | ".join(line))
            cl = {g: {sp: dict(zip(*np.unique(res[g]["D-gbm"]["cL"][res[g]["D-gbm"]["speed"] == sp], return_counts=True))) for sp in (0.05, 0.1, 0.2)} for g in TEST}
            print("  lower censoring counts per speed:", {g: {sp: {k: int(v) for k, v in d.items()} for sp, d in x.items()} for g, x in cl.items()})
        # repeat baselines (same string re-measured)
        for a, b in (("B_T2r1", "B_T2r2"), ("B_T2r2", "B_T2r1"), ("A1", "A2"), ("A2", "A1")):
            extra = {"Triv-nn": R.trivial_predictor(rows, (a,), 0), "Triv-5": R.trivial_predictor(rows, (a,), 2)}
            r2, _ = R.evaluate(dat, rows, (a,), (b,), t, z_of, extra)
            R.print_summary(f"K={K} t={t} train {a} -> test {b}", R.summary(r2))
            store[(K, t, f"{a}->{b}")] = r2
with open(OUT / "main_store.pkl", "wb") as fh:
    pickle.dump(store, fh)
