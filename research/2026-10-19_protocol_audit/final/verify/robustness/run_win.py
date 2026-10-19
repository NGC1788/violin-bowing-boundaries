"""Window-length sensitivity: 0.1 s and 0.4 s windows (labels only; Z from 0.2 s). Train A1+A2, test 5 (reviewer: robustness)."""
import pickle
from pathlib import Path

import numpy as np

import rb_core as R

OUT = Path(__file__).parent
TRAIN = ("A1", "A2")
TEST = ("B1", "C1", "D1", "B_T2r1", "B_T2r2")
store = {}
for w in ("0.1", "0.4"):
    dat = {t: dict(np.load(OUT / "cache_w" / f"{t}_w{w}.npz")) for t in R.ALL}
    z_of = {t: float(dat[t]["Z"][0]) for t in R.ALL}
    for t in (0.2, 0.0):
        rows = R.make_rows(dat, t, 2)
        extra = {"Triv-nn": R.trivial_predictor(rows, TRAIN, 0), "Triv-5": R.trivial_predictor(rows, TRAIN, 2)}
        res, phys = R.evaluate(dat, rows, TRAIN, TEST, t, z_of, extra)
        R.print_summary(f"window {w} s, t={t}, K=2, train A1+A2", R.summary(res))
        store[(w, t, "main")] = res
        for a, b in (("B_T2r1", "B_T2r2"), ("B_T2r2", "B_T2r1")):
            extra = {"Triv-5": R.trivial_predictor(rows, (a,), 2)}
            r2, _ = R.evaluate(dat, rows, (a,), (b,), t, z_of, extra)
            R.print_summary(f"window {w} s, t={t}: train {a} -> {b}", R.summary(r2))
with open(OUT / "win_store.pkl", "wb") as fh:
    pickle.dump(store, fh)
