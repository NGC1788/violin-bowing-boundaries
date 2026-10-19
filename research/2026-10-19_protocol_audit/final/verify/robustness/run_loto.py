"""Leave-one-type-out with all non-test specimens as training (reviewer: robustness)."""
import pickle
import sys
from pathlib import Path

import numpy as np

import rb_core as R

OUT = Path(__file__).parent
dat = R.get_data()
z_of = {t: float(dat[t]["Z"][0]) for t in R.ALL}
SPLITS = {"B": (("B1", "B_T2r1", "B_T2r2"), ("A1", "A2", "C1", "D1")),
          "C": (("C1",), ("A1", "A2", "B1", "D1", "B_T2r1", "B_T2r2")),
          "D": (("D1",), ("A1", "A2", "B1", "C1", "B_T2r1", "B_T2r2")),
          "A": (("A1", "A2"), ("B1", "C1", "D1", "B_T2r1", "B_T2r2"))}
store = {}
for t in (0.2, 0.0, 0.3):
    rows = R.make_rows(dat, t, 2)
    merged = {}
    for typ, (test, train) in SPLITS.items():
        extra = {"Triv-nn": R.trivial_predictor(rows, train, 0), "Triv-5": R.trivial_predictor(rows, train, 2)}
        res, phys = R.evaluate(dat, rows, train, test, t, z_of, extra)
        R.print_summary(f"LOTO t={t} hold out type {typ}: train {'+'.join(train)}", R.summary(res))
        print("   P-free lower params (c, a_v, b_beta, log sigma):", np.round(phys["P-free"][1][0], 3), " upper:", np.round(phys["P-free"][1][1], 3))
        merged.update(res)
    store[t] = merged
    five = {g: merged[g] for g in ("B1", "C1", "D1", "B_T2r1", "B_T2r2")}
    R.print_summary(f"LOTO t={t} — same 5 test specimens as the report (each from its own LOTO split)", R.summary(five))
with open(OUT / "loto_store.pkl", "wb") as fh:
    pickle.dump(store, fh)
