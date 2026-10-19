"""(e) window-length bootstrap of D-gbm - P-free; (f) anatomy of the GBM 'catastrophic failure' (reviewer: robustness)."""
import pickle
from pathlib import Path

import numpy as np

import rb_core as R

OUT = Path(__file__).parent
ws = pickle.load(open(OUT / "win_store.pkl", "rb"))
st = pickle.load(open(OUT / "main_store.pkl", "rb"))
TEST = ("B1", "C1", "D1", "B_T2r1", "B_T2r2")
rng = np.random.default_rng(3)


def boot(res, a, c, key, nb=4000):
    out = []
    for _ in range(nb):
        v = []
        for g in TEST:
            idx = rng.integers(0, 40, 40)
            col = res[g][a]["col"]
            sel = np.concatenate([np.flatnonzero(col == k) for k in idx])
            v.append(np.nanmedian(res[g][a][key][sel]) - np.nanmedian(res[g][c][key][sel]))
        out.append(np.mean(v))
    return np.array(out)


print("(e) window-length: paired bootstrap over beta columns, train A1+A2")
for (w, t, _), res in ws.items():
    for key, lab in (("eL", "lower"), ("eU", "upper")):
        for a, c in (("D-gbm", "P-free"), ("Triv-5", "D-gbm"), ("P-Sch", "P-free")):
            d = boot(res, a, c, key, 2000)
            print(f"  w={w} t={t} [{lab}] {a}-{c}: {np.mean(d):+.3f} CI95 [{np.percentile(d, 2.5):+.3f},{np.percentile(d, 97.5):+.3f}] P(<0) {np.mean(d < 0):.3f}", flush=True)

print("\n(f) GBM failure anatomy: train B_T2r2 -> test B_T2r1, t=0.2, K=2")
dat = R.get_data()
z_of = {t: float(dat[t]["Z"][0]) for t in R.ALL}
rows = R.make_rows(dat, 0.2, 2)
dm = R.C.data_models([dat["B_T2r2"]])
grid0 = np.log(0.03)
pred = [R.C.band_from_prob(dm["D-gbm"], r, 0.2) for r in rows["B_T2r1"]]
pl = np.array([p[0] for p in pred])
sp = np.array([r["speed"] for r in rows["B_T2r1"]])
fin = np.array([np.all(np.isfinite(r["lower"])) for r in rows["B_T2r1"]])
for s in (0.05, 0.1, 0.2):
    m = (sp == s) & fin
    print(f"  v{s}: finite cols {m.sum()}, predicted lower at the grid floor 0.03 N: {np.sum(np.isclose(pl[m], grid0))}, below the lowest measured force: {np.sum(pl[m] < np.array([r['fmin'] for r in rows['B_T2r1']])[m])}")
# same predictions with the grid clipped to the measured force range of each column
err_raw, err_clip = [], []
for r, p in zip(rows["B_T2r1"], pred):
    if not np.all(np.isfinite(r["lower"])):
        continue
    mid = 0.5 * sum(r["lower"])
    q = p[0] if np.isfinite(p[0]) else np.nan
    err_raw.append(abs(q - mid) if np.isfinite(q) else 3.0)
    qc = min(max(q, r["fmin"]), r["fmax"]) if np.isfinite(q) else r["fmax"]
    err_clip.append(abs(qc - mid))
print(f"  median |err| raw {np.median(err_raw):.3f}; with prediction clipped to the column's measured force range {np.median(err_clip):.3f}")
# training-side cause: share of B_T2r2 windows labelled H at the lowest force level per speed
d = dat["B_T2r2"]
for s in (0.05, 0.1, 0.2):
    m = np.isclose(d["speed"], s) & (d["force"] < 0.15)
    print(f"  B_T2r2 training windows with F<0.15 N at v{s}: n {m.sum()}, H share {d['H'][m].mean():.2f}")
d = dat["B_T2r1"]
for s in (0.05, 0.1, 0.2):
    m = np.isclose(d["speed"], s) & (d["force"] < 0.15)
    print(f"  B_T2r1 test windows with F<0.15 N at v{s}: n {m.sum()}, H share {d['H'][m].mean():.2f}")
