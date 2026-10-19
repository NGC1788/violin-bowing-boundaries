"""Extra checks (reviewer: robustness): low-force share of the lower boundary, errors restricted to boundaries >= 0.2 N,
within-specimen floors (self-fit P-free; leave-one-out neighbour smoother), repeat per speed, LOTO bootstrap."""
import pickle
from pathlib import Path

import numpy as np

import rb_core as R

OUT = Path(__file__).parent
st = pickle.load(open(OUT / "main_store.pkl", "rb"))
lo_st = pickle.load(open(OUT / "loto_store.pkl", "rb"))
dat = R.get_data()
z_of = {t: float(dat[t]["Z"][0]) for t in R.ALL}
TEST = ("B1", "C1", "D1", "B_T2r1", "B_T2r2")
t = 0.2
rows = R.make_rows(dat, t, 2)

print("(a) observed finite lower-boundary midpoints: median force [N] and share < 0.2 N, per speed")
for g in TEST + ("A1", "A2"):
    line = []
    for sp in (0.05, 0.1, 0.2):
        mids = np.array([0.5 * sum(r["lower"]) for r in rows[g] if r["speed"] == sp and np.all(np.isfinite(r["lower"]))])
        nleft = sum(1 for r in rows[g] if r["speed"] == sp and not np.isfinite(r["lower"][0]))
        line.append(f"v{sp}: med {np.exp(np.median(mids)):.3f} N, <0.2N {np.mean(mids < np.log(0.2)):.2f} (n {len(mids)}, left-cens {nleft})")
    print(f"  {g:7s} " + " | ".join(line))

res = st[(2, t, "main")]
print("\n(a2) lower |log err| restricted to columns whose observed midpoint >= 0.2 N (mean over 5 tests) vs all")
for m in res["B1"]:
    a, b = [], []
    for g in TEST:
        mids = np.array([0.5 * sum(r["lower"]) if np.all(np.isfinite(r["lower"])) else np.nan for r in rows[g]])
        e = res[g][m]["eL"]
        a.append(np.nanmedian(e)); b.append(np.nanmedian(np.where(mids >= np.log(0.2), e, np.nan)))
    print(f"  {m:8s} all {np.mean(a):.3f} | >=0.2N {np.mean(b):.3f}  [{' '.join(f'{x:.2f}' for x in b)}]")

print("\n(b) within-specimen floors at t=0.2, K=2")
for g in TEST + ("A1", "A2"):
    # self-fit P-free (form floor)
    phys = R.C.physics_models(rows[g], z_of)
    pr = [phys["P-free"][0](r, g) for r in rows[g]]
    e = R.per_column_errors(rows[g], pr)
    # leave-one-out neighbour smoother: median of finite-edge boundaries of +-2 neighbouring beta columns, same speed
    loo = []
    for r in rows[g]:
        nb = [x for x in rows[g] if x["speed"] == r["speed"] and 0 < abs(x["col"] - r["col"]) <= 2]
        L = [R.mid_or_edge(x["lower"]) for x in nb]; U = [R.mid_or_edge(x["upper"]) for x in nb]
        L = [v for v in L if np.isfinite(v)]; U = [v for v in U if np.isfinite(v)]
        loo.append((np.median(L) if L else np.inf, np.median(U) if U else np.nan))
    e2 = R.per_column_errors(rows[g], loo)
    print(f"  {g:7s} self-fit P-free lower {np.nanmedian(e['eL']):.3f} upper {np.nanmedian(e['eU']):.3f} | LOO neighbour smoother lower {np.nanmedian(e2['eL']):.3f} upper {np.nanmedian(e2['eU']):.3f}")

print("\n(c) repeat baseline per speed (Triv-5 and P-free, lower), t=0.2 K=2")
for key in ("B_T2r1->B_T2r2", "B_T2r2->B_T2r1"):
    r2 = st[(2, t, key)]
    g = list(r2)[0]
    for m in ("Triv-5", "P-free", "P-Sch", "D-gbm"):
        sp = r2[g][m]["speed"]
        print(f"  {key} {m:7s} " + " | ".join(f"v{s}: {np.nanmedian(r2[g][m]['eL'][sp == s]):.3f}" for s in (0.05, 0.1, 0.2))
              + f" || upper " + " | ".join(f"v{s}: {np.nanmedian(r2[g][m]['eU'][sp == s]):.3f}" for s in (0.05, 0.1, 0.2)))

print("\n(c2) cross-string error on B_T2 tests per speed (train A1+A2), for comparison")
for g in ("B_T2r1", "B_T2r2"):
    for m in ("Triv-5", "P-free", "P-Sch", "D-gbm"):
        sp = res[g][m]["speed"]
        print(f"  {g} {m:7s} " + " | ".join(f"v{s}: {np.nanmedian(res[g][m]['eL'][sp == s]):.3f}" for s in (0.05, 0.1, 0.2))
              + f" || upper " + " | ".join(f"v{s}: {np.nanmedian(res[g][m]['eU'][sp == s]):.3f}" for s in (0.05, 0.1, 0.2)))

print("\n(d) LOTO paired bootstrap over beta columns (5 report test specimens, each from its LOTO split), B=4000")
rng = np.random.default_rng(7)
for tt in (0.2, 0.0, 0.3):
    rl = {g: lo_st[tt][g] for g in TEST}
    for key, lab in (("eL", "lower"), ("eU", "upper")):
        boots = {m: [] for m in rl["B1"]}
        for b in range(4000):
            ii = [rng.integers(0, 40, 40) for _ in TEST]
            for m in boots:
                v = []
                for g, idx in zip(TEST, ii):
                    col = rl[g][m]["col"]
                    sel = np.concatenate([np.flatnonzero(col == c) for c in idx])
                    v.append(np.nanmedian(rl[g][m][key][sel]))
                boots[m].append(np.mean(v))
        boots = {m: np.array(v) for m, v in boots.items()}
        for a, c in (("D-gbm", "P-free"), ("Triv-5", "P-free"), ("Triv-nn", "D-gbm"), ("P-Sch", "P-Beigi"), ("P-Sch", "P-free")):
            d = boots[a] - boots[c]
            print(f"  t={tt} [{lab}] {a}-{c}: median {np.median(d):+.3f} CI95 [{np.percentile(d, 2.5):+.3f},{np.percentile(d, 97.5):+.3f}] P(<0) {np.mean(d < 0):.3f}", flush=True)
