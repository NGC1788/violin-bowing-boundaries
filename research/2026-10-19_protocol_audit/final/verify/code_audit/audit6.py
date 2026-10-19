"""code_audit: re-run compare.py's few-shot block exactly (same RNG order) with the D-model slope fixed; t_eval from argv."""
import sys, numpy as np
T = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
sys.argv = [sys.argv[0], str(T)]
sys.path.insert(0, "/Users/jeongjihoon/violin-rne/final")
import compare as C
assert C.T_EVAL == T
from data import load, TRAIN, TEST
from scipy.special import expit
train = [load(t) for t in TRAIN]; test = {t: load(t) for t in TEST}
z_of = {t: float(load(t)["Z"][0]) for t in TRAIN + TEST}
rows_by_tag = {}
for tag, d in list(zip(TRAIN, train)) + list(test.items()):
    rr = C.intervals(d, T)
    for r in rr: r["tag"] = tag
    rows_by_tag[tag] = rr
phys = C.physics_models(rows_by_tag["A1"] + rows_by_tag["A2"], z_of)
dmod = C.data_models(train)
preds_all = {n: (lambda r, tag, f=pf: f(r, tag)) for n, (pf, _) in phys.items()}
for n, pf in dmod.items():
    preds_all[n] = lambda r, tag, f=pf: C.band_from_prob(f, r, T)
def fit_slope_fixed(pred_fun):
    L, U, F, Y = [], [], [], []
    for d, tag in zip(train, TRAIN):
        for r in rows_by_tag[tag]:
            lmin, lmax = pred_fun(r, tag)
            for i in r["idx"]:
                L.append(lmin); U.append(lmax); F.append(np.log(max(d["force"][i], 1e-3))); Y.append(d["H"][i])
    L, U, F, Y = map(np.array, (L, U, F, Y))
    m = np.minimum(F - L, U - F); m = np.where(np.isnan(m), -50.0, m)   # no predicted band -> P(H) -> 0
    best = min(((k, -np.mean(Y * np.log(np.clip(expit(k * m), 1e-6, 1)) + (1 - Y) * np.log(np.clip(1 - expit(k * m), 1e-6, 1)))) for k in np.linspace(0.5, 15, 59)), key=lambda x: x[1])
    return best[0]
slope = {n: fit_slope_fixed(pf) for n, pf in preds_all.items()}
print("fixed slopes", {k: float(v) for k, v in slope.items()})
rng = np.random.default_rng(0)
out = {}
for tag, d in test.items():
    rows = rows_by_tag[tag]; all_idx = np.concatenate([r["idx"] for r in rows])
    for name, pf in preds_all.items():
        preds = [pf(r, tag) for r in rows]
        sb = C.score_boundaries(rows, preds)
        res = {"zero": (sb["lower"][0], sb["upper"][0])}
        for kshot in (5, 10, 20, 40):
            for mode in ("random", "grid"):
                scores = []
                for rep in range(20 if mode == "random" else 1):
                    if mode == "random":
                        chosen = rng.choice(all_idx, kshot, replace=False)
                    else:
                        order = np.argsort(d["column"][all_idx] * 1000 + d["folder"][all_idx] * 100000 + d["force"][all_idx])
                        chosen = all_idx[order[np.linspace(0, len(order) - 1, kshot).astype(int)]]
                    adj = C.recalibrate(d, rows, preds, slope[name], chosen)
                    s2 = C.score_boundaries(rows, adj)
                    scores.append((s2["lower"][0], s2["upper"][0]))
                res[f"k{kshot}_{mode}"] = tuple(np.nanmedian(np.array(scores, float), axis=0))
        out[(tag, name)] = res
for name in preds_all:
    keys = ["zero", "k10_random", "k40_random", "k10_grid", "k40_grid"]
    print(f"{name:8s} " + " | ".join(f"{k}: L {np.mean([out[(t, name)][k][0] for t in TEST]):.3f} U {np.mean([out[(t, name)][k][1] for t in TEST]):.3f}" for k in keys))
