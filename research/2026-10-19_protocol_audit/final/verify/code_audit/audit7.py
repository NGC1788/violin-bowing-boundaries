"""code_audit (diagnostic, post hoc): why k=10 recalibration hurts — share of chosen windows near a predicted boundary, and effect of prior width."""
import sys, numpy as np
sys.argv = [sys.argv[0]]
sys.path.insert(0, "/Users/jeongjihoon/violin-rne/final")
import compare as C
from data import load, TRAIN, TEST
from scipy.special import expit
from scipy.optimize import minimize
T = 0.2
dat = {t: load(t) for t in TRAIN + TEST}
z_of = {t: float(dat[t]["Z"][0]) for t in dat}
rows = {}
for t in dat:
    rows[t] = C.intervals(dat[t], T)
    for r in rows[t]: r["tag"] = t
phys = C.physics_models(rows["A1"] + rows["A2"], z_of)
pf = phys["P-free"][0]; k = 3.5
def recal(d, R, preds, chosen, sd):
    lookup = {i: pr for r, pr in zip(R, preds) for i in r["idx"]}
    F = np.array([np.log(max(d["force"][i], 1e-3)) for i in chosen]); L = np.array([lookup[i][0] for i in chosen]); U = np.array([lookup[i][1] for i in chosen]); Y = d["H"][chosen].astype(float)
    def nll(x):
        p = np.clip(expit(k * np.minimum(F - L - x[0], U + x[1] - F)), 1e-6, 1 - 1e-6)
        return -np.sum(Y * np.log(p) + (1 - Y) * np.log(1 - p)) + 0.5 * (x[0] ** 2 + x[1] ** 2) / sd ** 2
    x = minimize(nll, [0.0, 0.0], method="Nelder-Mead").x
    return [(a + x[0], b + x[1]) for a, b in preds], x
for sd in (1.0, 0.3, 0.15):
    agg = []; near = []
    for t in TEST:
        d = dat[t]; R = rows[t]; preds = [pf(r, t) for r in R]
        all_idx = np.concatenate([r["idx"] for r in R])
        lookup = {i: pr for r, pr in zip(R, preds) for i in r["idx"]}
        rng = np.random.default_rng(123); sc = []
        for rep in range(20):
            ch = rng.choice(all_idx, 10, replace=False)
            near.append(np.mean([min(abs(np.log(max(d["force"][i], 1e-3)) - lookup[i][0]), abs(np.log(max(d["force"][i], 1e-3)) - lookup[i][1])) < 0.3 for i in ch]))
            adj, x = recal(d, R, preds, ch, sd)
            s2 = C.score_boundaries(R, adj); sc.append((s2["lower"][0], s2["upper"][0], x[0], x[1]))
        sc = np.array(sc); agg.append((np.median(sc[:, 0]), np.median(sc[:, 1]), np.std(sc[:, 2]), np.std(sc[:, 3])))
    agg = np.array(agg)
    print(f"P-free k=10 random, prior SD {sd}: lower {agg[:,0].mean():.3f} upper {agg[:,1].mean():.3f} (zero-shot 0.322/0.202); SD of fitted shifts across reps L {agg[:,2].mean():.2f} U {agg[:,3].mean():.2f}; share of chosen windows within 0.3 log of a predicted boundary {np.mean(near):.2f}")
