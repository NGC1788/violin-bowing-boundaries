"""code_audit: oracle shift, few-shot with corrected D slope, plan-specified D-logit recalibration, IoU per speed, hits split."""
import sys, numpy as np
T = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
sys.argv = [sys.argv[0]]
sys.path.insert(0, "/Users/jeongjihoon/violin-rne/final")
import compare as C
from data import load, TRAIN, TEST
from scipy.special import expit
from scipy.optimize import minimize

ALL = TRAIN + TEST
dat = {t: load(t) for t in ALL}
z_of = {t: float(dat[t]["Z"][0]) for t in ALL}
rows = {}
for t in ALL:
    rows[t] = C.intervals(dat[t], T)
    for r in rows[t]:
        r["tag"] = t
phys = C.physics_models(rows["A1"] + rows["A2"], z_of)
dm = C.data_models([dat[t] for t in TRAIN])
preds_all = {n: (lambda r, tag, f=pf: f(r, tag)) for n, (pf, _) in phys.items()}
for n, pf in dm.items():
    preds_all[n] = lambda r, tag, f=pf: C.band_from_prob(f, r, T)
slope_code = {n: C.fit_slope([dat[t] for t in TRAIN], rows, pf) for n, pf in preds_all.items()}

def fit_slope_fixed(pf):
    L, U, F, Y = [], [], [], []
    for tag in TRAIN:
        d = dat[tag]
        for r in rows[tag]:
            lmin, lmax = pf(r, tag)
            for i in r["idx"]:
                L.append(lmin); U.append(lmax); F.append(np.log(max(d["force"][i], 1e-3))); Y.append(d["H"][i])
    L, U, F, Y = map(np.array, (L, U, F, Y))
    m = np.minimum(F - L, U - F); m = np.where(np.isnan(m), -50.0, m)
    ks = np.linspace(0.5, 15, 59)
    v = [-np.mean(Y * np.log(np.clip(expit(k * m), 1e-6, 1)) + (1 - Y) * np.log(np.clip(1 - expit(k * m), 1e-6, 1))) for k in ks]
    return ks[int(np.argmin(v))]
slope_fix = {n: fit_slope_fixed(pf) for n, pf in preds_all.items()}
print("t", T, "slope code", slope_code, "\nslope fixed", slope_fix)

print("== E. oracle constant shift (best achievable by any global recalibration), lower/upper median |log err|")
grid = np.linspace(-1.5, 1.5, 601)
for n, pf in preds_all.items():
    out = []
    for t in TEST:
        preds = [pf(r, t) for r in rows[t]]
        res = []
        for which, j in (("lower", 0), ("upper", 1)):
            pm = [(pr[j], 0.5 * (r[which][0] + r[which][1])) for r, pr in zip(rows[t], preds) if np.isfinite(r[which][0]) and np.isfinite(r[which][1])]
            p = np.array([a for a, b in pm]); mid = np.array([b for a, b in pm])
            fin = np.isfinite(p)
            base = np.median(np.where(fin, np.abs(p - mid), 3.0))
            best = min(grid, key=lambda s: np.median(np.where(fin, np.abs(p + s - mid), 3.0)))
            res.append((base, np.median(np.where(fin, np.abs(p + best - mid), 3.0)), best))
        out.append(res)
    o = np.array([[x[0][0], x[0][1], x[1][0], x[1][1]] for x in out])
    print(f" {n:8s} lower {o[:,0].mean():.3f} -> oracle {o[:,1].mean():.3f} | upper {o[:,2].mean():.3f} -> oracle {o[:,3].mean():.3f} | shifts L {[round(x[0][2],2) for x in out]} U {[round(x[1][2],2) for x in out]}")

print("== F. few-shot (random, 20 reps, k=10/40) with code slope vs fixed slope")
for n, pf in preds_all.items():
    line = []
    for slope_d, lab in ((slope_code, "code"), (slope_fix, "fixed")):
        acc = {10: [], 40: []}
        for t in TEST:
            d = dat[t]; R = rows[t]
            preds = [pf(r, t) for r in R]
            all_idx = np.concatenate([r["idx"] for r in R])
            for kshot in (10, 40):
                rng = np.random.default_rng(123)
                sc = []
                for rep in range(20):
                    ch = rng.choice(all_idx, kshot, replace=False)
                    adj = C.recalibrate(d, R, preds, slope_d[n], ch)
                    s2 = C.score_boundaries(R, adj)
                    sc.append((s2["lower"][0], s2["upper"][0]))
                acc[kshot].append(np.nanmedian(np.array(sc), axis=0))
        line.append(f"{lab}(k={slope_d[n]:.2f}): k10 L {np.mean([a[0] for a in acc[10]]):.3f} U {np.mean([a[1] for a in acc[10]]):.3f}; k40 L {np.mean([a[0] for a in acc[40]]):.3f} U {np.mean([a[1] for a in acc[40]]):.3f}")
    print(f" {n:8s} " + " || ".join(line))

print("== G. IoU pooled vs per-speed mean (plan says per speed)")
for n, pf in preds_all.items():
    pooled, per = [], []
    for t in TEST:
        d = dat[t]; R = rows[t]
        preds = [pf(r, t) for r in R]
        y, pr, sp = [], [], []
        for r, p in zip(R, preds):
            for i in r["idx"]:
                lf = np.log(max(d["force"][i], 1e-3))
                if n in dm:
                    q = float(dm[n](np.array([lf]), np.array([r["lv"]]), np.array([r["lb"]]), np.array([T]))[0]) >= 0.5
                else:
                    q = p[0] <= lf <= p[1]
                y.append(d["H"][i]); pr.append(q); sp.append(round(r["lv"], 3))
        y, pr, sp = map(np.array, (y, pr, sp))
        pooled.append(np.sum(y & pr) / np.sum(y | pr))
        per.append(np.mean([np.sum(y[sp == s] & pr[sp == s]) / np.sum(y[sp == s] | pr[sp == s]) for s in np.unique(sp)]))
    print(f" {n:8s} pooled {np.mean(pooled):.3f} per-speed mean {np.mean(per):.3f}")

print("== H. hit rate split: finite vs censored columns (zero-shot), mean over tests")
for n, pf in preds_all.items():
    fl, cl, fu, cu = [], [], [], []
    for t in TEST:
        R = rows[t]; preds = [pf(r, t) for r in R]
        for which, j, F_, Cn in (("lower", 0, fl, cl), ("upper", 1, fu, cu)):
            for r, pr in zip(R, preds):
                lo, hi = r[which]
                if np.isnan(lo) and np.isnan(hi): continue
                p = pr[j]
                ok = ((p >= lo - C.STEP) if np.isfinite(lo) else True) and ((p <= hi + C.STEP) if np.isfinite(hi) else True)
                ok = bool(ok) if np.isfinite(p) else (not np.isfinite(hi))
                (F_ if (np.isfinite(lo) and np.isfinite(hi)) else Cn).append(ok)
    print(f" {n:8s} lower hit finite {np.mean(fl):.2f} (n {len(fl)}) censored {np.mean(cl):.2f} (n {len(cl)}) | upper finite {np.mean(fu):.2f} (n {len(fu)}) censored {np.mean(cu):.2f} (n {len(cu)})")
