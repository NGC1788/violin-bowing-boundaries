"""code_audit: checks on intervals, slope fitting, fit convergence, penalties (reads final/ code, writes nothing there)."""
import sys, numpy as np
sys.argv = [sys.argv[0]]  # compare reads sys.argv[1]
sys.path.insert(0, "/Users/jeongjihoon/violin-rne/final")
import compare as C
from data import load, TRAIN, TEST
from scipy.special import expit
from scipy.optimize import minimize

T = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
ALL = TRAIN + TEST
dat = {t: load(t) for t in ALL}
z_of = {t: float(dat[t]["Z"][0]) for t in ALL}
rows = {}
for t in ALL:
    rows[t] = C.intervals(dat[t], T)
    for r in rows[t]:
        r["tag"] = t

print("== A. interval census at t=%.1f" % T)
for t in ALL:
    R = rows[t]
    lo = np.array([r["lower"][0] for r in R]); hi = np.array([r["lower"][1] for r in R])
    ulo = np.array([r["upper"][0] for r in R]); uhi = np.array([r["upper"][1] for r in R])
    fin = np.isfinite(lo) & np.isfinite(hi)
    norun = np.isnan(ulo)
    clipped = np.sum(fin & (lo < np.log(0.0011)))
    w = hi[fin] - lo[fin]
    ufin = np.isfinite(ulo) & np.isfinite(uhi)
    print(f" {t:7s} cols {len(R)} lower: finite {fin.sum()} leftcens {np.sum(~np.isfinite(lo))} norun(rightcens) {norun.sum()} | lo clipped at 1e-3: {clipped} | width median {np.median(w):.3f} max {w.max():.2f}"
          f" | upper: finite {ufin.sum()} rightcens {np.sum(np.isinf(uhi))}")

train_rows = rows["A1"] + rows["A2"]
phys = C.physics_models(train_rows, z_of)
dm = C.data_models([dat[t] for t in TRAIN])

print("== B. fit_slope on D models: NaN check")
for name, pf in dm.items():
    L, U, F, Y = [], [], [], []
    ninf = 0
    for tag in TRAIN:
        d = dat[tag]
        for r in rows[tag]:
            lmin, lmax = C.band_from_prob(pf, r, T)
            ninf += not np.isfinite(lmin)
            for i in r["idx"]:
                L.append(lmin); U.append(lmax); F.append(np.log(max(d["force"][i], 1e-3))); Y.append(d["H"][i])
    L, U, F, Y = map(np.array, (L, U, F, Y))
    m = np.minimum(F - L, U - F)
    print(f" {name}: training columns with no P>=0.5: {ninf}; windows with NaN margin: {np.isnan(m).sum()} of {len(m)}")
    ok = ~np.isnan(m)
    ks = np.linspace(0.5, 15, 59)
    def loss(k, mm, yy):
        return -np.mean(yy * np.log(np.clip(expit(k * mm), 1e-6, 1)) + (1 - yy) * np.log(np.clip(1 - expit(k * mm), 1e-6, 1)))
    vals = [loss(k, m, Y) for k in ks]
    print(f"   loss at k=0.5 with NaN included: {vals[0]}  -> min() returns k={min(zip(ks, vals), key=lambda x: x[1])[0]}")
    # proper: drop NaN-margin windows (or treat no-band as m=-inf -> p=0)
    m2 = np.where(np.isnan(m), -50.0, m)
    vals2 = [loss(k, m2, Y) for k in ks]
    vals3 = [loss(k, m[ok], Y[ok]) for k in ks]
    print(f"   corrected slope (no-band windows as p->0): k={ks[int(np.argmin(vals2))]:.2f}; dropping them: k={ks[int(np.argmin(vals3))]:.2f}")

print("== C. censored_fit convergence (Nelder-Mead from 0 vs multistart Powell/BFGS)")
from scipy.stats import norm
def nll_factory(rws, design, which):
    X = np.array([design(r)[0] for r in rws]); off = np.array([design(r)[1] for r in rws])
    lo = np.array([r[which][0] for r in rws]); hi = np.array([r[which][1] for r in rws])
    ok = np.isfinite(lo) | np.isfinite(hi)
    X, off, lo, hi = X[ok], off[ok], lo[ok], hi[ok]
    def nll(p):
        mu = X @ p[:-1] + off; s = np.exp(p[-1])
        pr = np.where(np.isfinite(hi), norm.cdf((hi - mu) / s), 1.0) - np.where(np.isfinite(lo), norm.cdf((lo - mu) / s), 0.0)
        return -np.sum(np.log(np.clip(pr, 1e-300, None)))
    return nll, X.shape[1]
designs = {
 "P-Sch lower": (lambda r: (np.array([1.0]), 2*np.log(z_of[r["tag"]]) + r["lv"] - 2*r["lb"]), "lower"),
 "P-Beigi lower": (lambda r: (np.array([1.0]), np.log(z_of[r["tag"]]) + r["lv"] - r["lb"]), "lower"),
 "Sch upper": (lambda r: (np.array([1.0]), np.log(z_of[r["tag"]]) + r["lv"] - r["lb"]), "upper"),
 "P-free lower": (lambda r: (np.array([1.0, r["lv"], r["lb"]]), 2*np.log(z_of[r["tag"]])), "lower"),
 "P-free upper": (lambda r: (np.array([1.0, r["lv"], r["lb"]]), np.log(z_of[r["tag"]])), "upper"),
}
for nm, (des, which) in designs.items():
    nll, p = nll_factory(train_rows, des, which)
    x0 = C.censored_fit(train_rows, des, which)
    best = None
    for start in range(6):
        rng = np.random.default_rng(start)
        s0 = np.r_[rng.normal(0, 2, p), np.log(0.5)]
        for meth in ("Powell", "BFGS"):
            rr = minimize(nll, s0, method=meth)
            if best is None or rr.fun < best.fun: best = rr
    print(f" {nm}: NM {np.round(x0,4)} nll {nll(x0):.4f} | best multistart {np.round(best.x,4)} nll {best.fun:.4f}")

print("== D. no-Helmholtz penalties (3.0) on test columns, D models")
for name, pf in dm.items():
    for t in TEST:
        preds = [C.band_from_prob(pf, r, T) for r in rows[t]]
        fin = [np.isfinite(r["lower"][0]) and np.isfinite(r["lower"][1]) for r in rows[t]]
        npen = sum(1 for p, f in zip(preds, fin) if f and not np.isfinite(p[0]))
        print(f" {name} {t}: penalised finite lower cols {npen}/{sum(fin)}; top-of-grid upper (>=log 5.9): {sum(1 for p in preds if np.isfinite(p[1]) and p[1] > np.log(5.9))}")
