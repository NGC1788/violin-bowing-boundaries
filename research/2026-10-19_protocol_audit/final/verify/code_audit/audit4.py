"""code_audit: (i) D models trained only on t_eval windows; (ii) D-logit leverage of force<=1e-3 clipped windows; (iii) paired column bootstrap of model differences."""
import sys, numpy as np
sys.argv = [sys.argv[0]]
sys.path.insert(0, "/Users/jeongjihoon/violin-rne/final")
import compare as C
from data import load, TRAIN, TEST
T = 0.2
ALL = TRAIN + TEST
dat = {t: load(t) for t in ALL}
z_of = {t: float(dat[t]["Z"][0]) for t in ALL}
rows = {}
for t in ALL:
    rows[t] = C.intervals(dat[t], T)
    for r in rows[t]:
        r["tag"] = t
for tag in TRAIN:
    d = dat[tag]
    print(tag, "windows with force<=1e-3 over all 4 times:", int(np.sum(d["force"] <= 1e-3)), " H among them:", int(np.sum(d["H"][d["force"] <= 1e-3])), " force<0.02:", int(np.sum(d["force"] < 0.02)))

def sub(d, mask):
    return {k: (v[mask] if (isinstance(v, np.ndarray) and v.shape[:1] == d["t"].shape) else v) for k, v in d.items()}

def score(dm):
    out = {}
    for name in dm:
        L, U = [], []
        for t in TEST:
            sb = C.score_boundaries(rows[t], [C.band_from_prob(dm[name], r, T) for r in rows[t]])
            L.append(sb["lower"][0]); U.append(sb["upper"][0])
        out[name] = (np.mean(L), np.mean(U))
    return out
base = C.data_models([dat[t] for t in TRAIN])
print("as coded (4 times):", {k: tuple(round(x, 3) for x in v) for k, v in score(base).items()})
only = C.data_models([sub(dat[t], np.isclose(dat[t]["t"], T)) for t in TRAIN])
print("t_eval windows only:", {k: tuple(round(x, 3) for x in v) for k, v in score(only).items()})
noclip = C.data_models([sub(dat[t], dat[t]["force"] > 0.02) for t in TRAIN])
print("drop force<=0.02 N windows (4 times):", {k: tuple(round(x, 3) for x in v) for k, v in score(noclip).items()})

print("== paired bootstrap over columns (within each test specimen), mean over 5 specimens of median |log err|; diff = model - P-free")
phys = C.physics_models(rows["A1"] + rows["A2"], z_of)
pf = {n: (lambda r, tag, f=p: f(r, tag)) for n, (p, _) in phys.items()}
for n, p in base.items():
    pf[n] = lambda r, tag, f=p: C.band_from_prob(f, r, T)
E = {}
for n, f in pf.items():
    for t in TEST:
        preds = [f(r, t) for r in rows[t]]
        for which, j in (("lower", 0), ("upper", 1)):
            e = np.array([abs(pr[j] - 0.5 * sum(r[which])) if np.isfinite(pr[j]) else 3.0 if (np.isfinite(r[which][0]) and np.isfinite(r[which][1])) else np.nan for r, pr in zip(rows[t], preds)])
            E[(n, t, which)] = e
rng = np.random.default_rng(0)
B = 2000
for which in ("lower", "upper"):
    for n in pf:
        if n == "P-free": continue
        diffs = []
        for b in range(B):
            dd = []
            for t in TEST:
                e1 = E[(n, t, which)]; e0 = E[("P-free", t, which)]
                ok = np.flatnonzero(~np.isnan(e0))
                s = rng.choice(ok, len(ok))
                dd.append(np.median(e1[s]) - np.median(e0[s]))
            diffs.append(np.mean(dd))
        pt = np.mean([np.nanmedian(E[(n, t, which)]) - np.nanmedian(E[("P-free", t, which)]) for t in TEST])
        print(f" {which} {n:8s} - P-free: {pt:+.3f} [{np.percentile(diffs, 2.5):+.3f}, {np.percentile(diffs, 97.5):+.3f}]")
