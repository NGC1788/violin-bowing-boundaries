"""Independent re-implementation of per-session boundary levels (post hoc refutation pass of
novelty/session_variance; Claude, 2026-10-20). All data already seen. Natural log.
Own column grouping (folder, beta index), own K=2 run detection, own interval-censored offset MLE,
bootstrap clustered on the beta INDEX (paired across the 3 speeds), plus a 5-column block bootstrap.
Variants: U_S1 (claim), U_E3 (empirical E3_free_Z), L_A12 (claim), L_c02 (lower, left-censored below 0.2 N),
L_onset (lower on the onset-force axis, ramp metadata; 11 sessions)."""
import csv, json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

H = Path.home() / "violin-rne"
F2 = json.loads((H / "novelty/thermal_upper/FROZEN_v2.json").read_text())
CACHE = {"A1": "final/cache/A1.npz", "A2": "final/cache/A2.npz", "A3": "novelty/thermal_upper/cache/TGT_2024-04-22_TypeA_sample3.npz",
         "A4": "novelty/thermal_upper/cache/TGT_2024-04-24_TypeA_sample4.npz", "B1": "final/cache/B1.npz", "C1": "final/cache/C1.npz",
         "D1": "final/cache/D1.npz", "D2": "novelty/thermal_upper/cache/TGT_2024-04-26_TypeD_sample2.npz", "A_T2": "final/cache/A_T2.npz",
         "B_T2s1": "final/cache/B_T2s1.npz", "B_T2r1": "final/cache/B_T2r1.npz", "B_T2r2": "final/cache/B_T2r2.npz",
         "C_T2": "final/cache/C_T2.npz", "D_T2": "novelty/thermal_upper/cache/D_T2.npz"}
GRP = {"A1": "A T1", "A2": "A T1", "A3": "A T1", "A4": "A T1", "B1": "B T1", "C1": "C T1", "D1": "D T1", "D2": "D T1",
       "A_T2": "A T2", "B_T2s1": "B T2", "B_T2r1": "B T2", "B_T2r2": "B T2", "C_T2": "C T2", "D_T2": "D T2"}
AC = H / "astra_20261018/archive_companion"
RAMP = {"A1": H / "claude_extra/ramp/ramp_meta_2024-03-25_TypeA_sample1.csv", "A2": H / "claude_extra/ramp_meta_A2.csv"}
for t, st in (("A3", "2024-04-22_TypeA_sample3"), ("A4", "2024-04-24_TypeA_sample4"), ("C1", "2024-04-12_TypeC_sample1"),
              ("D1", "2024-04-15_TypeD_sample1"), ("D2", "2024-04-26_TypeD_sample2"), ("A_T2", "2024-06-19_TypeA_T2_sample1"),
              ("B_T2s1", "2024-06-13_TypeB_T2_sample1"), ("C_T2", "2024-06-11_TypeC_T2_sample1"), ("D_T2", "2024-06-17_TypeD_T2_sample1")):
    RAMP[t] = AC / st / f"ramp_meta_{st}.csv"
LOG02 = np.log(0.2)


def load(tag, axis="current"):
    d = dict(np.load(H / CACHE[tag]))
    m = np.isclose(d["t"], 0.2)
    d = {k: (v[m] if k not in ("folders", "Z", "f0") else v) for k, v in d.items()}
    if axis == "onset":
        rm = {}
        for r in csv.DictReader(open(RAMP[tag])):
            if r["error"] or r["force_at_onset"] in ("", "nan"):
                continue
            rm[(re.search(r"v[123]$", r["folder"]).group(0), int(r["trial"]))] = float(r["force_at_onset"])
        sp = [re.search(r"v[123]$", str(d["folders"][k])).group(0) for k in d["folder"]]
        f = np.array([rm.get((s, int(tr)), np.nan) for s, tr in zip(sp, d["trial"])])
        ok = np.isfinite(f)
        d = {k: (v[ok] if k not in ("folders", "Z", "f0") else v) for k, v in d.items()}
        d["force"] = f[ok]
    return d


def columns(d, K=2):
    """one row per (speed folder, beta index): lv, lb, beta index, lower (lo,hi), upper (lo,hi)."""
    out = []
    key = d["folder"] * 1000 + d["column"]
    for k in np.unique(key):
        ii = np.flatnonzero(key == k)
        ii = ii[np.argsort(d["force"][ii], kind="stable")]
        f = np.log(np.maximum(d["force"][ii], 1e-3)); h = d["H"][ii].astype(bool)
        ok = h[:-1] & h[1:]
        if ok.any():
            s = int(np.argmax(ok)); e = int(len(ok) - 1 - np.argmax(ok[::-1])) + 1
            low = (-np.inf, f[0]) if s == 0 else (f[s - 1], f[s])
            up = (f[-1], np.inf) if e == len(f) - 1 else (f[e], f[e + 1])
        else:
            low, up = (f[-1], np.inf), (np.nan, np.nan)
        out.append(dict(lv=float(np.log(d["speed"][ii[0]])), lb=float(np.log(np.median(d["beta"][ii]))), col=int(d["column"][ii[0]]),
                        lower=low, upper=up))
    return out


def widen(lo, hi):
    lo, hi = lo.copy(), hi.copy()
    z = np.isfinite(lo) & np.isfinite(hi) & (hi - lo < 1e-9)
    F = np.exp(hi[z])
    lo[z] = np.where(F > 0.011, np.log(np.maximum(F - 0.01, 1e-12)), -np.inf); hi[z] = np.log(F + 0.01)
    return lo, hi


def cens02(lo, hi):
    lo = lo.copy(); hi = hi.copy()
    lo[lo < LOG02] = -np.inf
    hi[hi < LOG02] = LOG02
    return lo, hi


def nll(mu, s, lo, hi):
    pu = np.where(np.isfinite(hi), norm.cdf((hi - mu) / s), 1.0); pl = np.where(np.isfinite(lo), norm.cdf((lo - mu) / s), 0.0)
    return -np.sum(np.log(np.clip(pu - pl, 1e-300, None)))


def offset(mu, lo, hi):
    mid = np.where(np.isfinite(lo) & np.isfinite(hi), (lo + hi) / 2, np.where(np.isfinite(lo), lo, hi))
    x0 = [float(np.median(mid - mu)), np.log(0.3)]
    f = lambda p: nll(mu + p[0], np.exp(p[1]), lo, hi)
    cands = [minimize(f, x0, method="L-BFGS-B", bounds=[(-3, 3), (np.log(0.01), np.log(3))]),
             minimize(f, [0.0, np.log(0.5)], method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-9})]
    cands.append(minimize(f, min(cands, key=lambda r: r.fun).x, method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10}))
    return float(min(cands, key=lambda r: r.fun).x[0])


def free_fit(lv, lb, lo, hi):
    """censored OLS of log F on [1, lv, lb] (own fit; reference lower law)."""
    X = np.column_stack([np.ones_like(lv), lv, lb])
    mid = np.where(np.isfinite(lo) & np.isfinite(hi), (lo + hi) / 2, np.where(np.isfinite(lo), lo, hi))
    b0 = np.linalg.lstsq(X, mid, rcond=None)[0]
    r = minimize(lambda p: nll(X @ p[:3], np.exp(p[3]), lo, hi), np.r_[b0, np.log(0.3)], method="Nelder-Mead",
                 options={"maxiter": 40000, "xatol": 1e-9, "fatol": 1e-11})
    r = minimize(lambda p: nll(X @ p[:3], np.exp(p[3]), lo, hi), r.x, method="BFGS")
    return r.x[:3]


def arrays(rows, which, rule=None):
    rows = [r for r in rows if np.isfinite(r[which][0]) or np.isfinite(r[which][1])]
    lo = np.array([r[which][0] for r in rows], float); hi = np.array([r[which][1] for r in rows], float)
    lo, hi = widen(lo, hi)
    if rule == "c02":
        lo, hi = cens02(lo, hi)
    return rows, lo, hi


def make_ref(name, rowsA12=None):
    if name == "U_S1":
        c = F2["fits"]["S1_strict"]["theta"][0]
        return lambda rows, g: c + np.log(F2["z_published"][g][1]) + np.array([r["lv"] - r["lb"] for r in rows])
    if name == "U_E3":
        p = F2["fits"]["E3_free_Z"]["theta"]
        return lambda rows, g: p[0] + np.log(F2["z_published"][g][1]) + np.array([p[1] * r["lv"] + p[2] * r["lb"] for r in rows])
    which, rule = "lower", ("c02" if name == "L_c02" else None)
    rr, lo, hi = arrays(rowsA12, which, rule)
    p = free_fit(np.array([r["lv"] for r in rr]), np.array([r["lb"] for r in rr]), lo, hi)
    return lambda rows, g, p=p: p[0] + np.array([p[1] * r["lv"] + p[2] * r["lb"] for r in rows])


VARIANTS = {"U_S1": ("upper", None, "current"), "U_E3": ("upper", None, "current"), "L_A12": ("lower", None, "current"),
            "L_c02": ("lower", "c02", "current"), "L_onset": ("lower", None, "onset")}


def level_of(rows, which, rule, ref, g):
    rr, lo, hi = arrays(rows, which, rule)
    return offset(ref(rr, g), lo, hi)


def boot(rows, which, rule, ref, g, B, seed, block=1):
    cols = sorted({r["col"] for r in rows}); by = defaultdict(list)
    for r in rows:
        by[r["col"] // block].append(r)
    keys = sorted(by); rng = np.random.default_rng(seed); out = []
    for _ in range(B):
        kk = rng.integers(0, len(keys), len(keys))
        out.append(level_of([r for k in kk for r in by[keys[k]]], which, rule, ref, g))
    return float(np.std(out, ddof=1))


def main(B=200):
    res = {}
    for v, (which, rule, axis) in VARIANTS.items():
        tags = [t for t in CACHE if axis == "current" or t in RAMP]
        rows = {t: columns(load(t, axis)) for t in tags}
        ref = make_ref(v, rows["A1"] + rows["A2"]) if v.startswith("L") else make_ref(v)
        res[v] = {}
        for t in tags:
            y = level_of(rows[t], which, rule, ref, GRP[t])
            se = boot(rows[t], which, rule, ref, GRP[t], B, 41)
            se5 = boot(rows[t], which, rule, ref, GRP[t], B, 43, block=5) if v in ("U_S1", "L_A12") else None
            res[v][t] = dict(level=y, se=se, se_block5=se5, n=len(arrays(rows[t], which, rule)[0]))
            print(v, t, f"{y:+.3f} se {se:.3f}" + (f" block5 {se5:.3f}" if se5 else ""), flush=True)
    (Path(__file__).parent / "levels.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
