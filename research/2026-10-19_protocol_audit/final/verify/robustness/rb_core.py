"""Robustness review helpers (reviewer key: robustness). Imports final/compare.py read-only; never edits originals."""
import sys
from pathlib import Path

import numpy as np

FINAL = Path.home() / "violin-rne/final"
sys.argv = sys.argv[:1]  # compare.py reads sys.argv[1] at import
sys.path.insert(0, str(FINAL))
import compare as C  # noqa: E402
from data import load  # noqa: E402

ALL = ("A1", "A2", "B1", "C1", "D1", "B_T2r1", "B_T2r2")
MODELS = ("P-Sch", "P-Beigi", "P-free", "D-logit", "D-gbm")


def get_data(loader=load):
    return {t: loader(t) for t in ALL}


def make_rows(dat, t, K=2):
    C.K = K
    rows = {}
    for tag, d in dat.items():
        rs = C.intervals(d, t)
        for r in rs:
            r["tag"] = tag
            i0 = r["idx"][0]
            r["col"] = int(d["column"][i0])
            r["speed"] = round(float(d["speed"][i0]), 2)
        rows[tag] = rs
    return rows


def fit_models(dat, rows, train_tags, t, z_of):
    tr = sum((rows[x] for x in train_tags), [])
    phys = C.physics_models(tr, z_of)
    dm = C.data_models([dat[x] for x in train_tags])
    preds = {}
    for name in ("P-Sch", "P-Beigi", "P-free"):
        preds[name] = (lambda r, tag, f=phys[name][0]: f(r, tag))
    for name in ("D-logit", "D-gbm"):
        preds[name] = (lambda r, tag, f=dm[name]: C.band_from_prob(f, r, t))
    return preds, phys


def mid_or_edge(iv):
    lo, hi = iv
    if np.isnan(lo) and np.isnan(hi):
        return np.nan
    if np.isfinite(lo) and np.isfinite(hi):
        return 0.5 * (lo + hi)
    return hi if np.isfinite(hi) else lo  # censored: use finite edge


def trivial_predictor(rows, train_tags, halfwidth=0):
    """Predict each test column by the median (over training specimens) boundary of the same speed and nearest beta;
    halfwidth>0 pools +-halfwidth neighbouring beta columns per specimen before the median."""
    table = {}
    for tg in train_tags:
        for r in rows[tg]:
            table.setdefault((r["speed"]), []).append((r["lb"], mid_or_edge(r["lower"]), mid_or_edge(r["upper"]), tg, r["col"]))

    def pred(r, tag):
        cand = table[r["speed"]]
        outL, outU = [], []
        for tg in train_tags:
            c = [x for x in cand if x[3] == tg]
            if not c:
                continue
            c.sort(key=lambda x: x[4])
            j = int(np.argmin([abs(x[0] - r["lb"]) for x in c]))
            sel = c[max(0, j - halfwidth): j + halfwidth + 1]
            L = [x[1] for x in sel if np.isfinite(x[1])]
            U = [x[2] for x in sel if np.isfinite(x[2])]
            if L:
                outL.append(np.median(L))
            if U:
                outU.append(np.median(U))
        return (float(np.median(outL)) if outL else np.inf, float(np.median(outU)) if outU else np.nan)
    return pred


def per_column_errors(rows_t, preds):
    """Returns arrays per column: lower err (nan if not finite interval), upper err, speed, col, plus hit flags."""
    out = dict(eL=[], eU=[], hL=[], hU=[], speed=[], col=[], cL=[], cU=[])
    for r, pr in zip(rows_t, preds):
        out["speed"].append(r["speed"]); out["col"].append(r["col"])
        for which, j, ek, hk, ck in (("lower", 0, "eL", "hL", "cL"), ("upper", 1, "eU", "hU", "cU")):
            lo, hi = r[which]
            p = pr[j]
            if np.isnan(lo) and np.isnan(hi):
                out[ek].append(np.nan); out[hk].append(np.nan); out[ck].append("none"); continue
            if np.isfinite(lo) and np.isfinite(hi):
                out[ek].append(abs(p - 0.5 * (lo + hi)) if np.isfinite(p) else 3.0); out[ck].append("finite")
            else:
                out[ek].append(np.nan); out[ck].append("left" if not np.isfinite(lo) else "right")
            ok = (p >= lo - C.STEP if np.isfinite(lo) else True) and (p <= hi + C.STEP if np.isfinite(hi) else True)
            out[hk].append(float(bool(ok)) if np.isfinite(p) else float(not np.isfinite(hi)))
    return {k: np.array(v) for k, v in out.items()}


def evaluate(dat, rows, train_tags, test_tags, t, z_of, extra=None):
    preds, phys = fit_models(dat, rows, train_tags, t, z_of)
    if extra:
        preds.update(extra)
    res = {}
    for tag in test_tags:
        res[tag] = {}
        for name, pf in preds.items():
            pr = [pf(r, tag) for r in rows[tag]]
            res[tag][name] = per_column_errors(rows[tag], pr)
    return res, phys


def summary(res, models=None):
    models = models or list(next(iter(res.values())).keys())
    out = {}
    for m in models:
        L = [np.nanmedian(res[g][m]["eL"]) for g in res]
        U = [np.nanmedian(res[g][m]["eU"]) for g in res]
        HL = [np.nanmean(res[g][m]["hL"]) for g in res]
        HU = [np.nanmean(res[g][m]["hU"]) for g in res]
        out[m] = dict(L=float(np.mean(L)), U=float(np.mean(U)), HL=float(np.mean(HL)), HU=float(np.mean(HU)),
                      Lper={g: float(x) for g, x in zip(res, L)}, Uper={g: float(x) for g, x in zip(res, U)})
    return out


def print_summary(title, s):
    print(f"\n== {title}")
    for m, v in s.items():
        per = " ".join(f"{g}:{v['Lper'][g]:.3f}/{v['Uper'][g]:.3f}" for g in v["Lper"])
        print(f"  {m:10s} lower {v['L']:.3f} (hit {v['HL']:.2f})  upper {v['U']:.3f} (hit {v['HU']:.2f})   [{per}]", flush=True)
