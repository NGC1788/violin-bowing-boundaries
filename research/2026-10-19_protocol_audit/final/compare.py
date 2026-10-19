"""Physics model vs data-driven model: predicting the playability boundary of unseen strings
(Claude, 2026-10-18; protocol frozen in ANALYSIS_PLAN_FINAL.md). Train on A1+A2, test on B1, C1, D1, B_T2 x2.
Usage: python compare.py [t_eval=0.2]"""
import json
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from data import TEST, TRAIN, load

T_EVAL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
STEP = np.log(40.0) / 49  # one nominal force-grid step in log units
K = 2


def columns_of(d, t):
    m = np.isclose(d["t"], t)
    groups = defaultdict(list)
    for i in np.flatnonzero(m):
        groups[(int(d["folder"][i]), int(d["column"][i]))].append(i)
    return groups


def intervals(d, t):
    """Per column: (log v, log beta, lower lo, lower hi, upper lo, upper hi, window indices sorted by force)."""
    out = []
    for key, idx in columns_of(d, t).items():
        idx = sorted(idx, key=lambda i: d["force"][i])
        f = np.log(np.maximum(d["force"][idx], 1e-3)); h = d["H"][idx]
        runs = [i for i in range(len(h) - K + 1) if all(h[i:i + K])]
        if runs:
            s = runs[0]
            llo, lhi = (-np.inf, f[0]) if s == 0 else (f[s - 1], f[s])
            e = runs[-1] + K - 1
            ulo, uhi = (f[-1], np.inf) if e == len(h) - 1 else (f[e], f[e + 1])
        else:
            llo, lhi, ulo, uhi = f[-1], np.inf, np.nan, np.nan
        out.append(dict(lv=np.log(d["speed"][idx[0]]), lb=np.log(np.median(d["beta"][idx])), lower=(llo, lhi),
                        upper=(ulo, uhi), idx=idx, fmin=f[0], fmax=f[-1]))
    return out


def censored_fit(rows, design, which):
    """Interval-censored normal fit of log F = design(row) @ theta + offset(row)."""
    X = np.array([design(r)[0] for r in rows]); off = np.array([design(r)[1] for r in rows])
    lo = np.array([r[which][0] for r in rows]); hi = np.array([r[which][1] for r in rows])
    ok = np.isfinite(lo) | np.isfinite(hi)
    X, off, lo, hi = X[ok], off[ok], lo[ok], hi[ok]
    def nll(p):
        mu = X @ p[:-1] + off; s = np.exp(p[-1])
        pr = np.where(np.isfinite(hi), norm.cdf((hi - mu) / s), 1.0) - np.where(np.isfinite(lo), norm.cdf((lo - mu) / s), 0.0)
        return -np.sum(np.log(np.clip(pr, 1e-300, None)))
    r = minimize(nll, np.r_[np.zeros(X.shape[1]), np.log(0.5)], method="Nelder-Mead", options={"maxiter": 40000, "xatol": 1e-8, "fatol": 1e-10})
    return r.x


def physics_models(train_rows, z_of):
    """Each model returns predict(row, tag) -> (log Fmin, log Fmax)."""
    models = {}
    def mk(lower_design, upper_design):
        pl = censored_fit(train_rows, lambda r: lower_design(r, z_of[r["tag"]]), "lower")
        pu = censored_fit([r for r in train_rows if np.isfinite(r["upper"][0]) or np.isfinite(r["upper"][1])],
                          lambda r: upper_design(r, z_of[r["tag"]]), "upper")
        def pred(r, tag):
            xl, ol = lower_design(r, z_of[tag]); xu, ou = upper_design(r, z_of[tag])
            return float(np.dot(xl, pl[:-1]) + ol), float(np.dot(xu, pu[:-1]) + ou)
        return pred, (pl, pu)
    sch_u = lambda r, z: (np.array([1.0]), np.log(z) + r["lv"] - r["lb"])
    models["P-Sch"] = mk(lambda r, z: (np.array([1.0]), 2 * np.log(z) + r["lv"] - 2 * r["lb"]), sch_u)
    models["P-Beigi"] = mk(lambda r, z: (np.array([1.0]), np.log(z) + r["lv"] - r["lb"]), sch_u)
    models["P-free"] = mk(lambda r, z: (np.array([1.0, r["lv"], r["lb"]]), 2 * np.log(z)),
                          lambda r, z: (np.array([1.0, r["lv"], r["lb"]]), np.log(z)))
    return models


def features(lf, lv, lb, t):
    lf, lv, lb, t = map(np.asarray, (lf, lv, lb, t))
    return np.column_stack([lf, lv, lb, t, lf ** 2, lv ** 2, lb ** 2, lf * lv, lf * lb, lv * lb, lf * t])


def data_models(train):
    X = np.vstack([features(np.log(np.maximum(d["force"], 1e-3)), np.log(d["speed"]), np.log(d["beta"]), d["t"]) for d in train])
    y = np.concatenate([d["H"] for d in train]).astype(int)
    mu, sd = X.mean(0), X.std(0)
    logit = LogisticRegression(C=10.0, max_iter=5000).fit((X - mu) / sd, y)
    raw = np.vstack([np.column_stack([np.log(np.maximum(d["force"], 1e-3)), np.log(d["speed"]), np.log(d["beta"]), d["t"]]) for d in train])
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(raw, y)
    p_logit = lambda lf, lv, lb, t: logit.predict_proba((features(lf, lv, lb, t) - mu) / sd)[:, 1]
    p_gbm = lambda lf, lv, lb, t: gbm.predict_proba(np.column_stack([lf, lv, lb, t]))[:, 1]
    return {"D-logit": p_logit, "D-gbm": p_gbm}


def band_from_prob(p_fun, r, t):
    grid = np.linspace(np.log(0.03), np.log(6.0), 400)
    p = p_fun(grid, np.full_like(grid, r["lv"]), np.full_like(grid, r["lb"]), np.full_like(grid, t))
    above = p >= 0.5
    if not above.any():
        return np.inf, np.nan
    return float(grid[np.argmax(above)]), float(grid[len(above) - 1 - np.argmax(above[::-1])])


def score_boundaries(rows, preds):
    out = {}
    for which, j in (("lower", 0), ("upper", 1)):
        err, hits, n = [], 0, 0
        for r, pr in zip(rows, preds):
            lo, hi = r[which]
            if np.isnan(lo) and np.isnan(hi):
                continue
            p = pr[j]
            n += 1
            if np.isfinite(lo) and np.isfinite(hi):
                err.append(abs(p - 0.5 * (lo + hi)) if np.isfinite(p) else 3.0)
            ok = (p >= lo - STEP if np.isfinite(lo) else True) and (p <= hi + STEP if np.isfinite(hi) else True)
            hits += bool(ok) if np.isfinite(p) else bool(not np.isfinite(hi))
        out[which] = (float(np.median(err)) if err else np.nan, hits / max(n, 1), n)
    return out


def window_scores(d, rows, prob_fun):
    y, p = [], []
    for r in rows:
        for i in r["idx"]:
            y.append(d["H"][i]); p.append(prob_fun(np.log(max(d["force"][i], 1e-3)), r))
    y, p = np.array(y, bool), np.clip(np.array(p), 1e-4, 1 - 1e-4)
    pred = p >= 0.5
    iou = np.sum(pred & y) / max(np.sum(pred | y), 1)
    ll = -np.mean(y * np.log(p) + (~y) * np.log(1 - p))
    return float(iou), float(ll)


def band_prob(lf, lmin, lmax, k):
    return expit(k * np.minimum(lf - lmin, lmax - lf)) if np.isfinite(lmin) and np.isfinite(lmax) else (0.0 if not np.isfinite(lmin) else expit(k * (lf - lmin)))


def fit_slope(train, train_rows_by_tag, pred_fun):
    """Logistic slope k of the band model on training windows (same for all models' recalibration)."""
    L, U, F, Y = [], [], [], []
    for d, tag in zip(train, TRAIN):
        for r in train_rows_by_tag[tag]:
            lmin, lmax = pred_fun(r, tag)
            for i in r["idx"]:
                L.append(lmin); U.append(lmax); F.append(np.log(max(d["force"][i], 1e-3))); Y.append(d["H"][i])
    L, U, F, Y = map(np.array, (L, U, F, Y))
    m = np.minimum(F - L, U - F)
    best = min(((k, -np.mean(Y * np.log(np.clip(expit(k * m), 1e-6, 1)) + (1 - Y) * np.log(np.clip(1 - expit(k * m), 1e-6, 1)))) for k in np.linspace(0.5, 15, 59)), key=lambda x: x[1])
    return best[0]


def recalibrate(d, rows, preds, k, chosen):
    """Shift lower and upper boundaries (delta_L, delta_U) to fit the chosen windows; N(0,1) prior."""
    F, L, U, Y = [], [], [], []
    lookup = {}
    for r, pr in zip(rows, preds):
        for i in r["idx"]:
            lookup[i] = pr
    for i in chosen:
        pr = lookup[i]
        F.append(np.log(max(d["force"][i], 1e-3))); L.append(pr[0] if np.isfinite(pr[0]) else 3.0)
        U.append(pr[1] if np.isfinite(pr[1]) else (pr[0] + 1.0 if np.isfinite(pr[0]) else 3.0)); Y.append(d["H"][i])
    F, L, U, Y = map(np.array, (F, L, U, Y))
    def nll(x):
        p = np.clip(expit(k * np.minimum(F - L - x[0], U + x[1] - F)), 1e-6, 1 - 1e-6)
        return -np.sum(Y * np.log(p) + (1 - Y) * np.log(1 - p)) + 0.5 * (x[0] ** 2 + x[1] ** 2)
    x = minimize(nll, [0.0, 0.0], method="Nelder-Mead").x
    return [(pr[0] + x[0], pr[1] + x[1]) for pr in preds]


def main():
    train = [load(t) for t in TRAIN]
    test = {t: load(t) for t in TEST}
    z_of = {t: float(load(t)["Z"][0]) for t in TRAIN + TEST}
    rows_by_tag = {}
    for tag, d in list(zip(TRAIN, train)) + list(test.items()):
        rows = intervals(d, T_EVAL)
        for r in rows:
            r["tag"] = tag
        rows_by_tag[tag] = rows
    train_rows = rows_by_tag["A1"] + rows_by_tag["A2"]
    phys = physics_models(train_rows, z_of)
    dmod = data_models(train)
    results = defaultdict(dict)
    report = {"t_eval": T_EVAL, "Z": z_of, "physics_params": {}}
    for name, (pred_fun, params) in phys.items():
        report["physics_params"][name] = {"lower": [float(x) for x in params[0]], "upper": [float(x) for x in params[1]]}
    preds_all = {}
    for name, (pred_fun, _) in phys.items():
        preds_all[name] = lambda r, tag, f=pred_fun: f(r, tag)
    for name, p_fun in dmod.items():
        preds_all[name] = lambda r, tag, f=p_fun: band_from_prob(f, r, T_EVAL)
    slope = {name: fit_slope(train, rows_by_tag, pf) for name, pf in preds_all.items()}
    rng = np.random.default_rng(0)
    for tag, d in test.items():
        rows = rows_by_tag[tag]
        all_idx = np.concatenate([r["idx"] for r in rows])
        for name, pf in preds_all.items():
            preds = [pf(r, tag) for r in rows]
            sb = score_boundaries(rows, preds)
            if name in dmod:
                iou, ll = window_scores(d, rows, lambda lf, r, f=dmod[name]: float(f(np.array([lf]), np.array([r["lv"]]), np.array([r["lb"]]), np.array([T_EVAL]))[0]))
            else:
                pm = {id(r): pr for r, pr in zip(rows, preds)}
                iou, ll = window_scores(d, rows, lambda lf, r, k=slope[name]: band_prob(lf, *pm[id(r)], k))
            res = {"zero_shot": {"lower_med_abs_log": sb["lower"][0], "lower_hit": sb["lower"][1], "upper_med_abs_log": sb["upper"][0],
                                 "upper_hit": sb["upper"][1], "n_lower": sb["lower"][2], "n_upper": sb["upper"][2], "iou": iou, "logloss": ll}}
            for kshot in (5, 10, 20, 40):
                for mode in ("random", "grid"):
                    scores = []
                    reps = 20 if mode == "random" else 1
                    for rep in range(reps):
                        if mode == "random":
                            chosen = rng.choice(all_idx, kshot, replace=False)
                        else:
                            order = np.argsort(d["column"][all_idx] * 1000 + d["folder"][all_idx] * 100000 + d["force"][all_idx])
                            chosen = all_idx[order[np.linspace(0, len(order) - 1, kshot).astype(int)]]
                        adj = recalibrate(d, rows, preds, slope[name], chosen)
                        s2 = score_boundaries(rows, adj)
                        scores.append((s2["lower"][0], s2["lower"][1], s2["upper"][0], s2["upper"][1]))
                    sc = np.nanmedian(np.array(scores, float), axis=0)
                    res[f"k{kshot}_{mode}"] = {"lower_med_abs_log": sc[0], "lower_hit": sc[1], "upper_med_abs_log": sc[2], "upper_hit": sc[3]}
            results[tag][name] = res
            zs = res["zero_shot"]
            print(f"{tag:7s} {name:8s} zero-shot: lower |log err| {zs['lower_med_abs_log']:.3f} hit {zs['lower_hit']:.2f} | upper {zs['upper_med_abs_log']:.3f} hit {zs['upper_hit']:.2f} | IoU {zs['iou']:.3f} LL {zs['logloss']:.3f}"
                  f" || k10 random lower {res['k10_random']['lower_med_abs_log']:.3f} upper {res['k10_random']['upper_med_abs_log']:.3f}", flush=True)
    report["results"] = results
    report["band_slope"] = slope
    with open(f"compare_t{T_EVAL}.json", "w") as fh:
        json.dump(report, fh, indent=1, default=float)
    print("\n요약 (테스트 표본 5개 평균)")
    for name in preds_all:
        rows = [results[t][name] for t in TEST]
        f = lambda key, sub="zero_shot": np.nanmean([r[sub][key] for r in rows])
        print(f"  {name:8s} 0샷: 아래 {f('lower_med_abs_log'):.3f}/{f('lower_hit'):.2f}, 위 {f('upper_med_abs_log'):.3f}/{f('upper_hit'):.2f}, IoU {f('iou'):.3f}, LL {f('logloss'):.3f} | "
              f"10샷(무작위): 아래 {f('lower_med_abs_log', 'k10_random'):.3f}, 위 {f('upper_med_abs_log', 'k10_random'):.3f} | 40샷: 아래 {f('lower_med_abs_log', 'k40_random'):.3f}, 위 {f('upper_med_abs_log', 'k40_random'):.3f}")


if __name__ == "__main__":
    main()
