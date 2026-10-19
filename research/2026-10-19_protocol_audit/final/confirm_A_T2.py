"""Confirmatory test on the reserved specimen A_T2 (plan: CONFIRM_PLAN_A_T2.md, frozen before scoring).
(Claude, 2026-10-18)"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

import compare as C
import data as D

D.STEMS["A_T2"] = "2024-06-19_TypeA_T2_sample1"
T = 0.2
PUB = {"A1": 1.0592, "A2": 1.0592, "A_T2": 1.1884}
ONE = {"A1": 1.0, "A2": 1.0, "A_T2": 1.0}


def rows_for(tag):
    rs = C.intervals(D.load(tag), T)
    for r in rs:
        r["tag"] = tag
    return rs


def triv5(train_rows, r, which):
    vals = []
    for tag in ("A1", "A2"):
        cand = [x for x in train_rows if x["tag"] == tag and abs(x["lv"] - r["lv"]) < 1e-6
                and np.isfinite(x[which][0]) and np.isfinite(x[which][1])]
        cand.sort(key=lambda x: abs(x["lb"] - r["lb"]))
        vals += [0.5 * (x[which][0] + x[which][1]) for x in cand[:5]]
    return float(np.median(vals)) if vals else np.nan


def offset_fit(rows, base, which):
    """Interval-censored fit of one offset delta with the base law fixed: log F = base(r) + delta."""
    mu0 = np.array([base(r) for r in rows]); lo = np.array([r[which][0] for r in rows]); hi = np.array([r[which][1] for r in rows])
    ok = ~(np.isnan(lo) & np.isnan(hi))
    mu0, lo, hi = mu0[ok], lo[ok], hi[ok]
    def nll(p):
        mu = mu0 + p[0]; s = np.exp(p[1])
        pr = np.where(np.isfinite(hi), norm.cdf((hi - mu) / s), 1.0) - np.where(np.isfinite(lo), norm.cdf((lo - mu) / s), 0.0)
        return -np.sum(np.log(np.clip(pr, 1e-300, None)))
    return minimize(nll, [0.0, np.log(0.5)], method="Nelder-Mead").x[0]


def main():
    tr = rows_for("A1") + rows_for("A2")
    te = rows_for("A_T2")
    d_te = D.load("A_T2")
    print(f"A_T2: 창 {len(d_te['t'])}, 폴더 {list(d_te['folders'])}, flyback Z {d_te['Z'][0]:.3f} (공개 Z 1.1884)")
    pub = C.physics_models(tr, PUB)
    noz = C.physics_models(tr, ONE)
    gbm = C.data_models([D.load("A1"), D.load("A2")])["D-gbm"]
    preds = {
        "Triv-5": [(triv5(tr, r, "lower"), triv5(tr, r, "upper")) for r in te],
        "P-free (공개 Z)": [pub["P-free"][0](r, "A_T2") for r in te],
        "P-Beigi (공개 Z)": [pub["P-Beigi"][0](r, "A_T2") for r in te],
        "P-Sch (공개 Z)": [pub["P-Sch"][0](r, "A_T2") for r in te],
        "P-free (Z 없음)": [noz["P-free"][0](r, "A_T2") for r in te],
        "D-gbm": [C.band_from_prob(gbm, r, T) for r in te],
    }
    nl = [sum(np.isfinite(r["lower"][0]) and np.isfinite(r["lower"][1]) for r in te),
          sum(not np.isfinite(r["lower"][0]) for r in te), sum(not np.isfinite(r["lower"][1]) for r in te)]
    nu = [sum(np.isfinite(r["upper"][0]) and np.isfinite(r["upper"][1]) for r in te),
          sum((not np.isnan(r["upper"][0])) and not np.isfinite(r["upper"][1]) for r in te)]
    print(f"열 120개: 아래 경계 유한 {nl[0]}, 왼쪽검열 {nl[1]}, 오른쪽검열 {nl[2]} | 위 경계 유한 {nu[0]}, 오른쪽검열 {nu[1]}")
    scores = {}
    for name, pr in preds.items():
        sb = C.score_boundaries(te, pr)
        scores[name] = sb
        print(f"  {name:16s} 아래 |log 오차| {sb['lower'][0]:.3f} (적중 {sb['lower'][1]:.2f}) | 위 {sb['upper'][0]:.3f} (적중 {sb['upper'][1]:.2f})")
    h1 = scores["P-free (공개 Z)"]["lower"][0] < scores["Triv-5"]["lower"][0] and scores["P-free (공개 Z)"]["upper"][0] < scores["Triv-5"]["upper"][0]
    print(f"H1 (공개 Z P-free < Triv-5, 두 경계 모두): {'지지' if h1 else '기각'}")
    # Z exponent: offset of A_T2 relative to the A1+A2 no-Z P-free law
    rng = np.random.default_rng(0)
    for which, j in (("lower", 0), ("upper", 1)):
        base = lambda r: noz["P-free"][0](r, "A_T2")[j]
        delta = offset_fit(te, base, which)
        boots = [offset_fit([te[i] for i in rng.integers(0, len(te), len(te))], base, which) for _ in range(200)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        dA1 = offset_fit([r for r in tr if r["tag"] == "A1"], base, which)
        dA2 = offset_fit([r for r in tr if r["tag"] == "A2"], base, which)
        g = delta / np.log(1.1884 / 1.0592)
        print(f"  {which}: A_T2 수준 이동 {delta:+.3f} [{lo:+.3f}, {hi:+.3f}] log → Z 지수 g = {g:+.2f} [{lo / 0.1151:+.2f}, {hi / 0.1151:+.2f}]"
              f" (Schelleng {'2' if which == 'lower' else '1'}, Beigi {'1' if which == 'lower' else '-'}); 비교 잣대: A1 {dA1:+.3f}, A2 {dA2:+.3f} (A1–A2 차이 {dA1 - dA2:+.3f})")


if __name__ == "__main__":
    main()
