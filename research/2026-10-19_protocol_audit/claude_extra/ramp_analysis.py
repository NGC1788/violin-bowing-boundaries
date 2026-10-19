"""Does the attack protocol (bow acceleration before the plateau) vary across the IWK design, and does it
predict Helmholtz establishment beyond force and beta?  (Claude, 2026-10-18)

Inputs: ramp_meta_A2.csv (this folder) and ~/violin-rne/timeres/windows_full.csv.gz (0.2 s window labels).
Per stroke: final label = majority label of windows starting in the last 0.4 s of the plateau;
early label = label of the first window.
"""
import csv
import gzip
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

HERE = Path(__file__).parent
META = HERE / "ramp_meta_A2.csv"
WINDOWS = Path.home() / "violin-rne/timeres/windows_full.csv.gz"


def load():
    meta = {}
    with META.open() as fh:
        for r in csv.DictReader(fh):
            if r["error"]:
                continue
            meta[(r["folder"].split("/")[-1], int(r["trial"]))] = {k: float(r[k]) for k in
                                                                  ("beta", "v_max", "ramp_s", "accel", "ramp_linearity", "force_plateau",
                                                                   "force_at_onset", "force_cv_plateau", "onset_s", "plateau_on_s", "plateau_off_s")}
    win = defaultdict(list)
    with gzip.open(WINDOWS, "rt") as fh:
        for r in csv.DictReader(fh):
            if r["error"]:
                continue
            win[(r["condition"], int(r["trial"]))].append((float(r["t"]), r["label"], float(r["plateau_s"]), int(r["beta_level"]), int(r["force_rank"])))
    rows = []
    for key, ws in win.items():
        if key not in meta:
            continue
        ws.sort()
        plateau = ws[0][2]
        late = [w[1] for w in ws if w[0] >= plateau - 0.2 - 0.4]
        final = Counter(late).most_common(1)[0][0] if late else ws[-1][1]
        rows.append({"cond": key[0], "trial": key[1], "beta_level": ws[0][3], "force_rank": ws[0][4],
                     "early": ws[0][1], "final": final, "hh_frac": float(np.mean([w[1] == "helmholtz" for w in ws])), **meta[key]})
    return rows


def logit_fit(X, y):
    def nll(b):
        z = X @ b
        return float(np.sum(np.logaddexp(0, z) - y * z))
    def grad(b):
        p = 1 / (1 + np.exp(-(X @ b)))
        return X.T @ (p - y)
    res = minimize(nll, np.zeros(X.shape[1]), jac=grad, method="BFGS")
    p = 1 / (1 + np.exp(-(X @ res.x)))
    cov = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]) + 1e-9 * np.eye(X.shape[1]))
    return res.x, np.sqrt(np.diag(cov)), res.fun


def main():
    rows = load()
    print(f"긋기 {len(rows)}개 결합")
    for cond in ("v1", "v2", "v3"):
        R = [r for r in rows if r["cond"] == cond]
        a = np.array([r["accel"] for r in R]); f = np.array([r["force_plateau"] for r in R]); b = np.array([r["beta"] for r in R])
        fr = np.array([r["force_rank"] for r in R]); bl = np.array([r["beta_level"] for r in R])
        print(f"\n[{cond}] v={np.median([r['v_max'] for r in R]):.3f} m/s, 긋기 {len(R)}")
        print(f"  가속 분위수 (5/25/50/75/95%): {np.round(np.percentile(a, [5, 25, 50, 75, 95]), 3)} m/s²;  고유값 수(3자리) {len(set(np.round(a, 3)))}")
        print(f"  램프 선형성 R² 중앙값 {np.median([r['ramp_linearity'] for r in R]):.4f};  플래토 길이 분위수 {np.round(np.percentile([r['plateau_off_s'] - r['plateau_on_s'] for r in R], [5, 50, 95]), 2)} s")
        rho_f = spearmanr(a, f).correlation; rho_b = spearmanr(a, b).correlation
        rho_fr = spearmanr(a, fr).correlation
        print(f"  가속과의 순위상관: 힘 {rho_f:+.2f}, 힘 순위 {rho_fr:+.2f}, β {rho_b:+.2f}")
        # within-column variation: for each beta level, Spearman of accel vs force rank
        within = [spearmanr(a[bl == k], fr[bl == k]).correlation for k in sorted(set(bl)) if np.ptp(a[bl == k]) > 0]
        print(f"  같은 β 열 안에서 가속–힘순위 상관 중앙값 {np.nanmedian(within):+.2f} (열 {len(within)}개)")
        # Does acceleration predict final Helmholtz beyond force and beta? logistic with log F, log beta, (log F)^2, log a
        y = np.array([r["final"] == "helmholtz" for r in R], float)
        lf, lb, la = np.log(f), np.log(b), np.log(a)
        base = np.column_stack([np.ones_like(lf), lf, lb, lf ** 2, lf * lb])
        full = np.column_stack([base, la])
        _, _, nll0 = logit_fit(base, y)
        coef, se, nll1 = logit_fit(full, y)
        lr = 2 * (nll0 - nll1)
        print(f"  최종 헬름홀츠 로지스틱: log 가속 계수 {coef[-1]:+.3f} ± {se[-1]:.3f} (z = {coef[-1] / se[-1]:+.1f}), 우도비 χ² = {lr:.1f} (자유도 1)")
        y0 = np.array([r["early"] == "helmholtz" for r in R], float)
        _, _, e0 = logit_fit(base, y0)
        c2, s2, e1 = logit_fit(full, y0)
        print(f"  첫 창 헬름홀츠 로지스틱: log 가속 계수 {c2[-1]:+.3f} ± {s2[-1]:.3f}, 우도비 χ² = {2 * (e0 - e1):.1f}")


if __name__ == "__main__":
    main()
