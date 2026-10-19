"""Which minimum-bow-force law do the measured cello boundaries support?

Three published hypotheses disagree:
  Schelleng (1973)               F_min ∝ v / beta^2
  Schoonderwaldt et al. (2008)   F_min independent of bow speed (empirical)
  Beigi & Conneely (2026)        F_min ∝ Z v / beta   (simulation)
plus two composite forms. Each string gets its own prefactor (friction,
bridge resistance and so on differ between strings); the exponents are shared.
Scored by in-sample BIC and by leave-one-speed-out prediction, which is the
test that actually separates speed laws.
"""

import glob
import json
from pathlib import Path

import numpy as np

REPO = Path.home() / "violin-bowing-boundaries"
SPEED = {"v1": 0.05, "v2": 0.10, "v3": 0.20,
         "2024-03-27_r_v1": 0.05, "2024-03-27_r_v2": 0.10, "2024-03-27_r_v3": 0.20}


def load():
    rows = []
    for path in sorted(glob.glob(str(REPO / "reports/collection/*/regime_map/*/summary.json"))):
        string = Path(path).parts[-4].split("_Type")[1][0]
        for condition, info in json.loads(Path(path).read_text())["conditions"].items():
            fit = (info.get("boundaries") or {}).get("lower") or {}
            for beta, force in fit.get("points", []):
                if force > 0:
                    rows.append((string, SPEED[condition], beta, force))
    s = np.array([r[0] for r in rows])
    v, b, f = (np.array([r[i] for r in rows], dtype=float) for i in (1, 2, 3))
    return s, v, b, f


def design(model, s, v, b, strings):
    """Linear-in-log models: returns (X, offset) with log F = X @ theta + offset."""
    dummies = np.stack([(s == k).astype(float) for k in strings], axis=1)
    lv, lb = np.log(v), np.log(b)
    if model == "Schelleng v/β²":
        return dummies, lv - 2 * lb
    if model == "Beigi v/β":
        return dummies, lv - lb
    if model == "Schoonderwaldt (속도 무관)":
        return np.column_stack([dummies, lb]), 0 * lv
    if model == "자유 거듭제곱":
        return np.column_stack([dummies, lv, lb]), 0 * lv
    raise KeyError(model)


def fit_linear(model, s, v, b, f, strings):
    X, off = design(model, s, v, b, strings)
    theta, *_ = np.linalg.lstsq(X, np.log(f) - off, rcond=None)
    return theta


def predict_linear(model, theta, s, v, b, strings):
    X, off = design(model, s, v, b, strings)
    return X @ theta + off


def fit_sum(kind, s, v, b, f, strings):
    """F = v * (A_s/β² + B_s/β)  or  F = A_s v/β² + C_s, fitted per string in
    relative (≈log) error: divide both sides by F."""
    params = {}
    for k in strings:
        m = s == k
        if kind == "합성 v/β² + v/β":
            cols = np.column_stack([v[m] / b[m] ** 2, v[m] / b[m]])
        else:  # Schelleng + speed-independent floor
            cols = np.column_stack([v[m] / b[m] ** 2, np.ones(m.sum())])
        w = 1 / f[m]
        coef, *_ = np.linalg.lstsq(cols * w[:, None], f[m] * w, rcond=None)
        params[k] = np.maximum(coef, 1e-12)
    return params


def predict_sum(kind, params, s, v, b):
    out = np.empty_like(v)
    for k, coef in params.items():
        m = s == k
        cols = (np.column_stack([v[m] / b[m] ** 2, v[m] / b[m]]) if kind == "합성 v/β² + v/β"
                else np.column_stack([v[m] / b[m] ** 2, np.ones(m.sum())]))
        out[m] = np.log(np.maximum(cols @ coef, 1e-9))
    return out


def main():
    s, v, b, f = load()
    strings = sorted(set(s))
    print(f"아래 경계 점 {len(f)}개 (현 {strings}, 속도별 {[int(np.sum(v == x)) for x in (0.05, 0.1, 0.2)]})\n")
    linear = ["Schelleng v/β²", "Beigi v/β", "Schoonderwaldt (속도 무관)", "자유 거듭제곱"]
    sums = ["합성 v/β² + v/β", "Schelleng + 속도무관 바닥"]
    n = len(f)
    print(f"{'모형':28}{'파라미터':>8}{'RMS(log)':>10}{'BIC':>9}{'속도 하나 빼고 예측 RMS':>24}")
    for model in linear + sums:
        if model in linear:
            theta = fit_linear(model, s, v, b, f, strings)
            resid = np.log(f) - predict_linear(model, theta, s, v, b, strings)
            k = len(theta)
        else:
            params = fit_sum(model, s, v, b, f, strings)
            resid = np.log(f) - predict_sum(model, params, s, v, b)
            k = 2 * len(strings)
        rss = float(np.sum(resid ** 2))
        bic = n * np.log(rss / n) + k * np.log(n)
        cv = []
        for held in (0.05, 0.1, 0.2):
            tr, te = v != held, v == held
            if model in linear:
                th = fit_linear(model, s[tr], v[tr], b[tr], f[tr], strings)
                pred = predict_linear(model, th, s[te], v[te], b[te], strings)
            else:
                pr = fit_sum(model, s[tr], v[tr], b[tr], f[tr], strings)
                pred = predict_sum(model, pr, s[te], v[te], b[te])
            cv.append(np.log(f[te]) - pred)
        cv_rms = float(np.sqrt(np.mean(np.concatenate(cv) ** 2)))
        extra = ""
        if model == "자유 거듭제곱":
            extra = f"   (속도 지수 {theta[-2]:+.2f}, β 지수 {theta[-1]:+.2f})"
        if model == "Schoonderwaldt (속도 무관)":
            extra = f"   (β 지수 {theta[-1]:+.2f})"
        print(f"{model:28}{k:>8}{np.sqrt(rss / n):>10.3f}{bic:>9.1f}{cv_rms:>24.3f}{extra}")


if __name__ == "__main__":
    main()
