"""Finite-grid mathematical examples; no empirical data are read."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr

HERE = Path(__file__).resolve().parent


def lower_interval(log_force, label, k=2):
    for start in range(len(label) - k + 1):
        if np.all(label[start:start + k]):
            return (-np.inf if start == 0 else log_force[start - 1], log_force[start])
    return (log_force[-1], np.inf)


def fit_intervals(x, bounds):
    lo, hi = np.asarray(bounds).T
    design = np.column_stack([np.ones(len(x)), x])
    assert np.isfinite(lo).all() and np.isfinite(hi).all() and (hi > lo).all()

    def nll(par):
        mu = design @ par[:2]
        sig = np.exp(par[2])
        zlo, zhi = (lo - mu) / sig, (hi - mu) / sig
        # Use survival probabilities when both endpoints are positive.
        a = np.where(zlo > 0, log_ndtr(-zlo), log_ndtr(zhi))
        b = np.where(zlo > 0, log_ndtr(-zhi), log_ndtr(zlo))
        difference = b - a
        if np.any(difference >= 0):
            return np.inf
        return float(-np.sum(a + np.log(-np.expm1(difference))))

    ols = np.linalg.lstsq(design, (lo + hi) / 2, rcond=None)[0]
    start = np.r_[ols, 0.0]
    fit = minimize(nll, start, method="BFGS", options={"gtol": 1e-7})
    # A second deterministic optimizer verifies the same optimum.
    check = minimize(nll, start, method="Nelder-Mead", options={"maxiter": 5000, "xatol": 1e-10, "fatol": 1e-10})
    assert check.success and abs(fit.fun - check.fun) < 1e-7
    return {"intercept": float(check.x[0]), "beta_exponent": float(check.x[1]),
            "sigma": float(np.exp(check.x[2])), "nll": float(check.fun),
            "optimizer_difference": float(abs(fit.fun-check.fun)), "success": bool(check.success)}


def main():
    # Values are log(beta), all beta physically inside (0, 1).
    xs = np.array([0.0, 0.6, 1.8]) - 4
    # Replicates prevent the zero-variance MLE of three identical intervals.
    offsets = np.array([-2., -1., 0., 1., 2.])
    windows = []
    for col, x in enumerate(xs):
        for rep, offset in enumerate(offsets):
            for j, force in enumerate(np.arange(-1., 4.) + offset):
                h = 1 <= j <= 3
                y = 0.0 if (col == 0 and j == 3) else (0.5 if (col == 1 and j == 1) else 10.)
                windows.append((col, rep, j, x, force, h, y))
    a = np.array(windows)
    baseline = a[:, 5].astype(bool)
    result = {"kind": "synthetic finite-grid counterexample, not an empirical failure", "K": 2, "removal_fraction": 1/9, "variants": {}}
    for tilt in [0., 1.]:
        score = a[:, 6] - tilt * a[:, 3]
        cutoff = np.quantile(score[baseline], 1/9)
        h = baseline & (score >= cutoff)
        bounds, x = [], []
        for col in range(3):
            for rep in range(len(offsets)):
                mask = (a[:, 0] == col) & (a[:, 1] == rep)
                bounds.append(lower_interval(a[mask, 4], h[mask]))
                x.append(xs[col])
        removed = baseline & ~h
        result["variants"][str(tilt)] = {"removed_count": int(removed.sum()),
            "removed_mean_log_beta": float(a[removed, 3].mean()),
            "intervals": bounds, "fit": fit_intervals(np.asarray(x), bounds)}
    z, o = result["variants"]["0.0"], result["variants"]["1.0"]
    result["exponent_change"] = o["fit"]["beta_exponent"] - z["fit"]["beta_exponent"]
    assert o["removed_mean_log_beta"] > z["removed_mean_log_beta"]
    assert result["exponent_change"] < -0.05
    assert abs(z["fit"]["beta_exponent"]) < 1e-6
    # Equal all-window feature means are insufficient for neutrality.
    neutral = np.array([[-.5, 0, .5], [-1., .25, .75]])
    assert np.allclose(neutral.mean(axis=1), 0)
    labels = np.ones((2, 3), dtype=bool)
    labels[1, 0] = False
    neutral_bounds, original_bounds, neutral_x = [], [], []
    for col in range(2):
        for offset in offsets:
            log_force = np.arange(-1., 4.) + offset
            original_bounds.append(lower_interval(log_force, np.array([0, 1, 1, 1, 0], dtype=bool)))
            neutral_bounds.append(lower_interval(log_force, np.r_[False, labels[col], False]))
            neutral_x.append(float(col - 4))
    neutral_base = fit_intervals(np.asarray(neutral_x), original_bounds)
    neutral_fit = fit_intervals(np.asarray(neutral_x), neutral_bounds)
    assert abs(neutral_base["beta_exponent"]) < 1e-6
    assert abs(neutral_fit["beta_exponent"] - 1) < 1e-6
    result["neutral_counterexample"] = {"all_window_scaling_b": 0., "tilt": 0.,
        "log_features": neutral.tolist(), "removed": [1, 0],
        "first_K_starts": [0, 1], "finite_interval_base": neutral_base, "finite_interval_changed": neutral_fit,
        "conclusion": "The neutral feature slope need not preserve the lower boundary slope."}
    # Within-column permutation preserves every marginal feature summary but
    # changes which force position is removed and therefore the boundary effect.
    permuted = neutral.copy()
    permuted[1] = permuted[1, ::-1]
    assert np.allclose(permuted.mean(axis=1), neutral.mean(axis=1))
    assert np.allclose(permuted.std(axis=1), neutral.std(axis=1))
    perm_bounds = []
    for col in range(2):
        h = np.ones(3, dtype=bool)
        if col == 1:
            h[-1] = False
        for offset in offsets:
            perm_bounds.append(lower_interval(np.arange(-1., 4.) + offset, np.r_[False, h, False]))
    perm_fit = fit_intervals(np.asarray(neutral_x), perm_bounds)
    assert abs(perm_fit["beta_exponent"]) < 1e-6
    result["same_marginals_different_response"] = {
        "original_features": neutral.tolist(), "permuted_features": permuted.tolist(),
        "same": ["beta grid", "force grid", "base labels", "global feature slope", "within-column feature distributions", "removed fraction", "removed beta columns"],
        "different": "feature ordering within the force grid",
        "original_changed_exponent": neutral_fit["beta_exponent"],
        "permuted_changed_exponent": perm_fit["beta_exponent"]}
    (HERE / "counterexamples.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"exponent_change": result["exponent_change"], "fit0": z["fit"], "fit1": o["fit"]}, indent=2))


if __name__ == "__main__":
    main()
