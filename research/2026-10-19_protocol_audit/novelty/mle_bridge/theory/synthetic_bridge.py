"""Synthetic geometry/MLE distinctions; independent of empirical and production code.

Uses a stable Gaussian interval likelihood and direct SciPy refits. Numerical
results illustrate the accompanying analytic proofs; they do not replace them.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import log_ndtr


def log_mass(lo, hi, mu, sigma):
    a, b = (np.asarray(lo) - mu) / sigma, (np.asarray(hi) - mu) / sigma
    upper = log_ndtr(np.where(a > 0, -a, b))
    lower = log_ndtr(np.where(a > 0, -b, a))
    with np.errstate(divide="ignore", invalid="ignore"):
        return upper + np.log(-np.expm1(np.minimum(lower - upper, 0.0)))


def fit(x, lo, hi, p0=None):
    x, lo, hi = map(np.asarray, (x, lo, hi))
    mid = (lo + hi) / 2
    b, a = np.polyfit(x, mid, 1)
    if p0 is not None:
        b = p0
        a = float(np.mean(mid - b * x))
    sigma = max(float(np.std(mid - a - b * x)), 0.05)
    starts = [np.array([a, b, np.log(sigma * f)]) if p0 is None
              else np.array([a, np.log(sigma * f)]) for f in (0.5, 1.0, 2.0)]

    def obj(q):
        aa, bb = q[0], q[1] if p0 is None else p0
        ll = log_mass(lo, hi, aa + bb * x, np.exp(q[-1]))
        return float(-ll.sum()) if np.isfinite(ll).all() else 1e100

    bounds = [(None, None)] * (len(starts[0]) - 1) + [(-8.0, 5.0)]
    out = []
    for method in ("L-BFGS-B", "Powell"):
        for q in starts:
            opts = {"maxiter": 5000, "ftol": 1e-12}
            if method == "L-BFGS-B":
                opts["gtol"] = 1e-7
            else:
                opts["xtol"] = 1e-10
            r = minimize(obj, q, method=method, bounds=bounds, options=opts)
            if r.success and np.isfinite(r.fun) and r.fun < 1e90:
                out.append((float(r.fun), r.x, method))
    assert out, "no valid optimization result"
    val, q, method = min(out, key=lambda r: r[0])
    a, b, sigma = float(q[0]), float(q[1] if p0 is None else p0), float(np.exp(q[-1]))
    return {"a": a, "b": b, "sigma": sigma, "nll": val,
            "best_method": method, "converged_runs": len(out),
            "scale_bound": bool(q[-1] < -7.99 or q[-1] > 4.99),
            "lp": log_mass(lo, hi, a + b * x, sigma).tolist()}


def fit_original_design(lv, lb, lo, hi, p0=None):
    """Original intercept/speed/beta design, direct derivative-free cross-check."""
    full_x = np.column_stack((np.ones(len(lv)), lv, lb))
    design = full_x if p0 is None else full_x[:, :2]
    offset = np.zeros(len(lv)) if p0 is None else p0 * lb
    centers = (lo + hi) / 2 - offset
    init = np.linalg.lstsq(design, centers, rcond=None)[0]
    sigma = max(.05, float(np.std(centers - design @ init)))

    def nll(q):
        lp = log_mass(lo, hi, design @ q[:-1] + offset, np.exp(q[-1]))
        return float(-lp.sum()) if np.isfinite(lp).all() else 1e100

    successes = []
    for multiplier in (.5, 1., 2.):
        q0 = np.r_[init, np.log(sigma * multiplier)]
        bounds = [(None, None)] * design.shape[1] + [(-9., 3.5)]
        for method in ("L-BFGS-B", "Powell"):
            opts = {"maxiter": 5000, "ftol": 1e-12}
            opts["gtol" if method == "L-BFGS-B" else "xtol"] = 1e-9
            r = minimize(nll, q0, bounds=bounds, method=method, options=opts)
            if r.success and np.isfinite(r.fun) and r.fun < 1e90:
                successes.append((float(r.fun), r.x))
    assert successes
    value, q = min(successes, key=lambda z: z[0])
    coefficients = q[:-1].tolist() if p0 is None else [*q[:-1].tolist(), p0]
    return {"params": [*coefficients, float(q[-1])], "nll": value,
            "rank": int(np.linalg.matrix_rank(full_x)),
            "converged_runs": len(successes), "sigma": float(np.exp(q[-1]))}


def main(out_path):
    # Analytical counterexample: every fixed-sigma objective is even and convex
    # in the slope difference, so a free optimum exists at exactly p0.
    x = np.array([-2.0, -1.0, 1.0, 2.0])
    z = np.array([1.0, -1.0, -1.0, 1.0])
    counterexamples = []
    for p0 in (-2.0, -1.2, 0.0, 0.7):
        lo, hi = p0 * x + z - .05, p0 * x + z + .05
        null, alt = fit(x, lo, hi, p0), fit(x, lo, hi)
        lr = 2 * (null["nll"] - alt["nll"])
        delta = .5 * (max(lo - p0 * x) - min(hi - p0 * x))
        assert abs(lr) < 1e-8
        assert abs(alt["b"] - p0) < 1e-6
        assert abs(delta - .95) < 1e-12
        assert not null["scale_bound"] and not alt["scale_bound"]
        counterexamples.append({"p0": p0, "required_hard_tolerance": delta,
                                "profile_likelihood_ratio": lr,
                                "fixed": null, "free": alt})

    # Cross every beta with all three speed folders: unlike fitting one folder,
    # the full original c+a_v*log(v)+b*log(beta) design is identifiable.
    lv = np.repeat(np.log([.05, .10, .20]), 4)
    lb = np.tile(x - 3., 3)
    z12 = np.tile(z, 3)
    full_design = []
    for p0 in (-2., -1.2):
        center = .3 + .8 * lv + p0 * lb + z12
        lo, hi = center - .05, center + .05
        null = fit_original_design(lv, lb, lo, hi, p0)
        alt = fit_original_design(lv, lb, lo, hi)
        lr = 2 * (null["nll"] - alt["nll"])
        assert abs(lr) < 1e-8
        assert abs(alt["params"][2] - p0) < 1e-6
        assert abs(alt["params"][1] - .8) < 1e-6
        assert alt["rank"] == 3
        deltas = []
        for k in range(3):
            sel = slice(k * 4, (k + 1) * 4)
            # Free intercept at each folder makes the nuisance speed shift cancel.
            shifted_lo, shifted_hi = lo[sel] - p0 * lb[sel], hi[sel] - p0 * lb[sel]
            deltas.append(float(.5 * (max(shifted_lo) - min(shifted_hi))))
        assert max(abs(np.asarray(deltas) - .95)) < 1e-12
        full_design.append({"p0": p0, "n": 12, "speeds": 3, "betas_per_speed": 4,
                            "fixed": null, "free": alt, "profile_lr": lr,
                            "required_hard_tolerance_each_folder": deltas})

    full_x = np.column_stack((np.ones(12), lv, lb))
    degeneracy_ranks = {"original_twelve": int(np.linalg.matrix_rank(full_x)),
                        "same_folder_pair": int(np.linalg.matrix_rank(full_x[[0, 1]])),
                        "same_folder_triple": int(np.linalg.matrix_rank(full_x[[0, 1, 2]])),
                        "cross_folder_three": int(np.linalg.matrix_rank(full_x[[0, 1, 4]]))}
    assert degeneracy_ranks == {"original_twelve": 3, "same_folder_pair": 2,
                                 "same_folder_triple": 2, "cross_folder_three": 3}
    # A rank-three model on three rows can exactly interpolate the centers,
    # showing why full rank alone does not exclude sigma->0 degeneracy.
    ids = [0, 1, 4]
    interpolant = np.linalg.solve(full_x[ids], center[ids])
    assert np.max(np.abs(full_x[ids] @ interpolant - center[ids])) < 1e-12

    # Two finite intervals contain a common free regression line. A scale floor
    # would masquerade as a finite MLE, whereas the supremum lies at sigma->0.
    sigmas = [.1, .03, .01, .003]
    nlls = [float(-log_mass(np.array([-.05, -.05]), np.array([.05, .05]), 0., s).sum())
            for s in sigmas]
    assert all(a >= b for a, b in zip(nlls, nlls[1:]))
    assert nlls[-1] < 1e-50

    # Interval likelihood versus independent quadrature, including far tails.
    quadrature_error = 0.0
    for lo, hi, mu, sigma in [(-1.1, -.9, 0., 1.), (4., 4.1, -.1, 1.),
                               (-3., -2.9, .5, .8), (-.02, .03, 0., .2)]:
        q = quad(lambda y: np.exp(-.5 * ((y - mu) / sigma) ** 2) /
                 (sigma * np.sqrt(2 * np.pi)), lo, hi, epsabs=1e-18)[0]
        err = abs(float(log_mass(lo, hi, mu, sigma)) - float(np.log(q)))
        quadrature_error = max(quadrature_error, err)
    assert quadrature_error < 1e-10

    # A fixed-p intercept pair with disjoint intervals always costs >=log(4).
    bound_min_margin = float("inf")
    bound_checks = 0
    for a, sigma in itertools.product(np.linspace(-3., 3., 19), np.geomspace(.03, 30., 23)):
        val = -log_mass(np.array([-1.05, .95]), np.array([-.95, 1.05]), a, sigma).sum()
        bound_min_margin = min(bound_min_margin, float(val - np.log(4.)))
        bound_checks += 1
    assert bound_min_margin >= -1e-12

    # Direct attribution plus remaining-data reoptimization must reconstruct LR.
    x = np.linspace(-2., 2., 9)
    y = .4 + .8 * x + np.array([.4, -.25, .2, -.1, .15, -.3, .05, .15, -.1])
    lo, hi, p0 = y - .04, y + .04, .6
    null, alt = fit(x, lo, hi, p0), fit(x, lo, hi)
    full_lr = 2 * (null["nll"] - alt["nll"])
    deletion_cases, errors = [], []
    for j in range(len(x)):
        mask = np.arange(len(x)) != j
        null2, alt2 = fit(x[mask], lo[mask], hi[mask], p0), fit(x[mask], lo[mask], hi[mask])
        lp0, lp1 = np.asarray(null["lp"]), np.asarray(alt["lp"])
        gain0 = -null2["nll"] - lp0[mask].sum()
        gain1 = -alt2["nll"] - lp1[mask].sum()
        assert min(gain0, gain1) >= -1e-8
        direct = 2 * (lp1[j] - lp0[j])
        reopt = 2 * (gain1 - gain0)
        after = 2 * (null2["nll"] - alt2["nll"])
        reconstruction = full_lr - direct + reopt
        errors.append(abs(after - reconstruction))
        deletion_cases.append({"deleted_index": j, "after_lr": after,
                               "lr_change": after - full_lr,
                               "direct_removed_contrast": direct,
                               "refit_adjustment": float(reopt)})
    assert max(errors) < 1e-10
    assert min(r["lr_change"] for r in deletion_cases) < 0
    assert max(r["lr_change"] for r in deletion_cases) > 0

    result = {"status": "PASS", "synthetic_only": True,
              "hard_conflict_zero_lr": counterexamples,
              "original_three_speed_design": full_design,
              "rank_checks": degeneracy_ranks,
              "free_scale_collapse": {"sigma": sigmas, "nll": nlls},
              "quadrature_max_log_error": quadrature_error,
              "absolute_nll_pair_bound": {"checks": bound_checks,
                                          "minimum_margin": bound_min_margin},
              "deletion_decomposition": {"full_lr": full_lr,
                                          "max_identity_error": max(errors),
                                          "cases": deletion_cases}}
    Path(out_path).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(out_path),
                      "counterexamples": len(counterexamples),
                      "max_identity_error": max(errors)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("synthetic_results.json"))
    main(parser.parse_args().out)
