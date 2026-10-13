#!/usr/bin/env python3
"""Known-prior boundary transport diagnostic; not a new physical scaling law.

Fit one concave quadratic logit in log force using one source speed. Transport
its lower and upper zero crossings with fixed speed exponents. The four
exponent pairs are declared before evaluation, never selected on target labels.
The single-band model deliberately cannot represent disconnected playable sets.
No prospective A2 data are read.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits

import benchmark_data as bd
from benchmark_sets import blocked_folds, binary_iou, evaluate_panels
from set_envelope import label_family

PRIORS = {"uniform": (1., 1.), "static": (0., 0.),
          "asymmetric": (0., 1.), "reverse_falsification": (1., 0.)}
REGULARIZATIONS = (.001, .01, .1, 1.)


def validate_X(X):
    X = np.asarray(X, float)
    if X.ndim != 2 or X.shape[1] != 3 or not np.isfinite(X).all() or np.any(X <= 0):
        raise ValueError("X must contain finite positive beta, force and speed")
    return X


def objective(coef, design, target, regularization):
    """Convex logistic loss with a linear concavity constraint on coef[-1]."""
    logits = design @ coef
    penalty = coef.copy()
    penalty[0] = 0.
    loss = np.mean(np.logaddexp(0., logits) - target * logits)
    loss += .5 * regularization * np.dot(penalty, penalty)
    gradient = design.T @ (expit(logits) - target) / len(target) + regularization * penalty
    return float(loss), gradient


class BoundaryTransport:
    def __init__(self, regularization=.01, reference_speed=.1):
        if not np.isfinite(regularization) or regularization <= 0:
            raise ValueError("regularization must be positive")
        if not np.isfinite(reference_speed) or reference_speed <= 0:
            raise ValueError("reference_speed must be positive")
        self.regularization = float(regularization)
        self.reference_speed = float(reference_speed)

    def _coordinates(self, X):
        X = validate_X(X)
        beta = (np.log(X[:, 0]) - self.beta_center_) / self.beta_scale_
        force = np.log(X[:, 1]) - self.force_center_
        return beta, force

    def design(self, X):
        b, f = self._coordinates(X)
        return np.column_stack((np.ones(len(b)), b, b*b, f, b*f, f*f))

    def fit(self, X, target):
        X = validate_X(X)
        target = np.asarray(target, float)
        if target.shape != (len(X),) or not np.isfinite(target).all() or np.any((target < 0) | (target > 1)):
            raise ValueError("target must be finite in [0,1]")
        if len(X) < 2:
            raise ValueError("at least two source rows are required")
        self.beta_center_ = float(np.mean(np.log(X[:, 0])))
        self.beta_scale_ = float(max(np.std(np.log(X[:, 0])), 1e-8))
        self.force_center_ = float(np.median(np.log(X[:, 1])))
        self.training_force_range_ = tuple(np.log([X[:, 1].min(), X[:, 1].max()]))
        design = self.design(X)
        start = np.zeros(6)
        prevalence = np.clip(target.mean(), 1e-4, 1-1e-4)
        start[0] = np.log(prevalence/(1-prevalence))
        start[-1] = -.1
        result = minimize(objective, start, args=(design, target, self.regularization), jac=True,
                          method="L-BFGS-B", bounds=[(None, None)]*5 + [(None, -1e-6)],
                          options={"maxiter": 1500, "ftol": 1e-12, "gtol": 1e-8})
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f"source fit did not converge: {result.message}")
        self.coef_ = result.x
        self.optimization_ = {"success": bool(result.success), "iterations": int(result.nit),
                              "objective": float(result.fun), "message": str(result.message)}
        return self

    def geometry(self, X, exponents=(0., 1.)):
        b, _ = self._coordinates(X)
        alpha = np.asarray(exponents, float)
        if alpha.shape != (2,) or not np.isfinite(alpha).all():
            raise ValueError("two finite boundary exponents are required")
        c = -self.coef_[-1]
        a = self.coef_[0] + self.coef_[1]*b + self.coef_[2]*b*b
        linear = self.coef_[3] + self.coef_[4]*b
        center = linear/(2*c)
        width_squared = center*center + a/c
        real = width_squared > 0
        log_speed = np.log(np.asarray(X)[:, 2]/self.reference_speed)
        moved_center = center + .5*(alpha[0]+alpha[1])*log_speed
        moved_width = np.sqrt(np.maximum(width_squared, 0.)) + .5*(alpha[1]-alpha[0])*log_speed
        has_band = real & (moved_width > 0)
        # NaN means that the fitted classifier has no positive-width band, not
        # that a physical boundary has been observed or censored there.
        low = np.where(has_band, moved_center-moved_width+self.force_center_, np.nan)
        high = np.where(has_band, moved_center+moved_width+self.force_center_, np.nan)
        return {"curvature": c, "source_width_squared": width_squared,
                "source_has_real_roots": real, "center": moved_center,
                "signed_half_width": moved_width, "has_band": has_band,
                "lower_log_force": low, "upper_log_force": high,
                "crossed_after_transport": real & ~has_band}

    def decision_function(self, X, exponents=(0., 1.)):
        _, force = self._coordinates(X)
        g = self.geometry(X, exponents)
        distance2 = (force-g["center"])**2
        radius2 = g["signed_half_width"]**2
        logits = g["curvature"]*(radius2-distance2)
        # An absent source band cannot be created by translating imaginary roots.
        no_roots = ~g["source_has_real_roots"]
        logits[no_roots] = g["curvature"]*(g["source_width_squared"][no_roots]-distance2[no_roots])
        crossed = g["crossed_after_transport"]
        logits[crossed] = -g["curvature"]*(radius2[crossed]+distance2[crossed])-1e-12
        return logits

    def predict_proba(self, X, exponents=(0., 1.)):
        positive = expit(self.decision_function(X, exponents))
        return np.column_stack((1-positive, positive))


def source_boundary_audit(model, X, target, beta_levels):
    """Observed acquisition edges and multiple components are never filled in."""
    rows = []
    for level in np.unique(beta_levels):
        keep = np.flatnonzero(beta_levels == level)
        keep = keep[np.argsort(X[keep, 1])]
        labels = np.asarray(target[keep], bool)
        transitions = np.diff(np.r_[False, labels, False].astype(int))
        n_components = int(np.count_nonzero(transitions == 1))
        representative = np.array([[np.median(X[keep, 0]), np.median(X[keep, 1]), model.reference_speed]])
        geo = model.geometry(representative, (0., 0.))
        lo, hi = geo["lower_log_force"][0], geo["upper_log_force"][0]
        observed_lo, observed_hi = np.log(X[keep, 1].min()), np.log(X[keep, 1].max())
        rows.append({"beta_level": int(level), "n": len(keep), "positive_components": n_components,
                     "observed_low_edge_positive": bool(labels[0]), "observed_high_edge_positive": bool(labels[-1]),
                     "fitted_lower_log_force": float(lo) if np.isfinite(lo) else None,
                     "fitted_upper_log_force": float(hi) if np.isfinite(hi) else None,
                     "lower_outside_acquisition": bool(np.isfinite(lo) and (lo < observed_lo or lo > observed_hi)),
                     "upper_outside_acquisition": bool(np.isfinite(hi) and (hi < observed_lo or hi > observed_hi)),
                     "boundary_scope": "fitted roots; outside acquisition range means extrapolated, not observed"})
    return rows


def run(output: Path, sources=("A", "B", "C", "D")):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("preserve previous results: output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    # Written before loading any labels/evaluating any targets. Historical target
    # inspection cannot be erased: this is explicitly post-hoc development.
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "post-hoc retrospective known-prior baseline; all sample1 targets previously inspected",
                "not_novel": "Separate force-limit exponents are prior physics, not a newly discovered law.",
                "prior_source": "https://doi.org/10.3813/AAA.918070",
                "hypothesis": "Known asymmetric (lower=0, upper=1) transport may outperform uniform force/speed scaling.",
                "exponents": PRIORS, "regularization_grid": REGULARIZATIONS,
                "selection": "five source-only contiguous-beta folds; pooled nominal IoU; grid-order tie break",
                "training_speed": .1, "sources": sources,
                "source_speed_policy": "nominal 0.1 m/s used for fitting; measured target speed used for transport",
                "label_scope": "fixed A/0.1 observation rule; ambiguous nominal rows remain negative as in set benchmark",
                "structural_restriction": "one concave quadratic band per beta; holes and multiple components cannot be represented",
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    data = bd.load_dataset(include_ambiguous=True)
    X, y = data["X"], data["y_fixed"] == 1
    family = label_family(data["output_features"], data["reference_f0_hz"])
    results = {}
    saved_predictions = {"ids": data["ids"], "nominal": y}
    for source in sources:
        source_mask = (data["string"] == source) & (data["speed"] == .1)
        train_X, train_y = X[source_mask].copy(), y[source_mask]
        train_X[:, 2] = .1
        levels = data["beta_level"][source_mask]
        folds = blocked_folds(levels)
        candidates = []
        for reg in REGULARIZATIONS:
            oof = np.full(len(train_y), np.nan)
            for train, valid in folds:
                model = BoundaryTransport(reg).fit(train_X[train], train_y[train])
                oof[valid] = model.predict_proba(train_X[valid], (0., 0.))[:, 1]
            candidates.append({"regularization": reg, "source_oof_iou": binary_iou(oof >= .5, train_y),
                               "source_oof_brier": float(np.mean((oof-train_y)**2))})
        best = max(range(len(candidates)), key=lambda i: candidates[i]["source_oof_iou"])
        model = BoundaryTransport(candidates[best]["regularization"]).fit(train_X, train_y)
        result = {"source": source+"/0.1", "selection": candidates, "selected_index": best,
                  "training_ids_sha256": bd.id_hash(data["ids"][source_mask]),
                  "coefficient": model.coef_.tolist(), "beta_center": model.beta_center_,
                  "beta_scale": model.beta_scale_, "force_center": model.force_center_,
                  "optimization": model.optimization_,
                  "source_geometry": source_boundary_audit(model, train_X, train_y, levels), "priors": {}}
        for name, exponents in PRIORS.items():
            probability = model.predict_proba(X, exponents)[:, 1]
            geometry = model.geometry(X, exponents)
            test_mask = ~source_mask
            result["priors"][name] = {"exponents": exponents,
                "fitted_no_band_rows": int((~geometry["has_band"] & test_mask).sum()),
                "crossed_root_rows": int((geometry["crossed_after_transport"] & test_mask).sum()),
                **evaluate_panels(probability, y, family, data["group"], test_mask)}
            saved_predictions[source+"_"+name] = probability
        results[source] = result
        (output/"results.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
        print(json.dumps({"source": source, "source_cv_iou": candidates[best]["source_oof_iou"],
                          "macro_nominal_iou": {key: value["macro_panels"]["nominal_iou"]
                                                for key, value in result["priors"].items()}}), flush=True)
    np.savez_compressed(output/"predictions.npz", **saved_predictions)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", choices=list("ABCD"), default=list("ABCD"))
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.output, tuple(args.sources))
