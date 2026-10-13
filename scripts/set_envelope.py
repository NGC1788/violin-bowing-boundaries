"""Topology-free weak-label set learning in physical similarity coordinates.

Research prototype, not a validated physical regime detector. Label envelopes
describe a specified family of observation rules, not coverage of physical truth.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp


def similarity_coordinates(X, covariant=True, speed_exponent=1.0):
    """Input columns: beta, force in N, velocity in m/s; all strictly positive.

    Logarithms use fixed reference units beta=.1, force=1 N, speed=.1 m/s.
    Speed exponent is a declared physical prior, not identifiable at one speed.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3 or not np.isfinite(X).all() or np.any(X <= 0):
        raise ValueError("Expected finite positive beta, force and speed columns")
    beta, force, speed = np.log(X / np.array([0.1, 1.0, 0.1])).T
    return np.column_stack((beta, force - speed_exponent * speed)) if covariant else np.column_stack((beta, force, speed))


def label_family(features, reference_f0, fractions=(.3, .4, .5, .6, .7), periodicities=(.75, .8, .85)):
    """Reapply saved flyback ratios; does not retune hysteresis or recover raw signals."""
    per = np.asarray(features["periodicity"], float)
    f0 = np.asarray(features["f0_hz"], float)
    ratio = np.asarray(features["second_flyback_ratio"], float)
    count = np.asarray(features["flybacks_per_period"], float)
    valid = np.isfinite(per) & np.isfinite(f0) & np.isfinite(ratio) & (count >= 1)
    family = np.stack([valid & (per >= gate) & (f0 >= .75 * reference_f0) & (ratio < frac)
                       for frac in fractions for gate in periodicities], axis=1)
    return family


def label_envelope(features, reference_f0, fractions=(.3, .4, .5, .6, .7), periodicities=(.75, .8, .85)):
    family = label_family(features, reference_f0, fractions, periodicities)
    return family.all(axis=1), family.any(axis=1), family.mean(axis=1)


def family_ious(prediction, family, weights=None):
    p, family = np.asarray(prediction, bool), np.asarray(family, bool)
    if family.ndim != 2 or family.shape[0] != len(p):
        raise ValueError("Family shape mismatch")
    w = np.ones(len(p)) if weights is None else np.asarray(weights, float)
    numerator = np.sum(w[:, None] * (p[:, None] & family), axis=0)
    denominator = np.sum(w[:, None] * (p[:, None] | family), axis=0)
    return np.divide(numerator, denominator, out=np.ones_like(numerator), where=denominator > 0)


def iou_bounds(prediction, lower, upper, weights=None):
    """Exact extrema over all binary truth sets lower subset truth subset upper.

    Empty-set IoU is one when both sets are empty. Positive weights are fixed
    before examining the prediction. These are sensitivity bounds, not CIs.
    """
    p, lo, hi = (np.asarray(x, bool) for x in (prediction, lower, upper))
    if not (p.shape == lo.shape == hi.shape) or np.any(lo & ~hi):
        raise ValueError("Incompatible sets or lower is not a subset of upper")
    w = np.ones(p.shape) if weights is None else np.asarray(weights, float)
    if w.shape != p.shape or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("Weights must be finite positive and match the sets")
    union_max = np.sum(w * (p | hi))
    union_min = np.sum(w * (p | lo))
    return (float(np.sum(w * (p & lo)) / union_max) if union_max else 1.0,
            float(np.sum(w * (p & hi)) / union_min) if union_min else 1.0)


class SetEnvelopeModel:
    """RBF logit on similarity coordinates, with BCE or smooth worst-case IoU.

    Uses every observed cell, with no longest-run selection, gap filling or
    extrapolated midpoint boundary labels. No one-interval topology is imposed.
    All scaling and basis centers are fitted using training inputs only.
    """
    def __init__(self, objective="bce", covariant=True, regularization=.01, length_scale=.7,
                 speed_exponent=1., grid_size=9, maxiter=250):
        if objective not in {"bce", "robust_iou", "family_iou", "nominal_iou"}:
            raise ValueError("Unknown objective")
        self.objective, self.covariant = objective, covariant
        self.regularization, self.length_scale = regularization, length_scale
        self.speed_exponent, self.grid_size, self.maxiter = speed_exponent, grid_size, maxiter

    def _basis(self, X):
        raw = similarity_coordinates(X, self.covariant, self.speed_exponent)
        z = (raw - self.center_) / self.scale_
        distance = np.sum((z[:, None, :] - self.anchors_[None, :, :]) ** 2, axis=2)
        radial = np.exp(-distance / (2 * self.length_scale ** 2))
        return np.column_stack((np.ones(len(z)), z, radial))

    def fit(self, X, y, lower=None, upper=None, sample_weight=None, family=None):
        raw = similarity_coordinates(X, self.covariant, self.speed_exponent)
        y = np.asarray(y, float)
        if y.shape != (len(raw),) or not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
            raise ValueError("Training targets must be finite probabilities")
        self.center_ = np.mean(raw, axis=0)
        scale = np.std(raw, axis=0)
        self.scale_ = np.where(scale > .05, scale, 1.)
        z = (raw - self.center_) / self.scale_
        grids = [np.unique(np.quantile(z[:, j], np.linspace(0, 1, self.grid_size))) for j in range(z.shape[1])]
        # Constant training speed stays a single anchor axis, not many duplicates.
        grids = [np.array([0.]) if scale[j] < .05 else g for j, g in enumerate(grids)]
        self.anchors_ = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1).reshape(-1, z.shape[1])
        B = self._basis(X)
        weight = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        if weight.shape != y.shape or not np.isfinite(weight).all() or np.any(weight < 0) or weight.sum() <= 0:
            raise ValueError("Invalid training weights")
        weight = weight / weight.sum()
        lo = y if lower is None else np.asarray(lower, float)
        hi = y if upper is None else np.asarray(upper, float)
        if lo.shape != y.shape or hi.shape != y.shape or np.any(lo > hi) or np.any(lo < 0) or np.any(hi > 1):
            raise ValueError("Invalid label envelope")
        penalty = np.ones(B.shape[1]); penalty[0] = 0
        family = y[:, None] if family is None else np.asarray(family, float)
        if family.ndim != 2 or family.shape[0] != len(y) or not np.isfinite(family).all() or np.any((family<0)|(family>1)):
            raise ValueError("Invalid observation-rule family")

        def loss_gradient(coef, objective):
            score = B @ coef
            probability = expit(score)
            bce = np.sum(weight * (np.logaddexp(0, score) - y * score))
            grad = B.T @ (weight * (probability - y))
            loss = bce
            if objective == "robust_iou":
                numerator = np.sum(weight * probability * lo)
                denominator = max(np.sum(weight * (hi + probability * (1-hi))), 1e-12)
                derivative = -weight * (lo * denominator - numerator * (1-hi)) / denominator**2
                # Fixed small proper-loss anchor improves finite-sample optimisation.
                loss = -numerator / denominator + .1 * bce
                grad = B.T @ (derivative * probability * (1-probability)) + .1 * grad
            elif objective in {"family_iou", "nominal_iou"}:
                targets = y[:, None] if objective == "nominal_iou" else family
                numerator = np.sum(weight[:,None] * probability[:,None] * targets, axis=0)
                denominator = np.maximum(np.sum(weight[:,None] * (targets + probability[:,None]*(1-targets)), axis=0), 1e-12)
                jaccard = numerator / denominator
                tau = .05  # fixed smooth minimum temperature; no target-panel tuning
                adversary = np.exp(-jaccard/tau - logsumexp(-jaccard/tau))
                derivative = -weight[:,None] * (targets*denominator - (1-targets)*numerator) / denominator**2
                loss = tau * (logsumexp(-jaccard/tau) - np.log(targets.shape[1])) + .1*bce
                grad = B.T @ ((derivative@adversary) * probability*(1-probability)) + .1*grad
            loss += .5 * self.regularization * np.sum(penalty * coef**2)
            grad += self.regularization * penalty * coef
            return float(loss), grad

        initial = np.zeros(B.shape[1])
        result = minimize(lambda coef: loss_gradient(coef, "bce"), initial, jac=True, method="L-BFGS-B",
                          options={"maxiter": self.maxiter, "ftol": 1e-10})
        if self.objective != "bce":
            result = minimize(lambda coef: loss_gradient(coef, self.objective), result.x, jac=True, method="L-BFGS-B",
                              options={"maxiter": self.maxiter, "ftol": 1e-10})
        self.coef_, self.optimization_ = result.x, {"success": bool(result.success), "message": str(result.message),
            "iterations": int(result.nit), "objective": float(result.fun)}
        return self

    def predict_proba(self, X):
        p = expit(self._basis(X) @ self.coef_)
        return np.column_stack((1-p, p))

    def predict(self, X):
        return self.predict_proba(X)[:, 1] >= .5
