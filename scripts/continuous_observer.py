#!/usr/bin/env python3
"""Exact finite-state IoU extremes over a continuous two-threshold observer.

Labels are eligible & (ratio < fraction) & (periodicity >= cutoff). The same
threshold pair labels every row. A sweep and range-add/min segment tree solve
each fractional-program subproblem. This is an implementation of established
fractional programming/data structures, not a claim to invent those algorithms.
The result is conditional on this observer family, not on mechanical truth.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _add(values, lazy, node, left, right, first, amount):
    if right < first:
        return
    if first <= left:
        values[node] += amount
        lazy[node] += amount
        return
    # Suffix updates need only one partial path; every right sibling entered from
    # a left turn is fully covered. Avoid recursive JIT code: Numba's serialized
    # recursive relocations can crash in a fresh process reusing the cache.
    while left < right:
        middle = (left + right) // 2
        if first <= middle:
            sibling = node * 2 + 1
            values[sibling] += amount
            lazy[sibling] += amount
            node, right = node * 2, middle
        else:
            node, left = node * 2 + 1, middle + 1
    values[node] += amount
    lazy[node] += amount
    node //= 2
    while node >= 1:
        values[node] = lazy[node] + min(values[node * 2], values[node * 2 + 1])
        node //= 2


@njit(cache=True)
def _argmin(values, node, left, right):
    while left < right:
        middle = (left + right) // 2
        if values[node * 2] <= values[node * 2 + 1]:
            node, right = node * 2, middle
        else:
            node, left = node * 2 + 1, middle + 1
    return left


@njit(cache=True)
def _sweep(starts, q, weights, count, q_min, q_max):
    """q sorted descending. Minimize sum(weights*label) over every exact state."""
    tree = np.zeros(4 * count + 8)
    lazy = np.zeros_like(tree)
    i = 0
    while i < len(q) and q[i] >= q_max:
        _add(tree, lazy, 1, 0, count - 1, starts[i], weights[i])
        i += 1
    best = tree[1]
    best_index = _argmin(tree, 1, 0, count - 1)
    best_q = q_max
    # All equal-q trials enter together: no impossible intermediate operator.
    while i < len(q) and q[i] >= q_min:
        cutoff = q[i]
        while i < len(q) and q[i] == cutoff:
            _add(tree, lazy, 1, 0, count - 1, starts[i], weights[i])
            i += 1
        if tree[1] < best:
            best, best_q = tree[1], cutoff
            best_index = _argmin(tree, 1, 0, count - 1)
    return best, best_index, best_q


class ContinuousObserver:
    """Prepare geometry once; evaluate many prediction fields with the same labels."""

    def __init__(self, ratio, periodicity, eligible=None, *, fractions=(.3, .7), cutoffs=(.75, .85)):
        self.ratio = np.asarray(ratio, float)
        self.periodicity = np.asarray(periodicity, float)
        if self.ratio.ndim != 1 or self.periodicity.shape != self.ratio.shape or not len(self.ratio):
            raise ValueError("nonempty aligned one-dimensional feature arrays required")
        eligible = np.ones(len(self.ratio), bool) if eligible is None else np.asarray(eligible, bool)
        if eligible.shape != self.ratio.shape:
            raise ValueError("eligible mask shape differs")
        self.eligible = eligible & np.isfinite(self.ratio) & np.isfinite(self.periodicity)
        if not (len(fractions) == len(cutoffs) == 2 and np.isfinite([*fractions, *cutoffs]).all()
                and fractions[0] <= fractions[1] and cutoffs[0] <= cutoffs[1]):
            raise ValueError("threshold bounds must be finite ordered pairs")
        self.fractions, self.cutoffs = tuple(fractions), tuple(cutoffs)
        can_enter = self.eligible & (self.ratio < fractions[1]) & (self.periodicity >= cutoffs[0])
        rows = np.flatnonzero(can_enter)
        self.order = rows[np.argsort(-self.periodicity[rows], kind="stable")]
        internal = self.ratio[can_enter & (self.ratio >= fractions[0])]
        # Strict '<' changes immediately ABOVE an observed ratio, including the lower endpoint.
        self.f_states = np.unique(np.r_[fractions[0], np.nextafter(internal, np.inf), fractions[1]])
        self.starts = np.searchsorted(self.f_states, self.ratio[self.order], side="right")
        self.q = self.periodicity[self.order]
        self.q_states = np.unique(np.r_[cutoffs, self.q[(self.q >= cutoffs[0]) & (self.q <= cutoffs[1])]])

    @classmethod
    def from_features(cls, features, reference_f0, **kwargs):
        f0 = np.asarray(features["f0_hz"], float)
        count = np.asarray(features["flybacks_per_period"], float)
        valid = np.isfinite(f0) & np.isfinite(count) & (f0 >= .75 * reference_f0) & (count >= 1)
        return cls(features["second_flyback_ratio"], features["periodicity"], valid, **kwargs)

    def labels(self, fraction, cutoff):
        return self.eligible & (self.ratio < fraction) & (self.periodicity >= cutoff)

    def extreme(self, probability, *, maximum=False, sample_weight=None, tolerance=1e-12, maxiter=100):
        """Return globally extremal *soft* IoU (ordinary IoU for binary predictions).

        Exact candidate labels exhaust all states in the rectangle; optimization is
        checked to the declared floating-point residual tolerance. It is not a
        continuum certificate for the underlying string dynamics. Macro means of
        several panel IoUs are NOT covered by this single-ratio solver.
        """
        p = np.asarray(probability, float)
        w = np.ones(len(p)) if sample_weight is None else np.asarray(sample_weight, float)
        if (p.shape != self.ratio.shape or w.shape != p.shape or not np.isfinite(p).all()
                or not np.isfinite(w).all() or np.any((p < 0) | (p > 1)) or np.any(w < 0) or w.max() <= 0):
            raise ValueError("finite probabilities and nonnegative positive-total weights required")
        if tolerance <= 0 or maxiter < 1:
            raise ValueError("positive tolerance and iteration limit required")
        w = w / w.max()
        w = w / w.sum()
        base = float(w @ p)
        if base == 0:
            # Empty-union IoU is 1. For an empty prediction, every nonempty target scores 0.
            fraction = self.fractions[0 if maximum else 1]
            cutoff = self.cutoffs[1 if maximum else 0]
            label = self.labels(fraction, cutoff)
            score = float((w @ label) == 0)
            return {"iou": score, "fraction": fraction, "cutoff": cutoff, "iterations": 0,
                    "residual": 0., "label": label, "candidate_pairs": len(self.f_states)*len(self.q_states)}
        value = 0. if maximum else 1.
        sign = -1. if maximum else 1.
        for iteration in range(1, maxiter + 1):
            weights = sign * w * (p - value * (1-p))
            _, index, cutoff = _sweep(self.starts, self.q, weights[self.order], len(self.f_states), *self.cutoffs)
            fraction = float(self.f_states[index])
            label = self.labels(fraction, cutoff)
            numerator = float(w @ (p * label))
            denominator = base + float(w @ ((1-p) * label))
            residual = numerator - value * denominator
            updated = numerator / denominator
            # Recompute on the actual winning map, independent of the segment-tree sum.
            if abs(residual) <= tolerance * base:
                return {"iou": updated, "fraction": fraction, "cutoff": float(cutoff),
                        "iterations": iteration, "residual": residual, "label": label,
                        "ratio_gap_bound_exact_arithmetic": abs(residual) / base,
                        "candidate_pairs": len(self.f_states)*len(self.q_states)}
            value = updated
        raise RuntimeError("fractional-program iteration limit reached; no optimum was certified")
