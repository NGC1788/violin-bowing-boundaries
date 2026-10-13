"""Exhaustive finite checks for method_formalization.md (Python stdlib only).

This verifies algebra on finite examples; it is not an empirical research result.
"""

from __future__ import annotations

import itertools
import json
import math
import random


def ratio(numerator, denominator):
    return numerator / denominator if denominator else 1.0


def iou(pred, target, weights):
    return ratio(
        sum(w for p, t, w in zip(pred, target, weights) if p and t),
        sum(w for p, t, w in zip(pred, target, weights) if p or t),
    )


def soft_iou(pred, target, weights):
    return ratio(
        sum(w * p * t for p, t, w in zip(pred, target, weights)),
        sum(w * (p + t - p * t) for p, t, w in zip(pred, target, weights)),
    )


def complete(lower, upper):
    ambiguous = [i for i, (lo, hi) in enumerate(zip(lower, upper)) if not lo and hi]
    for labels in itertools.product((0, 1), repeat=len(ambiguous)):
        target = list(lower)
        for index, label in zip(ambiguous, labels):
            target[index] = label
        yield target


def prefix_minimum(pred, lower, upper, weights):
    ambiguous = [i for i, (lo, hi) in enumerate(zip(lower, upper)) if not lo and hi]
    ambiguous.sort(key=lambda i: pred[i] / (1 - pred[i]) if pred[i] < 1 else math.inf)
    target = list(lower)
    best = soft_iou(pred, target, weights)
    for index in ambiguous:
        target[index] = 1
        best = min(best, soft_iou(pred, target, weights))
    return best


def main():
    rng = random.Random(20261013)
    configurations = 0
    completions = 0
    largest_error = 0.0
    for n in range(1, 9):
        for _ in range(30):
            states = [rng.randrange(3) for _ in range(n)]
            lower = [int(s == 2) for s in states]
            upper = [int(s > 0) for s in states]
            pred = [rng.randrange(2) for _ in range(n)]
            soft = [rng.random() for _ in range(n)]
            weights = [rng.uniform(0.2, 2) for _ in range(n)]
            targets = list(complete(lower, upper))
            scores = [iou(pred, t, weights) for t in targets]
            lower_bound = ratio(
                sum(w for p, lo, w in zip(pred, lower, weights) if p and lo),
                sum(w for p, hi, w in zip(pred, upper, weights) if p or hi),
            )
            upper_bound = ratio(
                sum(w for p, hi, w in zip(pred, upper, weights) if p and hi),
                sum(w for p, lo, w in zip(pred, lower, weights) if p or lo),
            )
            exhaustive_soft = min(soft_iou(soft, t, weights) for t in targets)
            errors = (
                abs(min(scores) - lower_bound),
                abs(max(scores) - upper_bound),
                abs(exhaustive_soft - prefix_minimum(soft, lower, upper, weights)),
            )
            assert max(errors) < 1e-12, (n, states, errors)
            largest_error = max(largest_error, *errors)
            completions += len(targets)
            configurations += 1

    # Coherent family: all n+1 cells positive except one of the n uncertain cells.
    n = 99
    pred = [1] * (n + 1)
    family = [[1] + [int(k != j) for k in range(n)] for j in range(n)]
    coherent = min(iou(pred, target, [1] * (n + 1)) for target in family)
    rectangular = 1 / (n + 1)
    assert math.isclose(coherent, n / (n + 1))

    print(json.dumps({
        "status": "pass",
        "scope": "finite algebra checks, not data validation or a novelty claim",
        "random_seed": 20261013,
        "configurations": configurations,
        "exhaustive_completions": completions,
        "largest_numerical_error": largest_error,
        "coherent_gap_example": {
            "cells": n + 1, "operators": n,
            "coherent_worst_iou": coherent,
            "rectangular_worst_iou": rectangular,
        },
    }, indent=2))


if __name__ == "__main__":
    main()
