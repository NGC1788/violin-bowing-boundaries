"""Pilot: how many bow strokes does it take to recover a playability map?

The robot grids are exhaustive (40 positions x 50 forces per speed), so any
sequential measurement design can be simulated offline: a "measurement" is a
lookup of the existing label. We compare designs by the IoU of the predicted
Helmholtz region against the full-grid labels, as a function of strokes used.

Model: quadratic logistic in speed-normalised physics coordinates
u = log(beta), w = log(F / v). A band between two boundary lines is a
quadratic logit, so this one class covers the Schelleng picture.
"""

import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path.home() / "violin-bowing-boundaries"
GRIDS = {
    "A": "reports/collection/2024-03-25_TypeA_sample1/regime_map/20260917T135154Z_3982fcc7/regimes.csv",
    "B": "reports/collection/2024-04-10_TypeB_sample1__2024-04-10/regime_map/20260917T154209Z_aa60a05c/regimes.csv",
    "C": "reports/collection/2024-04-12_TypeC_sample1__2024-04-15/regime_map/20260917T163822Z_54d11e71/regimes.csv",
    "D": "reports/collection/2024-04-15_TypeD_sample1__2024-04-17/regime_map/20260917T172527Z_6dbcf479/regimes.csv",
}
BUDGETS = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
SEEDS = 10
TOL = 0.02  # roughly the spread the observer-rule audit found between plausible label rules


def load(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def panels(rows: list[dict]) -> dict:
    """condition -> (U, W, y) for usable cells."""
    out = {}
    for condition in sorted({r["condition"] for r in rows}):
        sub = [r for r in rows if r["condition"] == condition
               and float(r["c1_window_mean"]) > 0 and float(r["c2_window_mean"]) > 0
               and str(r.get("window_inside_plateau", "True")) in ("True", "true", "1")
               and r["label"] != "ambiguous"]
        beta = np.array([float(r["beta"]) for r in sub])
        force = np.array([float(r["c1_window_mean"]) for r in sub])
        speed = np.array([float(r["c2_window_mean"]) for r in sub])
        y = np.array([r["label"] == "helmholtz" for r in sub], dtype=float)
        out[condition] = (np.log(beta), np.log(force / speed), y, float(np.median(speed)))
    return out


# Fixed standardisation so that weights mean the same thing on every string.
U0, US, W0, WS = np.log(0.05), 0.7, np.log(1.0 / 0.1), 1.2


def features(u, w):
    a, b = (u - U0) / US, (w - W0) / WS
    return np.stack([np.ones_like(a), a, b, a * a, a * b, b * b], axis=1)


def fit(X, y, mean, precision, iters=200):
    """MAP logistic regression with a Gaussian prior N(mean, precision^-1)."""
    theta = mean.copy()
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ theta, -30, 30)))
        grad = X.T @ (p - y) + precision * (theta - mean)
        hess = (X * (p * (1 - p))[:, None]).T @ X + precision * np.eye(len(theta))
        step = np.linalg.solve(hess, grad)
        norm = np.max(np.abs(step))
        if norm > 2.0:                       # damped Newton: no wild jumps on near-separable samples
            step *= 2.0 / norm
        theta -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return theta


def iou(pred, y):
    inter = np.sum(pred & (y > 0.5))
    union = np.sum(pred | (y > 0.5))
    return inter / union if union else 1.0


def lattice(n_cells, u, w, n):
    """Evenly spread cells: what an experimenter would do with a fixed budget."""
    order = np.lexsort((w, u))
    pick = np.linspace(0, n_cells - 1, n).round().astype(int)
    return list(order[pick])


WEAK = np.full(6, 1e-2)


def run_design(X, y, design, budget, rng, prior_mean=None, prior_prec=None):
    n = len(y)
    mean = np.zeros(6) if prior_mean is None else prior_mean
    prec = WEAK if prior_prec is None else prior_prec
    if design == "lattice":
        chosen = lattice(n, X[:, 1], X[:, 2], budget)
    elif design == "random":
        chosen = list(rng.choice(n, size=budget, replace=False))
    else:  # uncertainty sampling, optionally starting from a transferred prior
        start = min(8, budget)
        chosen = lattice(n, X[:, 1], X[:, 2], start)
        fill = [c for c in lattice(n, X[:, 1], X[:, 2], 64) if c not in chosen]
        while len(chosen) < budget:
            if prior_mean is not None and len(chosen) % 4 == 0 and fill:   # keep some coverage
                chosen.append(fill.pop(0))
                continue
            handoff = prior_mean is not None and len(chosen) >= 16
            theta = fit(X[chosen], y[chosen], np.zeros(6) if handoff else mean, WEAK if handoff else prec)
            p = 1 / (1 + np.exp(-np.clip(X @ theta, -30, 30)))
            score = p * (1 - p)
            score[chosen] = -1
            chosen.append(int(np.argmax(score)))
    handoff = prior_mean is not None and len(chosen) >= 16
    theta = fit(X[chosen], y[chosen], np.zeros(6) if handoff else mean, WEAK if handoff else prec)
    return iou((X @ theta) > 0, y)


def main() -> None:
    data = {name: panels(load(REPO / rel)) for name, rel in GRIDS.items()}
    a2 = REPO / "research/2026-10-14/lockbox_A2/regimes.csv.gz"
    if a2.exists():
        try:
            data["A2"] = panels(load(a2))
        except KeyError as error:
            print(f"A2 skipped: missing column {error}", file=sys.stderr)

    results = []
    for target, conditions in data.items():
        # Prior from every other string, all speeds (leave-one-string-out).
        donors = [(u, w, y) for name, conds in data.items() if name != target and not
                  (target == "A2" and name == "A") and not (target == "A" and name == "A2")
                  for (u, w, y, _) in conds.values()]
        Xd = np.concatenate([features(u, w) for u, w, _ in donors])
        yd = np.concatenate([y for _, _, y in donors])
        prior = fit(Xd, yd, np.zeros(6), WEAK)
        for condition, (u, w, y, speed) in conditions.items():
            X = features(u, w)
            ceiling = iou((X @ fit(X, y, np.zeros(6), WEAK)) > 0, y)
            zero_shot = iou((X @ prior) > 0, y)
            row = {"target": target, "condition": condition, "speed": round(speed, 3),
                   "cells": int(len(y)), "ceiling": ceiling, "zero_shot": zero_shot, "curves": {}}
            for design, kwargs in (("lattice", {}), ("random", {}), ("active", {}),
                                   ("active+transfer", {"prior_mean": prior, "prior_prec": np.full(6, 2.0)})):
                base = design.split("+")[0]
                curve = []
                for budget in BUDGETS:
                    reps = SEEDS if base in ("random",) else 1
                    scores = [run_design(X, y, base, budget, np.random.default_rng(s), **kwargs) for s in range(reps)]
                    curve.append(float(np.mean(scores)))
                row["curves"][design] = curve
            results.append(row)
            print(f"{target:2} {condition:16} v={speed:.2f} cells={len(y)} ceiling={ceiling:.3f} zero-shot={zero_shot:.3f}",
                  flush=True)

    out = Path(__file__).with_name("few_stroke_results.json")
    out.write_text(json.dumps({"budgets": BUDGETS, "tol": TOL, "results": results}, indent=1))

    print("\n=== 패널 평균 IoU (예산별) ===")
    designs = ["lattice", "random", "active", "active+transfer"]
    print("strokes  " + "  ".join(f"{d:>16}" for d in designs))
    for i, budget in enumerate(BUDGETS):
        print(f"{budget:7}  " + "  ".join(f"{np.mean([r['curves'][d][i] for r in results]):16.3f}" for d in designs))
    print(f"ceiling(전체 {np.mean([r['cells'] for r in results]):.0f}칸 사용) {np.mean([r['ceiling'] for r in results]):.3f}"
          f" | zero-shot(0칸) {np.mean([r['zero_shot'] for r in results]):.3f}")

    print(f"\n=== 천장-{TOL} 도달에 필요한 긋기 수 (패널 중앙값) ===")
    for d in designs:
        need = []
        for r in results:
            hit = [b for b, s in zip(BUDGETS, r["curves"][d]) if s >= r["ceiling"] - TOL]
            need.append(hit[0] if hit else np.inf)
        reached = sum(np.isfinite(need))
        print(f"{d:16} 중앙값 {np.median(need):>6}  도달 {reached}/{len(need)} 패널")


if __name__ == "__main__":
    main()
