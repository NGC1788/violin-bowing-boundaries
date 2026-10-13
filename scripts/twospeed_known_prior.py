#!/usr/bin/env python3
"""Established asymmetric boundary prior on the published two-speed protocol.

Average two single-speed concave-band fits. Static/uniform/asymmetric priors
share folds and regularization candidates. This is a baseline, not a new method.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

import affine_transport as af
import benchmark_data as bd
from benchmark_sets import evaluate_panels, metrics
from boundary_transport import BoundaryTransport, REGULARIZATIONS
import multispeed_scaling as ms

PRIORS = {"static": (0., 0.), "uniform": (1., 1.), "asymmetric": (0., 1.)}


def freeze(model):
    return {"regularization": model.regularization, "reference_speed": model.reference_speed,
            "beta_center": model.beta_center_, "beta_scale": model.beta_scale_, "force_center": model.force_center_,
            "coef": model.coef_.tolist(), "training_force_range": model.training_force_range_,
            "optimization": model.optimization_}


def thaw(payload):
    model = BoundaryTransport(payload["regularization"], payload["reference_speed"])
    for name in ("beta_center", "beta_scale", "force_center", "training_force_range"):
        setattr(model, name + "_", payload[name])
    model.coef_ = np.asarray(payload["coef"])
    return model


def fit_pair(X, y, speeds, regularization):
    return [freeze(BoundaryTransport(regularization, float(speed)).fit(X[speeds == speed], y[speeds == speed]))
            for speed in sorted(np.unique(speeds))]


def predict_pair(models, X, exponents):
    if len(models) != 2:
        raise ValueError("exactly two source-speed models required")
    return np.mean([thaw(model).predict_proba(X, exponents)[:, 1] for model in models], axis=0)


def select_source(X, y, family, levels, speeds, ids, log_path, *, regularizations=REGULARIZATIONS):
    folds = ms.two_speed_folds(levels, speeds)
    candidates = []
    for reg in regularizations:
        oof = {name: np.full(len(y), np.nan) for name in PRIORS}
        for fold, (train, valid) in enumerate(folds):
            models = fit_pair(X[train], y[train], speeds[train], reg)
            scores = {}
            for name, exponents in PRIORS.items():
                oof[name][valid] = predict_pair(models, X[valid], exponents)
                scores[name] = metrics(oof[name][valid], y[valid], family[valid])
            ms.append_jsonl(log_path, {"regularization": reg, "fold": fold, "models": models,
                                       "training_ids_sha256": bd.id_hash(ids[train]),
                                       "validation_ids_sha256": bd.id_hash(ids[valid]),
                                       "validation_beta_levels": sorted(np.unique(levels[valid]).tolist()), "metrics": scores})
        candidates.append({"regularization": reg, "metrics": {name: metrics(probability, y, family)
                                                               for name, probability in oof.items()}})
    selected = {name: max(range(len(candidates)), key=lambda i: candidates[i]["metrics"][name]["family_min_iou"])
                for name in PRIORS}
    models = {str(i): fit_pair(X, y, speeds, candidates[i]["regularization"]) for i in sorted(set(selected.values()))}
    return {"candidates": candidates, "selected": selected, "models_by_candidate": models}


def run(output, flow_results):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("preserve existing runs")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "post-hoc known-prior comparator",
                "motivation": "Flow must be compared with the established different lower/upper speed dependencies.",
                "priors": PRIORS, "regularizations": REGULARIZATIONS, "ensemble": "arithmetic mean of two source-speed probabilities",
                "selection": "five shared contiguous beta blocks; pooled source OOF nominal IoU; one lambda per ensemble/prior",
                "target": "other two D-monochord speeds only; no E generalization score",
                "not_matched_complexity": "two six-parameter bands (12 base coefficients), versus nine total flow parameters",
                "all_models_frozen_before_target_scoring": True, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "flow_results_sha256": hashlib.sha256(Path(flow_results).read_bytes()).hexdigest()}
    ms.write_json(output / "protocol.json", protocol, exclusive=True)
    data = af.load_published()
    X, y, family = data["X"], data["y_fixed"] == 1, data["family"]
    ms.write_json(output / "input_manifest.json", data["manifest"], exclusive=True)
    frozen = {}
    for pair in itertools.combinations((.05, .1, .15, .2), 2):
        train, scopes = ms.build_masks(data, "D4", pair)
        key = f"published__D4__{pair[0]:g}_{pair[1]:g}"
        directory = output / key
        directory.mkdir()
        selection = select_source(X[train], y[train], family[train], data["beta_level"][train], data["speed"][train],
                                  data["ids"][train], directory / "source_cv.jsonl")
        payload = {"training_speeds": pair, "training_ids": data["ids"][train].tolist(),
                   "training_ids_sha256": bd.id_hash(data["ids"][train]),
                   "evaluation_ids_sha256": bd.id_hash(data["ids"][scopes["source_heldout_speed"]]), "selection": selection}
        path = directory / "frozen_models.json"
        ms.write_json(path, payload, exclusive=True)
        frozen[key] = (path, scopes["source_heldout_speed"])
    ms.write_json(output / "all_models_frozen.json", {key: {"path": str(path.relative_to(output)),
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for key, (path, _) in frozen.items()}, exclusive=True)
    # Previous flow target outcomes are only loaded after all known-prior fits freeze.
    previous = json.loads(Path(flow_results).read_text())
    results = {}
    for key, (path, test) in frozen.items():
        payload = json.loads(path.read_text())
        selection = payload["selection"]
        results[key] = {"training_speeds": payload["training_speeds"], "methods": {}}
        for name, exponents in PRIORS.items():
            index = selection["selected"][name]
            probability = predict_pair(selection["models_by_candidate"][str(index)], X, exponents)
            results[key]["methods"][name + "_ensemble"] = {"regularization": selection["candidates"][index]["regularization"],
                **evaluate_panels(probability, y, family, data["group"], test)}
        for method, record in previous[key]["methods"].items():
            results[key]["methods"][method] = record["source_heldout_speed"]
    ms.write_json(output / "results.json", results, exclusive=True)
    methods = list(next(iter(results.values()))["methods"])
    summary = {"status": protocol["status"], "source_tasks": len(results),
               "scope": protocol["target"], "overlap_warning": "six speed pairs are not independent repetitions",
               "mean_macro_nominal_iou": {method: float(np.mean([row["methods"][method]["macro_panels"]["nominal_iou"]
                    for row in results.values()])) for method in methods},
               "all_tasks": {key: {method: value["macro_panels"]["nominal_iou"] for method, value in row["methods"].items()}
                             for key, row in results.items()}, "seconds": time.monotonic() - started}
    flow_key, known_key = "affine_flow", "asymmetric_ensemble"
    difference = [row[flow_key] - row[known_key] for row in summary["all_tasks"].values()]
    summary["flow_minus_known_asymmetric"] = {"mean": float(np.mean(difference)),
                                               "wins": int(np.sum(np.array(difference) > 1e-12)),
                                               "losses": int(np.sum(np.array(difference) < -1e-12))}
    ms.write_json(output / "summary.json", summary, exclusive=True)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flow-results", type=Path, default=af.ROOT / "reports/affine_transport/2026-10-13-v2/results.json")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, args.flow_results)
