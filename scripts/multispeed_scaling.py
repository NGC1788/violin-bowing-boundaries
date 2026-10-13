#!/usr/bin/env python3
"""Source-only exponent selection from two observed speeds, retrospective sample1.

Known dimensional scaling is a baseline, not a novelty claim. All twelve source
fits and their immutable parameter files are completed before any target scoring.
The prospective A2 archive is never opened by this program.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import platform
import time
import warnings

import numpy as np
import sklearn
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits

import benchmark_data as bd
from benchmark_sets import blocked_folds, evaluate_panels, metrics
from set_envelope import label_family, similarity_coordinates

EXPONENTS = tuple(float(x) for x in np.arange(0., 2.00001, .25))
CS = (.1, 1., 10.)
SPEED_PAIRS = ((.05, .1), (.05, .2), (.1, .2))
METHODS = ("static_alpha0", "fixed_alpha1", "learned_alpha", "raw_3d")


def write_json(path, payload, *, exclusive=False):
    with Path(path).open("x" if exclusive else "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def append_jsonl(path, payload):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, allow_nan=False) + "\n")


def two_speed_folds(levels, nominal_speeds, n_splits=5):
    """The same beta column is held out across *both* source speeds together."""
    levels, speeds = np.asarray(levels), np.asarray(nominal_speeds, float)
    if levels.shape != speeds.shape or np.unique(speeds).size != 2:
        raise ValueError("exponent selection requires exactly two nominal source speeds")
    folds = blocked_folds(levels, n_splits)
    for train, valid in folds:
        if np.intersect1d(levels[train], levels[valid]).size:
            raise AssertionError("beta-column leakage")
        if np.unique(speeds[train]).size != 2 or np.unique(speeds[valid]).size != 2:
            raise ValueError("each source training and validation fold must contain both speeds")
    return folds


def coordinates(X, alpha):
    return similarity_coordinates(X, covariant=alpha is not None, speed_exponent=alpha if alpha is not None else 1.)


def fit_estimator(X, y, *, alpha, C):
    target = np.asarray(y, int)
    if np.unique(target).size != 2:
        raise ValueError("source fit requires both binary classes")
    estimator = make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(),
                              LogisticRegression(C=C, max_iter=2000, tol=1e-8, random_state=0))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator.fit(coordinates(X, alpha), target)
    return estimator


def freeze_estimator(estimator, alpha, C):
    poly, scaler, logistic = estimator.steps[0][1], estimator.steps[1][1], estimator.steps[2][1]
    return {"alpha": alpha, "C": C, "polynomial_powers": poly.powers_.tolist(),
            "polynomial_feature_mean": scaler.mean_.tolist(), "polynomial_feature_scale": scaler.scale_.tolist(),
            "logistic_coefficient": logistic.coef_[0].tolist(), "logistic_intercept": float(logistic.intercept_[0]),
            "solver_iterations": int(logistic.n_iter_[0]), "classes": logistic.classes_.tolist()}


def predict_frozen(model, X):
    """Reconstruct predictions from JSON coefficients, without a pickled estimator."""
    z = coordinates(X, model["alpha"])
    powers = np.asarray(model["polynomial_powers"], int)
    design = np.prod(z[:, None, :] ** powers[None, :, :], axis=2)
    design = (design - np.asarray(model["polynomial_feature_mean"])) / np.asarray(model["polynomial_feature_scale"])
    return expit(design @ np.asarray(model["logistic_coefficient"]) + model["logistic_intercept"])


def select_source(X, nominal, family, levels, speeds, ids, log_path, *, exponents=EXPONENTS, cs=CS):
    """Accept source arrays only; shared candidates avoid duplicate baseline fits."""
    if not 0. in exponents or not 1. in exponents:
        raise ValueError("the alpha grid must include the declared static and fixed baselines")
    folds = two_speed_folds(levels, speeds)
    candidates = []
    for alpha, C in itertools.chain(itertools.product(exponents, cs), ((None, c) for c in cs)):
        probability = np.full(len(X), np.nan)
        for fold, (train, valid) in enumerate(folds):
            started = time.monotonic()
            estimator = fit_estimator(X[train], nominal[train], alpha=alpha, C=C)
            probability[valid] = estimator.predict_proba(coordinates(X[valid], alpha))[:, 1]
            append_jsonl(log_path, {"phase": "source_cv", "alpha": alpha, "C": C, "fold": fold,
                                    "train_ids_sha256": bd.id_hash(ids[train]),
                                    "validation_ids_sha256": bd.id_hash(ids[valid]),
                                    "validation_beta_levels": sorted(np.unique(levels[valid]).tolist()),
                                    "training_speeds": sorted(np.unique(speeds[train]).tolist()),
                                    "validation_speeds": sorted(np.unique(speeds[valid]).tolist()),
                                    "metrics": metrics(probability[valid], nominal[valid], family[valid]),
                                    "fit": freeze_estimator(estimator, alpha, C),
                                    "seconds": time.monotonic() - started})
        if not np.isfinite(probability).all():
            raise AssertionError("each source row must receive exactly one out-of-fold probability")
        candidates.append({"alpha": alpha, "C": C, "metrics": metrics(probability, nominal, family)})
    allowed = {"static_alpha0": lambda c: c["alpha"] == 0.,
               "fixed_alpha1": lambda c: c["alpha"] == 1.,
               "learned_alpha": lambda c: c["alpha"] is not None,
               "raw_3d": lambda c: c["alpha"] is None}
    selected = {name: max((i for i, candidate in enumerate(candidates) if rule(candidate)),
                          key=lambda i: candidates[i]["metrics"]["family_min_iou"])
                for name, rule in allowed.items()}
    return candidates, selected


def build_masks(data, source, pair):
    pair = tuple(float(value) for value in pair)
    if len(pair) != 2 or len(set(pair)) != 2:
        raise ValueError("choose two different source speeds")
    source_rows = data["string"] == source
    train = source_rows & np.isin(data["speed"], pair)
    within_source = source_rows & ~np.isin(data["speed"], pair)
    other_specimens = ~source_rows
    if not train.any() or not within_source.any() or not other_specimens.any():
        raise ValueError("source training or evaluation partition is empty")
    return train, {"combined": ~train, "source_heldout_speed": within_source,
                   "other_specimens": other_specimens}


def run(output, *, sources=("A", "B", "C", "D"), pairs=SPEED_PAIRS, dataset=None, exponents=EXPONENTS, cs=CS):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("preserve previous results: output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "post-hoc retrospective development; all sample1 specimens already inspected",
                "motivation": "Single-speed fits cannot identify a speed exponent; use two source speeds before evaluating a third.",
                "not_novel": "Dimensional scaling and polynomial logistic regression are established baselines.",
                "source_specimens": sources, "training_speed_pairs": pairs, "alpha_grid": exponents, "C_grid": cs,
                "methods": METHODS, "predictors": ["log(beta/.1)", "log(force_N)", "log(measured_velocity/.1)"],
                "reduced_coordinates": "log(beta/.1), log(force_N)-alpha*log(measured_velocity/.1)",
                "basis": "degree-two polynomial, source-fold standardization, logistic BCE fit",
                "selection": "maximum pooled source OOF minimum IoU across 15 coherent observer rules; grid-order tie break",
                "folds": "five contiguous beta blocks shared across both training speeds",
                "target_scoring": "all source selections and fitted coefficients frozen before any target-score computation",
                "label_caveat": "Fixed nuisance/reference from A/0.1 for every experiment, including sources B/C/D; not independent label calibration.",
                "test_scopes": ["source specimen third speed", "other three specimens at all speeds", "combined ten panels"],
                "prospective_data": "No sample2/A2 access", "threads": 1,
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
                "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                  (Path(__file__), Path(bd.__file__), Path(__file__).with_name("set_envelope.py"),
                                   Path(__file__).with_name("benchmark_sets.py"))}}
    write_json(output / "protocol.json", protocol, exclusive=True)
    data = bd.load_dataset(include_ambiguous=True) if dataset is None else dataset
    X, nominal = data["X"], data["y_fixed"] == 1
    family = label_family(data["output_features"], data["reference_f0_hz"])
    if not np.array_equal(family[:, 7], nominal):
        raise ValueError("central observer rule must equal the nominal binary target")
    write_json(output / "input_manifest.json", data["manifest"], exclusive=True)
    frozen, evaluations = {}, {}
    for source, pair in itertools.product(sources, pairs):
        name = f"{source}__{pair[0]:g}_{pair[1]:g}"
        directory = output / name
        directory.mkdir()
        train, test = build_masks(data, source, pair)
        rows = np.flatnonzero(train)
        candidates, selected = select_source(X[train], nominal[train], family[train], data["beta_level"][train],
                                             data["speed"][train], data["ids"][train], directory / "fits.jsonl",
                                             exponents=exponents, cs=cs)
        payload = {"source": source, "training_speeds": pair, "training_ids": data["ids"][train].tolist(),
                   "training_ids_sha256": bd.id_hash(data["ids"][train]),
                   "evaluation_ids_sha256": {scope: bd.id_hash(data["ids"][mask]) for scope, mask in test.items()},
                   "selection": candidates, "selected_indices": selected, "models": {}}
        # Fit each unique winning parameter once; every method uses the same source rows.
        fitted = {}
        for index in sorted(set(selected.values())):
            candidate = candidates[index]
            estimator = fit_estimator(X[rows], nominal[rows], alpha=candidate["alpha"], C=candidate["C"])
            fitted[index] = freeze_estimator(estimator, candidate["alpha"], candidate["C"])
            append_jsonl(directory / "fits.jsonl", {"phase": "final_source_fit", "candidate_index": index,
                                                    "train_ids_sha256": payload["training_ids_sha256"],
                                                    "fit": fitted[index]})
        for method, index in selected.items():
            payload["models"][method] = fitted[index]
        write_json(directory / "frozen_models.json", payload, exclusive=True)
        frozen[name] = {"path": directory / "frozen_models.json", "source": source, "pair": pair, "test": test}
        print(f"FROZEN {name}: learned alpha={payload['models']['learned_alpha']['alpha']}, "
              f"C={payload['models']['learned_alpha']['C']}", flush=True)

    # A digest of every fitted source model is committed to disk before target scoring.
    model_manifest = {name: {"path": str(entry["path"].relative_to(output)),
                             "sha256": hashlib.sha256(entry["path"].read_bytes()).hexdigest()} for name, entry in frozen.items()}
    write_json(output / "all_models_frozen.json", model_manifest, exclusive=True)
    prediction_columns = {"ids": data["ids"], "nominal": nominal, "family": family, "groups": data["group"]}
    for name, entry in frozen.items():
        payload = json.loads(entry["path"].read_text())
        methods = {}
        for method, model in payload["models"].items():
            probability = predict_frozen(model, X)
            prediction_columns[f"{name}__{method}"] = probability
            methods[method] = {"selected_alpha": model["alpha"], "selected_C": model["C"],
                               "source_oof": payload["selection"][payload["selected_indices"][method]]["metrics"],
                               **{scope: evaluate_panels(probability, nominal, family, data["group"], mask)
                                  for scope, mask in entry["test"].items()}}
        evaluations[name] = {"source": entry["source"], "training_speeds": entry["pair"], "methods": methods}
    write_json(output / "results.json", evaluations, exclusive=True)
    np.savez_compressed(output / "predictions.npz", **prediction_columns)
    aggregate = {}
    for method in METHODS:
        aggregate[method] = {}
        for scope in ("combined", "source_heldout_speed", "other_specimens"):
            aggregate[method][scope] = {"mean_panel_macro_nominal_iou": float(np.mean([
                result["methods"][method][scope]["macro_panels"]["nominal_iou"] for result in evaluations.values()])),
                "mean_panel_macro_family_min_iou": float(np.mean([
                    result["methods"][method][scope]["macro_panels"]["family_min_iou"] for result in evaluations.values()])),
                "mean_global_same_rule_family_min_iou": float(np.mean([
                    result["methods"][method][scope]["global_same_rule"]["family_min_iou"] for result in evaluations.values()]))}
    summary = {"scope": protocol["scope"], "source_experiments": len(evaluations),
               "aggregate_scope": "unweighted mean over prespecified source/pair experiments; overlapping evaluation panels are not independent replicates",
               "aggregate": aggregate, "selected_exponents": {name: result["methods"]["learned_alpha"]["selected_alpha"]
                                                               for name, result in evaluations.items()},
               "fit_records": sum(sum(1 for _ in (output / name / "fits.jsonl").open()) for name in frozen),
               "seconds": time.monotonic() - started}
    write_json(output / "summary.json", summary, exclusive=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output)
