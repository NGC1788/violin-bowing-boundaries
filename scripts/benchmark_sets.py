#!/usr/bin/env python3
"""Retrospective, source-selected comparison of coherent weak-label set learners.

No target labels are used by the fit/select loop. Existing physics predictions
are separately flagged historical target-adapted diagnostics. This script never
downloads data or opens the prospective lockbox.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import scipy
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

import benchmark_data as bd
from set_envelope import SetEnvelopeModel, similarity_coordinates, label_family, family_ious, iou_bounds


def binary_iou(prediction, target):
    p, t = np.asarray(prediction, bool), np.asarray(target, bool)
    union = np.count_nonzero(p | t)
    return float(np.count_nonzero(p & t) / union) if union else 1.0


def metrics(probability, nominal, family):
    probability = np.asarray(probability, float)
    if probability.shape != (len(nominal),) or not np.isfinite(probability).all():
        raise ValueError("invalid prediction probabilities")
    prediction = probability >= .5
    coherent = family_ious(prediction, family)
    lo, hi = family.all(axis=1), family.any(axis=1)
    boxlo, boxhi = iou_bounds(prediction, lo, hi)
    return {"n": int(len(nominal)), "nominal_iou": binary_iou(prediction, nominal),
            "family_min_iou": float(coherent.min()), "family_mean_iou": float(coherent.mean()),
            "family_max_iou": float(coherent.max()), "box_lower_iou": boxlo, "box_upper_iou": boxhi,
            "brier_nominal": float(np.mean((probability - nominal) ** 2)),
            "predicted_positive": int(prediction.sum()), "family_ious": coherent.tolist()}


def blocked_folds(beta_levels, n_splits=5):
    """Hold out whole contiguous beta columns; indexes are local to source rows."""
    levels = np.unique(beta_levels)
    if len(levels) < n_splits:
        raise ValueError("too few beta columns for blocked folds")
    folds = []
    for hold in np.array_split(levels, n_splits):
        valid = np.flatnonzero(np.isin(beta_levels, hold))
        train = np.flatnonzero(~np.isin(beta_levels, hold))
        folds.append((train, valid))
    return folds


def specifications():
    result = []
    for covariant in (False, True):
        prefix = "covariant" if covariant else "rawlog"
        for kind in ("logistic", "svc", "extra_trees"):
            grid = [{"C": c} for c in (.1, 1., 10.)] if kind in ("logistic", "svc") else [{}]
            result.append({"name": f"{prefix}_{kind}", "kind": kind, "covariant": covariant, "grid": grid})
    result.append({"name": "rawlog_hgb", "kind": "hgb", "covariant": False, "grid": [{}]})
    grid = [{"length_scale": length, "regularization": reg}
            for length, reg in itertools.product((.4, .8, 1.2), (.001, .01, .1))]
    for objective in ("nominal_bce", "family_mean_bce", "consensus_bce", "nominal_iou", "box_robust", "family_robust"):
        result.append({"name": f"covariant_{objective}", "kind": "set", "objective": objective,
                       "covariant": True, "grid": grid})
    result.append({"name": "rawlog_family_robust", "kind": "set", "objective": "family_robust",
                   "covariant": False, "grid": grid})
    return result


class StandardModel:
    def __init__(self, estimator, covariant):
        self.estimator, self.covariant = estimator, covariant

    def fit(self, X, y):
        self.estimator.fit(similarity_coordinates(X, self.covariant), y)
        return self

    def predict_proba(self, X):
        z = similarity_coordinates(X, self.covariant)
        probability = self.estimator.predict_proba(z)
        classes = self.estimator.classes_
        if len(classes) == 1:
            positive = np.full(len(z), float(classes[0] == 1))
            return np.column_stack((1-positive, positive))
        return probability


def fit_model(spec, params, X, nominal, family):
    kind = spec["kind"]
    if kind == "set":
        objective = spec["objective"]
        target = nominal.astype(float)
        lower, upper = family.all(axis=1), family.any(axis=1)
        if objective == "family_mean_bce":
            target = family.mean(axis=1)
        keep = lower == upper if objective == "consensus_bce" else np.ones(len(X), bool)
        if not keep.any():
            raise ValueError("consensus-only learner has no identified source labels")
        mapped = {"nominal_bce": "bce", "family_mean_bce": "bce", "consensus_bce": "bce",
                  "nominal_iou": "nominal_iou", "box_robust": "robust_iou", "family_robust": "family_iou"}[objective]
        model = SetEnvelopeModel(objective=mapped, covariant=spec["covariant"], **params)
        # Zero weights mask unidentified labels while every variant retains the same
        # source-fitted normalization and RBF basis (including unknown cells' X).
        model.fit(X, target, lower=lower, upper=upper, family=family, sample_weight=keep.astype(float))
        return model, {"fit_rows": int(keep.sum()), "basis_rows": len(X), "optimization": model.optimization_,
                       "basis_center": model.center_.tolist(), "basis_scale": model.scale_.tolist(),
                       "basis_anchors": model.anchors_.tolist(), "coefficients": model.coef_.tolist()}
    if kind == "logistic":
        estimator = make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(),
                                  LogisticRegression(C=params["C"], max_iter=1000, random_state=0))
    elif kind == "svc":
        estimator = make_pipeline(StandardScaler(), SVC(C=params["C"], kernel="rbf", gamma="scale",
                                                        probability=True, random_state=0))
    elif kind == "extra_trees":
        estimator = ExtraTreesClassifier(n_estimators=200, min_samples_leaf=3, max_features=1.0,
                                         n_jobs=2, random_state=0)
    elif kind == "hgb":
        estimator = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15, l2_regularization=1.,
                                                   early_stopping=False, random_state=0)
    else:
        raise ValueError(f"unknown estimator: {kind}")
    model = StandardModel(estimator, spec["covariant"]).fit(X, nominal.astype(int))
    return model, {"fit_rows": len(X), "estimator_settings": {k: str(v) for k, v in estimator.get_params(deep=True).items()}}


def append_jsonl(path, record):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def cross_validate(spec, X, nominal, family, levels, ids, log_path):
    """Only source arrays are accepted: target outputs cannot enter model selection."""
    results = []
    folds = blocked_folds(levels)
    for params in spec["grid"]:
        predicted = np.full(len(X), np.nan)
        for fold, (train, valid) in enumerate(folds):
            start = time.monotonic()
            model, diagnostic = fit_model(spec, params, X[train], nominal[train], family[train])
            probability = model.predict_proba(X[valid])[:, 1]
            predicted[valid] = probability
            append_jsonl(log_path, {"phase": "source_cv", "model": spec["name"], "params": params,
                                   "fold": fold, "train_ids_sha256": bd.id_hash(ids[train]),
                                   "validation_ids_sha256": bd.id_hash(ids[valid]),
                                   "seconds": round(time.monotonic()-start, 4),
                                   "metrics": metrics(probability, nominal[valid], family[valid]), **diagnostic})
        result = {"params": params, "metrics": metrics(predicted, nominal, family)}
        results.append(result)
        print(f"CV {spec['name']} {params} family={result['metrics']['family_min_iou']:.4f} nominal={result['metrics']['nominal_iou']:.4f}", flush=True)
    # Stable tie breaking follows the declared grid order, not target performance.
    primary = max(range(len(results)), key=lambda i: results[i]["metrics"]["family_min_iou"])
    nominal_best = max(range(len(results)), key=lambda i: results[i]["metrics"]["nominal_iou"])
    return results, primary, nominal_best


def evaluate_panels(probability, nominal, family, groups, mask):
    panels = {str(group): metrics(probability[mask & (groups == group)], nominal[mask & (groups == group)],
                                  family[mask & (groups == group)]) for group in sorted(set(groups[mask]))}
    scalar_keys = ("nominal_iou", "family_min_iou", "family_mean_iou", "family_max_iou", "box_lower_iou", "box_upper_iou", "brier_nominal")
    macro = {key: float(np.mean([panel[key] for panel in panels.values()])) for key in scalar_keys}
    macro["worst_panel_family_min_iou"] = min(p["family_min_iou"] for p in panels.values())
    # Global uses the SAME observation rule for all test rows, unlike mean per-panel minima.
    global_score = metrics(probability[mask], nominal[mask], family[mask])
    specimens = {}
    for specimen in sorted({group.split("/")[0] for group in panels}):
        selected = [p for group, p in panels.items() if group.startswith(specimen + "/")]
        specimens[specimen] = {key: float(np.mean([p[key] for p in selected])) for key in scalar_keys}
    return {"panels": panels, "macro_panels": macro, "specimen_panel_means": specimens,
            "global_same_rule": global_score}


def run(out: Path, source="A", selected_models=None):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("output directory must be empty; preserve previous benchmark artifacts")
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    data = bd.load_dataset(include_ambiguous=True)
    X, nominal = data["X"], data["y_fixed"] == 1
    family = label_family(data["output_features"], data["reference_f0_hz"])
    if not np.array_equal(family[:, 7], nominal):
        raise ValueError("central observation family does not reproduce fixed nominal labels")
    source_mask = (data["string"] == source) & (data["speed"] == .1)
    if not source_mask.any():
        raise ValueError("source panel is absent")
    test_mask = ~source_mask
    specs = [spec for spec in specifications() if not selected_models or spec["name"] in selected_models]
    if not specs:
        raise ValueError("no model specifications selected")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "scope": "retrospective development",
                "data": data["manifest"], "source": f"{source}/0.1", "source_ids_sha256": bd.id_hash(data["ids"][source_mask]),
                "evaluation_ids_sha256": bd.id_hash(data["ids"][test_mask]),
                "classifier_reference": "A/0.1 fixed for all sources; other source sweeps are conditional on this label definition",
                "nominal_target": "y_fixed == 1; ambiguous cells count negative at the nominal rule but remain in all 15 family rules",
                "selection": "pooled source-only out-of-fold minimum IoU over one coherent rule; separately preserve nominal-IoU winner",
                "models": specs, "versions": {"python": platform.python_version(), "numpy": np.__version__,
                                               "scipy": scipy.__version__, "sklearn": sklearn.__version__},
                "script_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                  (Path(__file__), Path(bd.__file__), Path(__file__).with_name("set_envelope.py"))},
                "threads": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    predictions, results, selections = {}, {}, {}
    log_path = out/"fits.jsonl"
    for spec in specs:
        cv, primary, alternative = cross_validate(spec, X[source_mask], nominal[source_mask], family[source_mask],
                                                   data["beta_level"][source_mask], data["ids"][source_mask], log_path)
        selections[spec["name"]] = {"cv": cv, "primary_index": primary, "nominal_index": alternative}
        for index, selection in [(primary, "family_cv")] + ([(alternative, "nominal_cv")] if primary != alternative else []):
            name = spec["name"] if selection == "family_cv" else spec["name"] + "__nominal_cv"
            fit_start = time.monotonic()
            params = cv[index]["params"]
            model, diagnostic = fit_model(spec, params, X[source_mask], nominal[source_mask], family[source_mask])
            probability = model.predict_proba(X)[:, 1]
            predictions[name] = probability
            append_jsonl(log_path, {"phase": "final_source_fit", "model": name, "params": params,
                                   "selection": selection, "train_ids_sha256": manifest["source_ids_sha256"],
                                   "seconds": round(time.monotonic()-fit_start,4), **diagnostic})
            results[name] = {"selection": selection, "params": params, "historical_diagnostic": False,
                             **evaluate_panels(probability, nominal, family, data["group"], test_mask)}
            (out/"results.partial.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
        (out/"selection.json").write_text(json.dumps(selections, indent=2, allow_nan=False)+"\n")
    constants = {"always_negative": 0., "always_positive": 1., "source_prevalence": float(nominal[source_mask].mean())}
    for name, value in constants.items():
        probability = np.full(len(X), value)
        predictions[name] = probability
        results[name] = {"selection": "fixed", "params": {"probability": value}, "historical_diagnostic": False,
                         **evaluate_panels(probability, nominal, family, data["group"], test_mask)}
    for name, key in (("historical_physics_legacy", "legacy_sim_prediction"),
                      ("historical_physics_fixed_classifier", "reclassified_sim_prediction")):
        probability = data[key].astype(float)
        predictions[name] = probability
        results[name] = {"selection": "not fitted here", "params": {}, "historical_diagnostic": True,
                         "warning": "Target-derived impedance/pitch/noise and known waveguide numerical defect; excluded from fair main ranking.",
                         **evaluate_panels(probability, nominal, family, data["group"], test_mask)}
    # Fixed prior-exponent sensitivity: same source-selected hyperparameters, never tune on test.
    if "covariant_family_robust" in selections:
        entry = selections["covariant_family_robust"]
        params = entry["cv"][entry["primary_index"]]["params"]
        spec = next(s for s in specs if s["name"] == "covariant_family_robust")
        for exponent in (0., .5, 1.5):
            name = f"covariant_family_robust__exponent_{exponent:g}"
            model, diagnostic = fit_model(spec, {**params, "speed_exponent": exponent}, X[source_mask], nominal[source_mask], family[source_mask])
            probability = model.predict_proba(X)[:, 1]
            predictions[name] = probability
            append_jsonl(log_path, {"phase": "fixed_exponent_sensitivity", "model": name, "params": {**params, "speed_exponent": exponent},
                                   "train_ids_sha256": manifest["source_ids_sha256"], **diagnostic})
            results[name] = {"selection": "fixed prior sensitivity; no target selection", "params": {**params, "speed_exponent": exponent},
                             "historical_diagnostic": False, "prior_sensitivity": True,
                             **evaluate_panels(probability, nominal, family, data["group"], test_mask)}
    np.savez_compressed(out/"predictions.npz", ids=data["ids"], source_mask=source_mask, family=family,
                        nominal=nominal, groups=data["group"], beta_level=data["beta_level"], **predictions)
    (out/"results.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
    rows = []
    for name, result in results.items():
        for panel, metric in result["panels"].items():
            rows.append({"model": name, "panel": panel, "historical_diagnostic": result["historical_diagnostic"],
                         **{k:v for k,v in metric.items() if k != "family_ious"}})
    with (out/"panel_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {"status": "PASS", "seconds": round(time.monotonic()-start,2), "source_rows": int(source_mask.sum()),
               "evaluation_rows": int(test_mask.sum()), "model_outputs": len(results),
               "fit_records": sum(1 for _ in log_path.open()), "main_primary_macro_ranking": sorted(
                   [{"model": name, **r["macro_panels"]} for name,r in results.items()
                    if not r["historical_diagnostic"] and not r.get("prior_sensitivity") and r["selection"] != "nominal_cv"],
                   key=lambda r: -r["family_min_iou"])}
    (out/"run_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=list("ABCD"), default="A")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", action="append")
    args = parser.parse_args(argv)
    with threadpool_limits(limits=1):
        run(args.out, args.source, args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
