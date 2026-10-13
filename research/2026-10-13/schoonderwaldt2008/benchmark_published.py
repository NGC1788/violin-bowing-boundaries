#!/usr/bin/env python3
"""Recorded-plan external comparison on published violin D-string classifications.

No own recordings, raw signals, synthetic relabeling, gray interiors, or target
selection. This is retrospective: the extraction and marker audit preceded it.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_sets import blocked_folds, binary_iou
from boundary_transport import BoundaryTransport, PRIORS, REGULARIZATIONS
from speed_prior_audit import estimator

C_GRID = (.1, 1., 10.)
SOURCE_SPEEDS = (.1, .05, .15, .2)
SPECS = (
    ("quadratic_static", "quadratic_logistic", 0.),
    ("quadratic_uniform", "quadratic_logistic", 1.),
    ("quadratic_raw3d", "quadratic_logistic", None),
    ("rbf_static", "rbf_svm", 0.),
    ("rbf_uniform", "rbf_svm", 1.),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def cell_id(r):
    return f"fig{r['figure']}{r['panel']}_b{r['beta_column_0based']}_f{r['force_rank_0based']}"


def load_cells(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len({cell_id(r) for r in rows}) != len(rows):
        raise ValueError("duplicate published grid IDs")
    for r in rows:
        r["id"] = cell_id(r)
    selected = [r for r in rows if r["string"] == "D4" and r["strict_systematically_analyzed"] == "True"]
    X = np.array([[float(r["beta_from_reported_rounded_distance"]),
                   float(r["bow_force_nominal_N"]), float(r["bow_speed_nominal_m_s"])] for r in selected])
    y = np.array([int(r["binary_helmholtz"]) for r in selected], dtype=bool)
    beta = np.array([int(r["beta_column_0based"]) for r in selected])
    ids = np.array([r["id"] for r in selected])
    return rows, selected, X, y, beta, ids


def coordinates(X, exponent, source_speed):
    b, f, v = np.log(X / [.1, 1., source_speed]).T
    return np.column_stack((b, f, v)) if exponent is None else np.column_stack((b, f - exponent * v))


def scores(p, y):
    p, y = np.asarray(p, float), np.asarray(y, bool)
    if len(y) == 0 or p.shape != y.shape or not np.isfinite(p).all():
        raise ValueError("empty or invalid score arrays")
    h = p >= .5
    tp, fp, fn, tn = (int(np.count_nonzero(q)) for q in (h & y, h & ~y, ~h & y, ~h & ~y))
    return {"n": len(y), "positive": int(y.sum()), "iou": binary_iou(h, y),
            "brier": float(np.mean((p-y)**2)), "error_rate": float(np.mean(h != y)),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def audit_grid(all_rows):
    """Exact adjacent-label brackets only; missing ranks never bridge components."""
    result = []
    for speed in sorted(SOURCE_SPEEDS):
        for beta in range(11):
            rows = sorted((r for r in all_rows if r["string"] == "D4" and
                           float(r["bow_speed_nominal_m_s"]) == speed and
                           int(r["beta_column_0based"]) == beta), key=lambda r: int(r["force_rank_0based"]))
            if len(rows) != 24:
                raise ValueError("incomplete slot table")
            labels = [int(r["binary_helmholtz"]) if r["strict_systematically_analyzed"] == "True" else None for r in rows]
            force = [float(r["bow_force_nominal_N"]) for r in rows]
            runs, lower_brackets, upper_brackets = [], [], []
            for k, label in enumerate(labels):
                if label == 1 and (k == 0 or labels[k-1] != 1):
                    runs.append([k, k])
                elif label == 1:
                    runs[-1][1] = k
                if k and labels[k-1] == 0 and label == 1:
                    lower_brackets.append([force[k-1], force[k]])
                if k and labels[k-1] == 1 and label == 0:
                    upper_brackets.append([force[k-1], force[k]])
            result.append({"speed": speed, "beta_column": beta, "unknown_force_ranks": [i for i, t in enumerate(labels) if t is None],
                           "positive_runs_without_gap_filling": runs,
                           "lower_adjacent_transition_brackets_N": lower_brackets,
                           "upper_adjacent_transition_brackets_N": upper_brackets,
                           "low_acquisition_edge_H": labels[0] == 1,
                           "high_acquisition_edge_H": labels[-1] == 1,
                           "positive_adjacent_to_unknown": any(t == 1 and
                               ((k > 0 and labels[k-1] is None) or (k < 23 and labels[k+1] is None)) for k,t in enumerate(labels)),
                           "interpretation": "Observed author-label adjacency; not continuous or physical boundary certification"})
    return result


def run(output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output must be empty to preserve prior execution")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Retrospective external published-author-classification comparison; not preregistered or blinded",
        "source_doi": "10.3813/AAA.918070", "source_figure": 4,
        "physical_scope": "One real violin D4 string on rigid monochord; not assembled violin",
        "primary_source_speed": .1, "sensitivity_source_speeds": SOURCE_SPEEDS,
        "target_speeds": "All other D4 speeds for each source; no source scores in target summary",
        "strict_filter": "D4 and strict_systematically_analyzed True; unknown/gray omitted, never negatives",
        "E_exclusion": "Published systematically analyzed cells concentrate at exterior/boundaries; gray interior deliberately unanalyzed",
        "class_semantics": "author Helmholtz versus all other strictly analyzed published classes",
        "observer_family": "none: exactly one author classification per cell",
        "models": [{"name": name, "learner": kind, "exponent": alpha} for name, kind, alpha in SPECS]
                  + [{"name": "boundary_"+name, "exponents": pair} for name,pair in PRIORS.items()],
        "C_grid": C_GRID, "boundary_regularization_grid": REGULARIZATIONS,
        "source_selection": "Five source-only contiguous beta-column folds; pooled source OOF IoU at 0.5; first grid tie wins",
        "boundary_prior_fairness": "One concave source fit and same selected regularization shared by all four fixed priors",
        "source_normalization": "log(beta/.1),log(force/1N),log(speed/source_speed); static/uniform identical source coordinates",
        "threshold": .5, "primary_metric": "equal mean of target-speed IoUs",
        "uncertainty": "No independent-instrument CI; one string, spatially dependent conditions, approximate reported beta",
        "novelty": "Fixed known physics priors and standard classifiers; no new law or method claimed",
        "plan_written_before_target_prediction": True,
        "hashes": {str(path.relative_to(ROOT)): sha(path) for path in
                   (Path(__file__), HERE/"published_cells.csv", ROOT/"scripts/boundary_transport.py",
                    ROOT/"scripts/speed_prior_audit.py", ROOT/"scripts/benchmark_sets.py")},
    }
    save_json(output/"manifest.json", manifest)
    all_rows, rows, X, y, beta, ids = load_cells(HERE/"published_cells.csv")
    source_plans = []
    for source in SOURCE_SPEEDS:
        src = X[:,2] == source
        source_ids = ids[src]
        folds = blocked_folds(beta[src])
        source_plans.append({"source_speed": source, "source_ids": source_ids.tolist(), "target_ids": ids[~src].tolist(),
                             "folds": [{"fold": j, "train_ids": source_ids[t].tolist(), "validation_ids": source_ids[v].tolist()}
                                       for j,(t,v) in enumerate(folds)]})
    save_json(output/"source_and_fold_ids.json", source_plans)
    save_json(output/"published_grid_audit.json", audit_grid(all_rows))
    records, prediction_rows, beta_rows = [], [], []

    def evaluate(name, source, probability, selection, diagnostic):
        target = X[:,2] != source
        panels = {str(v): scores(probability[(X[:,2] == v)], y[(X[:,2] == v)])
                  for v in sorted(SOURCE_SPEEDS) if v != source}
        records.append({"source_speed": source, "model": name, "source_selection": selection,
                        "target_panels": panels, "macro_target_iou": float(np.mean([v["iou"] for v in panels.values()])),
                        "macro_target_brier": float(np.mean([v["brier"] for v in panels.values()])),
                        "pooled_targets": scores(probability[target], y[target]), "fit_diagnostic": diagnostic})
        for i in np.flatnonzero(target):
            prediction_rows.append({"source_speed": source, "model": name, "cell_id": ids[i],
                                    "target_speed": X[i,2], "beta_column": beta[i],
                                    "force_N": X[i,1], "author_H": int(y[i]),
                                    "probability_H": probability[i], "prediction_H": int(probability[i] >= .5)})
        for v in sorted(SOURCE_SPEEDS):
            if v == source:
                continue
            for b in np.unique(beta):
                mask = (X[:,2] == v) & (beta == b)
                beta_rows.append({"source_speed": source, "model": name, "target_speed": v,
                                  "beta_column": int(b), **scores(probability[mask], y[mask])})

    for source in SOURCE_SPEEDS:
        src = X[:,2] == source
        xs, ys, levels = X[src], y[src], beta[src]
        folds = blocked_folds(levels)
        for name, kind, exponent in SPECS:
            z = coordinates(X, exponent, source)
            zs = z[src]
            candidate = []
            for c in C_GRID:
                oof = np.full(len(ys), np.nan)
                for tr, va in folds:
                    fit = estimator(kind, c).fit(zs[tr], ys[tr])
                    oof[va] = fit.predict_proba(zs[va])[:,1]
                candidate.append({"C": c, **scores(oof, ys)})
            choice = max(range(len(candidate)), key=lambda k: candidate[k]["iou"])
            fit = estimator(kind, candidate[choice]["C"]).fit(zs, ys)
            evaluate(name, source, fit.predict_proba(z)[:,1], {"candidates": candidate, "selected_index": choice},
                     {"training_speed_variance": float(np.var(zs[:,-1])) if exponent is None else None,
                      "source_coordinates_dimensions": z.shape[1]})
        candidate = []
        for reg in REGULARIZATIONS:
            oof = np.full(len(ys), np.nan)
            for tr, va in folds:
                fit = BoundaryTransport(reg, source).fit(xs[tr], ys[tr])
                oof[va] = fit.predict_proba(xs[va], (0.,0.))[:,1]
            candidate.append({"regularization": reg, **scores(oof, ys)})
        choice = max(range(len(candidate)), key=lambda k: candidate[k]["iou"])
        fit = BoundaryTransport(candidate[choice]["regularization"], source).fit(xs, ys)
        for name, exponents in PRIORS.items():
            g = fit.geometry(X, exponents)
            target = ~src
            observed_min, observed_max = np.log(xs[:,1].min()), np.log(xs[:,1].max())
            source_g = fit.geometry(xs, (0.,0.))
            evaluate("boundary_"+name, source, fit.predict_proba(X, exponents)[:,1],
                     {"candidates": candidate, "selected_index": choice, "shared_across_four_priors": True},
                     {"coefficient": fit.coef_.tolist(), "optimization": fit.optimization_,
                      "target_no_fitted_band_count": int(np.sum(~g["has_band"] & target)),
                      "target_crossed_root_count": int(np.sum(g["crossed_after_transport"] & target)),
                      "source_lower_extrapolated_beta_columns": sorted(set(levels[(source_g["lower_log_force"] < observed_min) |
                                                                                                  (source_g["lower_log_force"] > observed_max)].tolist())),
                      "source_upper_extrapolated_beta_columns": sorted(set(levels[(source_g["upper_log_force"] < observed_min) |
                                                                                                  (source_g["upper_log_force"] > observed_max)].tolist()))})
        print(json.dumps({"source_speed": source, "macro_target_iou": {r["model"]: round(r["macro_target_iou"],6)
                                                                       for r in records if r["source_speed"] == source}}), flush=True)
    save_json(output/"results.json", {"elapsed_seconds": time.perf_counter()-started, "results": records})
    for file, data in (("predictions.csv", prediction_rows), ("per_beta_errors.csv", beta_rows)):
        with (output/file).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    summary = [{"source_speed": r["source_speed"], "model": r["model"],
                "macro_target_iou": r["macro_target_iou"], "macro_target_brier": r["macro_target_brier"],
                "pooled_target_iou": r["pooled_targets"]["iou"]} for r in records]
    with (output/"summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.output)
