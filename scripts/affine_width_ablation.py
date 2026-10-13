#!/usr/bin/env python3
"""Decisive matched k=0 ablation of the affine-width hypothesis.

Reuse the exact full-flow objective, basis, normalization and source CV protocol;
fix only k=0. No production full-flow source is modified. All tasks are retained.
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
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

import affine_transport as af
import benchmark_data as bd
from benchmark_sets import evaluate_panels, metrics
import multispeed_scaling as ms
from set_envelope import label_family


def fit_translation(X, y, lam, warp_lam):
    coords, norm = af.raw_coordinates(X), af.normalization(X)
    y = np.asarray(y, float)
    if np.unique(y).size != 2:
        raise ValueError("source must contain both classes")
    theta = np.zeros(9)
    theta[0] = np.log(np.mean(y) / (1 - np.mean(y)))
    theta[5], theta[6] = -.1, 1.
    bounds = list(af.BOUNDS)
    bounds[8] = (0., 0.)
    result = minimize(af.objective, theta, args=(coords, y, norm, lam, warp_lam), method="L-BFGS-B", jac=True,
                      bounds=bounds, options={"maxiter": 500, "ftol": 1e-11, "gtol": 2e-7, "maxls": 40})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"fixed-width source optimization failed: {result.message}")
    return {"theta": result.x.tolist(), "normalization": norm, "lambda": lam, "warp_lambda": warp_lam,
            "fixed_k": 0., "start_k": 0., "unique_starts": 1, "success": bool(result.success),
            "objective": float(result.fun), "iterations": int(result.nit), "message": str(result.message)}


def select_source(X, y, family, levels, speeds, ids, log_path, *, lambdas=af.LAMBDAS, warp_lambdas=af.WARP_LAMBDAS):
    folds = ms.two_speed_folds(levels, speeds)
    candidates = []
    for lam, warp_lam in itertools.product(lambdas, warp_lambdas):
        probability = np.full(len(y), np.nan)
        for fold, (train, valid) in enumerate(folds):
            model = fit_translation(X[train], y[train], lam, warp_lam)
            probability[valid] = af.predict_frozen(model, X[valid])
            ms.append_jsonl(log_path, {"lambda": lam, "warp_lambda": warp_lam, "fold": fold,
                "train_ids_sha256": bd.id_hash(ids[train]), "valid_ids_sha256": bd.id_hash(ids[valid]),
                "heldout_beta_levels": sorted(np.unique(levels[valid]).tolist()), "fit": model,
                "metrics": metrics(probability[valid], y[valid], family[valid])})
        if not np.isfinite(probability).all():
            raise AssertionError("missing OOF predictions")
        candidates.append({"lambda": lam, "warp_lambda": warp_lam, "metrics": metrics(probability, y, family)})
    selected = max(range(len(candidates)), key=lambda i: candidates[i]["metrics"]["family_min_iou"])
    best = candidates[selected]
    return {"model": fit_translation(X, y, best["lambda"], best["warp_lambda"]),
            "source_candidates": candidates, "selected_index": selected}


def full_start_audit(full_run):
    gaps = []
    for path in Path(full_run).glob("*/flow_cv.jsonl"):
        for line in path.open():
            entry = json.loads(line)["fit"]
            chosen = entry["starts"][entry["selected_start"]]
            zero = next(s for s in entry["starts"] if s["initial_k"] == 0.)
            gaps.append(zero["objective"] - chosen["objective"])
    return {"cv_fits": len(gaps), "maximum_objective_gain_from_other_start": float(max(gaps)),
            "fits_other_start_improves_objective_over_1e_6": int(np.sum(np.asarray(gaps) > 1e-6)),
            "caveat": "full flow uses three distinct k starts; k=0 has one unique initialization (repeating it would be duplicate work)"}


def run(output, full_run):
    output, full_run = Path(output), Path(full_run)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("preserve existing results")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(), "scope": "POST-HOC matched width ablation, not new candidate search",
                "hypothesis": "Does nonzero k improve transfer over beta-dependent translation with k fixed to zero?",
                "same": ["source rows", "shared-beta folds", "objective", "basis", "normalization", "lambda grids", "a0/a1 bounds", "selection metric"],
                "changed": "k fixed0 instead of bounded[-2,2]; 8 free parameters instead of9",
                "initialization": "one unique k=0 start versus full flow k=-.5,0,.5; no duplicated identical starts",
                "lambda_grid": af.LAMBDAS, "warp_lambda_grid": af.WARP_LAMBDAS,
                "published_headline": "held-out D speeds only; E conditional observed-cell diagnostic excluded",
                "no_prospective_data": True, "full_run": str(full_run),
                "code_sha256": {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                (Path(__file__), Path(af.__file__), Path(ms.__file__))}}
    ms.write_json(output / "protocol.json", protocol, exclusive=True)
    cello = bd.load_dataset(include_ambiguous=True)
    cello["family"] = label_family(cello["output_features"], cello["reference_f0_hz"])
    datasets = {"cello": cello, "published": af.load_published()}
    prior_manifest = json.loads((full_run / "input_manifests.json").read_text())
    if datasets["published"]["manifest"]["sha256"] != prior_manifest["published"]["sha256"]:
        raise ValueError("published data changed since full-flow freeze")
    tasks = [("cello", s, pair) for s, pair in itertools.product(("A", "B", "C", "D"), ms.SPEED_PAIRS)]
    tasks += [("published", "D4", pair) for pair in itertools.combinations((.05, .1, .15, .2), 2)]
    frozen = {}
    for domain, source, pair in tasks:
        data = datasets[domain]
        train, test = ms.build_masks(data, source, pair)
        if domain == "published":
            test["combined"] = test["source_heldout_speed"].copy()
        key = f"{domain}__{source}__{pair[0]:g}_{pair[1]:g}"
        directory = output / key
        directory.mkdir()
        prior_path = full_run / key / "frozen_models.json"
        prior = json.loads(prior_path.read_text())
        train_hash = bd.id_hash(data["ids"][train])
        if prior["training_ids_sha256"] != train_hash:
            raise ValueError("source ID mismatch")
        selection = select_source(data["X"][train], data["y_fixed"][train] == 1, data["family"][train],
                                  data["beta_level"][train], data["speed"][train], data["ids"][train], directory / "source_cv.jsonl")
        payload = {"domain": domain, "source": source, "training_speeds": pair,
                   "training_ids": data["ids"][train].tolist(), "training_ids_sha256": train_hash,
                   "target_ids_sha256": {scope: bd.id_hash(data["ids"][mask]) for scope, mask in test.items()},
                   "translation": selection, "full_flow_model": prior["flow"]["model"],
                   "full_flow_frozen_file_sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest()}
        path = directory / "frozen_models.json"
        ms.write_json(path, payload, exclusive=True)
        frozen[key] = (path, domain, test)
    ms.write_json(output / "all_models_frozen.json", {key: {"path": str(item[0].relative_to(output)),
                  "sha256": hashlib.sha256(item[0].read_bytes()).hexdigest()} for key, item in frozen.items()}, exclusive=True)
    results = {}
    for key, (path, domain, test) in frozen.items():
        data, payload = datasets[domain], json.loads(path.read_text())
        results[key] = {"domain": domain, "training_speeds": payload["training_speeds"],
                        "selected_translation_regularization": {name: payload["translation"]["model"][name]
                                                                for name in ("lambda", "warp_lambda")}, "methods": {}}
        for method, model in (("full_flow", payload["full_flow_model"]), ("translation_k0", payload["translation"]["model"])):
            probability = af.predict_frozen(model, data["X"])
            results[key]["methods"][method] = {scope: evaluate_panels(probability, data["y_fixed"] == 1, data["family"],
                data["group"], mask) for scope, mask in test.items()}
    ms.write_json(output / "results.json", results, exclusive=True)
    aggregate = {}
    for domain in datasets:
        group = [r for r in results.values() if r["domain"] == domain]
        aggregate[domain] = {"tasks": len(group), "methods": {}}
        for method in ("full_flow", "translation_k0"):
            aggregate[domain]["methods"][method] = {scope: {aggregation: {metric: float(np.mean([
                r["methods"][method][scope][aggregation][metric] for r in group]))
                for metric in ("nominal_iou", "family_min_iou")} for aggregation in ("macro_panels", "global_same_rule")}
                for scope in ("combined", "source_heldout_speed", "other_specimens")}
        differences = [r["methods"]["full_flow"]["combined"]["macro_panels"]["family_min_iou"] -
                       r["methods"]["translation_k0"]["combined"]["macro_panels"]["family_min_iou"] for r in group]
        aggregate[domain]["full_minus_k0"] = {"mean": float(np.mean(differences)),
            "wins": int(np.sum(np.asarray(differences) > 1e-12)), "losses": int(np.sum(np.asarray(differences) < -1e-12))}
    summary = {"scope": protocol["scope"], "aggregate": aggregate, "full_flow_multistart_audit": full_start_audit(full_run),
               "all_tasks": {key: {m: v["combined"]["macro_panels"]["family_min_iou"] for m, v in row["methods"].items()}
                             for key, row in results.items()}, "seconds": time.monotonic() - started,
               "overlap_warning": "source/pair tasks are not independent replicates"}
    ms.write_json(output / "summary.json", summary, exclusive=True)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-run", type=Path, default=af.ROOT / "reports/affine_transport/2026-10-13-v2")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, args.full_run)
