#!/usr/bin/env python3
"""Post-hoc affine log-force transport hypothesis; known flow mathematics.

Source-only fits, complete pre-target parameter manifests, no prospective A2.
The quadratic logit is concave in transformed force at every bridge position.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, exprel
from threadpoolctl import threadpool_limits

import benchmark_data as bd
from benchmark_sets import evaluate_panels, metrics
import multispeed_scaling as ms
from set_envelope import label_family

ROOT = Path(__file__).resolve().parents[1]
LAMBDAS = (.001, .01, .1)
WARP_LAMBDAS = (.001, .01, .1)
START_K = (-.5, 0., .5)
BOUNDS = [(None, None)] * 5 + [(None, -1e-6), (-1., 3.), (-2., 2.), (-2., 2.)]


def raw_coordinates(X):
    X = np.asarray(X, float)
    if X.ndim != 2 or X.shape[1] != 3 or not np.isfinite(X).all() or np.any(X <= 0):
        raise ValueError("expected finite positive beta, force, speed")
    return np.log(X[:, 0] / .1), np.log(X[:, 1]), np.log(X[:, 2] / .1)


def transport(b, u, t, a0, a1, k, *, gradient=False):
    """Backward flow to reference speed: z=e^-kt u-a(b)(1-e^-kt)/k."""
    b, u, t = np.broadcast_arrays(np.asarray(b, float), np.asarray(u, float), np.asarray(t, float))
    q = k * t
    e = np.exp(-q)
    j = t * exprel(-q)
    z = e * u - (a0 + a1 * b) * j
    if not gradient:
        return z
    small = abs(q) < 1e-4
    dj = np.empty_like(t)
    dj[small] = (-.5 * t[small] ** 2 + k * t[small] ** 3 / 3 - k ** 2 * t[small] ** 4 / 8
                 + k ** 3 * t[small] ** 5 / 30)
    dj[~small] = (q[~small] * e[~small] + np.expm1(-q[~small])) / k ** 2
    return z, np.column_stack((-j, -b * j, -t * e * u - (a0 + a1 * b) * dj))


def normalization(X):
    b, u, t = raw_coordinates(X)
    prior = np.column_stack((b, u - t))
    scale = np.std(prior, axis=0)
    scale[scale < 1e-8] = 1.
    return {"mean": np.mean(prior, axis=0).tolist(), "scale": scale.tolist(),
            "scope": "source rows only; fixed prior coordinates b,u-t, independent of learned warp"}


def design(theta, coordinates, norm, *, derivatives=False):
    b, u, t = coordinates
    z, dz = transport(b, u, t, *theta[6:], gradient=True)
    B = (b - norm["mean"][0]) / norm["scale"][0]
    Z = (z - norm["mean"][1]) / norm["scale"][1]
    basis = np.column_stack((np.ones_like(B), B, Z, B * B, B * Z, Z * Z))
    score = basis @ theta[:6]
    if derivatives:
        warp = (theta[2] + theta[4] * B + 2 * theta[5] * Z)[:, None] * dz / norm["scale"][1]
        return score, np.column_stack((basis, warp))
    return score


def objective(theta, coordinates, y, norm, lam, warp_lam):
    score, jac = design(theta, coordinates, norm, derivatives=True)
    warp_delta = theta[6:] - np.array([1., 0., 0.])
    value = np.mean(np.logaddexp(0., score) - y * score)
    value += .5 * lam * np.dot(theta[1:6], theta[1:6]) + .5 * warp_lam * np.dot(warp_delta, warp_delta)
    grad = jac.T @ (expit(score) - y) / len(y)
    grad[1:6] += lam * theta[1:6]
    grad[6:] += warp_lam * warp_delta
    return float(value), grad


def fit_source(X, y, lam, warp_lam, *, starts=START_K, maxiter=500):
    coords, norm = raw_coordinates(X), normalization(X)
    y = np.asarray(y, float)
    if np.unique(y).size != 2:
        raise ValueError("source must contain both classes")
    fits = []
    for k in starts:
        theta = np.zeros(9)
        theta[0] = np.log(np.mean(y) / (1 - np.mean(y)))
        theta[5], theta[6], theta[8] = -.1, 1., k
        result = minimize(objective, theta, args=(coords, y, norm, lam, warp_lam), method="L-BFGS-B",
                          jac=True, bounds=BOUNDS, options={"maxiter": maxiter, "ftol": 1e-11, "gtol": 2e-7,
                                                          "maxls": 40})
        fits.append({"initial_k": k, "success": bool(result.success), "message": str(result.message),
                     "objective": float(result.fun), "iterations": int(result.nit),
                     "gradient_infinity_norm": float(np.max(abs(result.jac))), "theta": result.x.tolist()})
    eligible = [i for i, item in enumerate(fits) if item["success"] and np.isfinite(item["objective"])]
    if not eligible:
        raise RuntimeError("all three source optimizations failed: " + repr(fits))
    chosen = min(eligible, key=lambda i: fits[i]["objective"])
    return {"theta": fits[chosen]["theta"], "normalization": norm, "lambda": lam, "warp_lambda": warp_lam,
            "selected_start": chosen, "starts": fits, "constraint": "theta_z_squared <= -1e-6",
            "reference_speed": .1}


def predict_frozen(model, X):
    return expit(design(np.asarray(model["theta"]), raw_coordinates(X), model["normalization"]))


def select_source(X, y, family, levels, speeds, ids, log_path, *, lambdas=LAMBDAS, warp_lambdas=WARP_LAMBDAS):
    folds = ms.two_speed_folds(levels, speeds)
    candidates = []
    for lam, warp_lam in itertools.product(lambdas, warp_lambdas):
        probabilities = np.full(len(y), np.nan)
        for fold, (train, valid) in enumerate(folds):
            fit = fit_source(X[train], y[train], lam, warp_lam)
            probabilities[valid] = predict_frozen(fit, X[valid])
            ms.append_jsonl(log_path, {"lambda": lam, "warp_lambda": warp_lam, "fold": fold,
                                       "train_ids_sha256": bd.id_hash(ids[train]),
                                       "valid_ids_sha256": bd.id_hash(ids[valid]),
                                       "heldout_beta_levels": sorted(np.unique(levels[valid]).tolist()),
                                       "fit": fit, "metrics": metrics(probabilities[valid], y[valid], family[valid])})
        if not np.isfinite(probabilities).all():
            raise AssertionError("missing source OOF probabilities")
        candidates.append({"lambda": lam, "warp_lambda": warp_lam,
                           "metrics": metrics(probabilities, y, family)})
    best = max(range(len(candidates)), key=lambda i: candidates[i]["metrics"]["family_min_iou"])
    chosen = candidates[best]
    final = fit_source(X, y, chosen["lambda"], chosen["warp_lambda"])
    return {"model": final, "source_candidates": candidates, "selected_index": best}


def load_published():
    path = ROOT / "research/2026-10-13/schoonderwaldt2008/published_cells.csv"
    rows = [r for r in csv.DictReader(path.open()) if r["strict_systematically_analyzed"] == "True"]
    data = {"X": np.asarray([[float(r["beta_from_reported_rounded_distance"]), float(r["bow_force_nominal_N"]),
                              float(r["bow_speed_nominal_m_s"])] for r in rows]),
            "y_fixed": np.asarray([int(r["binary_helmholtz"]) for r in rows]),
            "string": np.asarray([r["string"] for r in rows]),
            "speed": np.asarray([float(r["bow_speed_nominal_m_s"]) for r in rows]),
            "beta_level": np.asarray([int(r["beta_column_0based"]) for r in rows]),
            "ids": np.asarray([f"fig{r['figure']}{r['panel']}:r{r['force_rank_0based']}:c{r['beta_column_0based']}" for r in rows]),
            "group": np.asarray([f"{r['string']}/{r['bow_speed_nominal_m_s']}" for r in rows]),
            "manifest": {"source": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "strict_only": True, "scope": "published classifications, selected steady intervals, monochord not intact violin"}}
    data["family"] = data["y_fixed"][:, None].astype(bool)
    return data


def run(output, *, cello_baselines, include_published=True):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output must be empty; preserve historical runs")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "POST-HOC DEVELOPMENT, not confirmatory",
                "hypothesis": "Affine log-force flow allows a band's width and center to evolve differently across speed.",
                "novelty": "Flow/group mathematics, polynomial classifiers and shape constraints are established; novelty not established.",
                "lambda_grid": LAMBDAS, "warp_lambda_grid": WARP_LAMBDAS, "initial_k": START_K,
                "parameter_bounds": BOUNDS, "basis": "1,B,Z,B^2,BZ,Z^2; concave in Z",
                "normalization": "source-only, fixed prior b,u-t coordinates",
                "source_selection": "five shared contiguous beta blocks; maximize pooled minimum-family OOF IoU",
                "starts_selection": "lowest successful TRAINING objective only",
                "optimization": "analytic gradient L-BFGS-B, maxiter500, ftol1e-11, gtol2e-7",
                "cello_label_caveat": "15 observer rules all use fixed A-derived nuisance; retrospective sample1 only",
                "published_label_caveat": "single published class per strict cell; gray and absent cells excluded",
                "published_evaluation": "combined headline ONLY held-out D speeds; E other_specimens is a conditional boundary-focused diagnostic, not ordinary generalization IoU",
                "baseline_caveat": "baseline regularization/basis prior differs; improvement would not isolate flow alone",
                "prospective_data": "A2 never accessed", "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    ms.write_json(output / "protocol.json", protocol, exclusive=True)
    cello = bd.load_dataset(include_ambiguous=True)
    cello["family"] = label_family(cello["output_features"], cello["reference_f0_hz"])
    datasets = {"cello": cello}
    tasks = [("cello", source, pair) for source, pair in itertools.product(("A", "B", "C", "D"), ms.SPEED_PAIRS)]
    if include_published:
        datasets["published"] = load_published()
        tasks += [("published", "D4", pair) for pair in itertools.combinations((.05, .1, .15, .2), 2)]
    ms.write_json(output / "input_manifests.json", {name: data["manifest"] for name, data in datasets.items()}, exclusive=True)
    frozen = {}
    for domain, source, pair in tasks:
        data = datasets[domain]
        X, y, family = data["X"], data["y_fixed"] == 1, data["family"]
        train, test = ms.build_masks(data, source, pair)
        if domain == "published":
            # The E figure systematically omits interior H classifications, so
            # its observed-cell distribution is not a standard full-grid target.
            test["combined"] = test["source_heldout_speed"].copy()
        key = f"{domain}__{source}__{pair[0]:g}_{pair[1]:g}"
        directory = output / key
        directory.mkdir()
        selection = select_source(X[train], y[train], family[train], data["beta_level"][train], data["speed"][train],
                                  data["ids"][train], directory / "flow_cv.jsonl")
        training_hash = bd.id_hash(data["ids"][train])
        payload = {"domain": domain, "source": source, "training_speeds": pair,
                   "training_ids": data["ids"][train].tolist(), "training_ids_sha256": training_hash,
                   "target_ids_sha256": {scope: bd.id_hash(data["ids"][mask]) for scope, mask in test.items()},
                   "flow": selection, "baseline_models": {}}
        if domain == "cello":
            existing = Path(cello_baselines) / f"{source}__{pair[0]:g}_{pair[1]:g}" / "frozen_models.json"
            baseline = json.loads(existing.read_text())
            if baseline["training_ids_sha256"] != training_hash:
                raise ValueError("existing cello baseline source IDs differ")
            payload["baseline_provenance"] = {"path": str(existing), "sha256": hashlib.sha256(existing.read_bytes()).hexdigest()}
            payload["baseline_models"] = {method: baseline["models"][method] for method in ("learned_alpha", "raw_3d")}
            payload["baseline_source_selection"] = baseline["selection"]
        else:
            candidates, selected = ms.select_source(X[train], y[train], family[train], data["beta_level"][train],
                                                     data["speed"][train], data["ids"][train], directory / "baseline_cv.jsonl")
            payload["baseline_source_selection"] = candidates
            for method in ("learned_alpha", "raw_3d"):
                c = candidates[selected[method]]
                model = ms.fit_estimator(X[train], y[train], alpha=c["alpha"], C=c["C"])
                payload["baseline_models"][method] = ms.freeze_estimator(model, c["alpha"], c["C"])
        path = directory / "frozen_models.json"
        ms.write_json(path, payload, exclusive=True)
        frozen[key] = {"path": path, "domain": domain, "test": test}
        print(f"FROZEN {key}: warp={selection['model']['theta'][6:]}, "
              f"lambda={selection['model']['lambda']}, warp_lambda={selection['model']['warp_lambda']}", flush=True)
    ms.write_json(output / "all_models_frozen.json", {key: {"path": str(value["path"].relative_to(output)),
                  "sha256": hashlib.sha256(value["path"].read_bytes()).hexdigest()} for key, value in frozen.items()}, exclusive=True)
    results = {}
    for key, item in frozen.items():
        if not (output / "all_models_frozen.json").exists():
            raise AssertionError("all source models must be frozen before target scoring")
        data = datasets[item["domain"]]
        payload = json.loads(item["path"].read_text())
        probabilities = {"affine_flow": predict_frozen(payload["flow"]["model"], data["X"])}
        probabilities.update({name: ms.predict_frozen(model, data["X"]) for name, model in payload["baseline_models"].items()})
        results[key] = {"domain": item["domain"], "source": payload["source"], "training_speeds": payload["training_speeds"],
                        "warp": payload["flow"]["model"]["theta"][6:], "methods": {}}
        for method, probability in probabilities.items():
            results[key]["methods"][method] = {scope: evaluate_panels(probability, data["y_fixed"] == 1,
                data["family"], data["group"], mask) for scope, mask in item["test"].items()}
    ms.write_json(output / "results.json", results, exclusive=True)
    aggregates = {}
    for domain in datasets:
        selected = [r for r in results.values() if r["domain"] == domain]
        aggregates[domain] = {"source_experiments": len(selected), "methods": {}}
        if domain == "published":
            aggregates[domain]["scope_warning"] = "combined=held-out D speeds only; other_specimens=E boundary-focused observed-cell diagnostic, not generalization headline"
        for method in ("affine_flow", "learned_alpha", "raw_3d"):
            aggregates[domain]["methods"][method] = {}
            for scope in ("combined", "source_heldout_speed", "other_specimens"):
                aggregates[domain]["methods"][method][scope] = {aggregation: {metric: float(np.mean([
                    r["methods"][method][scope][aggregation][metric] for r in selected]))
                    for metric in ("nominal_iou", "family_min_iou")} for aggregation in ("macro_panels", "global_same_rule")}
        aggregates[domain]["paired_combined_family_macro"] = {baseline: {
            "mean_difference": float(np.mean([r["methods"]["affine_flow"]["combined"]["macro_panels"]["family_min_iou"] -
                r["methods"][baseline]["combined"]["macro_panels"]["family_min_iou"] for r in selected])),
            "wins": sum(r["methods"]["affine_flow"]["combined"]["macro_panels"]["family_min_iou"] >
                r["methods"][baseline]["combined"]["macro_panels"]["family_min_iou"] + 1e-12 for r in selected),
            "losses": sum(r["methods"]["affine_flow"]["combined"]["macro_panels"]["family_min_iou"] <
                r["methods"][baseline]["combined"]["macro_panels"]["family_min_iou"] - 1e-12 for r in selected)}
            for baseline in ("learned_alpha", "raw_3d")}
    summary = {"status": protocol["status"], "aggregate": aggregates,
               "independence_caveat": "overlapping targets and source speed pairs are not independent replicates",
               "selected_warps": {key: row["warp"] for key, row in results.items()}, "seconds": time.monotonic() - started}
    ms.write_json(output / "summary.json", summary, exclusive=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cello-baselines", type=Path, default=ROOT / "reports/multispeed_scaling/2026-10-13")
    parser.add_argument("--cello-only", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, cello_baselines=args.cello_baselines, include_published=not args.cello_only)
