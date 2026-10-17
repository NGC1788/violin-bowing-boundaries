#!/usr/bin/env python3
"""Exact shared-observer ranking sensitivity of existing development predictions.

No fitting and no sample2 access. A difference of separately minimized scores
is a valid maximin preference, but is not a uniform-dominance certificate.
The exhaustive 2D histogram/cumulative-sum calculation is established machinery.
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
from threadpoolctl import threadpool_limits

import affine_transport as af
import benchmark_data as bd
import benchmark_sets as bs
from continuous_observer import ContinuousObserver
import multispeed_scaling as ms
from set_envelope import family_ious, label_family

ROOT = bd.ROOT
TOLERANCE = 1e-12
GRID = tuple(itertools.product((.3, .4, .5, .6, .7), (.75, .8, .85)))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, payload):
    with Path(path).open("x") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


class DenseObserver:
    """Exact IoUs for every threshold state; one shared geometry for all models."""

    def __init__(self, observer, *, max_states=1_000_000):
        self.observer = observer
        self.fractions = observer.f_states
        self.cutoffs = observer.q_states[::-1]
        self.shape = len(self.cutoffs), len(self.fractions)
        if np.prod(self.shape) > max_states:
            raise ValueError("declared dense laptop state budget exceeded")
        self.rows = observer.order
        self.q_index = np.searchsorted(-self.cutoffs, -observer.periodicity[self.rows], side="left")
        self.f_index = observer.starts

    def counts(self, weights):
        result = np.zeros(self.shape, dtype=np.float64)
        np.add.at(result, (self.q_index, self.f_index), np.asarray(weights)[self.rows])
        np.cumsum(result, axis=0, out=result)
        np.cumsum(result, axis=1, out=result)
        return result

    def scores(self, predictions, subset=None):
        p = np.asarray(predictions)
        if p.ndim != 2 or p.shape[1] != len(self.observer.ratio) or not np.isin(p, (0, 1)).all():
            raise ValueError("predictions must be an aligned model-by-row binary array")
        p = p.astype(bool)
        keep = np.ones(p.shape[1], bool) if subset is None else np.asarray(subset, bool)
        if keep.shape != (p.shape[1],) or not keep.any():
            raise ValueError("each evaluated subset must contain rows")
        target_count = self.counts(keep)
        result = np.empty((len(p), *self.shape))
        for index, prediction in enumerate(p):
            tp = self.counts(prediction & keep)
            union = np.count_nonzero(prediction & keep) + target_count - tp
            np.divide(tp, union, out=result[index], where=union > 0)
            result[index][union == 0] = 1.
        return result

    def witness(self, flat_index):
        q, f = np.unravel_index(flat_index, self.shape)
        return float(self.fractions[f]), float(self.cutoffs[q])

    def grid_indices(self):
        return [(np.searchsorted(-self.cutoffs, -q, side="right") - 1,
                 np.searchsorted(self.fractions, f, side="right") - 1) for f, q in GRID]


def is_reversal(lower, upper, margin=TOLERANCE):
    return bool(lower < -margin and upper > margin)


def ranking_opposed(rank_difference, lower, upper):
    return bool((rank_difference > TOLERANCE and lower < -TOLERANCE)
                or (rank_difference < -TOLERANCE and upper > TOLERANCE))


def pair_records(names, dense, grid, geometry, *, task, scope, original_macro=None):
    dense = np.asarray(dense).reshape(len(names), -1)
    grid = np.asarray(grid)
    rows = []
    for a, b in itertools.combinations(range(len(names)), 2):
        difference = dense[a] - dense[b]
        imin, imax = int(np.argmin(difference)), int(np.argmax(difference))
        lower, upper = float(difference[imin]), float(difference[imax])
        grid_difference = grid[a] - grid[b]
        grid_min, grid_max = float(grid_difference.min()), float(grid_difference.max())
        independent = float(dense[a].min() - dense[b].min())
        grid_independent = float(grid[a].min() - grid[b].min())
        fmin, qmin = geometry.witness(imin)
        fmax, qmax = geometry.witness(imax)
        row = {"task": task, "scope": scope, "model_a": names[a], "model_b": names[b],
            "continuous_difference_min": lower, "continuous_difference_max": upper,
            "grid_difference_min": grid_min, "grid_difference_max": grid_max,
            "continuous_reversal": is_reversal(lower, upper), "grid_reversal": is_reversal(grid_min, grid_max),
            "grid_misses_continuous_reversal": is_reversal(lower, upper) and not is_reversal(grid_min, grid_max),
            "continuous_independent_min_difference": independent,
            "grid_independent_min_difference": grid_independent,
            "independent_min_ranking_has_opposing_rule": ranking_opposed(independent, lower, upper),
            "grid_independent_min_ranking_has_opposing_rule": ranking_opposed(grid_independent, grid_min, grid_max),
            "individual_min_a": float(dense[a].min()), "individual_min_b": float(dense[b].min()),
            "minimum_witness_fraction": fmin, "minimum_witness_periodicity": qmin,
            "minimum_witness_iou_a": float(dense[a, imin]), "minimum_witness_iou_b": float(dense[b, imin]),
            "maximum_witness_fraction": fmax, "maximum_witness_periodicity": qmax,
            "maximum_witness_iou_a": float(dense[a, imax]), "maximum_witness_iou_b": float(dense[b, imax]),
            "candidate_threshold_pairs": int(dense.shape[1]),
            "reversal_beyond_001": is_reversal(lower, upper, .001),
            "reversal_beyond_005": is_reversal(lower, upper, .005),
            "reversal_beyond_010": is_reversal(lower, upper, .01)}
        if original_macro is not None:
            delta = float(original_macro[a] - original_macro[b])
            row.update(original_grid_mean_panel_min_difference=delta,
                       original_grid_macro_ranking_has_opposing_shared_rule=ranking_opposed(delta, lower, upper))
        rows.append(row)
    return rows


def audit_task(data, probabilities, mask, task):
    names = list(probabilities)
    p = np.asarray([probabilities[name][mask] >= .5 for name in names])
    features = {key: value[mask] for key, value in data["output_features"].items()}
    groups = data["group"][mask]
    family = label_family(features, data["reference_f0_hz"])
    observer = ContinuousObserver.from_features(features, data["reference_f0_hz"])
    geometry = DenseObserver(observer)
    dense = geometry.scores(p)
    grid = np.asarray([family_ious(prediction, family) for prediction in p])
    mapped = np.asarray([[row[q, f] for q, f in geometry.grid_indices()] for row in dense])
    np.testing.assert_allclose(mapped, grid, atol=1e-13, rtol=0)
    records = pair_records(names, dense, grid, geometry, task=task, scope="pooled")
    del dense
    macro = np.zeros((len(names), *geometry.shape))
    macro_grid = np.zeros_like(grid)
    original_macro = np.zeros(len(names))
    unique_groups = np.unique(groups)
    for group in unique_groups:
        panel = groups == group
        panel_dense = geometry.scores(p, panel)
        macro += panel_dense / len(unique_groups)
        del panel_dense
        # Smaller panel-specific geometry enumerates the same label possibilities
        # and avoids reporting duplicate states from other panels' features.
        panel_observer = ContinuousObserver.from_features({key: value[panel] for key, value in features.items()},
                                                         data["reference_f0_hz"])
        panel_geometry = DenseObserver(panel_observer)
        panel_dense = panel_geometry.scores(p[:, panel])
        panel_grid = np.asarray([family_ious(prediction[panel], family[panel]) for prediction in p])
        panel_mapped = np.asarray([[row[q, f] for q, f in panel_geometry.grid_indices()] for row in panel_dense])
        np.testing.assert_allclose(panel_mapped, panel_grid, atol=1e-13, rtol=0)
        macro_grid += panel_grid / len(unique_groups)
        original_macro += panel_grid.min(axis=1) / len(unique_groups)
        records.extend(pair_records(names, panel_dense, panel_grid, panel_geometry, task=task, scope=f"panel:{group}"))
    records.extend(pair_records(names, macro, macro_grid, geometry, task=task, scope="macro_shared",
                                original_macro=original_macro))
    return records, {"task": task, "n": int(mask.sum()), "panels": len(unique_groups), "models": names,
        "threshold_state_shape": list(geometry.shape), "candidate_threshold_pairs": int(np.prod(geometry.shape)),
        "evaluation_ids_sha256": bd.id_hash(data["ids"][mask])}


def benchmark_tasks(base, data):
    for source in "ABCD":
        directory = Path(base) / f"benchmark_{source}"
        saved = np.load(directory / "predictions.npz", allow_pickle=False)
        np.testing.assert_array_equal(saved["ids"], data["ids"])
        np.testing.assert_array_equal(saved["family"], label_family(data["output_features"], data["reference_f0_hz"]))
        expected_source = (data["string"] == source) & (data["speed"] == .1)
        np.testing.assert_array_equal(saved["source_mask"], expected_source)
        results = json.loads((directory / "results.json").read_text())
        names = [name for name, result in results.items() if result["selection"] == "family_cv"
                 and not result["historical_diagnostic"]]
        if len(names) != 14:
            raise ValueError("exactly the fourteen originally declared models are required")
        selection = json.loads((directory / "selection.json").read_text())
        standard = [name for name in names if name.endswith(("_logistic", "_svc", "_extra_trees", "_hgb"))]
        baseline = max(standard, key=lambda name: selection[name]["cv"][selection[name]["primary_index"]]["metrics"]["family_min_iou"])
        yield f"benchmark_{source}", {name: saved[name] for name in names}, ~expected_source, {
            "selected_simple_baseline": baseline, "comparison_candidate": "covariant_family_robust",
            "input_sha256": {name: digest(directory / name) for name in ("predictions.npz", "results.json", "selection.json", "manifest.json")}}


def transport_tasks(full_run, k0_run, data):
    full_run, k0_run = Path(full_run), Path(k0_run)
    manifests = {run: json.loads((run / "all_models_frozen.json").read_text()) for run in (full_run, k0_run)}
    for source, pair in itertools.product("ABCD", ms.SPEED_PAIRS):
        key = f"cello__{source}__{pair[0]:g}_{pair[1]:g}"
        payloads, hashes = [], {}
        for run in (full_run, k0_run):
            entry = manifests[run][key]
            path = (run / entry["path"]).resolve()
            if not path.is_relative_to(run.resolve()) or digest(path) != entry["sha256"]:
                raise ValueError("transport development model hash/path mismatch")
            payloads.append(json.loads(path.read_text()))
            hashes["flow" if run == full_run else "k0"] = entry["sha256"]
        full, zero = payloads
        train, masks = ms.build_masks(data, source, pair)
        if any(p["training_ids_sha256"] != bd.id_hash(data["ids"][train]) for p in payloads):
            raise ValueError("transport source membership changed")
        if zero["full_flow_model"] != full["flow"]["model"]:
            raise ValueError("matched k0 comparison refers to another full-flow model")
        p = {"affine_flow": af.predict_frozen(full["flow"]["model"], data["X"]),
             "translation_k0": af.predict_frozen(zero["translation"]["model"], data["X"])}
        p.update({name: ms.predict_frozen(model, data["X"]) for name, model in full["baseline_models"].items()})
        candidates = full["baseline_source_selection"]
        baseline_scores = {}
        for method, model in full["baseline_models"].items():
            matching = [c for c in candidates if c["alpha"] == model["alpha"] and c["C"] == model["C"]]
            if len(matching) != 1:
                raise ValueError("baseline source-selection provenance ambiguous")
            baseline_scores[method] = matching[0]["metrics"]["family_min_iou"]
        baseline = max(("learned_alpha", "raw_3d"), key=baseline_scores.__getitem__)
        yield key, p, masks["combined"], {"selected_simple_baseline": baseline,
            "comparison_candidate": "affine_flow", "input_sha256": hashes}


def summarize(rows):
    return {"comparisons": len(rows), "grid_reversals": sum(r["grid_reversal"] for r in rows),
        "continuous_reversals": sum(r["continuous_reversal"] for r in rows),
        "grid_missed_reversals": sum(r["grid_misses_continuous_reversal"] for r in rows),
        "independent_min_ranking_opposed": sum(r["independent_min_ranking_has_opposing_rule"] for r in rows),
        "reversals_beyond_001": sum(r["reversal_beyond_001"] for r in rows),
        "reversals_beyond_005": sum(r["reversal_beyond_005"] for r in rows),
        "reversals_beyond_010": sum(r["reversal_beyond_010"] for r in rows),
        "maximum_grid_minimum_optimism": max((r["grid_difference_min"] - r["continuous_difference_min"] for r in rows), default=0.),
        "maximum_grid_maximum_pessimism": max((r["continuous_difference_max"] - r["grid_difference_max"] for r in rows), default=0.)}


def oriented_comparison(rows, task, scope, candidate, baseline):
    row = next(r for r in rows if r["task"] == task and r["scope"] == scope
               and {r["model_a"], r["model_b"]} == {candidate, baseline})
    same = row["model_a"] == candidate
    return {"task": task, "scope": scope, "candidate": candidate, "baseline": baseline,
        "continuous_difference_min": row["continuous_difference_min"] if same else -row["continuous_difference_max"],
        "continuous_difference_max": row["continuous_difference_max"] if same else -row["continuous_difference_min"],
        "grid_difference_min": row["grid_difference_min"] if same else -row["grid_difference_max"],
        "grid_difference_max": row["grid_difference_max"] if same else -row["grid_difference_min"],
        "independent_min_difference": row["continuous_independent_min_difference"] * (1 if same else -1),
        "reversal": row["continuous_reversal"], "grid_missed_reversal": row["grid_misses_continuous_reversal"]}


def run(out, benchmark_base, full_run, k0_run):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("preserve previous audits; choose an empty directory")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(), "scope": "retrospective shared-observer audit; no sample2 access",
        "prediction": "threshold frozen probabilities at .5; no predictor fitting or selection",
        "comparisons": "all unordered model pairs: original 14-model family-CV winners for A-D; all four transport models for all 12 source/speed pairs",
        "observer": "fixed A-derived eligibility plus ratio < f, periodicity >= q; rectangle f=[.3,.7], q=[.75,.85]",
        "aggregation": "pooled IoU; mean panel IoU under ONE shared operator; each individual panel",
        "grid": GRID, "exact": "enumerate all attainable threshold states by 2D histogram and cumulative sums; duplicate label states allowed",
        "empty_union_iou": 1., "reversal_tolerance": TOLERANCE, "fixed_effect_margins": [.001, .005, .01],
        "not_claimed": "Maximin preference is not erroneous; it simply does not imply same-observer dominance. No calibrated truth coverage, significance or new algorithm claim.",
        "source_hashes": {name: digest(ROOT / "scripts" / name) for name in
                          (Path(__file__).name, "continuous_observer.py", "benchmark_data.py", "affine_transport.py", "multispeed_scaling.py")}}
    save(out / "protocol.json", protocol)
    data = bd.load_dataset(include_ambiguous=True)
    rows, tasks = [], []
    for task, probabilities, mask, provenance in itertools.chain(benchmark_tasks(benchmark_base, data),
                                                               transport_tasks(full_run, k0_run, data)):
        begin = time.perf_counter()
        records, metadata = audit_task(data, probabilities, mask, task)
        metadata.update(provenance, seconds=time.perf_counter() - begin)
        rows.extend(records)
        tasks.append(metadata)
        print(json.dumps({"task": task, "seconds": metadata["seconds"],
                          **summarize([r for r in records if r["scope"] == "pooled"])}), flush=True)
    summaries = {}
    for group in ("benchmark", "transport"):
        subset = [r for r in rows if r["task"].startswith("benchmark") == (group == "benchmark")]
        summaries[group] = {scope: summarize([r for r in subset if
            (r["scope"].startswith("panel:") if scope == "panels" else r["scope"] == scope)])
            for scope in ("pooled", "macro_shared", "panels")}
    highlights = []
    for task in tasks:
        for scope in ("pooled", "macro_shared"):
            highlights.append(oriented_comparison(rows, task["task"], scope, task["comparison_candidate"], task["selected_simple_baseline"]))
            if task["task"].startswith("cello"):
                highlights.append(oriented_comparison(rows, task["task"], scope, "affine_flow", "translation_k0"))
    output = {"protocol": protocol, "counts": summaries, "tasks": tasks, "declared_contrasts": highlights,
              "total_pair_scope_comparisons": len(rows), "seconds": time.perf_counter() - started}
    save(out / "summary.json", output)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with (out / "all_pairs.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"counts": summaries, "seconds": output["seconds"]}, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--benchmark-base", type=Path, default=ROOT / "reports/research_20261013")
    parser.add_argument("--full-run", type=Path, default=ROOT / "reports/affine_transport/2026-10-13-v2")
    parser.add_argument("--k0-run", type=Path, default=ROOT / "reports/affine_width_ablation/2026-10-13")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.out, args.benchmark_base, args.full_run, args.k0_run)
