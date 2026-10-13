#!/usr/bin/env python3
"""Search friction and loss parameters so the simulated regimes match the measured ones.

The waveguide model's friction curve and reflection filter are not measured quantities; the
literature values are a starting point. This searches them against a measured diagram and
writes every evaluation to a JSONL file, so a long run can be stopped and resumed and the
whole search is auditable.

Nothing is written per stroke: each candidate is simulated in memory, given the measured
observation noise, labelled with the same classifier as the measurements, and scored on the
same cells. The score is the Helmholtz intersection over union on the sampled cells, with
label agreement reported beside it.

Calibrate on part of the data and test on the rest: --conditions picks the training bow
speeds and --hold-out reports the score on the remaining ones without fitting them.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import hashlib
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

import bowed_string as bs
import regime_map
import simulate_grid
import trial_audit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RATE = trial_audit.PUBLISHED_RATE_HZ
MODEL_VERSION = "passive-point-bow-waveguide-v2"
# (low, high, log-scaled) for the search
SPACE = {"mu_s": (0.45, 1.20, False), "mu_d": (0.15, 0.55, False), "v0": (0.01, 0.40, True),
         "q1": (300.0, 6000.0, True), "corner_hz": (1500.0, 20000.0, True)}


def sample_cells(rows, levels: int, ranks: int, seed: int = 0) -> list[dict]:
    """Whole force columns at evenly spaced beta levels: enough to see where a boundary sits."""
    by_level: dict[int, list[dict]] = {}
    for row in rows:
        by_level.setdefault(int(row["beta_level"]), []).append(row)
    chosen_levels = np.unique(np.linspace(0, len(by_level) - 1, min(levels, len(by_level))).astype(int))
    picked = []
    for index in chosen_levels:
        level = sorted(by_level)[index]
        column = sorted(by_level[level], key=lambda r: int(r["force_rank"]))
        take = np.unique(np.linspace(0, len(column) - 1, min(ranks, len(column))).astype(int))
        picked += [column[i] for i in take]
    return picked


def load_cells(reports: Path, cache_root: Path, diagram: str, levels: int, ranks: int):
    """Measured rows (with labels and grid position) plus the profiles needed to drive them."""
    _, _, rows, regime, labels = simulate_grid.load_measured(reports, diagram)
    positions = grid_positions(reports, diagram)
    by_condition: dict[str, list[dict]] = {}
    for row in rows:
        key = (row["condition"], int(row["trial"]))
        if key not in labels or key not in positions:
            continue
        entry = dict(row)
        entry["label"] = labels[key]
        entry.update(positions[key])
        by_condition.setdefault(row["condition"], []).append(entry)
    selected = {name: sample_cells(group, levels, ranks) for name, group in by_condition.items()}
    reference_f0 = {name: regime["conditions"][name]["reference_f0_hz"] for name in selected}
    settings = {"impedance": regime["flyback_law"]["impedance_kg_s"]["median"],
                "noise": regime["noise_floor"].get("control_median") or 0.0,
                "floor": regime["noise_floor"]["min_amplitude"], "reference_f0": reference_f0}
    profiles = {}
    for name, group in selected.items():
        directory = cache_root / diagram / name
        for row in group:
            number = int(row["trial"])
            profiles[(name, number)] = np.load(trial_audit.cache_path(directory, number, "profile"))
    return selected, profiles, settings


def grid_positions(reports: Path, diagram: str) -> dict:
    """beta level and force rank per trial, from the measured regime table."""
    import csv
    run = simulate_grid.latest_run(reports / diagram / "regime_map")
    positions = {}
    with (run / "regimes.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            positions[(row["condition"], int(row["trial"]))] = {
                "beta_level": int(row["beta_level"]), "force_rank": int(row["force_rank"]),
                "beta": float(row["beta"])}
    return positions


def simulate_labels(backend, params_list, cells, profiles, settings, rate, seed, batch_size=256):
    """Labels for the sampled cells, for every parameter set at once; no files are written.

    Each candidate is another stroke in the same batch, because the per-step cost is dominated
    by launch overhead: evaluating 16 parameter sets together costs about the same as one.
    """
    sets = len(params_list)
    labels: list[dict] = [dict() for _ in params_list]
    thresholds = {**regime_map.DEFAULT_THRESHOLDS, "min_amplitude": settings["floor"]}
    for name, group in cells.items():
        group = sorted(group, key=lambda r: int(r["window_start"]))
        for offset in range(0, len(group), batch_size):
            batch = group[offset:offset + batch_size]
            width = len(batch)
            numbers = [int(r["trial"]) for r in batch]
            stack = [profiles[(name, n)] for n in numbers]
            length = max(p.shape[0] for p in stack)
            padded = np.stack([np.concatenate([p, np.repeat(p[-1:], length - p.shape[0], axis=0)]) for p in stack], axis=1)
            padded = np.tile(padded, (1, sets, 1))
            onsets = []
            for p in stack:
                moving = np.flatnonzero(p[:, 1] > simulate_grid.ONSET_SPEED)
                onsets.append(int(moving[0]) * trial_audit.PROFILE_DECIMATION if moving.size else 0)
            starts = [int(r["window_start"]) for r in batch]
            ends = [int(r["window_end"]) for r in batch]
            plan = simulate_grid.batch_window(starts, ends, onsets, rate)
            per_stroke = lambda key: np.repeat([p[key] for p in params_list], width)  # noqa: E731
            string = bs.StringParams(f0=settings["reference_f0"][name], impedance=settings["impedance"],
                                     q1=per_stroke("q1"), corner_hz=per_stroke("corner_hz"))
            friction = bs.FrictionParams(per_stroke("mu_s"), per_stroke("mu_d"), per_stroke("v0"))
            bridge = bs.simulate(backend, string, friction, np.tile([float(r["beta"]) for r in batch], sets),
                                 padded[:, :, 1], padded[:, :, 0], RATE / trial_audit.PROFILE_DECIMATION, rate,
                                 plan["first_sample"] / RATE, plan["steps"], plan["every"], plan["record_from"])
            for k in range(sets):
                for i, (n, start, end) in enumerate(zip(numbers, starts, ends)):
                    window = bridge[start - plan["record_start"]:end + 1 - plan["record_start"], k * width + i].astype(float)
                    window = window + np.random.default_rng(seed + n).normal(0.0, settings["noise"], window.size)
                    periodicity, f0 = regime_map.dominant_period(window)
                    shape = regime_map.flyback_structure(window, RATE / f0 if math.isfinite(f0) else math.nan)
                    labels[k][(name, n)] = regime_map.classify(float(np.std(window)), periodicity, f0,
                                                               shape["flybacks_per_period"],
                                                               settings["reference_f0"][name], thresholds)
    return labels


def score(cells, simulated) -> dict:
    """Helmholtz IoU (the objective) and plain label agreement, over the sampled cells."""
    total = both = either = agree = 0
    for name, group in cells.items():
        for row in group:
            key = (name, int(row["trial"]))
            if key not in simulated:
                continue
            total += 1
            measured, predicted = row["label"], simulated[key]
            agree += measured == predicted
            both += measured == predicted == "helmholtz"
            either += "helmholtz" in (measured, predicted)
    return {"cells": total, "helmholtz_iou": round(both / either, 4) if either else 0.0,
            "agreement": round(agree / total, 4) if total else 0.0}


def parse_space(overrides, space=SPACE) -> dict:
    """``name=low:high`` strings override the search box; the scale of the original entry is kept."""
    updated = dict(space)
    for item in overrides or []:
        name, _, bounds = item.partition("=")
        if name not in space or bounds.count(":") != 1:
            raise ValueError(f"--space expects name=low:high with name in {sorted(space)}: {item!r}")
        low, high = (float(v) for v in bounds.split(":"))
        if not 0 < low < high:
            raise ValueError(f"--space needs 0 < low < high: {item!r}")
        updated[name] = (low, high, space[name][2])
    return updated


def draw(rng, space=SPACE, center: dict | None = None, factor: float = 1.6) -> dict:
    """A random set inside the box; with ``center``, inside a factor of that set (refinement)."""
    values = {}
    for name, (low, high, log) in space.items():
        if center and name in center:
            low, high, log = max(low, center[name] / factor), min(high, center[name] * factor), True
            if not low < high:
                low, high, log = space[name][0], space[name][1], space[name][2]
        u = rng.random()
        values[name] = float(math.exp(math.log(low) + u * (math.log(high) - math.log(low))) if log
                             else low + u * (high - low))
    if values["mu_d"] >= values["mu_s"]:  # a falling friction curve is required for stick-slip
        values["mu_d"] = values["mu_s"] * 0.5
    return values


def evaluate(backend, params_list, cells, profiles, settings, rate, seed) -> list[dict]:
    began = time.monotonic()
    passive = [all(bool(np.all(bs.reflection_is_passive(bs.StringParams(f0=f0, q1=p["q1"],
                    corner_hz=p["corner_hz"]), rate))) for name in cells
                   for f0 in [settings["reference_f0"][name]]) for p in params_list]
    valid_params = [p for p, valid in zip(params_list, passive) if valid]
    simulated = iter(simulate_labels(backend, valid_params, cells, profiles, settings, rate, seed) if valid_params else [])
    elapsed = round((time.monotonic() - began) / max(1, len(params_list)), 1)
    results = []
    for params, valid in zip(params_list, passive):
        if not valid:
            results.append({"params": {k: float(v) for k,v in params.items()}, "status": "rejected_nonpassive",
                "helmholtz_iou": -1., "agreement": 0., "cells": 0, "seconds": 0., "simulated_classes": {}})
            continue
        labels = next(simulated)
        counts: dict[str, int] = {}
        for label in labels.values():
            counts[label] = counts.get(label, 0) + 1
        results.append({"params": {k: float(v) for k, v in params.items()}, "status": "evaluated", **score(cells, labels),
                        "seconds": elapsed, "simulated_classes": counts})
    return results


def best_of(paths) -> dict | None:
    rows = [json.loads(line) for path in paths if Path(path).exists()
            for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(row.get("model_version") != "passive-point-bow-waveguide-v2" for row in rows):
        raise ValueError("Cannot seed corrected-model calibration from legacy or unversioned scores")
    return max((r for r in rows if r.get("status") != "rejected_nonpassive"), key=lambda r: r["helmholtz_iou"], default=None)


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False,
                                    default=lambda item: item.item() if isinstance(item, np.generic) else str(item)).encode()).hexdigest()


def run_specification(diagram, reports, cells, profiles, settings, backend, rate, levels, ranks,
                      seed, train, test, space, around, center, seed_from):
    """Bind a calibration sequence to actual numerical inputs, data files and code.

    Iteration budget and candidate batch size are intentionally excluded so an
    unchanged search can be extended or rebatched. Refinement center is frozen.
    """
    rows = {name: sorted(group, key=lambda row: int(row["trial"])) for name,group in sorted(cells.items())}
    profile_digest = hashlib.sha256()
    for (name, number), profile in sorted(profiles.items()):
        array = np.ascontiguousarray(profile)
        profile_digest.update(json.dumps([name, int(number), list(array.shape), array.dtype.str], separators=(",", ":")).encode()+b"\n")
        profile_digest.update(array.tobytes())
    # Record the source report bytes as well as the selected numerical values.
    source_files = []
    source_root = Path(reports) / diagram
    for pattern in ("trial_audit/*/trials.csv", "trial_audit/*/summary.json", "regime_map/*/regimes.csv", "regime_map/*/summary.json"):
        for path in sorted(source_root.glob(pattern)):
            source_files.append({"path": path.relative_to(Path(reports)).as_posix(),
                                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    seeds = [{"name": Path(p).name, "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()}
             for p in seed_from or [] if Path(p).exists()]
    specification = {"model_version": MODEL_VERSION, "diagram": diagram, "rate": rate, "levels": levels,
                     "ranks": ranks, "seed": seed, "train_conditions": train, "test_conditions": test,
                     "space": space, "refinement": {"factor": around, "center": center, "sources": seeds},
                     "backend": {"name": backend.name, "dtype": str(getattr(backend, "dtype", "unspecified")),
                                 "device": str(getattr(backend, "device", "cpu"))},
                     "selected_rows_sha256": _json_hash(rows), "profiles_sha256": profile_digest.hexdigest(),
                     "settings": settings, "source_files": source_files,
                     "split_ids": {"train": [[name,int(row["trial"])] for name in train for row in rows[name]],
                                   "test": [[name,int(row["trial"])] for name in test for row in rows[name]]},
                     "implementation_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                                for name in ("calibrate.py","bowed_string.py","simulate_grid.py","regime_map.py","trial_audit.py")}}
    # Round-trip to ensure the serialized form, not a tuple/array representation, is hashed.
    specification = json.loads(json.dumps(specification, default=lambda x:x.item() if isinstance(x,np.generic) else str(x), allow_nan=False))
    return specification, _json_hash(specification)


def run(diagram: str, reports: Path, cache_root: Path, out_path: Path, backend, rate: int, levels: int, ranks: int,
        iterations: int, seed: int, conditions: list[str] | None, hold_out: bool, progress=print,
        batch_params: int = 16, space: dict | None = None, around: float | None = None,
        seed_from: list | None = None) -> dict:
    if iterations < 0 or levels < 1 or ranks < 1 or batch_params < 1:
        raise ValueError("iterations must be nonnegative; levels, ranks and batch_params must be positive")
    if around is not None and (not math.isfinite(around) or around <= 1):
        raise ValueError("refinement factor must be finite and greater than one")
    cells, profiles, settings = load_cells(reports, cache_root, diagram, levels, ranks)
    all_conditions = sorted(cells)
    train = [c for c in all_conditions if not conditions or c in conditions]
    if not train:
        raise ValueError(f"no conditions selected; available: {all_conditions}")
    test = [c for c in all_conditions if c not in train] if hold_out else []
    train_cells = {c: cells[c] for c in train}
    test_cells = {c: cells[c] for c in test}
    progress(f"CALIBRATE {diagram}: train {train} ({sum(len(v) for v in train_cells.values())} cells), "
             f"test {test or 'none'}, rate {rate}, backend {backend.name}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = []
    manifest_path = out_path.with_suffix(out_path.suffix + ".run.json")
    previous_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if out_path.exists():
        done = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if any(row.get("model_version") != "passive-point-bow-waveguide-v2" for row in done):
            raise ValueError("Cannot resume corrected-model calibration into legacy or unversioned scores; use a new output file")
        if done and (previous_manifest is None or any(not row.get("run_fingerprint") for row in done)):
            raise ValueError("Cannot resume legacy/unbound scores without a complete run fingerprint; use a new output file")
        if [row.get("index") for row in done] != list(range(len(done))):
            raise ValueError("Calibration log indices are not a complete ordered sequence")
        progress(f"  resuming after {len(done)} evaluations")
    best = max((r for r in done if r.get("status") != "rejected_nonpassive"), key=lambda r: r["helmholtz_iou"], default=None)
    space = space or SPACE
    center = None
    if around:
        if previous_manifest:
            center = previous_manifest["specification"]["refinement"]["center"]
        else:
            source = best_of(list(seed_from or []) + [out_path])
            if not source:
                raise ValueError("--around-best needs earlier evaluations; pass --seed-from with a calibration file")
            center = source["params"]
        progress(f"  refining within a factor of {around} around fixed center: {center}")
    specification, fingerprint = run_specification(diagram, reports, cells, profiles, settings, backend, rate,
                                                   levels, ranks, seed, train, test, space, around, center, seed_from)
    if previous_manifest:
        if (previous_manifest.get("run_fingerprint") != fingerprint
                or _json_hash(previous_manifest.get("specification")) != fingerprint
                or any(row.get("run_fingerprint") != fingerprint for row in done)):
            raise ValueError("Calibration run fingerprint mismatch: data, split, settings, code or search configuration changed; use a new output file")
    else:
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump({"run_fingerprint": fingerprint, "specification": specification},handle,indent=2,allow_nan=False)
    rng = np.random.default_rng(seed)
    for _ in range(len(done)):
        draw(rng, space, center, around or 1.6)  # keep the sequence aligned with the saved evaluations
    literature = {"mu_s": bs.FrictionParams.mu_s, "mu_d": bs.FrictionParams.mu_d, "v0": bs.FrictionParams.v0,
                  "q1": bs.StringParams.q1, "corner_hz": bs.StringParams.corner_hz}
    index = len(done)
    while index < iterations:
        # index 0 uses the literature set but still draws, so resuming stays aligned with the sequence
        drawn = [draw(rng, space, center, around or 1.6) for _ in range(min(batch_params, iterations - index))]
        group = [literature if index + k == 0 and not center else candidate for k, candidate in enumerate(drawn)]
        results = evaluate(backend, group, train_cells, profiles, settings, rate, seed)
        for k, result in enumerate(results):
            result.update(index=index + k, kind="literature" if index + k == 0 and not center else "random",
                          model_version="passive-point-bow-waveguide-v2",
                          run_fingerprint=fingerprint,
                          utc=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
            if result.get("status") != "rejected_nonpassive" and (best is None or result["helmholtz_iou"] > best["helmholtz_iou"]):
                best = result
                result["best_so_far"] = True
            with out_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            mark = "  <= best" if result.get("best_so_far") else ""
            progress(f"  [{result['index']}] IoU {result['helmholtz_iou']:.3f} agree {result['agreement']:.3f} "
                     f"({result['seconds']:.0f}s/set) {result['params']}{mark}")
            if result.get("hold_out"):
                progress(f"        hold-out {test}: IoU {result['hold_out']['helmholtz_iou']:.3f} "
                         f"agree {result['hold_out']['agreement']:.3f}")
        index += len(group)
    progress(f"BEST: IoU {best['helmholtz_iou']:.3f} agree {best['agreement']:.3f} params {best['params']}"
             if best else "no evaluations")
    # Evaluate excluded speeds once after the whole search, not after every improvement.
    if best and test_cells:
        held = evaluate(backend, [best["params"]], test_cells, profiles, settings, rate, seed)[0]
        best = {**best, "hold_out": {key:value for key,value in held.items() if key != "params"},
                "hold_out_scope": "Post-search diagnostic; historical source settings may still be target-adapted"}
        if held["status"] == "evaluated":
            progress(f"POST-SEARCH HOLD-OUT {test}: IoU {held['helmholtz_iou']:.3f} agreement {held['agreement']:.3f}")
        else:
            progress(f"POST-SEARCH HOLD-OUT {test}: unavailable ({held['status']}); training result preserved")
    summary_path = out_path.with_suffix(out_path.suffix + ".summary.json")
    summary = {"status": "complete" if best else "no_valid_candidate", "model_version": MODEL_VERSION,
               "run_fingerprint": fingerprint, "evaluations": index, "requested_iterations": iterations,
               "train_conditions": train, "test_conditions": test, "best": best,
               "diagnostic_scope": "Conditional calibration with historically target-adapted nuisance settings; not a fair untouched transfer claim",
               "nuisance_settings": settings}
    summary_path.write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    progress(f"Calibration summary: {summary_path}")
    return best or {}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--diagram", default="2024-03-25_TypeA_sample1")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--levels", type=int, default=6, help="beta levels sampled")
    parser.add_argument("--ranks", type=int, default=25, help="force ranks sampled per level")
    parser.add_argument("--rate", type=int, default=100_000)
    parser.add_argument("--conditions", action="append", help="training folders (repeatable); default: all")
    parser.add_argument("--hold-out", action="store_true", help="score the untrained folders too")
    parser.add_argument("--backend", choices=("auto", "torch", "numpy"), default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--space", action="append",
                        help="widen or narrow the search box: name=low:high (repeatable)")
    parser.add_argument("--around-best", type=float, default=None, metavar="FACTOR",
                        help="refine: sample within this factor of the best set found so far")
    parser.add_argument("--seed-from", action="append", type=Path,
                        help="calibration files to take the refinement centre from (repeatable)")
    parser.add_argument("--batch-params", type=int, default=16,
                        help="parameter sets simulated in one batch (nearly free: the step cost is launch-bound)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    backend = simulate_grid.pick_backend(args.backend, args.device, args.dtype)
    out = args.out or PROJECT_ROOT / "reports" / "calibration" / f"{args.diagram}_rate{args.rate}.jsonl"
    try:
        run(args.diagram, PROJECT_ROOT / "reports" / "collection", PROJECT_ROOT / "data" / "cache", out, backend,
            args.rate, args.levels, args.ranks, args.iterations, args.seed, args.conditions, args.hold_out,
            lambda line: print(line, flush=True), args.batch_params, parse_space(args.space),
            args.around_best, args.seed_from)
        return 0
    except (FileNotFoundError, KeyError, ValueError, regime_map.RegimeError) as error:
        print("CALIBRATE: FAIL —", error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
