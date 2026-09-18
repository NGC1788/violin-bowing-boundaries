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


def simulate_labels(backend, string_kwargs, friction, cells, profiles, settings, rate, seed, batch_size=512):
    """Labels for the sampled cells under one parameter set; no files are written."""
    labels = {}
    for name, group in cells.items():
        string = bs.StringParams(f0=settings["reference_f0"][name], impedance=settings["impedance"], **string_kwargs)
        thresholds = {**regime_map.DEFAULT_THRESHOLDS, "min_amplitude": settings["floor"]}
        group = sorted(group, key=lambda r: int(r["window_start"]))
        for offset in range(0, len(group), batch_size):
            batch = group[offset:offset + batch_size]
            numbers = [int(r["trial"]) for r in batch]
            stack = [profiles[(name, n)] for n in numbers]
            length = max(p.shape[0] for p in stack)
            padded = np.stack([np.concatenate([p, np.repeat(p[-1:], length - p.shape[0], axis=0)]) for p in stack], axis=1)
            onsets = []
            for p in stack:
                moving = np.flatnonzero(p[:, 1] > simulate_grid.ONSET_SPEED)
                onsets.append(int(moving[0]) * trial_audit.PROFILE_DECIMATION if moving.size else 0)
            starts = [int(r["window_start"]) for r in batch]
            ends = [int(r["window_end"]) for r in batch]
            plan = simulate_grid.batch_window(starts, ends, onsets, rate)
            bridge = bs.simulate(backend, string, friction, [float(r["beta"]) for r in batch], padded[:, :, 1],
                                 padded[:, :, 0], RATE / trial_audit.PROFILE_DECIMATION, rate,
                                 plan["first_sample"] / RATE, plan["steps"], plan["every"], plan["record_from"])
            for i, (n, start, end) in enumerate(zip(numbers, starts, ends)):
                window = bridge[start - plan["record_start"]:end + 1 - plan["record_start"], i].astype(float)
                window = window + np.random.default_rng(seed + n).normal(0.0, settings["noise"], window.size)
                periodicity, f0 = regime_map.dominant_period(window)
                shape = regime_map.flyback_structure(window, RATE / f0 if math.isfinite(f0) else math.nan)
                labels[(name, n)] = regime_map.classify(float(np.std(window)), periodicity, f0,
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


def draw(rng, space=SPACE) -> dict:
    values = {}
    for name, (low, high, log) in space.items():
        u = rng.random()
        values[name] = float(math.exp(math.log(low) + u * (math.log(high) - math.log(low))) if log
                             else low + u * (high - low))
    if values["mu_d"] >= values["mu_s"]:  # a falling friction curve is required for stick-slip
        values["mu_d"] = values["mu_s"] * 0.5
    return values


def evaluate(backend, params, cells, profiles, settings, rate, seed) -> dict:
    friction = bs.FrictionParams(params["mu_s"], params["mu_d"], params["v0"])
    string_kwargs = {"q1": params["q1"], "corner_hz": params["corner_hz"]}
    began = time.monotonic()
    simulated = simulate_labels(backend, string_kwargs, friction, cells, profiles, settings, rate, seed)
    result = {"params": {k: round(v, 5) for k, v in params.items()}, **score(cells, simulated),
              "seconds": round(time.monotonic() - began, 1)}
    counts: dict[str, int] = {}
    for label in simulated.values():
        counts[label] = counts.get(label, 0) + 1
    result["simulated_classes"] = counts
    return result


def run(diagram: str, reports: Path, cache_root: Path, out_path: Path, backend, rate: int, levels: int, ranks: int,
        iterations: int, seed: int, conditions: list[str] | None, hold_out: bool, progress=print) -> dict:
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
    if out_path.exists():
        done = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        progress(f"  resuming after {len(done)} evaluations")
    best = max(done, key=lambda r: r["helmholtz_iou"], default=None)
    rng = np.random.default_rng(seed)
    for _ in range(len(done)):
        draw(rng)  # keep the sequence aligned with the saved evaluations
    literature = {"mu_s": bs.FrictionParams.mu_s, "mu_d": bs.FrictionParams.mu_d, "v0": bs.FrictionParams.v0,
                  "q1": bs.StringParams.q1, "corner_hz": bs.StringParams.corner_hz}
    for index in range(len(done), iterations):
        params = literature if index == 0 else draw(rng)
        result = evaluate(backend, params, train_cells, profiles, settings, rate, seed)
        result.update(index=index, kind="literature" if index == 0 else "random",
                      utc=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        if best is None or result["helmholtz_iou"] > best["helmholtz_iou"]:
            best = result
            if test_cells:
                best["hold_out"] = score(test_cells, simulate_labels(
                    backend, {"q1": params["q1"], "corner_hz": params["corner_hz"]},
                    bs.FrictionParams(params["mu_s"], params["mu_d"], params["v0"]),
                    test_cells, profiles, settings, rate, seed))
            result["best_so_far"] = True
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        mark = "  <= best" if result.get("best_so_far") else ""
        progress(f"  [{index}] IoU {result['helmholtz_iou']:.3f} agree {result['agreement']:.3f} "
                 f"({result['seconds']:.0f}s) {result['params']}{mark}")
        if result.get("best_so_far") and "hold_out" in best:
            progress(f"        hold-out {test}: IoU {best['hold_out']['helmholtz_iou']:.3f} "
                     f"agree {best['hold_out']['agreement']:.3f}")
    progress(f"BEST: IoU {best['helmholtz_iou']:.3f} agree {best['agreement']:.3f} params {best['params']}"
             if best else "no evaluations")
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
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    backend = simulate_grid.pick_backend(args.backend, args.device, args.dtype)
    out = args.out or PROJECT_ROOT / "reports" / "calibration" / f"{args.diagram}_rate{args.rate}.jsonl"
    try:
        run(args.diagram, PROJECT_ROOT / "reports" / "collection", PROJECT_ROOT / "data" / "cache", out, backend,
            args.rate, args.levels, args.ranks, args.iterations, args.seed, args.conditions, args.hold_out,
            lambda line: print(line, flush=True))
        return 0
    except (FileNotFoundError, KeyError, ValueError, regime_map.RegimeError) as error:
        print("CALIBRATE: FAIL —", error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
