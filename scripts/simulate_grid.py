#!/usr/bin/env python3
"""Simulate a measured Schelleng diagram with the waveguide model and map it with the same classifier.

Each simulated stroke uses the measured beta, the measured bow-force and bow-velocity history of
the same robot stroke (1 kHz profile from the collection cache) and the measured analysis
window. String pitch is each folder's measured reference f0 and the impedance is the diagram's
flyback-law Z. White noise with the measured no-contact window std is added (observation model),
so tiny periodic motion is judged like the measurements. The simulated windows are written in
the collection cache format with an audit table copied from the measurement, then regime-map
runs unchanged and the labels are compared cell by cell with the measured ones.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np

import bowed_string as bs
import regime_map
import trial_audit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RATE = trial_audit.PUBLISHED_RATE_HZ
ONSET_SPEED = 1e-3      # m/s: a profile block above this counts as motion
ONSET_MARGIN = 2500     # samples simulated before the earliest motion in a batch


def latest_run(directory: Path) -> Path:
    runs = sorted(p for p in directory.iterdir() if p.is_dir()) if directory.is_dir() else []
    if not runs:
        raise FileNotFoundError(f"no runs under {directory}")
    return runs[-1]


def load_measured(reports: Path, diagram: str):
    base = reports / diagram
    audit_run = Path(json.loads((base / "trial_audit" / "latest.json").read_text(encoding="utf-8"))["run_dir"])
    audit = json.loads((audit_run / "summary.json").read_text(encoding="utf-8"))
    with (audit_run / "trials.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, [r for r in reader if not r["error"]]
    regime_run = latest_run(base / "regime_map")
    regime = json.loads((regime_run / "summary.json").read_text(encoding="utf-8"))
    with (regime_run / "regimes.csv").open(newline="", encoding="utf-8") as handle:
        labels = {(r["condition"], int(r["trial"])): r["label"] for r in csv.DictReader(handle)}
    return audit, fields, rows, regime, labels


def batch_window(starts, ends, onsets, rate: int) -> dict:
    """Sample bookkeeping for one batch: first simulated sample, steps, recording offset."""
    if rate % RATE:
        raise ValueError(f"rate must be a multiple of {RATE}")
    every = rate // RATE
    first = max(0, min(onsets) - ONSET_MARGIN)
    first = min(first, min(starts))
    return {"every": every, "first_sample": first, "record_start": min(starts),
            "steps": (max(ends) + 1 - first) * every, "record_from": (min(starts) - first) * every}


def simulate_condition(backend, string, friction, rows, cache_dir: Path, out_dir: Path, rate: int, noise: float,
                       seed: int, batch_size: int, progress=print) -> float:
    rows = sorted(rows, key=lambda r: int(r["window_start"]))
    seconds = 0.0
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset + batch_size]
        numbers = [int(r["trial"]) for r in batch]
        profiles = [np.load(trial_audit.cache_path(cache_dir, n, "profile")) for n in numbers]
        length = max(p.shape[0] for p in profiles)
        padded = np.stack([np.concatenate([p, np.repeat(p[-1:], length - p.shape[0], axis=0)]) for p in profiles], axis=1)
        onsets = []
        for p in profiles:
            moving = np.flatnonzero(p[:, 1] > ONSET_SPEED)
            onsets.append(int(moving[0]) * trial_audit.PROFILE_DECIMATION if moving.size else 0)
        starts = [int(r["window_start"]) for r in batch]
        ends = [int(r["window_end"]) for r in batch]
        plan = batch_window(starts, ends, onsets, rate)
        began = time.monotonic()
        bridge = bs.simulate(backend, string, friction, [float(r["beta"]) for r in batch], padded[:, :, 1],
                             padded[:, :, 0], RATE / trial_audit.PROFILE_DECIMATION, rate,
                             plan["first_sample"] / RATE, plan["steps"], plan["every"], plan["record_from"])
        seconds += time.monotonic() - began
        for i, (n, start, end) in enumerate(zip(numbers, starts, ends)):
            window = np.load(trial_audit.cache_path(cache_dir, n)).astype(np.float32)
            simulated = bridge[start - plan["record_start"]:end + 1 - plan["record_start"], i].astype(np.float64)
            simulated += np.random.default_rng(seed + n).normal(0.0, noise, simulated.size)
            window[:, 2] = simulated
            trial_audit.write_window(out_dir, n, window)
        progress(f"  {cache_dir.name}: {offset + len(batch)}/{len(rows)} strokes, {plan['steps']:,} steps, "
                 f"{seconds:.0f} s so far")
    return seconds


def direct_labels(out_root: Path, rows, reference_f0: dict, floor: float) -> dict:
    thresholds = {**regime_map.DEFAULT_THRESHOLDS, "min_amplitude": floor}
    labels = {}
    for r in rows:
        window = np.load(trial_audit.cache_path(out_root / r["condition"], int(r["trial"])))[:, 2].astype(float)
        periodicity, f0 = regime_map.dominant_period(window)
        shape = regime_map.flyback_structure(window, RATE / f0 if math.isfinite(f0) else math.nan)
        labels[(r["condition"], int(r["trial"]))] = regime_map.classify(
            float(np.std(window)), periodicity, f0, shape["flybacks_per_period"], reference_f0[r["condition"]], thresholds)
    return labels


def compare(measured: dict, simulated: dict) -> dict:
    keys = sorted(set(measured) & set(simulated))
    confusion: dict = {}
    for key in keys:
        cell = confusion.setdefault(measured[key], {})
        cell[simulated[key]] = cell.get(simulated[key], 0) + 1
    both = sum(measured[k] == simulated[k] == "helmholtz" for k in keys)
    either = sum("helmholtz" in (measured[k], simulated[k]) for k in keys)
    return {"strokes": len(keys), "agreement": round(sum(measured[k] == simulated[k] for k in keys) / max(1, len(keys)), 4),
            "helmholtz_iou": round(both / either, 4) if either else None,
            "confusion_measured_to_simulated": confusion}


def run(diagram: str, reports: Path, cache_root: Path, out_data: Path, out_reports: Path, backend, string_overrides: dict,
        friction, rate: int, batch_size: int, limit: int | None, noise: float | None, workers: int, plot: bool,
        seed: int = 0, progress=print) -> tuple[dict, Path]:
    audit, fields, rows, regime, measured_labels = load_measured(reports, diagram)
    impedance = string_overrides.get("impedance") or regime["flyback_law"]["impedance_kg_s"]["median"]
    noise = (regime["noise_floor"].get("control_median") or 0.0) if noise is None else noise
    by_condition: dict = {}
    for r in rows:
        by_condition.setdefault(r["condition"], []).append(r)
    if limit:
        for name, group in by_condition.items():
            group.sort(key=lambda r: int(r["trial"]))
            by_condition[name] = [group[i] for i in np.linspace(0, len(group) - 1, min(limit, len(group))).astype(int)]
    reference_f0 = {name: regime["conditions"][name]["reference_f0_hz"] for name in by_condition}
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    sim_cache, sim_reports = out_data / diagram / run_id, out_reports / diagram / run_id
    parameters = {"rate": rate, "impedance": impedance, "q1": string_overrides.get("q1", bs.StringParams.q1),
                  "corner_hz": string_overrides.get("corner_hz", bs.StringParams.corner_hz),
                  "friction": vars(friction), "observation_noise": noise, "reference_f0": reference_f0,
                  "backend": backend.name, "limit": limit, "seed": seed}
    progress(f"SIMULATE {diagram}: {sum(map(len, by_condition.values()))} strokes, {parameters}")
    seconds = 0.0
    for name, group in sorted(by_condition.items()):
        string = bs.StringParams(f0=reference_f0[name], impedance=impedance, q1=parameters["q1"], corner_hz=parameters["corner_hz"])
        seconds += simulate_condition(backend, string, friction, group, cache_root / diagram / name, sim_cache / name,
                                      rate, noise, seed, batch_size, progress)
    selected = [r for group in by_condition.values() for r in group]
    audit_dir = sim_reports / "trial_audit"
    audit_run = audit_dir / run_id
    audit_run.mkdir(parents=True)
    with (audit_run / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(selected, key=lambda r: (r["condition"], int(r["trial"]))))
    (audit_run / "summary.json").write_text(json.dumps({
        "status": "PASS", "created_utc": run_id[:16], "root": str(sim_cache), "window_cache": str(sim_cache),
        "simulation": parameters, "measured_audit": audit.get("merged_from") or audit.get("root"),
        "trials_audited": len(selected), "trials_failed": 0, "conditions": [], "warnings": []}, indent=2), encoding="utf-8")
    (audit_dir / "latest.json").write_text(json.dumps({"status": "PASS", "run_dir": str(audit_run)}), encoding="utf-8")
    result = {"diagram": diagram, "simulation": parameters, "simulation_seconds": round(seconds, 1)}
    if limit:
        floor = regime["noise_floor"]["min_amplitude"]
        result["comparison"] = compare(measured_labels, direct_labels(sim_cache, selected, reference_f0, floor))
    else:
        _, regime_run = regime_map.run(audit_dir, sim_reports / "regime_map", workers, dict(regime_map.DEFAULT_THRESHOLDS),
                                       None, plot, progress)
        simulated = json.loads((regime_run / "summary.json").read_text(encoding="utf-8"))
        with (regime_run / "regimes.csv").open(newline="", encoding="utf-8") as handle:
            sim_labels = {(r["condition"], int(r["trial"])): r["label"] for r in csv.DictReader(handle)}
        result["comparison"] = compare(measured_labels, sim_labels)
        result["regime_run"] = str(regime_run)
        result["boundaries"] = {name: {kind: {side: {k: summary["conditions"][name]["boundaries"][side][k] for k in ("slope", "slope_se", "n")}
                                              for side in ("lower", "upper")}
                                       for kind, summary in (("measured", regime), ("simulated", simulated))}
                                for name in sorted(by_condition)}
    sim_reports.mkdir(parents=True, exist_ok=True)
    (sim_reports / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    c = result["comparison"]
    progress(f"SIMULATION vs MEASUREMENT: {c['strokes']} strokes, agreement {c['agreement']}, Helmholtz IoU {c['helmholtz_iou']}, "
             f"{seconds:.0f} s simulated")
    for measured, row in sorted(c["confusion_measured_to_simulated"].items()):
        progress(f"  measured {measured}: " + ", ".join(f"{k} {v}" for k, v in sorted(row.items())))
    for name, b in result.get("boundaries", {}).items():
        progress(f"  {name} lower measured {b['measured']['lower']} simulated {b['simulated']['lower']} | "
                 f"upper measured {b['measured']['upper']} simulated {b['simulated']['upper']}")
    progress(f"Comparison: {sim_reports / 'comparison.json'}")
    return result, sim_reports


def pick_backend(name: str, device: str, dtype: str):
    if name == "auto":
        try:
            import torch
            name = "torch" if torch.cuda.is_available() else "numpy"
        except ImportError:
            name = "numpy"
    return bs.Backend(name, device if name == "torch" else "cpu", dtype)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--diagram", default="2024-03-25_TypeA_sample1")
    parser.add_argument("--backend", choices=("auto", "torch", "numpy"), default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--rate", type=int, default=100_000, help="simulation rate, a multiple of 50 kHz")
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=None, help="evenly spaced strokes per folder (no map, direct labels)")
    parser.add_argument("--impedance", type=float, default=None, help="default: measured flyback-law Z")
    parser.add_argument("--q1", type=float, default=bs.StringParams.q1)
    parser.add_argument("--corner-hz", type=float, default=bs.StringParams.corner_hz)
    parser.add_argument("--mu-s", type=float, default=bs.FrictionParams.mu_s)
    parser.add_argument("--mu-d", type=float, default=bs.FrictionParams.mu_d)
    parser.add_argument("--v0", type=float, default=bs.FrictionParams.v0)
    parser.add_argument("--noise", type=float, default=None, help="observation noise std; default: measured no-contact median")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    backend = pick_backend(args.backend, args.device, args.dtype)
    friction = bs.FrictionParams(args.mu_s, args.mu_d, args.v0)
    overrides = {"impedance": args.impedance, "q1": args.q1, "corner_hz": args.corner_hz}
    try:
        run(args.diagram, PROJECT_ROOT / "reports" / "collection", PROJECT_ROOT / "data" / "cache",
            PROJECT_ROOT / "data" / "simulation", PROJECT_ROOT / "reports" / "simulation", backend, overrides, friction,
            args.rate, args.batch_size, args.limit, args.noise, args.workers, not args.no_plot, args.seed,
            lambda line: print(line, flush=True))
        return 0
    except (FileNotFoundError, KeyError, ValueError, regime_map.RegimeError) as error:
        print("SIMULATE: FAIL —", error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
