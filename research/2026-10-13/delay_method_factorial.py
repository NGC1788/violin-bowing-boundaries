#!/usr/bin/env python3
"""Matched established-DSP ablation: linear vs first-order Thiran feedback delays.

Physical/filter parameters, contact law and bridge observation are identical. Linear
outputs from convergence_factorial.json are reused; allpass outcomes are never selected
on labels. Allpass is an established numerical comparator, not a novel physical model.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location("factorial", HERE / "convergence_factorial.py")
factorial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(factorial)


def free_decay_comparison():
    betas = [.03, .1, .2]
    t = (np.arange(4002) + .5) / 1000
    velocity = np.full((len(t), 3), .1)
    force = np.broadcast_to(np.where(t[:, None] < .02, 4., 0.), velocity.shape).copy()
    string = factorial.bs.StringParams()
    rows = []
    for method in ("linear", "thiran1"):
        waveform = factorial.simulate_cpu(string, factorial.bs.FrictionParams(), betas, velocity, force,
                                          1000., 50_000, 0., 200_000, 5, feedback_delay=method)
        for j, beta in enumerate(betas):
            amplitudes, centers = [], []
            for start in np.arange(.3, 3.4, .2):
                lo = int(start * 10_000)
                tt = np.arange(lo, lo + 4000) / 10_000
                weight = np.hanning(4000)
                basis = np.column_stack([np.cos(2 * np.pi * string.f0 * tt),
                                         np.sin(2 * np.pi * string.f0 * tt), np.ones(4000)])
                fitted = np.linalg.lstsq(weight[:, None] * basis, weight * waveform[lo:lo + 4000, j], rcond=None)[0]
                amplitudes.append(np.hypot(fitted[0], fitted[1]))
                centers.append(start + .2)
            q = -np.pi * string.f0 / np.polyfit(centers, np.log(amplitudes), 1)[0]
            rows.append({"delay": method, "beta": beta, "rate": 50_000, "requested_q": 2000., "fitted_q": float(q)})
    return rows


def main():
    started = time.perf_counter()
    path = ROOT / "reports/physics_audit/2026-10-13"
    old = json.loads((path / "convergence_factorial.json").read_text())
    keys = ("case", "string_pitch", "f0", "beta", "speed", "force_ratio_beta_F_over_Zv", "normal_force", "force_rank")
    cases = [{key: row[key] for key in keys} for row in old["rows"]["duration0.6_rate100000"]]
    rates, durations = (100_000, 200_000, 400_000, 800_000), (.6, 1.2)
    baseline = {(d, r): old["rows"][f"duration{d}_rate{r}"] for d in durations for r in rates}
    allpass = {(d, r): factorial.run(cases, r, d, feedback_delay="thiran1") for d in durations for r in rates}
    summaries, details = [], []

    def add(first, second, metadata):
        summary, changes = factorial.compare(first, second)
        summaries.append({**metadata, **summary})
        details.append({**metadata, "changes": changes})

    for method, rows in (("linear", baseline), ("thiran1", allpass)):
        for duration in durations:
            for first, second in zip(rates[:-1], rates[1:]):
                add(rows[duration, first], rows[duration, second], {"comparison": "rate", "delay": method,
                    "duration": duration, "first": first, "second": second})
        for rate in rates:
            add(rows[.6, rate], rows[1.2, rate], {"comparison": "duration", "delay": method,
                "rate": rate, "first": .6, "second": 1.2})
    for duration in durations:
        for rate in rates:
            add(baseline[duration, rate], allpass[duration, rate], {"comparison": "delay_method",
                "duration": duration, "rate": rate, "first": "linear", "second": "thiran1"})
    summary = {"scope": "fixed-grid numerical mechanism ablation, not experimental violin validation",
               "new_thiran1_strokes": 768, "reused_linear_strokes": 768, "prespecified_cases": 96,
               "feedback_change": "first-order allpass replaces only the two feedback fractional-delay interpolators",
               "kept_identical": ["physical parameters", "friction law", "bow profiles", "observation windows",
                                  "linear bridge observation interpolation", "50kHz box-averaged output"],
               "limitations": ["Thiran1 is an established magnitude-preserving allpass with phase dispersion.",
                               "Box averaging alone is not a high-rejection anti-aliasing filter.",
                               "Historical shape labels are provisional; no continuum certificate follows from rate agreement."],
               "comparisons": summaries, "free_decay": free_decay_comparison(),
               "elapsed_seconds": time.perf_counter() - started}
    (path / "delay_method_factorial.json").write_text(json.dumps(factorial.benchmark.finite_json(
        {"summary": summary, "allpass_rows": {f"duration{d}_rate{r}": rows for (d, r), rows in allpass.items()},
         "details": details}), indent=2, allow_nan=False) + "\n")
    (HERE / "delay_method_summary.json").write_text(json.dumps(factorial.benchmark.finite_json(summary), indent=2,
                                                              allow_nan=False) + "\n")
    for row in summaries:
        if row["comparison"] == "rate":
            print(row["delay"], row["duration"], row["first"], row["second"], "changed", row["changed_labels"],
                  "median_stick_change", row["stick_fraction_abs_change"]["median"],
                  "median_spectral_tv", row["spectral_total_variation"]["median"], flush=True)
    print("elapsed_seconds", summary["elapsed_seconds"])


if __name__ == "__main__":
    main()
