#!/usr/bin/env python3
"""Prespecified 96-case rate x duration audit of the passive point-bow solver.

Rates: 50,100,200,400,800 kHz. Durations: .6 and 1.2 s. Observe the final .4 s
through the same 50 kHz box-averaged output. No parameter selection uses outcomes.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import bowed_string as bs
from bowed_string_cpu import simulate_cpu

spec = importlib.util.spec_from_file_location("violin_benchmark", Path(__file__).with_name("violin_synthetic_benchmark.py"))
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def spectral_features(waveform, f0):
    n = waveform.size
    centered = waveform - np.mean(waveform)
    spectrum = abs(np.fft.rfft(centered * np.hanning(n))) ** 2
    frequencies = np.fft.rfftfreq(n, 1 / 50_000)
    total = float(spectrum.sum())
    # Fixed nominal-frequency bins: do not align spectra to the estimated period.
    # Include energy up to 20.5*f0 and one catch-all high-frequency bin.
    edges = np.r_[np.arange(0, 21) + .5, np.inf] * f0
    bands = [float(spectrum[(frequencies >= lo) & (frequencies < hi)].sum())
             for lo, hi in zip(np.r_[0., edges[:-1]], edges)]
    distribution = np.asarray(bands) / max(total, 1e-30)
    return {"amplitude_std": float(np.std(centered)), "mean_bridge_force": float(np.mean(waveform)),
            "nominal_frequency_band_power": distribution.tolist()}


def run(cases, rate, duration, feedback_delay="linear"):
    rows, valid = [], []
    for case in cases:
        try:
            bs.prepare_waveguide(bs.StringParams(f0=case["f0"], impedance=1., length=.328, q1=200., corner_hz=6000.),
                                 [case["beta"]], rate)
        except ValueError as error:
            rows.append({**case, "rate": rate, "duration": duration, "status": "rejected", "reason": str(error)})
        else:
            valid.append(case)
    for offset in range(0, len(valid), 64):
        group = valid[offset:offset + 64]
        f0, beta, speed, normal = (np.asarray([case[key] for case in group]) for key in ("f0", "beta", "speed", "normal_force"))
        t = (np.arange(int(duration * 1000) + 2) + .5) / 1000
        velocity = np.minimum(t[:, None] / .1, 1.) * speed
        force = np.broadcast_to(normal, velocity.shape).copy()
        string = bs.StringParams(f0=f0, impedance=1., length=.328, q1=200., corner_hz=6000.)
        waveform, truth = simulate_cpu(string, bs.FrictionParams(), beta, velocity, force, 1000., rate, 0.,
                                       round(duration * rate), rate // 50_000, round((duration - .4) * rate),
                                       return_diagnostics=True, feedback_delay=feedback_delay)
        for j, case in enumerate(group):
            rows.append({**case, "rate": rate, "duration": duration, "feedback_delay": feedback_delay,
                         "status": "simulated",
                         **benchmark.describe(waveform[:, j], case["f0"]),
                         **spectral_features(waveform[:, j], case["f0"]),
                         "internal_stick_fraction": float(1 - truth["slip_fraction"][j]),
                         "raw_contact_onsets_per_period": float(truth["slip_onsets"][j] / (.4 * case["f0"]))})
    print(f"delay={feedback_delay}, rate={rate}, duration={duration}: {len(valid)} simulated", flush=True)
    return sorted(rows, key=lambda row: row["case"])


def compare(first, second):
    changes = []
    for a, b in zip(first, second):
        assert a["case"] == b["case"]
        if a["status"] != "simulated" or b["status"] != "simulated":
            continue
        first_h, second_h = a["shape_label"] == "helmholtz", b["shape_label"] == "helmholtz"
        # Post-hoc diagnostic only. Historical labels remain unchanged.
        first_gated_h = first_h and a["estimated_f0"] is not None and a["estimated_f0"] <= 1.25 * a["f0"]
        second_gated_h = second_h and b["estimated_f0"] is not None and b["estimated_f0"] <= 1.25 * b["f0"]
        changes.append({"case": a["case"], "label_changed": a["shape_label"] != b["shape_label"],
                        "first_label": a["shape_label"], "second_label": b["shape_label"],
                        "first_h": first_h, "second_h": second_h,
                        "binary_h_changed": first_h != second_h,
                        "first_gated_h": first_gated_h, "second_gated_h": second_gated_h,
                        "binary_gated_h_changed": first_gated_h != second_gated_h,
                        "stick_fraction_abs_change": abs(a["internal_stick_fraction"] - b["internal_stick_fraction"]),
                        "periodicity_abs_change": abs(a["periodicity"] - b["periodicity"]),
                        "relative_frequency_abs_change": abs(a["estimated_f0"] - b["estimated_f0"]) / a["f0"],
                        "relative_amplitude_abs_change": abs(a["amplitude_std"] - b["amplitude_std"]) /
                                                         max(a["amplitude_std"], b["amplitude_std"], 1e-30),
                        "spectral_total_variation": float(.5 * np.sum(abs(np.asarray(a["nominal_frequency_band_power"]) -
                                                                            np.asarray(b["nominal_frequency_band_power"]))))})
    summary = {"n": len(changes), "changed_labels": sum(row["label_changed"] for row in changes)}
    for prefix, key in (("binary_h", "h"), ("binary_gated_h", "gated_h")):
        first_positive = sum(row[f"first_{key}"] for row in changes)
        second_positive = sum(row[f"second_{key}"] for row in changes)
        intersection = sum(row[f"first_{key}"] and row[f"second_{key}"] for row in changes)
        union = first_positive + second_positive - intersection
        summary[prefix] = {"changed": sum(row[f"{prefix}_changed"] for row in changes),
                           "first_positive": first_positive, "second_positive": second_positive,
                           "intersection": intersection, "union": union,
                           "iou": intersection / union if union else None}
    for key in ("stick_fraction_abs_change", "periodicity_abs_change", "relative_frequency_abs_change",
                "relative_amplitude_abs_change", "spectral_total_variation"):
        values = np.asarray([row[key] for row in changes])
        summary[key] = {"median": float(np.median(values)), "p90": float(np.percentile(values, 90)),
                        "max": float(np.max(values))}
    summary["stick_change_gt_0_02"] = sum(row["stick_fraction_abs_change"] > .02 for row in changes)
    summary["spectral_tv_gt_0_10"] = sum(row["spectral_total_variation"] > .1 for row in changes)
    return summary, changes


def main():
    start = time.perf_counter()
    cases_path = ROOT / "reports/physics_audit/2026-10-13/violin_synthetic_cases.json"
    all_cases = json.loads(cases_path.read_text())["main"]
    keys = ("case", "string_pitch", "f0", "beta", "speed", "force_ratio_beta_F_over_Zv", "normal_force", "force_rank")
    cases = [{key: row[key] for key in keys} for row in all_cases if row["force_rank"] in (6, 16)]
    rates = (50_000, 100_000, 200_000, 400_000, 800_000)
    durations = (.6, 1.2)
    results = {(duration, rate): run(cases, rate, duration) for duration in durations for rate in rates}
    aggregates, details = [], []
    for duration in durations:
        for first, second in zip(rates[:-1], rates[1:]):
            aggregate, changes = compare(results[duration, first], results[duration, second])
            metadata = {"comparison": "rate", "duration": duration, "first": first, "second": second}
            aggregates.append({**metadata, **aggregate})
            details.append({**metadata, "changes": changes})
    for rate in rates:
        aggregate, changes = compare(results[.6, rate], results[1.2, rate])
        metadata = {"comparison": "duration", "rate": rate, "first": .6, "second": 1.2}
        aggregates.append({**metadata, **aggregate})
        details.append({**metadata, "changes": changes})
    report = {"scope": "numerical audit of assumed generic strings, no experimental violin validation",
              "rates": rates, "durations": durations, "selected_cases": len(cases),
              "total_simulated_strokes": sum(row["status"] == "simulated" for rows in results.values() for row in rows),
              "selection": "same force ranks 6 and 16 chosen before the earlier main-grid results",
              "observation": "final .4s, common50kHz output, no added noise; fixed nominal-frequency spectrum bands",
              "comparisons": aggregates, "elapsed_seconds": time.perf_counter() - start}
    path = ROOT / "reports/physics_audit/2026-10-13"
    (path / "convergence_factorial.json").write_text(json.dumps(benchmark.finite_json(
        {"summary": report, "rows": {f"duration{duration}_rate{rate}": rows for (duration, rate), rows in results.items()},
         "details": details}), indent=2, allow_nan=False) + "\n")
    Path(__file__).with_name("convergence_summary.json").write_text(json.dumps(benchmark.finite_json(report), indent=2,
                                                                            allow_nan=False) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    for duration, color in ((.6, "#197278"), (1.2, "#c75b12")):
        selected = [row for row in aggregates if row["comparison"] == "rate" and row["duration"] == duration]
        x = [row["second"] / 1000 for row in selected]
        axes[0].plot(x, [100 * row["binary_h"]["changed"] / row["n"] for row in selected], "o-", color=color,
                     label=f"Duration {duration}s")
        axes[1].plot(x, [row["stick_fraction_abs_change"]["median"] for row in selected], "o-", color=color)
        axes[2].plot(x, [row["spectral_total_variation"]["median"] for row in selected], "o-", color=color)
    for ax, title, ylabel in zip(axes, ("Binary H/non-H instability", "Internal contact stability", "Bridge spectrum stability"),
                                ("Changed H membership (%)", "Median absolute stick-fraction difference", "Median spectral total variation")):
        ax.set(xscale="log", xlabel="Higher rate in adjacent-rate pair (kHz)", title=title, ylabel=ylabel)
        ax.set_xticks([100, 200, 400, 800], ["100", "200", "400", "800"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(alpha=.2)
    axes[1].set_yscale("log")
    axes[0].legend(frameon=False)
    fig.suptitle("Fixed 96-case synthetic rate and duration audit; no continuum guarantee")
    fig.tight_layout()
    fig.savefig(path / "convergence_factorial.png", dpi=160)
    plt.close(fig)
    for row in aggregates:
        print({key: row[key] for key in row if key not in ("stick_fraction_abs_change", "periodicity_abs_change",
                                                         "relative_frequency_abs_change", "relative_amplitude_abs_change",
                                                         "spectral_total_variation")})
    print("elapsed_seconds", report["elapsed_seconds"])


if __name__ == "__main__":
    main()
