#!/usr/bin/env python3
"""960 generic bowed-string cases at violin open-string frequencies; CPU only.

This is a numerical model audit, NOT experimental violin validation. Z, Q and
loss corner are declared assumptions, not measurements of violin strings.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import bowed_string as bs
from bowed_string_cpu import simulate_cpu
import regime_map as rm

RATE_OUT = 50_000


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    return value


def describe(waveform, reference):
    periodicity, f0 = rm.dominant_period(waveform, rate=RATE_OUT, f0_range=(0.4 * reference, 3.5 * reference))
    shape = rm.flyback_structure(waveform, RATE_OUT / f0)
    label = rm.classify(float(np.std(waveform)), periodicity, f0, shape["flybacks_per_period"],
                        reference, rm.DEFAULT_THRESHOLDS)
    return {"shape_label": label, "periodicity": periodicity, "estimated_f0": f0,
            "flyback_ratio": shape["second_flyback_ratio"], "flyback_height": shape["flyback_height"],
            "flyback_duration_fraction_of_cycle": shape["flyback_ms"] * f0 / 1000}


def run_cases(cases, rate):
    results = []
    eligible = []
    for case in cases:
        try:
            bs.prepare_waveguide(bs.StringParams(f0=case["f0"], impedance=1., length=.328, q1=200., corner_hz=6000.),
                                 [case["beta"]], rate)
        except ValueError as error:
            results.append({**case, "rate": rate, "status": "rejected_before_simulation", "reason": str(error)})
        else:
            eligible.append(case)
    for offset in range(0, len(eligible), 64):
        group = eligible[offset:offset + 64]
        f0 = np.asarray([c["f0"] for c in group])
        beta = np.asarray([c["beta"] for c in group])
        speed = np.asarray([c["speed"] for c in group])
        fn = np.asarray([c["normal_force"] for c in group])
        t = (np.arange(602) + .5) / 1000
        velocity = np.minimum(t[:, None] / .1, 1) * speed
        force = np.broadcast_to(fn, velocity.shape).copy()
        string = bs.StringParams(f0=f0, impedance=1., length=.328, q1=200., corner_hz=6000.)
        waveform, truth = simulate_cpu(string, bs.FrictionParams(), beta, velocity, force, 1000,
                                       rate, 0., int(.6 * rate), rate // RATE_OUT, int(.2 * rate),
                                       return_diagnostics=True)
        losses = bs.linear_decay_diagnostics(string, beta, rate)
        for j, case in enumerate(group):
            features = describe(waveform[:, j], case["f0"])
            results.append({**case, "rate": rate, "status": "simulated", **features,
                            "internal_stick_fraction": float(1 - truth["slip_fraction"][j]),
                            "raw_slip_onsets_per_nominal_period": float(truth["slip_onsets"][j] / (.4 * case["f0"])),
                            "max_contact_residual": float(truth["max_load_line_residual"][j]),
                            "minimum_friction_dissipation": float(truth["min_friction_dissipation_watts"][j]),
                            "effective_q_at_nominal_f0": float(losses["effective_q_at_nominal_f0"][j])})
        print(f"rate={rate}: {min(offset + 64, len(eligible))}/{len(eligible)} eligible", flush=True)
    return sorted(results, key=lambda row: row["case"])


def main():
    started = time.perf_counter()
    frequencies = {name: 440 * 2 ** (semitones / 12) for name, semitones in (("G3", -14), ("D4", -7), ("A4", 0), ("E5", 7))}
    force_ratios = np.geomspace(.1, 12., 20)
    cases = []
    for name, f0 in frequencies.items():
        for beta in (.03, .05, .1, .2):
            for speed in (.05, .1, .2):
                for rank, ratio in enumerate(force_ratios):
                    cases.append({"case": len(cases), "string_pitch": name, "f0": f0, "beta": beta,
                                  "speed": speed, "force_ratio_beta_F_over_Zv": float(ratio),
                                  "normal_force": float(ratio * speed / beta), "force_rank": rank})
    rows = run_cases(cases, 100_000)
    # Preselected force ranks, before any output is inspected: cover lower and upper areas.
    subset = [case for case in cases if case["force_rank"] in (6, 16)]
    low, high = run_cases(subset, 50_000), run_cases(subset, 200_000)
    middle = {row["case"]: row for row in rows}
    comparisons = []
    for a, b in zip(low, high):
        m = middle[a["case"]]
        comparisons.append({"case": a["case"], "string_pitch": a["string_pitch"],
                            "labels": [r.get("shape_label") for r in (a, m, b)],
                            "stick_fractions": [r.get("internal_stick_fraction") for r in (a, m, b)]})
    low_comparisons = [row for row in comparisons if row["labels"][0] is not None]
    high_comparisons = [row for row in comparisons if row["labels"][2] is not None]
    summary = {"scope": "synthetic generic strings at violin frequencies, no experimental validation",
               "parameter_assumptions": {"Z_kg_s": 1., "length_m": .328, "requested_Q1": 200.,
                                         "corner_hz": 6000., "friction": vars(bs.FrictionParams()),
                                         "equal_tempered_A4_hz": 440., "frequencies": frequencies},
               "protocol": {"duration_s": .6, "record_from_s": .2, "ramp_s": .1,
                            "output_rate_hz": RATE_OUT, "main_simulation_rate": 100_000,
                            "observation_noise": 0., "batch": 64, "threads": 2,
                            "frequency_search_relative_to_reference": [.4, 3.5],
                            "rate_subset_selection": "force ranks 6 and 16 of 20, for every pitch/beta/speed"},
               "main_cases": len(rows), "multirate_cases": len(subset),
               "total_simulated_strokes": sum(r["status"] == "simulated" for r in rows + low + high),
               "rate50k_rejected": sum(r["status"] != "simulated" for r in low),
               "rate200k_rejected": sum(r["status"] != "simulated" for r in high),
               "main_shape_label_counts": dict(Counter(row["shape_label"] for row in rows)),
               "rate_50k_100k_label_comparisons": len(low_comparisons),
               "rate_100k_200k_label_comparisons": len(high_comparisons),
               "rate_50k_100k_label_disagreements": sum(row["labels"][0] != row["labels"][1] for row in low_comparisons),
               "rate_100k_200k_label_disagreements": sum(row["labels"][1] != row["labels"][2] for row in high_comparisons),
               "rate_all_three_agree": sum(len(set(row["labels"])) == 1 for row in low_comparisons),
               "max_contact_residual": max(row["max_contact_residual"] for row in rows),
               "min_friction_dissipation": min(row["minimum_friction_dissipation"] for row in rows),
               "elapsed_seconds": time.perf_counter() - started,
               "caveats": ["Shape labels use the historical classifier, known to accept pure sine and triangle signals.",
                           "Internal stick fraction and slip onsets are simulated states, not empirical truth.",
                           "No body, stiffness, torsion, finite bow width, thermal friction or real string calibration.",
                           "Rate disagreement prevents a convergence claim from this experiment alone."]}
    path = ROOT / "reports/physics_audit/2026-10-13"
    path.mkdir(parents=True, exist_ok=True)
    (path / "violin_synthetic_summary.json").write_text(json.dumps(finite_json(summary), indent=2, allow_nan=False) + "\n")
    (path / "violin_synthetic_cases.json").write_text(json.dumps(finite_json({"main": rows, "rate50k": low,
                                                                          "rate200k": high,
                                                                          "comparisons": comparisons}), indent=2,
                                                               allow_nan=False) + "\n")
    # Small public aggregate; full synthetic case table stays in ignored reports/.
    (Path(__file__).with_name("violin_synthetic_summary.json")).write_text(json.dumps(finite_json(summary), indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.7), sharex=True, sharey=True)
    for ax, name in zip(axes, frequencies):
        points = [row for row in rows if row["string_pitch"] == name and row["speed"] == .1]
        for label in rm.CLASSES:
            selected = [row for row in points if row["shape_label"] == label]
            ax.scatter([r["beta"] for r in selected], [r["force_ratio_beta_F_over_Zv"] for r in selected],
                       c=rm.COLORS[label], label=label, s=24)
        ax.set(xscale="log", yscale="log", xlabel="Relative bow position beta", title=name)
        ax.set_xticks([.03, .05, .1, .2], ["0.03", "0.05", "0.10", "0.20"])
        ax.xaxis.set_minor_locator(NullLocator())
    axes[0].set_ylabel("Dimensionless normal force beta F / (Z v)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Synthetic passive strings at violin pitches; provisional waveform labels")
    fig.tight_layout(rect=[0, .17, 1, .93])
    fig.savefig(path / "violin_synthetic_maps.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
