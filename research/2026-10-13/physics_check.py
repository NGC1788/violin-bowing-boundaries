#!/usr/bin/env python3
"""Small, reproducible numerical audit; synthetic signals only, no raw data download.

Run from the repository with .venv-mac/bin/python research/2026-10-13/physics_check.py.
Results describe implementation validity and rate sensitivity, not real-data accuracy.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import bowed_string as bs
from bowed_string_cpu import simulate_cpu
import regime_map as rm


def waveform_features(waveform, reference_f0=98.08):
    periodicity, frequency = rm.dominant_period(waveform)
    shape = rm.flyback_structure(waveform, 50_000 / frequency)
    return {"label": rm.classify(float(np.std(waveform)), periodicity, frequency,
                                shape["flybacks_per_period"], reference_f0, rm.DEFAULT_THRESHOLDS),
            "periodicity": periodicity, "frequency_hz": frequency,
            "flybacks": shape["flybacks_per_period"], "flyback_ms": shape["flyback_ms"],
            "flyback_height": shape["flyback_height"]}


def reflection_audit():
    rows = []
    for rate in (50_000, 100_000, 200_000):
        for name, q1, corner in (("literature_default", 2000., 6000.),
                                  ("historical_calibration_best", 3726.62, 1896.41)):
            string = bs.StringParams(q1=q1, corner_hz=corner)
            gain, pole = bs.reflection_design(string, rate)
            omega = 2 * np.pi * string.f0 / rate
            intended = abs(gain * (1 - pole) / (1 - pole * np.exp(-1j * omega)))
            legacy = abs(gain * (1 - pole) / (1 - gain * pole * np.exp(-1j * omega)))
            rows.append({"parameters": name, "rate_hz": rate, "requested_q1": q1,
                         "gain": float(gain), "pole": float(pole), "passive": bool(gain <= 1),
                         "intended_reflection_magnitude_f0": intended,
                         "legacy_reflection_magnitude_f0": legacy,
                         "legacy_reflection_dc_gain": float(gain * (1 - pole) / (1 - gain * pole)),
                         "legacy_effective_q_before_interpolation": float(-np.pi / (2 * np.log(legacy)))})
    return rows


def main():
    started = time.perf_counter()
    t = np.arange(20_000) / 50_000
    phase = (98.08 * t) % 1
    controls = {"sine_negative_control": np.sin(2 * np.pi * 98.08 * t),
                "symmetric_triangle_negative_control": 2 * np.abs(2 * phase - 1) - 1,
                "ideal_sawtooth_positive_shape_control": 2 * phase - 1}
    report = {"scope": "synthetic numerical audit only; no experimental accuracy claims",
              "reflection": reflection_audit(),
              "observation_controls": {name: waveform_features(signal) for name, signal in controls.items()}}
    rng = np.random.default_rng(20261013)
    free = rng.uniform(-3, 3, 10_000)
    normal = rng.uniform(0, 10, free.size)
    force, sticking, slip = bs.friction_step(bs.Backend(), free, 1 / (2 * 0.946), normal,
                                           rng.random(free.size) < 0.5, bs.FrictionParams())
    report["contact_equation"] = {"random_cases": free.size,
                                  "max_load_line_residual": float(np.max(abs(free - force / (2 * 0.946) - slip))),
                                  "min_friction_dissipation_watts": float(np.min(force * slip))}
    cases = [(beta, speed, fn) for beta in (.03, .1, .2) for speed in (.05, .1)
             for fn in (.25, 1., 2., 4., 6., 10.)]
    beta, speed, normal = np.asarray(cases).T
    profile_time = (np.arange(602) + .5) / 1000
    velocity = np.minimum(profile_time[:, None] / .1, 1) * speed
    normal = np.broadcast_to(normal, velocity.shape).copy()
    per_rate = {}
    for rate in (50_000, 100_000, 200_000):
        every = rate // 50_000
        waveform, contact = simulate_cpu(bs.StringParams(), bs.FrictionParams(), beta, velocity, normal,
                                          1000, rate, 0., int(.6 * rate), every, int(.2 * rate),
                                          return_diagnostics=True)
        rows = []
        for column, parameters in enumerate(cases):
            rows.append({"beta": parameters[0], "speed_m_s": parameters[1], "normal_force_n": parameters[2],
                         **waveform_features(waveform[:, column]),
                         "contact_slip_fraction": float(contact["slip_fraction"][column]),
                         "raw_contact_slip_onsets": int(contact["slip_onsets"][column]),
                         "max_load_line_residual": float(contact["max_load_line_residual"][column])})
        per_rate[str(rate)] = rows
    report["rate_sweep"] = {"cases_per_rate": len(cases), "rates": per_rate,
                            "same_legacy_classifier_at_all_rates": True,
                            "noise_added": False}
    report["rate_comparison"] = []
    for low, high in ((50_000, 100_000), (100_000, 200_000)):
        labels = [(a["label"], b["label"]) for a, b in zip(per_rate[str(low)], per_rate[str(high)])]
        changed = [i for i, (a, b) in enumerate(labels) if a != b]
        report["rate_comparison"].append({"low_rate": low, "high_rate": high,
                                          "changed_labels": len(changed), "total": len(labels),
                                          "changed_case_indices": changed})
    report["elapsed_seconds"] = time.perf_counter() - started
    path = ROOT / "reports/physics_audit/2026-10-13/physics_check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    def finite_json(value):
        if isinstance(value, dict):
            return {key: finite_json(item) for key, item in value.items()}
        if isinstance(value, list):
            return [finite_json(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    report = finite_json(report)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(path)
    print(json.dumps({"observation_controls": report["observation_controls"],
                      "rate_comparison": report["rate_comparison"],
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
