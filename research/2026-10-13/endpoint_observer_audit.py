#!/usr/bin/env python3
"""Known endpoint-wave reconstruction baseline; no novel-method claim.

Copies the frozen linear-feedback contact solver to expose both endpoint forces
and contact velocity without changing production outputs. This is a synthetic
identifiability audit, not mechanical labels for the experimental database.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import bowed_string as bs
from bowed_string_cpu import _read, simulate_cpu


@njit(cache=True, parallel=True)
def _endpoints(gain, pole, db, dn, hb, hn, z, y, size, mu_s, mu_d, v0,
               velocity, normal_force, profile_rate, rate, steps):
    batch = z.size
    result = np.zeros((steps, batch, 3))
    stick = np.zeros((steps, batch), dtype=np.uint8)
    for k in prange(batch):
        bb, bn = np.zeros(size), np.zeros(size)
        sb, sn = 0., 0.
        sticking = True
        for n in range(steps):
            position = n / rate * profile_rate - .5
            j = min(max(int(math.floor(position)), 0), velocity.shape[0] - 2)
            fraction = min(max(position - j, 0.), 1.)
            vb = velocity[j, k] * (1 - fraction) + velocity[j + 1, k] * fraction
            fn = max(normal_force[j, k] * (1 - fraction) + normal_force[j + 1, k] * fraction, 0.)
            sb = gain[k] * (1 - pole[k]) * _read(bb, db[k], n) + pole[k] * sb
            sn = gain[k] * (1 - pole[k]) * _read(bn, dn[k], n) + pole[k] * sn
            free_slip = vb + sb + sn
            magnitude = abs(free_slip)
            a = y[k] * fn
            b = v0 + a * mu_d - magnitude
            c = v0 * (a * mu_s - magnitude)
            discriminant = b * b - 4 * c
            rooted = math.sqrt(max(discriminant, 0.))
            root = -2 * c / (b + rooted) if b > 0 else (-b + rooted) / 2
            slip_exists = (c < 0) or ((discriminant >= 0) and (b < 0) and (root > 0))
            can_stick = magnitude <= mu_s * fn * y[k]
            sticking = (can_stick or not slip_exists) if sticking else not slip_exists
            root = max(root, 0.)
            direction = 1. if free_slip > 0 else (-1. if free_slip < 0 else 0.)
            mu = mu_d + (mu_s - mu_d) * v0 / (v0 + root)
            force = free_slip / y[k] if sticking else direction * fn * mu
            bb[n % size] = -sn + y[k] * force
            bn[n % size] = -sb + y[k] * force
            result[n, k, 0] = 2 * z[k] * _read(bb, hb[k], n)
            result[n, k, 1] = 2 * z[k] * _read(bn, hn[k], n)
            result[n, k, 2] = -sb - sn + y[k] * force
            stick[n, k] = sticking
    return result, stick


def simulate_endpoints(string, betas, speeds, forces, rate=100_000, duration=.6):
    """Constant-force / .1 s speed ramp, full-rate double-precision outputs."""
    betas, speeds, forces = map(lambda x: np.asarray(x, float), (betas, speeds, forces))
    cfg = bs.prepare_waveguide(string, betas, rate)
    steps = round(duration * rate)
    if steps * betas.size * (3 * 8 + 1) > 512 * 1024 ** 2:
        raise ValueError("prototype output exceeds 512 MiB")
    t = (np.arange(math.ceil(duration * 1000) + 2) + .5) / 1000
    velocity = np.minimum(t[:, None] / .1, 1) * speeds
    force = np.broadcast_to(forces, velocity.shape).copy()
    friction = bs.FrictionParams()
    hn = (1 - betas) * rate / (2 * np.asarray(string.f0))
    before = get_num_threads()
    try:
        set_num_threads(2)
        outputs = _endpoints(cfg["gain"], cfg["pole"], cfg["delay_bridge"], cfg["delay_nut"],
                             cfg["delay_half"], hn, cfg["impedance"], cfg["admittance"], cfg["size"],
                             friction.mu_s, friction.mu_d, friction.v0, velocity, force, 1000., rate, steps)
    finally:
        set_num_threads(before)
    return outputs, cfg, velocity, force


def delay(values, samples):
    """Linear fractional delay; negative samples advances. Zero outside recording."""
    x = np.arange(values.size, dtype=float)
    return np.interp(x - samples, x, values, left=0., right=0.)


def reflection(values, gain, pole, samples):
    return lfilter([gain * (1 - pole)], [1., -pole], delay(values, samples))


def estimate_contact(endpoint_force, z, half_delay, round_delay, gain, pole, *, naive=False):
    """Known stable forward filter, followed by offline adjoint delay alignment.

    In the frozen model this equals K*vc, K=O^T*O, away from record edges.
    It does not invert O and does not claim exact instantaneous velocity.
    """
    outgoing_delayed = endpoint_force / (2 * z)
    returned = delay(outgoing_delayed, 2 * half_delay) if naive else reflection(
        outgoing_delayed, gain, pole, round_delay)
    return delay(outgoing_delayed - returned, -half_delay)


def smoothing(values, half_delay):
    """Exact 3-tap O^T O away from the finite recording boundaries."""
    fraction = half_delay - math.floor(half_delay)
    a = fraction * (1 - fraction)
    out = (1 - 2 * a) * values.copy()
    out[1:] += a * values[:-1]
    out[:-1] += a * values[1:]
    return out


def case_metrics(outputs, stick, cfg, k, case, rate):
    fb, fn, vc = outputs[:, k].T
    hb = cfg["delay_half"][k]
    hn = (1 - case["beta"]) * rate / (2 * case["f0"])
    g, p, z = cfg["gain"][k], cfg["pole"][k], cfg["impedance"][k]
    vb = case["speed"]
    selection = slice(round(.2 * rate), round(.58 * rate))
    estimates, smoothed, naive = [], [], []
    for signal, h, d in ((fb, hb, cfg["delay_bridge"][k]), (fn, hn, cfg["delay_nut"][k])):
        estimates.append(estimate_contact(signal, z, h, d, g, p))
        smoothed.append(smoothing(vc, h))
        naive.append(estimate_contact(signal, z, h, d, g, p, naive=True))
    # Matching filters remove the differing interpolation kernels from the
    # dual-end residual. Agreement cannot control a shared model error.
    residual = smoothing(estimates[0], hn) - smoothing(estimates[1], hb)
    true_stick = stick[selection, k].astype(bool)
    truth = vc[selection]
    def metrics(estimate):
        e = estimate[selection]
        error = (e - truth) / vb
        inferred = abs(vb - e) <= .01 * vb
        return {"velocity_rmse_over_speed": float(np.sqrt(np.mean(error ** 2))),
                "velocity_max_error_over_speed": float(np.max(abs(error))),
                "stick_fraction_abs_error_at_1pct_speed": float(abs(inferred.mean() - true_stick.mean())),
                "stick_indicator_error_at_1pct_speed": float(np.mean(inferred != true_stick)),
                "false_stick_samples": int(np.sum(inferred & ~true_stick)),
                "missed_stick_samples": int(np.sum(~inferred & true_stick))}
    methods = {"known_bridge": metrics(estimates[0]), "known_nut": metrics(estimates[1]),
               "known_average": metrics((estimates[0] + estimates[1]) / 2),
               "naive_bridge": metrics(naive[0]), "naive_nut": metrics(naive[1]),
               "shared_Z_10pct_high": metrics((estimates[0] + estimates[1]) / 2 / 1.1)}
    # Preregistered bounded alternating noise, with different phases at the ends.
    epsilon_force = .002 * z * vb
    phase = np.arange(fb.size)
    noisy = []
    for signal, h, d, phase_shift in ((fb, hb, cfg["delay_bridge"][k], 0),
                                     (fn, hn, cfg["delay_nut"][k], 3)):
        bounded_noise = epsilon_force * np.sign(np.sin((phase + phase_shift) * .713))
        noisy.append(estimate_contact(signal + bounded_noise, z, h, d, g, p))
    noise_bound = (1 + g) * epsilon_force / (2 * z)
    methods["bounded_noise_average"] = metrics((noisy[0] + noisy[1]) / 2)
    # Conditional interpolation error bound uses actual adjacent increments.
    # This oracle bound is NOT available from the empirical measurements.
    increment = np.r_[0., abs(np.diff(vc))] + np.r_[abs(np.diff(vc)), 0.]
    bounds = [(h % 1) * (1 - h % 1) * increment for h in (hb, hn)]
    max_smoothing_violation = max(float(np.max(abs(estimates[i][selection] - truth) - bounds[i][selection]))
                                  for i in (0, 1))
    return {**case, "rate": rate, "true_stick_fraction": float(true_stick.mean()), "methods": methods,
            "known_vs_exact_smoothed_truth_max_over_speed": max(
                float(np.max(abs(estimates[i][selection] - smoothed[i][selection])) / vb) for i in (0, 1)),
            "crossfiltered_endpoint_residual_max_over_speed": float(np.max(abs(residual[selection])) / vb),
            "shared_Z_error_crossfiltered_residual_max_over_speed": float(np.max(abs(residual[selection] / 1.1)) / vb),
            "noise_bound_over_speed": float(noise_bound / vb),
            "observed_noise_error_max_over_speed": max(
                float(np.max(abs(noisy[i][selection] - estimates[i][selection]))) / vb for i in (0, 1)),
            "oracle_local_smoothing_bound_max_violation_over_speed": max_smoothing_violation / vb}


def ideal_toy():
    """Integer-delay wave identity, plus two non-identifiability counterexamples."""
    rng = np.random.default_rng(0)
    outgoing = rng.normal(size=1000)
    h, z = 17, .946
    vc = outgoing - delay(outgoing, 2 * h)
    force = 2 * z * delay(outgoing, h)
    recovered = (delay(force, -h) - delay(force, h)) / (2 * z)
    middle = slice(2 * h + 2, -2 * h - 2)
    # O with half-sample delay annihilates the Nyquist alternation in its interior.
    alternation = (-1.) ** np.arange(1000)
    hidden = delay(alternation, 17.5)[middle]
    return {"integer_delay_identity_max_error": float(np.max(abs(recovered[middle] - vc[middle]))),
            "half_sample_observer_hidden_Nyquist_max_amplitude": float(np.max(abs(hidden))),
            "hidden_velocity_original_max_amplitude": float(np.max(abs(alternation))),
            "shared_impedance_scale_counterexample": "Both velocity reconstructions scale by 1/1.1 if Z is 10% too high; their crossfiltered residual remains zero."}


def aggregate(rows):
    result = {}
    for method in rows[0]["methods"]:
        result[method] = {}
        for name in rows[0]["methods"][method]:
            values = [row["methods"][method][name] for row in rows]
            result[method][name] = {"median": float(np.median(values)), "max": float(np.max(values)),
                                    "mean": float(np.mean(values))}
    scalar = ("known_vs_exact_smoothed_truth_max_over_speed", "crossfiltered_endpoint_residual_max_over_speed",
              "shared_Z_error_crossfiltered_residual_max_over_speed", "noise_bound_over_speed",
              "observed_noise_error_max_over_speed", "oracle_local_smoothing_bound_max_violation_over_speed")
    return {"n": len(rows), "methods": result,
            "checks": {key: max(row[key] for row in rows) for key in scalar}}


def main():
    started = time.perf_counter()
    cases = []
    for f0 in (98.08, 195.99771799087463, 440., 659.2551138257398):
        for beta in (.03, .05, .1, .2):
            for speed in (.05, .1, .2):
                for force_ratio in (.45, 5.6):
                    cases.append({"case": len(cases), "f0": f0, "beta": beta, "speed": speed,
                                  "normal_force": force_ratio * speed / beta, "beta_F_over_Zv": force_ratio})
    folder = ROOT / "reports/physics_audit/2026-10-13"
    folder.mkdir(parents=True, exist_ok=True)
    protocol = {"cases": cases, "rates": [100_000, 200_000], "duration": .6, "analysis_window": [.2, .58],
                "q1": 200., "corner_hz": 6000., "impedance": 1., "noise_force_bound_over_Zv": .002,
                "threshold_for_near_stick_over_v": .01, "shared_Z_perturbation": 1.1,
                "scope": "Conditional known-parameter synthetic reconstruction, no real data or labels"}
    (folder / "endpoint_observer_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    rows = []
    parity_max = 0.
    for rate in protocol["rates"]:
        for offset in range(0, len(cases), 16):
            group = cases[offset:offset + 16]
            string = bs.StringParams(f0=np.array([c["f0"] for c in group]), impedance=1., q1=200., corner_hz=6000.)
            beta, speed, force = [np.array([c[key] for c in group]) for key in ("beta", "speed", "normal_force")]
            (signals, stick), cfg, velocity, force_profiles = simulate_endpoints(string, beta, speed, force, rate)
            if offset == 0:
                reference = simulate_cpu(string, bs.FrictionParams(), beta, velocity, force_profiles,
                                         1000., rate, 0., round(.6 * rate))
                parity_max = max(parity_max, float(np.max(abs(signals[:, :, 0].astype(np.float32) - reference))))
            rows.extend(case_metrics(signals, stick, cfg, k, case, rate) for k, case in enumerate(group))
        print(f"endpoint observer: {rate} Hz, {len(cases)} cases", flush=True)
    summary = {"protocol": {key: value for key, value in protocol.items() if key != "cases"},
               "cases_per_rate": len(cases), "toy_checks": ideal_toy(), "production_bridge_float32_max_parity_error": parity_max,
               "by_rate": {str(rate): aggregate([row for row in rows if row["rate"] == rate]) for rate in protocol["rates"]},
               "elapsed_seconds": time.perf_counter() - started,
               "limitations": ["No empirical endpoint signal was read or fitted.",
                               "Endpoint forces follow the frozen lumped-loss observation convention, not a validated transducer transfer function.",
                               "Known model parameters and time alignment are oracle inputs.",
                               "Near-zero velocity does not distinguish exact mathematical sticking from sufficiently slow slip.",
                               "Dual-end agreement cannot certify shared parameter accuracy or recover a common observation nullspace.",
                               "Oracle local smoothing error bound is unavailable without a justified physical smoothness bound.",
                               "Wave reconstruction is established prior work, not a new method claim."]}
    (folder / "endpoint_observer_cases.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    (folder / "endpoint_observer_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    Path(__file__).with_name("endpoint_observer_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"parity": parity_max, "seconds": summary["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
