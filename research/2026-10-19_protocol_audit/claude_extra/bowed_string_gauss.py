#!/usr/bin/env python3
"""Copy of scripts/bowed_string_cpu.py with a Gaussian Stribeck friction curve (Claude, 2026-10-18).

FrictionParams is reinterpreted: mu_s = static limit muS, mu_d = Coulomb level muC, v0 = Stribeck speed vS;
mu(w) = muC + (muS - muC) exp(-(w/vS)^2) (Matusiak & Chatziioannou 2024, eq. 12, steady state only).

Original docstring: Compiled CPU implementation of the *corrected passive* bowed-string waveguide.

No fastmath, no CUDA, and at most two worker threads by default. The reference
numpy solver remains in bowed_string.py; parity tests compare actual waveforms.
Contact diagnostics are simulator states, not experimental ground-truth labels.
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "violin-bowing-boundaries/scripts"))
from bowed_string import FrictionParams, StringParams, prepare_waveguide, validate_simulation_inputs


@njit(cache=True, inline="always")
def _mu_gauss(w, mu_s, mu_c, vs):
    return mu_c + (mu_s - mu_c) * math.exp(-(w / vs) ** 2)


@njit(cache=True)
def _gauss_root(m, a, mu_s, mu_c, vs):
    """Largest slip speed w in (0, m) on the load line m - w = a mu(w) for a Gaussian Stribeck curve.

    For w > m - a mu_c the load line lies below the curve, so the largest root is in (0, m - a mu_c];
    scan down from there in 64 steps to the first sign change, then bisect 40 times.
    """
    if a <= 0:
        return m > 0, m
    top = m - a * mu_c
    if top <= 0:
        return False, 0.0
    hi = top
    if m - hi - a * _mu_gauss(hi, mu_s, mu_c, vs) >= 0:
        return True, hi
    n = 64
    for i in range(1, n + 1):
        lo = top * (1.0 - i / n)
        if m - lo - a * _mu_gauss(lo, mu_s, mu_c, vs) >= 0:
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if m - mid - a * _mu_gauss(mid, mu_s, mu_c, vs) >= 0:
                    lo = mid
                else:
                    hi = mid
            root = 0.5 * (lo + hi)
            return root > 0, root
        hi = lo
    return False, 0.0


@njit(cache=True, inline="always")
def _read(buffer, delay, n):
    position = n - delay
    base = math.floor(position)
    fraction = position - base
    size = buffer.size
    return (1 - fraction) * buffer[base % size] + fraction * buffer[(base + 1) % size]


@njit(cache=True, inline="always")
def _thiran_step(value, previous_input, previous_output, coefficient):
    return coefficient * value + previous_input - coefficient * previous_output


@njit(cache=True, parallel=True)
def _solve(gain, pole, d_bridge, d_nut, d_half, impedance, admittance, size, mu_s, mu_d, v0,
           velocity, normal_force, profile_rate, rate, start_time, steps, every, record_from, use_thiran):
    batch = impedance.size
    count = (steps - record_from) // every
    output = np.zeros((count, batch), dtype=np.float32)
    slip_output = np.zeros((count, batch), dtype=np.float32)
    slip_counts = np.zeros(batch, dtype=np.int64)
    slip_onsets = np.zeros(batch, dtype=np.int64)
    maximum_slip = np.zeros(batch)
    minimum_dissipation = np.full(batch, np.inf)
    maximum_load_error = np.zeros(batch)
    for k in prange(batch):
        buffer_b, buffer_n = np.zeros(size), np.zeros(size)
        state_b, state_n = 0.0, 0.0
        previous_input_b, previous_input_n = 0.0, 0.0
        previous_delay_b, previous_delay_n = 0.0, 0.0
        integer_b, integer_n = int(math.floor(d_bridge[k])), int(math.floor(d_nut[k]))
        fractional_b, fractional_n = d_bridge[k] - integer_b, d_nut[k] - integer_n
        allpass_b = (1 - fractional_b) / (1 + fractional_b)
        allpass_n = (1 - fractional_n) / (1 + fractional_n)
        sticking = True
        accumulator, slip_accumulator, filled = 0.0, 0.0, 0
        for n in range(steps):
            position = (start_time + n / rate) * profile_rate - 0.5
            j = min(max(int(math.floor(position)), 0), velocity.shape[0] - 2)
            fraction = min(max(position - j, 0.0), 1.0)
            vb = velocity[j, k] * (1 - fraction) + velocity[j + 1, k] * fraction
            fn = max(normal_force[j, k] * (1 - fraction) + normal_force[j + 1, k] * fraction, 0.0)
            if use_thiran:
                input_b = buffer_b[(n - integer_b) % size]
                input_n = buffer_n[(n - integer_n) % size]
                # Exact integer bypass avoids an uncancelled pole at -1 for delta=0.
                delayed_b = input_b if fractional_b == 0 else _thiran_step(input_b, previous_input_b, previous_delay_b, allpass_b)
                delayed_n = input_n if fractional_n == 0 else _thiran_step(input_n, previous_input_n, previous_delay_n, allpass_n)
                previous_input_b, previous_input_n = input_b, input_n
                previous_delay_b, previous_delay_n = delayed_b, delayed_n
            else:
                delayed_b = _read(buffer_b, d_bridge[k], n)
                delayed_n = _read(buffer_n, d_nut[k], n)
            state_b = gain[k] * (1 - pole[k]) * delayed_b + pole[k] * state_b
            state_n = gain[k] * (1 - pole[k]) * delayed_n + pole[k] * state_n
            free_slip = vb + state_b + state_n
            magnitude = abs(free_slip)
            a = admittance[k] * fn
            slip_exists, root = _gauss_root(magnitude, a, mu_s[k], mu_d[k], v0[k])
            can_stick = magnitude <= mu_s[k] * fn * admittance[k]
            previous_sticking = sticking
            sticking = (can_stick or not slip_exists) if sticking else not slip_exists
            root = max(root, 0.0)
            direction = 1.0 if free_slip > 0 else (-1.0 if free_slip < 0 else 0.0)
            mu = mu_d[k] + (mu_s[k] - mu_d[k]) * math.exp(-(root / v0[k]) ** 2)
            force = free_slip / admittance[k] if sticking else direction * fn * mu
            slip = 0.0 if sticking else direction * root
            buffer_b[n % size] = -state_n + admittance[k] * force
            buffer_n[n % size] = -state_b + admittance[k] * force
            if n >= record_from:
                accumulator += 2 * impedance[k] * _read(buffer_b, d_half[k], n)
                slip_accumulator += not sticking
                slip_counts[k] += not sticking
                slip_onsets[k] += previous_sticking and not sticking
                maximum_slip[k] = max(maximum_slip[k], abs(slip))
                minimum_dissipation[k] = min(minimum_dissipation[k], force * slip)
                maximum_load_error[k] = max(maximum_load_error[k], abs(free_slip - admittance[k] * force - slip))
                if (n - record_from + 1) % every == 0:
                    output[filled, k] = accumulator / every
                    slip_output[filled, k] = slip_accumulator / every
                    accumulator, slip_accumulator = 0.0, 0.0
                    filled += 1
    return output, slip_output, slip_counts, slip_onsets, maximum_slip, minimum_dissipation, maximum_load_error


def simulate_cpu(string: StringParams, friction: FrictionParams, betas, profile_velocity, profile_force,
                 profile_rate: float, rate: float, start_time: float, steps: int, output_every: int = 1,
                 record_from: int = 0, *, threads: int = 2, return_diagnostics: bool = False,
                 feedback_delay: str = "linear"):
    """Same force-array contract as bowed_string.simulate, with optional contact states.

    The returned slip_fraction_by_output averages the slip indicator over each output
    block. It is exact at output_every=1. No waveform-based classifier enters this
    state calculation. The first call includes numba compilation time. Optional thiran1
    changes only the feedback fractional-delay realization, retaining linear bridge
    observation interpolation and the same box-averaged output. The first-order allpass
    preserves magnitude but introduces phase dispersion; it is an established comparator.
    """
    if not isinstance(threads, int) or not 1 <= threads <= 2:
        raise ValueError("threads must be 1 or 2 for the laptop CPU budget")
    if feedback_delay not in ("linear", "thiran1"):
        raise ValueError("feedback_delay must be linear or thiran1")
    velocity, force = validate_simulation_inputs(betas, profile_velocity, profile_force, profile_rate, rate,
                                                start_time, steps, output_every, record_from, friction)
    config = prepare_waveguide(string, betas, rate)
    batch = np.asarray(betas).size
    spread = lambda x: np.broadcast_to(np.asarray(x, float), (batch,)).copy()
    # Include both outputs, profiles and worst-case simultaneously active line buffers.
    required = 2 * max(0, (steps - record_from) // output_every) * batch * 4
    required += velocity.nbytes + force.nbytes + 2 * threads * config["size"] * 8
    if required > 1024 ** 3:
        raise ValueError("simulation arrays exceed the 1 GiB laptop memory budget; reduce the batch/output window")
    old_threads = get_num_threads()
    started = time.perf_counter()
    try:
        set_num_threads(threads)
        result = _solve(config["gain"], config["pole"], config["delay_bridge"], config["delay_nut"],
                        config["delay_half"], config["impedance"], config["admittance"], config["size"],
                        spread(friction.mu_s), spread(friction.mu_d), spread(friction.v0),
                        np.ascontiguousarray(velocity), np.ascontiguousarray(force), profile_rate, rate,
                        start_time, steps, output_every, record_from, feedback_delay == "thiran1")
    finally:
        set_num_threads(old_threads)
    if not return_diagnostics:
        return result[0]
    recorded_steps = steps - record_from
    return result[0], {"slip_fraction_by_output": result[1], "slip_fraction": result[2] / max(recorded_steps, 1),
                       "slip_onsets": result[3], "max_slip_speed": result[4],
                       "min_friction_dissipation_watts": result[5], "max_load_line_residual": result[6],
                       "elapsed_seconds_including_possible_compilation": time.perf_counter() - started,
                       "threads": threads, "model": "passive-point-bow-waveguide-v2",
                       "feedback_delay": feedback_delay,
                       "extra_delay_state": "two previous inputs and two previous outputs per stroke" if feedback_delay == "thiran1" else "none",
                       "truth_scope": "internal simulator contact states, not measured regime labels"}
