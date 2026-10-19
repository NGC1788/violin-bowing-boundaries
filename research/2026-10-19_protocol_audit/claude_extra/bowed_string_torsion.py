"""Bowed-string waveguide with a torsional polarisation (Claude, 2026-10-18).

Same transverse waveguide, reflection filters, point bow and hyperbolic friction as
scripts/bowed_string_cpu.py (linear fractional delay), plus a second, torsional waveguide that sees the same
friction force at the bow point (McIntyre-Schumacher-Woodhouse; Woodhouse 2003). The contact is solved with the
combined admittance Y = 1/(2 Z) + 1/(2 Z_T): the bow drives the string surface, whose velocity is the
transverse velocity plus r * angular velocity. Torsional waves travel faster (c_T) and are heavily damped
(Q_T ~ 30), which is the classic reason real strings settle into Helmholtz motion within a few periods.

Default torsion (cello G, van Walstijn et al. 2026 Table 1 / Matusiak & Chatziioannou 2024 Table II):
  K_T 2.9e-4 N m^2, P_T 4.2e-10 kg m, r 0.487 mm  ->  Z_T = sqrt(K_T P_T) / r^2 = 1.47 kg/s,
  c_T = sqrt(K_T / P_T) = 830 m/s on L = 0.7 m  ->  torsional fundamental 593 Hz, Q_T 30 (vW) / 34 (Matusiak).
Each torsional reflection: gain exp(-pi / (2 Q_T)), no low-pass. Bridge-force output is transverse only.
Interface mirrors bowed_string_cpu.simulate_cpu (extra keyword: torsion=dict(z=, f=, q=) or None).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

sys.path.insert(0, str(Path.home() / "violin-bowing-boundaries/scripts"))
from bowed_string import FrictionParams, StringParams, prepare_waveguide, validate_simulation_inputs  # noqa: E402

TORSION = dict(z=1.47, f=593.0, q=30.0)


@njit(cache=True, inline="always")
def _read(buffer, delay, n):
    position = n - delay
    base = math.floor(position)
    fraction = position - base
    size = buffer.size
    return (1 - fraction) * buffer[base % size] + fraction * buffer[(base + 1) % size]


@njit(cache=True, parallel=True)
def _solve(gain, pole, d_bridge, d_nut, d_half, impedance, admittance, size, mu_s, mu_d, v0,
           velocity, normal_force, profile_rate, rate, start_time, steps, every, record_from,
           t_gain, t_bridge, t_nut, t_adm, t_size):
    batch = impedance.size
    count = (steps - record_from) // every
    output = np.zeros((count, batch), dtype=np.float32)
    slip_output = np.zeros((count, batch), dtype=np.float32)
    for k in prange(batch):
        buffer_b, buffer_n = np.zeros(size), np.zeros(size)
        tbuf_b, tbuf_n = np.zeros(t_size), np.zeros(t_size)
        state_b, state_n = 0.0, 0.0
        sticking = True
        accumulator, slip_accumulator, filled = 0.0, 0.0, 0
        Y = admittance[k] + t_adm
        for n in range(steps):
            position = (start_time + n / rate) * profile_rate - 0.5
            j = min(max(int(math.floor(position)), 0), velocity.shape[0] - 2)
            fraction = min(max(position - j, 0.0), 1.0)
            vb = velocity[j, k] * (1 - fraction) + velocity[j + 1, k] * fraction
            fn = max(normal_force[j, k] * (1 - fraction) + normal_force[j + 1, k] * fraction, 0.0)
            state_b = gain[k] * (1 - pole[k]) * _read(buffer_b, d_bridge[k], n) + pole[k] * state_b
            state_n = gain[k] * (1 - pole[k]) * _read(buffer_n, d_nut[k], n) + pole[k] * state_n
            tstate_b = t_gain * _read(tbuf_b, t_bridge[k], n)
            tstate_n = t_gain * _read(tbuf_n, t_nut[k], n)
            free_slip = vb + state_b + state_n + tstate_b + tstate_n
            magnitude = abs(free_slip)
            a = Y * fn
            b = v0[k] + a * mu_d[k] - magnitude
            c = v0[k] * (a * mu_s[k] - magnitude)
            discriminant = b * b - 4 * c
            rooted = math.sqrt(max(discriminant, 0.0))
            root = -2 * c / (b + rooted) if b > 0 else (-b + rooted) / 2
            slip_exists = (c < 0) or ((discriminant >= 0) and (b < 0) and (root > 0))
            can_stick = magnitude <= mu_s[k] * fn * Y
            sticking = (can_stick or not slip_exists) if sticking else not slip_exists
            root = max(root, 0.0)
            direction = 1.0 if free_slip > 0 else (-1.0 if free_slip < 0 else 0.0)
            mu = mu_d[k] + (mu_s[k] - mu_d[k]) * v0[k] / (v0[k] + root)
            force = free_slip / Y if sticking else direction * fn * mu
            buffer_b[n % size] = -state_n + admittance[k] * force
            buffer_n[n % size] = -state_b + admittance[k] * force
            tbuf_b[n % t_size] = -tstate_n + t_adm * force
            tbuf_n[n % t_size] = -tstate_b + t_adm * force
            if n >= record_from:
                accumulator += 2 * impedance[k] * _read(buffer_b, d_half[k], n)
                slip_accumulator += not sticking
                if (n - record_from + 1) % every == 0:
                    output[filled, k] = accumulator / every
                    slip_output[filled, k] = slip_accumulator / every
                    accumulator, slip_accumulator = 0.0, 0.0
                    filled += 1
    return output, slip_output


def simulate_cpu(string: StringParams, friction: FrictionParams, betas, profile_velocity, profile_force,
                 profile_rate: float, rate: float, start_time: float, steps: int, output_every: int = 1,
                 record_from: int = 0, *, threads: int = 2, return_diagnostics: bool = False,
                 feedback_delay: str = "linear", torsion=TORSION):
    velocity, force = validate_simulation_inputs(betas, profile_velocity, profile_force, profile_rate, rate,
                                                start_time, steps, output_every, record_from, friction)
    config = prepare_waveguide(string, betas, rate)
    betas = np.asarray(betas, float)
    batch = betas.size
    spread = lambda x: np.broadcast_to(np.asarray(x, float), (batch,)).copy()
    if torsion is None:
        t_gain, t_adm, t_bridge, t_nut, t_size = 0.0, 0.0, np.full(batch, 2.0), np.full(batch, 2.0), 8
    else:
        t_gain = math.exp(-math.pi / (2 * torsion["q"]))
        t_adm = 1 / (2 * torsion["z"])
        t_bridge = betas * rate / torsion["f"]
        t_nut = (1 - betas) * rate / torsion["f"]
        if np.any(t_bridge < 2):
            raise ValueError("torsional bridge-side delay < 2 samples; raise the rate or beta")
        t_size = int(math.ceil(t_nut.max())) + 3
    old = get_num_threads()
    try:
        set_num_threads(threads)
        out, slip = _solve(config["gain"], config["pole"], config["delay_bridge"], config["delay_nut"],
                           config["delay_half"], config["impedance"], config["admittance"], config["size"],
                           spread(friction.mu_s), spread(friction.mu_d), spread(friction.v0),
                           np.ascontiguousarray(velocity), np.ascontiguousarray(force), profile_rate, rate,
                           start_time, steps, output_every, record_from, t_gain, t_bridge, t_nut, t_adm, t_size)
    finally:
        set_num_threads(old)
    if not return_diagnostics:
        return out
    return out, {"slip_fraction_by_output": slip}
