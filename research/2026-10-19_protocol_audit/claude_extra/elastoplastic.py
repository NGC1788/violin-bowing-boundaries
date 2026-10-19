"""Bowed-string waveguide with an elasto-plastic (bristle) bow-string contact (Claude, 2026-10-18).

Same waveguide, reflection filters and point bow as scripts/bowed_string_cpu.py (Codex's corrected
passive version); only the contact law differs. Model and parameters: Matusiak & Chatziioannou,
JASA 156, 1135 (2024), eqs. (8)-(12), Table III S-model:
    dz/dt = w (1 - alpha(w, z) z / z_ss(w)),   z_ss(w) = sign(w) (F_N / s0) mu(w)
    mu(w) = muC + (muS - muC) exp(-(w / vS)^2),  z_ba = 0.7 muC F_N / s0
    alpha = 0 (|z| <= z_ba), 1/2 (1 + sign(z) sin(pi h)) (z_ba < |z| < |z_ss|), 1 (|z| >= |z_ss|),
            and alpha = 0 whenever sign(w) != sign(z)
    friction force on the string  f = s0 z + s1 dz/dt;  slip w = free_slip - Y f  (Y = admittance)
Each step is solved implicitly (backward Euler for z) by bisection in z; the bracket always
contains a sign change of the residual (it is dominated by (z - z_prev)/dt at both ends).
Not included (unlike Matusiak 2024): bow-hair compliance, torsion. Slip indicator for contact-state
labels: slip_threshold > 0 -> |w| > threshold; == 0 -> raw w returned; < 0 (default) -> saturated bristle.

FrictionParams is reinterpreted: mu_s = muS, mu_d = muC, v0 = vS. Interface mirrors
bowed_string_cpu.simulate_cpu so sim_contact2.py can use it (FRICTION_CURVE=ep).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

sys.path.insert(0, str(Path.home() / "violin-bowing-boundaries/scripts"))
from bowed_string import FrictionParams, StringParams, prepare_waveguide, validate_simulation_inputs  # noqa: E402

SIGMA0 = 3.1860e5      # N/m   bristle stiffness (Table III, S-model)
SIGMA1 = 0.0027        # kg/s  bristle damping
CHI = 0.7              # z_ba = CHI * muC * F_N / s0
SLIP_THRESHOLD = -0.95  # < 0: slip = bristle saturated (|z| >= 0.95 |z_ss(w)|, w z > 0). An absolute |w| threshold
                        # (e.g. 0.1 m/s) is invalid: slow sliding of a decaying string never exceeds v_b.


@njit(cache=True, inline="always")
def _read(buffer, delay, n):
    position = n - delay
    base = math.floor(position)
    fraction = position - base
    size = buffer.size
    return (1 - fraction) * buffer[base % size] + fraction * buffer[(base + 1) % size]


@njit(cache=True, inline="always")
def _residual(z, z_prev, dt, P, Q, fn, s0, mu_s, mu_c, vs, chi):
    w = P - Q * z
    mu = mu_c + (mu_s - mu_c) * math.exp(-(w / vs) ** 2)
    zss_mag = fn * mu / s0
    zba = chi * mu_c * fn / s0
    alpha = 0.0
    if w != 0.0 and z != 0.0 and (w > 0) == (z > 0):
        az = abs(z)
        if az >= zss_mag:
            alpha = 1.0
        elif az > zba and zss_mag > zba:
            sz = 1.0 if z > 0 else -1.0
            h = (z - sz * (zss_mag + zba) / 2.0) / (zss_mag - zba)
            alpha = 0.5 * (1.0 + sz * math.sin(math.pi * h))
    zss = zss_mag if w > 0 else -zss_mag
    ratio = (alpha * z / zss) if zss != 0.0 else 0.0
    return (z - z_prev) / dt - w * (1.0 - ratio)


@njit(cache=True, parallel=True)
def _solve(gain, pole, d_bridge, d_nut, d_half, impedance, admittance, size, mu_s, mu_c, vs,
           velocity, normal_force, profile_rate, rate, start_time, steps, every, record_from, s0, s1, chi, thr):
    batch = impedance.size
    count = (steps - record_from) // every
    output = np.zeros((count, batch), dtype=np.float32)
    slip_output = np.zeros((count, batch), dtype=np.float32)
    dt = 1.0 / rate
    for k in prange(batch):
        buffer_b, buffer_n = np.zeros(size), np.zeros(size)
        state_b, state_n = 0.0, 0.0
        z = 0.0
        accumulator, slip_accumulator, filled = 0.0, 0.0, 0
        Y = admittance[k]
        for n in range(steps):
            position = (start_time + n / rate) * profile_rate - 0.5
            j = min(max(int(math.floor(position)), 0), velocity.shape[0] - 2)
            fraction = min(max(position - j, 0.0), 1.0)
            vb = velocity[j, k] * (1 - fraction) + velocity[j + 1, k] * fraction
            fn = max(normal_force[j, k] * (1 - fraction) + normal_force[j + 1, k] * fraction, 0.0)
            delayed_b = _read(buffer_b, d_bridge[k], n)
            delayed_n = _read(buffer_n, d_nut[k], n)
            state_b = gain[k] * (1 - pole[k]) * delayed_b + pole[k] * state_b
            state_n = gain[k] * (1 - pole[k]) * delayed_n + pole[k] * state_n
            free_slip = vb + state_b + state_n
            if fn <= 0.0:
                z = 0.0
                force = 0.0
                w = free_slip
            else:
                z_prev = z
                P = free_slip + Y * s1 * z_prev / dt
                Q = Y * (s0 + s1 / dt)
                zmax = abs(z_prev) + abs(P) / Q + 10.0 * fn * mu_s[k] / s0 + 1e-9
                lo, hi = -zmax, zmax
                for _ in range(48):
                    mid = 0.5 * (lo + hi)
                    if _residual(mid, z_prev, dt, P, Q, fn, s0, mu_s[k], mu_c[k], vs[k], chi) > 0:
                        hi = mid
                    else:
                        lo = mid
                z = 0.5 * (lo + hi)
                force = s0 * z + s1 * (z - z_prev) / dt
                w = free_slip - Y * force
            buffer_b[n % size] = -state_n + Y * force
            buffer_n[n % size] = -state_b + Y * force
            if n >= record_from:
                accumulator += 2 * impedance[k] * _read(buffer_b, d_half[k], n)
                if thr > 0:
                    slip_accumulator += abs(w) > thr
                elif thr == 0:
                    slip_accumulator += w
                else:  # gross sliding = saturated bristle (|z| >= -thr * |z_ss(w)|, same sign as w)
                    zss_w = fn * (mu_c[k] + (mu_s[k] - mu_c[k]) * math.exp(-(w / vs[k]) ** 2)) / s0
                    slip_accumulator += (fn > 0.0) and (w * z > 0.0) and (abs(z) >= -thr * zss_w)
                if (n - record_from + 1) % every == 0:
                    output[filled, k] = accumulator / every
                    slip_output[filled, k] = slip_accumulator / every
                    accumulator, slip_accumulator = 0.0, 0.0
                    filled += 1
    return output, slip_output


def simulate_cpu(string: StringParams, friction: FrictionParams, betas, profile_velocity, profile_force,
                 profile_rate: float, rate: float, start_time: float, steps: int, output_every: int = 1,
                 record_from: int = 0, *, threads: int = 2, return_diagnostics: bool = False,
                 feedback_delay: str = "linear", slip_threshold: float = SLIP_THRESHOLD):
    velocity, force = validate_simulation_inputs(betas, profile_velocity, profile_force, profile_rate, rate,
                                                start_time, steps, output_every, record_from, friction)
    config = prepare_waveguide(string, betas, rate)
    batch = np.asarray(betas).size
    spread = lambda x: np.broadcast_to(np.asarray(x, float), (batch,)).copy()
    old = get_num_threads()
    try:
        set_num_threads(threads)
        out, slip = _solve(config["gain"], config["pole"], config["delay_bridge"], config["delay_nut"],
                           config["delay_half"], config["impedance"], config["admittance"], config["size"],
                           spread(friction.mu_s), spread(friction.mu_d), spread(friction.v0),
                           np.ascontiguousarray(velocity), np.ascontiguousarray(force), profile_rate, rate,
                           start_time, steps, output_every, record_from, SIGMA0, SIGMA1, CHI, slip_threshold)
    finally:
        set_num_threads(old)
    if not return_diagnostics:
        return out
    return out, {"slip_fraction_by_output": slip}
