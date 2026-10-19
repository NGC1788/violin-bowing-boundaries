"""Bowed-string waveguide with the thermal elasto-plastic contact of van Walstijn et al. (Claude, 2026-10-18).

Model: van Walstijn, Chatziioannou, Lampis & Matusiak, Acta Acustica 10, 47 (2026), doi 10.1051/aacus/2026042, eqs. (8)-(14), (22),
(30)-(32), (B.2), Table 1 (parameters fitted to the IWK cello rig, only at v = 0.1 m/s in that paper).
    friction force on the string   f = f_b sigma0 z                       (eq. 8 times bow width)
    dz/dt = w (1 - alpha(z, w, tau) z / z_ss(w, tau)),  z_ss = sign(w) y(tau) mu_str(|w|; F_b) / sigma0
    z_ba = chi y(tau) mu_d / sigma0,  alpha as in Matusiak 2024 (sinusoidal adhesion map)
    a_tau dtau/dt + [b_tau sqrt(|w| / F_b) + c_tau] tau = sigma0 z varsigma,  varsigma = alpha z w / z_ss
    y(tau) = (1 + y_a s) / (1 + s),  mu_str(tau) = (mu_s + mu_d y_a s) / (1 + y_a s),  s = (tau / tau_g)^xi
    mu_str(v; F_b) = mu_str(tau_ss(v; F_b)), with v(tau) from (B.2); F_b = f_b / W_b (force per unit length)
Simplifications (same as elastoplastic.py): point bow (W_b only enters through F_b), no bow-hair
compliance, no torsion; tau is updated semi-implicitly after the implicit z step.
Interface mirrors bowed_string_cpu.simulate_cpu; the FrictionParams argument is ignored (Table 1 values).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

sys.path.insert(0, str(Path.home() / "violin-bowing-boundaries/scripts"))
from bowed_string import StringParams, prepare_waveguide, validate_simulation_inputs  # noqa: E402

PARAMS = dict(mu_s=1.05, mu_d=0.3, sigma0=1.0e6, tau_g=25.0, y_a=0.4, xi=2.0, chi=0.5,
              a_tau=1.0e-6, b_tau=0.22, c_tau=1.0e-4, width=0.01)
SLIP_THRESHOLD = -0.95  # < 0: slip = saturated bristle (see elastoplastic.py); > 0: |w| threshold in m/s
LOGV = np.linspace(-8.0, 1.5, 761)   # log10 |w|
LOGF = np.linspace(-2.0, 3.5, 111)   # log10 F_b (N/m)


def mu_str_table(p=PARAMS):
    """mu_str(|w|; F_b) on the (LOGF, LOGV) grid via the steady-state map (B.2)."""
    tau = np.concatenate([[0.0], np.geomspace(1e-6, 1e5, 6000)])
    s = (tau / p["tau_g"]) ** p["xi"]
    mu_ss = (p["mu_s"] + p["mu_d"] * p["y_a"] * s) / (1 + s)
    mu_str = (p["mu_s"] + p["mu_d"] * p["y_a"] * s) / (1 + p["y_a"] * s)
    table = np.empty((LOGF.size, LOGV.size))
    for i, lf in enumerate(LOGF):
        fb = 10.0 ** lf
        root = (p["b_tau"] * tau + np.sqrt(p["b_tau"] ** 2 * tau ** 2 + 4 * p["c_tau"] * tau * mu_ss * fb)) / (2 * mu_ss)
        v = root ** 2 / fb
        assert np.all(np.diff(v) > 0), "v(tau) must be increasing"
        table[i] = np.interp(LOGV, np.log10(np.maximum(v, 1e-300)), mu_str, left=p["mu_s"], right=mu_str[-1])
    return table


@njit(cache=True, inline="always")
def _read(buffer, delay, n):
    position = n - delay
    base = math.floor(position)
    fraction = position - base
    size = buffer.size
    return (1 - fraction) * buffer[base % size] + fraction * buffer[(base + 1) % size]


@njit(cache=True, inline="always")
def _mu(table, row, frac_f, w, lv0, dlv):
    x = (math.log10(abs(w) + 1e-12) - lv0) / dlv
    x = min(max(x, 0.0), table.shape[1] - 1.000001)
    j = int(x)
    fx = x - j
    a = table[row, j] * (1 - fx) + table[row, j + 1] * fx
    b = table[row + 1, j] * (1 - fx) + table[row + 1, j + 1] * fx
    return a * (1 - frac_f) + b * frac_f


@njit(cache=True, inline="always")
def _alpha_ratio(z, w, zss_mag, zba):
    alpha = 0.0
    if w != 0.0 and z != 0.0 and (w > 0) == (z > 0):
        az = abs(z)
        if az >= zss_mag:
            alpha = 1.0
        elif az > zba:
            sz = 1.0 if z > 0 else -1.0
            h = (z - sz * (zss_mag + zba) / 2.0) / (zss_mag - zba)
            alpha = 0.5 * (1.0 + sz * math.sin(math.pi * h))
    return alpha


@njit(cache=True, parallel=True)
def _solve(gain, pole, d_bridge, d_nut, d_half, impedance, admittance, size, table, lv0, dlv, lf0, dlf,
           velocity, normal_force, profile_rate, rate, start_time, steps, every, record_from, prm, thr):
    mu_d, s0, tau_g, y_a, xi, chi, a_tau, b_tau, c_tau, width = prm[1], prm[2], prm[3], prm[4], prm[5], prm[6], prm[7], prm[8], prm[9], prm[10]
    batch = impedance.size
    count = (steps - record_from) // every
    output = np.zeros((count, batch), dtype=np.float32)
    slip_output = np.zeros((count, batch), dtype=np.float32)
    temp_output = np.zeros((count, batch), dtype=np.float32)
    dt = 1.0 / rate
    for k in prange(batch):
        buffer_b, buffer_n = np.zeros(size), np.zeros(size)
        state_b, state_n = 0.0, 0.0
        z, tau = 0.0, 0.0
        accumulator, slip_accumulator, temp_accumulator, filled = 0.0, 0.0, 0.0, 0
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
            zss_mag = 0.0
            if fn <= 0.0:
                z = 0.0
                force = 0.0
                w = free_slip
                tau = tau / (1.0 + dt * c_tau / a_tau)
            else:
                fb_len = fn / width
                xf = (math.log10(fb_len) - lf0) / dlf
                xf = min(max(xf, 0.0), table.shape[0] - 1.000001)
                row = int(xf)
                frac_f = xf - row
                sq = (tau / tau_g) ** xi
                y = (1.0 + y_a * sq) / (1.0 + sq)
                zba = chi * y * mu_d / s0
                z_prev = z
                P = free_slip
                Q = Y * fn * s0
                zmax = abs(z_prev) + abs(P) / Q + 10.0 * prm[0] / s0 + 1e-12
                lo, hi = -zmax, zmax
                for _ in range(52):
                    mid = 0.5 * (lo + hi)
                    wm = P - Q * mid
                    zss_mag = y * _mu(table, row, frac_f, wm, lv0, dlv) / s0
                    alpha = _alpha_ratio(mid, wm, zss_mag, zba)
                    zss = zss_mag if wm > 0 else -zss_mag
                    r = (mid - z_prev) / dt - wm * (1.0 - alpha * mid / zss)
                    if r > 0:
                        hi = mid
                    else:
                        lo = mid
                z = 0.5 * (lo + hi)
                w = P - Q * z
                force = fn * s0 * z
                zss_mag = y * _mu(table, row, frac_f, w, lv0, dlv) / s0
                alpha = _alpha_ratio(z, w, zss_mag, zba)
                vs = alpha * abs(z) * abs(w) / zss_mag
                heat = s0 * abs(z) * vs
                tau = (tau + dt / a_tau * heat) / (1.0 + dt / a_tau * (b_tau * math.sqrt(abs(w) / fb_len) + c_tau))
            buffer_b[n % size] = -state_n + Y * force
            buffer_n[n % size] = -state_b + Y * force
            if n >= record_from:
                accumulator += 2 * impedance[k] * _read(buffer_b, d_half[k], n)
                if thr > 0:
                    slip_accumulator += abs(w) > thr
                elif thr == 0:
                    slip_accumulator += w
                elif fn > 0.0:  # gross sliding = saturated bristle (zss_mag was evaluated at the final w)
                    slip_accumulator += (w * z > 0.0) and (abs(z) >= -thr * zss_mag)
                temp_accumulator += tau
                if (n - record_from + 1) % every == 0:
                    output[filled, k] = accumulator / every
                    slip_output[filled, k] = slip_accumulator / every
                    temp_output[filled, k] = temp_accumulator / every
                    accumulator, slip_accumulator, temp_accumulator = 0.0, 0.0, 0.0
                    filled += 1
    return output, slip_output, temp_output


_TABLE = None


def simulate_cpu(string: StringParams, friction, betas, profile_velocity, profile_force,
                 profile_rate: float, rate: float, start_time: float, steps: int, output_every: int = 1,
                 record_from: int = 0, *, threads: int = 2, return_diagnostics: bool = False,
                 feedback_delay: str = "linear", slip_threshold: float = SLIP_THRESHOLD):
    global _TABLE
    from bowed_string import FrictionParams
    velocity, force = validate_simulation_inputs(betas, profile_velocity, profile_force, profile_rate, rate,
                                                start_time, steps, output_every, record_from, FrictionParams())
    if _TABLE is None:
        _TABLE = mu_str_table()
    config = prepare_waveguide(string, betas, rate)
    p = PARAMS
    prm = np.array([p["mu_s"], p["mu_d"], p["sigma0"], p["tau_g"], p["y_a"], p["xi"], p["chi"],
                    p["a_tau"], p["b_tau"], p["c_tau"], p["width"]])
    old = get_num_threads()
    try:
        set_num_threads(threads)
        out, slip, temp = _solve(config["gain"], config["pole"], config["delay_bridge"], config["delay_nut"],
                                 config["delay_half"], config["impedance"], config["admittance"], config["size"],
                                 _TABLE, LOGV[0], LOGV[1] - LOGV[0], LOGF[0], LOGF[1] - LOGF[0],
                                 np.ascontiguousarray(velocity), np.ascontiguousarray(force), profile_rate, rate,
                                 start_time, steps, output_every, record_from, prm, slip_threshold)
    finally:
        set_num_threads(old)
    if not return_diagnostics:
        return out
    return out, {"slip_fraction_by_output": slip, "contact_temperature": temp}
