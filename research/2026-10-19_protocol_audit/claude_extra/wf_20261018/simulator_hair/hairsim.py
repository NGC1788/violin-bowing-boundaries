"""Point-bow waveguide with optional physical elements (Claude, 2026-10-18, wf_20261018/simulator_hair).

A NEW solver (copied structure, no existing file edited). With every new element switched off it executes the
same floating-point operations as scripts/bowed_string_cpu.py (feedback_delay="linear"), so the outputs agree
bit for bit (test_parity.py). Elements, each optional and each taken from the literature (no tuning):

  damping  "onepole"  existing one-pole reflection filters (Q1, corner_hz) and linear-interpolated delays.
           "physical" distributed string damping sigma(k) = r0 + r1 k^2 of Matusiak & Chatziioannou, JASA 156,
                      1135 (2024) eq. (1), Table I (r0 1.537 s^-1, r1 0.0087 m^2/s), applied per string side in
                      proportion to path length (bridge-side round trip beta/f0, nut side (1-beta)/f0); realised
                      as a linear-phase FIR that also carries the fractional delay; optional bending dispersion
                      (EI from Table I: E 1.37e10 Pa, r 0.5 mm) with kappa^2 = EI / (rho A).
  torsion  second waveguide that sees the same friction force (series admittance 1/(2 Z_T)); "gain" mode is the
           constant per-reflection gain of claude_extra/bowed_string_torsion.py (parity), "physical" mode is a
           constant-Q (Q_T) FIR per side. Matusiak Table I: c_T 849.53 m/s, theta 4.2e-10 kg m,
           K_T 3.0312e-4 N m^2, Q 29, r 0.5 mm -> Z_T = sqrt(K_T theta) / r^2 = 1.427 kg/s, f_T = 606.8 Hz.
  hair     bow-hair compliance in series with the contact (bow stick rigid, moving at v_b; hair surface moves at
           v_b - u, u = d(eta)/dt), point bow -> totals over the hair width W = 0.01 m:
             "kv"  massless Kelvin-Voigt, Matusiak eq. (3): f = k eta + d u, Table I K 4.8297e6 N/m^2,
                   D 5.7674e3 kg/(s m) -> k 4.83e4 N/m, d 57.7 kg/s;
             "msd" mass-spring-damper, van Walstijn et al. Acta Acustica 10, 47 (2026) eq. (54), Table 1
                   (from Pitteroff & Woodhouse): D_h 0.62 kg/m, K_h 11e6 N/m^2, zeta_h 1000 kg/(s m)
                   -> m 6.2 g, k 1.1e5 N/m, r 10 kg/s.
           Trapezoidal rule (energy-conserving for spring and mass; the discrete energy balance
           dH = dt (fbar ubar - r ubar^2) holds to round-off, reported as a diagnostic).
  friction "hyperbolic" (closed-form root, repo law) or "ep" elasto-plastic S-model (Matusiak Table III:
           sigma0 3.186e5 N/m, sigma1 0.0027 kg/s, vS 0.228, muC 0.5071, muS 1.0207), backward-Euler bristle state
           solved by bisection exactly as claude_extra/elastoplastic.py.

Series connection at the bow: string transverse, torsion and hair all carry the same force f, their velocities
add, so the contact sees Y = Y_string + Y_torsion + Y_hair and a free slip velocity that includes the hair's
history term. Contact states (slip flag per sample) are simulator states, not measured labels.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

sys.path.insert(0, str(Path.home() / "violin-bowing-boundaries/scripts"))
from bowed_string import FrictionParams, StringParams, prepare_waveguide, validate_simulation_inputs  # noqa: E402

# ------------------------------------------------------------------ literature parameter sets
MAT_STRING = dict(r0=1.53714, r1=0.0087, L=0.7, rho=10059.0, radius=5e-4, T=149.6, E=1.37e10)
MAT_STRING["A"] = math.pi * MAT_STRING["radius"] ** 2
MAT_STRING["EI"] = MAT_STRING["E"] * math.pi * MAT_STRING["radius"] ** 4 / 4
MAT_STRING["kappa2"] = MAT_STRING["EI"] / (MAT_STRING["rho"] * MAT_STRING["A"])
MAT_TORSION = dict(z=math.sqrt(3.0312e-4 * 4.2e-10) / (5e-4) ** 2, f=849.53 / (2 * 0.7), q=29.0)
VW_TORSION = dict(z=math.sqrt(2.9e-4 * 4.2e-10) / (0.487e-3) ** 2, f=math.sqrt(2.9e-4 / 4.2e-10) / (2 * 0.7), q=30.0)
HAIR_KV_MAT = dict(kind="kv", k=4.8297e6 * 0.01, d=5.7674e3 * 0.01)
HAIR_MSD_VW = dict(kind="msd", m=0.62 * 0.01, k=11.0e6 * 0.01, r=1000.0 * 0.01)
EP_MAT = dict(s0=3.1860e5, s1=0.0027, chi=0.7, thr=-0.95)
FRICTION_MORES = FrictionParams()
FRICTION_MAT_HYP = FrictionParams(mu_s=1.0207, mu_d=0.5071, v0=0.19)
FRICTION_MAT_EP = FrictionParams(mu_s=1.0207, mu_d=0.5071, v0=0.228)  # reinterpreted: muS, muC, vS


# ------------------------------------------------------------------ FIR design
def _side_response(omega_s, D, rate, sigma, extra_delay):
    """Desired side transfer function (without the bulk integer delay)."""
    return np.exp(-sigma * D / rate) * np.exp(-1j * omega_s * extra_delay)


def design_fir(D, rate, sigma_fn, dispersion_fn, M, nfft=1 << 14):
    """Taps h[0..L-1] such that y[n] = sum_j h[j] x[n - n0 - j] approximates the side response.

    D: total side delay (samples); sigma_fn(omega_rad_per_s) -> decay rate (1/s) applied over D / rate seconds;
    dispersion_fn(omega) -> extra delay in samples (negative = earlier). The response is centred at
    residual = D - n0 = M + frac. Returns (taps, n0, max|H|, rms error over 0..10 kHz).
    """
    n0 = int(math.floor(D)) - M
    if n0 < 1:
        raise ValueError(f"FIR half-length {M} too long for side delay {D:.2f}")
    residual = D - n0
    L = 2 * M + 2
    w = 2 * np.pi * np.arange(nfft // 2 + 1) / nfft          # rad/sample
    om = w * rate                                            # rad/s
    G = np.exp(-sigma_fn(om) * D / rate) * np.exp(-1j * w * (residual + dispersion_fn(om)))
    h = np.fft.irfft(G, nfft)[:L].copy()
    # mild taper of the outermost 3 samples on each side (the kernel is ~0 there when M is adequate)
    taper = np.ones(L)
    edge = 3
    ramp = 0.5 * (1 - np.cos(np.pi * (np.arange(edge) + 0.5) / edge))
    taper[:edge] = ramp
    taper[-edge:] = ramp[::-1]
    h *= taper
    H = np.fft.rfft(h, nfft) * np.exp(1j * w * 0)  # response of taps (delay n0 not included)
    peak = float(np.abs(H).max())
    if peak > 1.0:
        h /= peak * (1 + 1e-12)
        H = np.fft.rfft(h, nfft)
    band = om < 2 * np.pi * 10000
    err = float(np.sqrt(np.mean(np.abs(H[band] - G[band]) ** 2)))
    return h, n0, float(np.abs(H).max()), err


DISPERSION_CAP_HZ = 5000.0  # bending dispersion is modelled up to this frequency (constant extra delay above)


def transverse_firs(betas, rate, f0, r0, r1, c, kappa2=0.0, max_m=(40, 110)):
    """Per-stroke bridge/nut FIRs for distributed damping (and optional bending dispersion)."""
    def sigma(om):
        k2 = _k2(om, c, kappa2)
        return r0 + r1 * k2

    def k_of(om):
        return np.sqrt(_k2(om, c, kappa2))

    out = []
    for beta in np.asarray(betas, float):
        sides = []
        for frac_len, mmax in ((beta, max_m[0]), (1 - beta, max_m[1])):
            D = frac_len * rate / f0
            if kappa2 > 0:
                cap = 2 * np.pi * DISPERSION_CAP_HZ
                disp = lambda om, D=D: D * (c * k_of(np.clip(om, 1e-6, cap)) / np.clip(om, 1e-6, cap) - 1.0)
                mmax = mmax * 2
            else:
                disp = lambda om: 0.0 * om
            # loss acts over the travel time of the envelope; with weak dispersion use D / rate
            sides.append(_trimmed(D, rate, sigma, disp, mmax))
        out.append(sides)
    return out


def _trimmed(D, rate, sigma_fn, disp_fn, mmax, rel=1e-6):
    """Design with the longest allowed half-length, then shorten to the support where |h| > rel * max."""
    M = min(mmax, int(math.floor(D)) - 2)
    h, n0, peak, err = design_fir(D, rate, sigma_fn, disp_fn, M)
    centre = D - n0
    idx = np.flatnonzero(np.abs(h) > rel * np.abs(h).max())
    support = int(math.ceil(max(abs(idx[0] - centre), abs(idx[-1] - centre))))
    need = support + max(6, support // 5)
    if need < M:
        return design_fir(D, rate, sigma_fn, disp_fn, max(need, 4))
    return h, n0, peak, err


def _k2(om, c, kappa2):
    if kappa2 <= 0:
        return (om / c) ** 2
    return (-c * c + np.sqrt(c ** 4 + 4 * kappa2 * om * om)) / (2 * kappa2)


def torsion_firs(betas, rate, f_t, q_t, max_m=(30, 60)):
    out = []
    for beta in np.asarray(betas, float):
        sides = []
        for frac_len, mmax in ((beta, max_m[0]), (1 - beta, max_m[1])):
            D = frac_len * rate / f_t
            sides.append(_trimmed(D, rate, lambda om: np.abs(om) / (2 * q_t), lambda om: 0.0 * om, mmax))
        out.append(sides)
    return out


def _pack(firs):
    """List of per-stroke (h, n0, ...) -> padded tap matrix, n0 vector, length vector."""
    L = max(len(f[0]) for f in firs)
    taps = np.zeros((len(firs), L))
    n0 = np.zeros(len(firs), np.int64)
    ln = np.zeros(len(firs), np.int64)
    for i, f in enumerate(firs):
        taps[i, :len(f[0])] = f[0]
        n0[i] = f[1]
        ln[i] = len(f[0])
    return taps, n0, ln


# ------------------------------------------------------------------ kernel
@njit(cache=True, inline="always")
def _read(buffer, delay, n):
    position = n - delay
    base = math.floor(position)
    fraction = position - base
    size = buffer.size
    return (1 - fraction) * buffer[base % size] + fraction * buffer[(base + 1) % size]


@njit(cache=True, inline="always")
def _fir(buffer, taps, n0, length, n):
    size = buffer.size
    acc = 0.0
    base = n - n0
    for j in range(length):
        acc += taps[j] * buffer[(base - j) % size]
    return acc


@njit(cache=True, inline="always")
def _ep_residual(z, z_prev, dt, P, Q, fn, s0, mu_s, mu_c, vs, chi):
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
def _solve(gain, pole, d_bridge, d_nut, d_half, impedance, admittance, size,
           use_fir, taps_b, n0_b, len_b, taps_n, n0_n, len_n,
           t_mode, t_gain, t_bridge, t_nut, t_adm, t_size, ttaps_b, tn0_b, tlen_b, ttaps_n, tn0_n, tlen_n,
           h_kind, h_m, h_k, h_r,
           f_kind, mu_s, mu_d, v0, s0, s1, chi, thr,
           velocity, normal_force, profile_rate, rate, start_time, steps, every, record_from):
    batch = impedance.size
    count = (steps - record_from) // every
    output = np.zeros((count, batch), dtype=np.float32)
    slip_flags = np.zeros((steps - record_from, batch), dtype=np.uint8)
    min_diss = np.full(batch, np.inf)
    max_hair_res = np.zeros(batch)
    max_hair_energy = np.zeros(batch)
    work_in = np.zeros(batch)
    diss_friction = np.zeros(batch)
    dt = 1.0 / rate
    for k in prange(batch):
        buffer_b, buffer_n = np.zeros(size), np.zeros(size)
        tbuf_b, tbuf_n = np.zeros(t_size), np.zeros(t_size)
        state_b, state_n = 0.0, 0.0
        tstate_b, tstate_n = 0.0, 0.0
        sticking = True
        z = 0.0
        eta, u_h, f_prev = 0.0, 0.0, 0.0
        # hair instantaneous admittance and recurrence constants (trapezoidal rule)
        y_h = 0.0
        alpha_h, beta_h = 0.0, 0.0
        if h_kind == 1:
            y_h = 1.0 / (h_k * dt / 2 + h_r)
        elif h_kind == 2:
            alpha_h = h_m / dt
            beta_h = h_r / 2 + h_k * dt / 4
            y_h = 1.0 / (2 * (alpha_h + beta_h))
        Y = admittance[k] + t_adm
        Y = Y + y_h
        accumulator, filled = 0.0, 0
        for n in range(steps):
            position = (start_time + n / rate) * profile_rate - 0.5
            j = min(max(int(math.floor(position)), 0), velocity.shape[0] - 2)
            fraction = min(max(position - j, 0.0), 1.0)
            vb = velocity[j, k] * (1 - fraction) + velocity[j + 1, k] * fraction
            fn = max(normal_force[j, k] * (1 - fraction) + normal_force[j + 1, k] * fraction, 0.0)
            if use_fir:
                state_b = _fir(buffer_b, taps_b[k], n0_b[k], len_b[k], n)
                state_n = _fir(buffer_n, taps_n[k], n0_n[k], len_n[k], n)
            else:
                delayed_b = _read(buffer_b, d_bridge[k], n)
                delayed_n = _read(buffer_n, d_nut[k], n)
                state_b = gain[k] * (1 - pole[k]) * delayed_b + pole[k] * state_b
                state_n = gain[k] * (1 - pole[k]) * delayed_n + pole[k] * state_n
            free_slip = vb + state_b + state_n
            if t_mode == 1:
                tstate_b = t_gain * _read(tbuf_b, t_bridge[k], n)
                tstate_n = t_gain * _read(tbuf_n, t_nut[k], n)
                free_slip = free_slip + tstate_b + tstate_n
            elif t_mode == 2:
                tstate_b = _fir(tbuf_b, ttaps_b[k], tn0_b[k], tlen_b[k], n)
                tstate_n = _fir(tbuf_n, ttaps_n[k], tn0_n[k], tlen_n[k], n)
                free_slip = free_slip + tstate_b + tstate_n
            h_free = 0.0
            if h_kind == 1:
                h_free = y_h * (h_k * eta + h_k * dt / 2 * u_h)
            elif h_kind == 2:
                h_free = -((u_h * (alpha_h - beta_h) + f_prev / 2 - h_k * eta) / (alpha_h + beta_h))
            if h_kind != 0:
                free_slip = free_slip + h_free
            slip_flag = False
            if f_kind == 0:
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
                slip = 0.0 if sticking else direction * root
                slip_flag = not sticking
            else:
                if fn <= 0.0:
                    z = 0.0
                    force = 0.0
                    slip = free_slip
                else:
                    z_prev = z
                    P = free_slip + Y * s1 * z_prev / dt
                    Q = Y * (s0 + s1 / dt)
                    zmax = abs(z_prev) + abs(P) / Q + 10.0 * fn * mu_s[k] / s0 + 1e-9
                    lo, hi = -zmax, zmax
                    for _ in range(48):
                        mid = 0.5 * (lo + hi)
                        if _ep_residual(mid, z_prev, dt, P, Q, fn, s0, mu_s[k], mu_d[k], v0[k], chi) > 0:
                            hi = mid
                        else:
                            lo = mid
                    z = 0.5 * (lo + hi)
                    force = s0 * z + s1 * (z - z_prev) / dt
                    slip = free_slip - Y * force
                zss_w = fn * (mu_d[k] + (mu_s[k] - mu_d[k]) * math.exp(-(slip / v0[k]) ** 2)) / s0
                slip_flag = (fn > 0.0) and (slip * z > 0.0) and (abs(z) >= -thr * zss_w)
            # hair update (trapezoidal), energy balance check
            if h_kind == 1:
                u_new = y_h * force - h_free
                eta_new = eta + dt / 2 * (u_new + u_h)
                e_old = 0.5 * h_k * eta * eta
                e_new = 0.5 * h_k * eta_new * eta_new
                ub = 0.5 * (u_new + u_h)
                fb = 0.5 * (force + f_prev)
                res = (e_new - e_old) - dt * (fb * ub - h_r * ub * ub)
                max_hair_res[k] = max(max_hair_res[k], abs(res))
                max_hair_energy[k] = max(max_hair_energy[k], e_new)
                eta, u_h = eta_new, u_new
            elif h_kind == 2:
                u_new = y_h * force - h_free
                eta_new = eta + dt / 2 * (u_new + u_h)
                e_old = 0.5 * h_m * u_h * u_h + 0.5 * h_k * eta * eta
                e_new = 0.5 * h_m * u_new * u_new + 0.5 * h_k * eta_new * eta_new
                ub = 0.5 * (u_new + u_h)
                fb = 0.5 * (force + f_prev)
                res = (e_new - e_old) - dt * (fb * ub - h_r * ub * ub)
                max_hair_res[k] = max(max_hair_res[k], abs(res))
                max_hair_energy[k] = max(max_hair_energy[k], e_new)
                eta, u_h = eta_new, u_new
            f_prev = force
            buffer_b[n % size] = -state_n + admittance[k] * force
            buffer_n[n % size] = -state_b + admittance[k] * force
            if t_mode != 0:
                tbuf_b[n % t_size] = -tstate_n + t_adm * force
                tbuf_n[n % t_size] = -tstate_b + t_adm * force
            if n >= record_from:
                accumulator += 2 * impedance[k] * _read(buffer_b, d_half[k], n)
                slip_flags[n - record_from, k] = slip_flag
                min_diss[k] = min(min_diss[k], force * slip)
                work_in[k] += vb * force * dt
                diss_friction[k] += force * slip * dt
                if (n - record_from + 1) % every == 0:
                    output[filled, k] = accumulator / every
                    accumulator = 0.0
                    filled += 1
    return output, slip_flags, min_diss, max_hair_res, max_hair_energy, work_in, diss_friction


# ------------------------------------------------------------------ wrapper
def simulate(string: StringParams, friction: FrictionParams, betas, profile_velocity, profile_force,
             profile_rate: float, rate: float, start_time: float, steps: int, output_every: int = 1,
             record_from: int = 0, *, threads: int = 2, damping="onepole", damping_params=None, dispersion=False,
             torsion=None, torsion_mode="physical", hair=None, friction_kind="hyperbolic", ep=None,
             return_designs=False):
    """Returns (bridge_force[outputs, batch] float32, diagnostics dict incl. per-sample slip flags uint8)."""
    if not isinstance(threads, int) or not 1 <= threads <= 2:
        raise ValueError("threads must be 1 or 2")
    velocity, force = validate_simulation_inputs(betas, profile_velocity, profile_force, profile_rate, rate,
                                                start_time, steps, output_every, record_from, friction)
    config = prepare_waveguide(string, betas, rate)
    betas = np.asarray(betas, float)
    batch = betas.size
    spread = lambda x: np.broadcast_to(np.asarray(x, float), (batch,)).copy()
    designs = {}
    if damping == "onepole":
        use_fir = False
        taps_b = taps_n = np.zeros((batch, 1)); n0_b = n0_n = len_b = len_n = np.ones(batch, np.int64)
    elif damping == "physical":
        use_fir = True
        p = dict(MAT_STRING) if damping_params is None else {**MAT_STRING, **damping_params}
        c = 2 * float(np.asarray(string.length)) * float(np.asarray(string.f0))
        firs = transverse_firs(betas, rate, float(np.asarray(string.f0)), p["r0"], p["r1"], c,
                               p["kappa2"] if dispersion else 0.0)
        taps_b, n0_b, len_b = _pack([f[0] for f in firs])
        taps_n, n0_n, len_n = _pack([f[1] for f in firs])
        designs["transverse"] = [(f[0][2], f[0][3], f[1][2], f[1][3]) for f in firs]
        need = int(max(n0_b.max() + len_b.max(), n0_n.max() + len_n.max())) + 3
        if need > config["size"]:
            config["size"] = need
    else:
        raise ValueError("damping must be onepole or physical")
    if torsion is None:
        t_mode, t_gain, t_adm, t_bridge, t_nut, t_size = 0, 0.0, 0.0, np.full(batch, 2.0), np.full(batch, 2.0), 8
        tt_b = tt_n = np.zeros((batch, 1)); tn0_b = tn0_n = tl_b = tl_n = np.ones(batch, np.int64)
    else:
        t_adm = 1 / (2 * torsion["z"])
        t_bridge = betas * rate / torsion["f"]
        t_nut = (1 - betas) * rate / torsion["f"]
        if torsion_mode == "gain":
            t_mode, t_gain = 1, math.exp(-math.pi / (2 * torsion["q"]))
            if np.any(t_bridge < 2):
                raise ValueError("torsional bridge-side delay < 2 samples")
            t_size = int(math.ceil(t_nut.max())) + 3
            tt_b = tt_n = np.zeros((batch, 1)); tn0_b = tn0_n = tl_b = tl_n = np.ones(batch, np.int64)
        else:
            t_mode, t_gain = 2, 0.0
            tf = torsion_firs(betas, rate, torsion["f"], torsion["q"])
            tt_b, tn0_b, tl_b = _pack([f[0] for f in tf])
            tt_n, tn0_n, tl_n = _pack([f[1] for f in tf])
            designs["torsion"] = [(f[0][2], f[0][3], f[1][2], f[1][3]) for f in tf]
            t_size = int(max(tn0_b.max() + tl_b.max(), tn0_n.max() + tl_n.max())) + 3
    if hair is None:
        h_kind, h_m, h_k, h_r = 0, 0.0, 0.0, 0.0
    elif hair["kind"] == "kv":
        h_kind, h_m, h_k, h_r = 1, 0.0, float(hair["k"]), float(hair["d"])
    elif hair["kind"] == "msd":
        h_kind, h_m, h_k, h_r = 2, float(hair["m"]), float(hair["k"]), float(hair["r"])
    else:
        raise ValueError("hair kind must be kv or msd")
    if friction_kind == "hyperbolic":
        f_kind, e = 0, dict(s0=1.0, s1=0.0, chi=0.7, thr=-0.95)
    elif friction_kind == "ep":
        f_kind, e = 1, dict(EP_MAT) if ep is None else {**EP_MAT, **ep}
    else:
        raise ValueError("friction_kind must be hyperbolic or ep")
    old = get_num_threads()
    started = time.perf_counter()
    try:
        set_num_threads(threads)
        res = _solve(config["gain"], config["pole"], config["delay_bridge"], config["delay_nut"], config["delay_half"],
                     config["impedance"], config["admittance"], config["size"],
                     use_fir, np.ascontiguousarray(taps_b), n0_b, len_b, np.ascontiguousarray(taps_n), n0_n, len_n,
                     t_mode, t_gain, np.asarray(t_bridge, float), np.asarray(t_nut, float), t_adm, t_size,
                     np.ascontiguousarray(tt_b), tn0_b, tl_b, np.ascontiguousarray(tt_n), tn0_n, tl_n,
                     h_kind, h_m, h_k, h_r,
                     f_kind, spread(friction.mu_s), spread(friction.mu_d), spread(friction.v0),
                     float(e["s0"]), float(e["s1"]), float(e["chi"]), float(e["thr"]),
                     np.ascontiguousarray(velocity), np.ascontiguousarray(force), profile_rate, rate, start_time,
                     steps, output_every, record_from)
    finally:
        set_num_threads(old)
    out, flags, min_diss, hair_res, hair_e, work, diss = res
    diag = {"slip_flags": flags, "min_friction_dissipation_watts": min_diss, "max_hair_energy_residual_J": hair_res,
            "max_hair_energy_J": hair_e, "bow_work_J": work, "friction_dissipation_J": diss,
            "elapsed_s": time.perf_counter() - started, "designs": designs,
            "contact_admittance": float(config["admittance"][0] + t_adm + (0 if h_kind == 0 else
                                        (1 / (h_k / rate / 2 + h_r) if h_kind == 1 else 1 / (2 * (h_m * rate + h_r / 2 + h_k / rate / 4)))))}
    return out, diag
