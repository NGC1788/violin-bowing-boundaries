#!/usr/bin/env python3
"""Batched bowed-string simulator: digital waveguide string, point bow, hyperbolic friction curve.

Stage-1 physics model (McIntyre, Schumacher & Woodhouse 1983 family) for comparison with the
robot measurements. Velocity waves travel between the bow and each termination in fractional
delay lines; each round trip is inverted and passed through a one-pole low-pass reflection
filter whose DC gain sets the damping of the fundamental and whose cut-off rounds the
Helmholtz corner. At the bow the string presents 2Z to the friction force by construction.
Friction follows mu(w) = mu_d + (mu_s - mu_d) v0 / (v0 + |w|); a sticking stroke stays stuck
while it can and a slipping stroke keeps slipping while a slip solution exists (hysteresis).
The bridge force is 2Z times the velocity wave arriving at the bridge.

A previous modal stiff-string implementation failed its time-step/admittance checks and
was rejected. That implementation failure is not a claim that modal string formulations
cannot converge; independent energy-consistent modal formulations remain valid comparators.

Every stroke in a batch has its own beta, bow-velocity and bow-force histories. String and
friction values may be scalars (one string for the batch) or arrays with one value per
stroke, which lets many parameter sets run in the same batch: the per-step cost is dominated
by launch overhead, not by the batch size. Array code runs on numpy or torch through a small backend shim; the per-step cost
is independent of the number of string modes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class StringParams:
    f0: float = 98.08               # Hz
    impedance: float = 0.946        # kg/s, Z; measured flyback law for Part 1 A1
    length: float = 0.7             # m
    q1: float = 2000.0              # quality factor of the fundamental from the reflection gain
    corner_hz: float = 6000.0       # one-pole cut-off of each round-trip reflection (corner rounding)

    @property
    def wave_speed(self) -> float:
        return 2 * self.length * self.f0


@dataclass
class FrictionParams:
    mu_s: float = 0.6725             # Mores (DAGA 2016) hyperbolic fit, limit at zero slip
    mu_d: float = 0.346
    v0: float = 0.06                 # m/s


class Backend:
    """The few array operations the time loop needs, for numpy or torch."""

    def __init__(self, name: str = "numpy", device: str = "cpu", dtype: str = "float64"):
        self.name = name
        if name == "torch":
            import torch
            self.torch = torch
            self.device = torch.device(device)
            self.dtype = getattr(torch, dtype)
        else:
            self.dtype = np.dtype(dtype)

    def asarray(self, value):
        if self.name == "torch":
            return self.torch.as_tensor(np.asarray(value), dtype=self.dtype, device=self.device)
        return np.asarray(value, dtype=self.dtype)

    def index(self, value):
        if self.name == "torch":
            return self.torch.as_tensor(np.asarray(value), dtype=self.torch.long, device=self.device)
        return np.asarray(value, dtype=np.int64)

    def zeros(self, shape):
        if self.name == "torch":
            return self.torch.zeros(shape, dtype=self.dtype, device=self.device)
        return np.zeros(shape, dtype=self.dtype)

    def ones_bool(self, n):
        if self.name == "torch":
            return self.torch.ones(n, dtype=self.torch.bool, device=self.device)
        return np.ones(n, dtype=bool)

    def where(self, condition, a, b):
        return self.torch.where(condition, a, b) if self.name == "torch" else np.where(condition, a, b)

    def clamp_min(self, x, value):
        return self.torch.clamp(x, min=value) if self.name == "torch" else np.maximum(x, value)

    def sqrt(self, x):
        return self.torch.sqrt(x) if self.name == "torch" else np.sqrt(x)

    def sign(self, x):
        return self.torch.sign(x) if self.name == "torch" else np.sign(x)

    def floor_index(self, x):
        if self.name == "torch":
            base = self.torch.floor(x)
            return base, base.long()
        base = np.floor(x)
        return base, base.astype(np.int64)

    def stack(self, values):
        return np.stack(values) if self.name == "numpy" else self.to_numpy(self.torch.stack(values))

    def to_numpy(self, x):
        return x.detach().cpu().numpy() if self.name == "torch" else np.asarray(x)


def friction_step(backend: Backend, free_slip, admittance, normal_force, sticking, friction: FrictionParams):
    """Bow force on the string for one step.

    ``free_slip`` W0 = v_bow - v_h, where the string velocity at the bow is v_h + Y f. Stick needs
    |W0 / Y| <= mu_s F_N. The slip speed w > 0 solves w + Y F_N mu(w) = |W0|, a quadratic for the
    hyperbolic curve; the larger root is used. Returns (force, sticking, slip_velocity).
    """
    xp = backend
    force_n = xp.clamp_min(normal_force, 0.0)
    a = admittance * force_n
    magnitude = abs(free_slip)
    b = friction.v0 + a * friction.mu_d - magnitude
    c = friction.v0 * (a * friction.mu_s - magnitude)
    disc = b * b - 4 * c
    rooted = xp.sqrt(xp.clamp_min(disc, 0.0))
    root = xp.where(b > 0, -2 * c / (b + rooted + (b <= 0)), (-b + rooted) / 2)  # no cancellation for b > 0
    slip_exists = (c < 0) | ((disc >= 0) & (b < 0) & (root > 0))
    can_stick = magnitude <= friction.mu_s * force_n * admittance
    stick = xp.where(sticking, can_stick | ~slip_exists, ~slip_exists)
    root = xp.clamp_min(root, 0.0)
    mu = friction.mu_d + (friction.mu_s - friction.mu_d) * friction.v0 / (friction.v0 + root)
    direction = xp.sign(free_slip)
    force = xp.where(stick, free_slip / admittance, direction * force_n * mu)
    slip = xp.where(stick, 0.0 * free_slip, direction * root)
    return force, stick, slip


def reflection_design(string: StringParams, rate: float):
    """Unconstrained coefficients for H(z)=g(1-p)/(1-p z^-1).

    Q1 is the fundamental decay target *before* fractional-delay interpolation losses.
    A first-order low-pass cannot realize every Q1/corner pair passively. Inspect the
    gain before simulation; changing the recurrence or clipping the gain silently
    would change the meaning of these physical parameters.
    """
    values = np.broadcast_arrays(np.asarray(string.f0, float), np.asarray(string.q1, float),
                                 np.asarray(string.corner_hz, float))
    if not (math.isfinite(rate) and rate > 0):
        raise ValueError("rate must be finite and positive")
    if any(np.any(~np.isfinite(v)) or np.any(v <= 0) for v in values):
        raise ValueError("f0, q1 and corner_hz must be finite and positive")
    if np.any(values[0] >= rate / 2):
        raise ValueError("f0 must lie below the sampling Nyquist frequency")
    pole = np.exp(-2 * np.pi * np.asarray(string.corner_hz, dtype=float) / rate)
    omega = 2 * np.pi * np.asarray(string.f0, dtype=float) / rate
    lowpass = np.abs((1 - pole) / (1 - pole * np.exp(-1j * omega)))
    gain = np.exp(-np.pi / (2 * np.asarray(string.q1, dtype=float))) / lowpass
    return gain, pole


def reflection_is_passive(string: StringParams, rate: float):
    """Boolean scalar/array: whether the requested loss filter has gain <= 1 at DC."""
    gain, _ = reflection_design(string, rate)
    return gain <= 1.0


def reflection_coefficients(string: StringParams, rate: float):
    """Passive (gain, pole), for y[n]=gain*(1-pole)*x[n]+pole*y[n-1].

    Two reflections per fundamental period provide the specified Q1 in the absence
    of interpolation loss. A requested nonpassive pair is rejected, never clipped.
    """
    gain, pole = reflection_design(string, rate)
    if np.any(gain > 1.0):
        raise ValueError("nonpassive reflection: requested q1/corner_hz needs DC gain > 1; "
                         "reduce q1 or increase corner_hz, then recalibrate")
    return gain, pole


def prepare_waveguide(string: StringParams, betas, rate: float):
    """Shared numpy geometry and passive filter design for reference and CPU solvers."""
    betas = np.asarray(betas, dtype=float)
    if betas.ndim != 1 or not betas.size or np.any(~np.isfinite(betas)) or np.any((betas <= 0) | (betas >= 1)):
        raise ValueError("betas must be a nonempty vector strictly between 0 and 1")
    batch = betas.size
    spread = lambda value: np.broadcast_to(np.asarray(value, dtype=float), (batch,)).copy()
    impedance = spread(string.impedance)
    lengths = spread(string.length)
    if np.any(~np.isfinite(impedance)) or np.any(impedance <= 0) or np.any(~np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("string impedance and length must be finite and positive")
    gain, pole = reflection_coefficients(string, rate)
    gain, pole = spread(gain), spread(pole)
    lag = pole / (1 - pole)
    f0 = spread(string.f0)
    delay_bridge = betas * rate / f0 - lag
    delay_nut = (1 - betas) * rate / f0 - lag
    if np.any(delay_bridge < 2) or np.any(delay_nut < 2):
        raise ValueError("beta too small for this rate/filter: each termination round trip must be >= 2 samples")
    size = int(math.ceil(max(delay_bridge.max(), delay_nut.max()))) + 3
    return {"gain": gain, "pole": pole, "delay_bridge": delay_bridge, "delay_nut": delay_nut,
            "delay_half": betas * rate / f0 / 2, "size": size, "impedance": impedance,
            "admittance": 1 / (2 * impedance)}


def linear_decay_diagnostics(string: StringParams, betas, rate: float):
    """Nominal-f0 loop attenuation including linear-interpolation losses.

    These are frequency-response diagnostics, not a fitted free-decay measurement.
    ``effective_q_at_nominal_f0`` can be below requested Q1 and depend on beta/rate.
    Phase error is relative to one full fundamental-period round trip.
    """
    config = prepare_waveguide(string, betas, rate)
    omega = 2 * np.pi * np.asarray(string.f0) / rate
    response = config["gain"] * (1 - config["pole"]) / (1 - config["pole"] * np.exp(-1j * omega))
    loop = response ** 2
    for key in ("delay_bridge", "delay_nut"):
        delay = config[key]
        base = np.floor(delay)
        frac = delay - base
        loop = loop * np.exp(-1j * omega * base) * ((1 - frac) + frac * np.exp(-1j * omega))
    return {"reflection_dc_gain": config["gain"], "reflection_pole": config["pole"],
            "loop_magnitude_at_nominal_f0": np.abs(loop),
            "effective_q_at_nominal_f0": -np.pi / np.log(np.abs(loop)),
            "loop_phase_error_radians_at_nominal_f0": np.angle(loop)}


def validate_simulation_inputs(betas, velocity, force, profile_rate, rate, start_time, steps, output_every, record_from,
                               friction: FrictionParams):
    """Fail before allocating state for malformed profiles and undefined friction."""
    batch = np.asarray(betas).size
    velocity, force = np.asarray(velocity, float), np.asarray(force, float)
    if velocity.ndim != 2 or velocity.shape != force.shape or velocity.shape[1] != batch or velocity.shape[0] < 2:
        raise ValueError("velocity/force profiles must have matching (at least 2 samples, batch) shapes")
    if np.any(~np.isfinite(velocity)) or np.any(~np.isfinite(force)):
        raise ValueError("profiles must contain finite values")
    if not (math.isfinite(profile_rate) and profile_rate > 0 and math.isfinite(start_time)):
        raise ValueError("profile_rate must be positive and start_time finite")
    if any(not isinstance(n, (int, np.integer)) for n in (steps, output_every, record_from)):
        raise ValueError("steps, output_every and record_from must be integers")
    if steps < 0 or output_every < 1 or not 0 <= record_from <= steps:
        raise ValueError("require steps >= record_from >= 0 and output_every >= 1")
    mu_s, mu_d, v0 = np.broadcast_arrays(np.asarray(friction.mu_s, float), np.asarray(friction.mu_d, float),
                                        np.asarray(friction.v0, float))
    if any(np.any(~np.isfinite(v)) for v in (mu_s, mu_d, v0)) or np.any(mu_s < mu_d) or np.any(mu_d < 0) or np.any(v0 <= 0):
        raise ValueError("friction requires finite mu_s >= mu_d >= 0 and v0 > 0")
    return velocity, force


def simulate(backend: Backend, string: StringParams, friction: FrictionParams, betas, profile_velocity,
             profile_force, profile_rate: float, rate: float, start_time: float, steps: int, output_every: int = 1,
             record_from: int = 0, progress=None):
    """Simulate a batch of strokes and return the bridge force.

    ``profile_velocity`` and ``profile_force`` have shape (P, B): bow velocity and bow force
    sampled at ``profile_rate`` with sample j at time (j + 0.5) / profile_rate (block means),
    linearly interpolated and held beyond the ends. Step n is at time ``start_time + n / rate``.
    The bridge force is averaged over ``output_every`` steps from step ``record_from``;
    returns (outputs, B) float32.
    """
    xp = backend
    betas = np.asarray(betas, dtype=float)
    batch = betas.size
    profile_velocity, profile_force = validate_simulation_inputs(
        betas, profile_velocity, profile_force, profile_rate, rate, start_time, steps, output_every, record_from, friction)
    spread = lambda value: np.broadcast_to(np.asarray(value, dtype=float), (batch,)).astype(float)  # noqa: E731
    config = prepare_waveguide(string, betas, rate)
    gain, pole, size = config["gain"], config["pole"], config["size"]
    impedance, admittance = config["impedance"], config["admittance"]
    gain, pole = xp.asarray(gain), xp.asarray(pole)
    impedance_t, admittance_t = xp.asarray(impedance), xp.asarray(admittance)
    rows = xp.index(np.arange(batch))
    buffer_b, buffer_n = xp.zeros((batch, size)), xp.zeros((batch, size))
    state_b, state_n = xp.zeros((batch,)), xp.zeros((batch,))
    d_b, d_n, d_half = (xp.asarray(config[key]) for key in ("delay_bridge", "delay_nut", "delay_half"))
    sticking = xp.ones_bool(batch)
    # Per-stroke friction values must live on the backend too, or a numpy array meets a device tensor.
    friction = FrictionParams(*(xp.asarray(spread(value)) for value in (friction.mu_s, friction.mu_d, friction.v0)))
    velocity = xp.asarray(profile_velocity)
    force_profile = xp.asarray(profile_force)
    profile_len = int(profile_velocity.shape[0])

    def read(buffer, delay, n):
        base, i0 = xp.floor_index(n - delay)
        frac = (n - delay) - base
        i0 = i0 % size
        return (1 - frac) * buffer[rows, i0] + frac * buffer[rows, (i0 + 1) % size]

    outputs = max(0, (steps - record_from) // output_every)
    recorded = np.zeros((outputs, batch), dtype=np.float32)
    chunk, filled = [], 0
    accumulator = xp.zeros((batch,))
    for n in range(steps):
        position = (start_time + n / rate) * profile_rate - 0.5
        j = min(max(int(math.floor(position)), 0), profile_len - 2)
        frac = min(max(position - j, 0.0), 1.0)
        vb = velocity[j] * (1 - frac) + velocity[j + 1] * frac
        fn = force_profile[j] * (1 - frac) + force_profile[j + 1] * frac
        state_b = gain * (1 - pole) * read(buffer_b, d_b, n) + pole * state_b
        state_n = gain * (1 - pole) * read(buffer_n, d_n, n) + pole * state_n
        force, sticking, _ = friction_step(xp, vb + state_b + state_n, admittance_t, fn, sticking, friction)
        column = n % size
        buffer_b[:, column] = -state_n + admittance_t * force  # toward the bridge: from the nut side plus the bow
        buffer_n[:, column] = -state_b + admittance_t * force
        if n >= record_from:
            accumulator = accumulator + 2 * impedance_t * read(buffer_b, d_half, n)
            if (n - record_from + 1) % output_every == 0:
                chunk.append(accumulator / output_every)
                accumulator = accumulator * 0
                if len(chunk) == 4096 or filled + len(chunk) == outputs:
                    recorded[filled:filled + len(chunk)] = xp.stack(chunk)
                    filled += len(chunk)
                    chunk = []
        if progress and n % 50000 == 0:
            progress(n, steps)
    return recorded
