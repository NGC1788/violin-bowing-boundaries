# Simulation v5 — the measurement's start-from-rest ramp with the v4 string (pre-registered)

Claude, 2026-10-21 (before any v5 code; the commit time is the record). Post hoc w.r.t. the measured boundaries.

## Question
v4 (maintenance from an ideal Helmholtz start) with the measured, frequency-dependent damping reproduced the measured
level at β 0.1 (0.335 N vs 0.27–0.40 N) but fell as β^−4.1 and could not sustain Helmholtz motion at β ≤ 0.03, while
the data fall as β^−1.23 (onset force) / β^−1.54 (window force). The robot does not start from Helmholtz motion: the
string is at rest and the bow speed ramps up linearly (median 0.104 m/s² at 0.1 m/s), and Helmholtz motion is judged
0.2–0.4 s after the plateau starts. Does that start (establishment) change the β dependence?

## Model (fixed; everything else exactly v4)
v4 waveguide and amplitude-checked label, v4 minimum-phase 1024-tap filter for the smooth power-law damping (C1s; gate
already passed), D = 4000, bridge-side loss, Coulomb μs 1.2 / μd 0.6, stick-if-possible, 32-force grid, K = 2 rule.
Start: string at rest, bow speed ramped linearly from 0 to v at acceleration a, then held; label window 0.2–0.4 s after
the plateau starts (the data's window).
- **R1 (reading)**: C1s, a = 0.104 m/s², v = 0.1, β 0.02/0.03/0.05/0.08/0.12/0.2 (+0.1).
- R1 at v = 0.05 (a 0.088) and 0.2 (a 0.417), β 0.1 → speed exponent.
- **R1-fast** (secondary): a = 1.0 m/s² (the data contain fast attacks too).
- **R1-low** (sensitivity): v4 C1s-low filter.
- **R0** (control, expected to fail as in v1): C0f with the slow ramp.
- Convergence: D = 8000 (2048 taps) at β 0.03 and 0.12 for R1 (pass if ≤ 1 grid step).
Seen before this plan: v1 C0 ramp (never Helmholtz), the verifier's uninterpretable modal C1 ramp numbers, all v4 results.

## Readings (fixed)
- b_β(R1) over the six β (needs F_min at ≥ 3 β, else "undefined"):
  **b_β ≥ −2.0 → "the ramp restores the measured flattening"**; −3.0 < b_β < −2.0 → partial; **b_β ≤ −3.0 → "it does not"**.
- Level at β 0.1: "matches" if F_min(R1) in [0.2, 0.55] N (measured 0.27–0.40 N), else report the ratio.
- Also reported: β at which R1 first sustains Helmholtz motion, b_v, R1-fast and R1-low against R1, R0.
