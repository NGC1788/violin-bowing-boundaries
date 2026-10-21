# Simulation v6 — smooth (velocity-dependent) friction curve (pre-registered)

Claude, 2026-10-21 (before any v6 code; the commit time is the record). Post hoc w.r.t. the measured boundaries.

## Question
v5: with the abrupt Coulomb friction (1.2 → 0.6), the damped string almost never establishes Helmholtz motion from rest
under the robot's ramp. Does a standard velocity-dependent friction curve fix establishment, and what does it do to the
maintenance minimum and its β dependence?

## Model (fixed; everything else as v4/v5)
Friction during slip: μ(u) = μ∞ + (μs − μ∞) v₀/(v₀ + u), u = |slip speed|, μs = 1.2, μ∞ = 0.35, v₀ = 0.1 m/s (the v1 C2
choice); stick-if-possible; the slip speed is the unique positive root of the quadratic from v = w + Y f.
- **H0m** (gate): frequency-independent loss with ζ(1) of the power law (as v4 C0f), ideal Helmholtz start.
- **H1m**: v4 smooth measured damping (1024-tap minimum-phase filter), ideal start.
- **H1r (main)**: H1m string, start from rest, ramp 0.104 m/s², label 0.2–0.4 s after the plateau starts (as v5).
- **H0r** (control): H0m string with the ramp.
β 0.02/0.03/0.05/0.08/0.12/0.2 (+0.1), v = 0.1; ideal start: initial stick force μ(v/β) F.

## Readings (fixed)
- Gate: H0m b_β in −2 ± 0.3 and F_min(0.1)/F_S in [0.5, 4] with F_S using Δμ = μs − μ(v/β) at β 0.1 (≈ 0.77) and
  Z/R = π ζ(1). If the gate fails, nothing is interpreted.
- Establishment: H1r has F_min at ≥ 3 β? If not, "smooth friction does not restore establishment".
  If yes: b_β(H1r) ≥ −2.0 → "restores the measured flattening"; ≤ −3.0 → "does not"; else partial; level "matches" if
  F_min(H1r, β 0.1) in [0.2, 0.55] N.
- Maintenance: r = F_min(H1m)/F_min(H0m) at β 0.1 (≥ 3 substantial, ≤ 1.5 not) and b_β(H1m) against v4's −4.1.
