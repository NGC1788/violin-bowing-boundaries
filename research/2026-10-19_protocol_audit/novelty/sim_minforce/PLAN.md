# Which ingredient lifts the minimum bow force above Schelleng? (simulation ablation, pre-registered)

Claude, written 2026-10-20 ≈23:30 KST, before any simulation exists or has been run. Post hoc with respect to the
measured boundaries (seen). Builder and verifier are separate workflow agents; this file is not edited after launch
(results go to RESULT.md and verify/).

## Background (what is already known)
- Measured: on this rig the minimum force is 5–20 × Schelleng's F_S filled with same-rig pluck damping, friction jump
  and Z (`novelty/string_damping/`), with β exponent −1.23 (onset force) / −1.54 (window force). Schoonderwaldt et al.
  2008 found 6–11 × on a violin and blamed ripple / corner rounding, which Schelleng's purely resistive termination ignores.
- Mansour, Woodhouse & Scavone 2017: time-domain simulations agree with the (revised) analytic minimum force; torsion
  changes little; frequency-dependent damping, bending stiffness, torsion produce "Schelleng ripples", "rabbit ears",
  precursors in the friction force.
- van Walstijn et al. 2026 (this rig, a cello G string): a full model (modal string with frequency-dependent damping,
  torsion, bow-hair compliance, finite bow width, thermal elasto-plastic friction) reproduced the measured Helmholtz
  region "reasonably well", including patchiness near the minimum-force line. They did not decompose which ingredient
  sets the minimum force relative to Schelleng's formula.
- Open question here: with this rig's own string parameters, does **frequency-dependent string damping alone**
  (corner rounding) lift the simulated minimum force by an order of magnitude over F_S and flatten its β exponent?
  Or is the gap carried by the friction law / bow?

## Model (fixed)
Single-polarisation transverse modal string, rigid ends, point bow, constant normal force, bow velocity ramped
linearly from 0 to v at the median measured acceleration for that speed (0.104 m/s² at 0.1 m/s; 0.088 at 0.05;
0.417 at 0.2), then held for 0.5 s. String A T1 (POMA Table 1): T = 145.31 N, μ = 7.721 g/m, Z = 1.0592 kg/s,
EI = 3.032e-4 N m², f0 = 97.8 Hz (Lampis 2024) → L = sqrt(T/μ)/(2 f0).
Modal damping ζ_n from `novelty/string_damping/damping_digitized.json` (panel "A T1"); missing modes by log-linear
interpolation in n; modes beyond the last digitized one by a power law log ζ = a + p log n fitted to all digitized modes.

Configurations (the ablation ladder):
- **C0 validation**: Coulomb-like friction (μs = 1.2 static, constant μd = 0.6 when slipping, i.e. Δμ = 0.6), ζ_n = ζ₁/n
  (frequency-independent resistive loss with Z/R = π ζ₁), no stiffness. Must reproduce Schelleng's form:
  F_min ∝ β^−2, and F_min / [Z² v / (2 R β² Δμ)] within a factor 2. If C0 fails, nothing else is interpreted.
- **C1 measured damping**: as C0 but measured ζ_n (frequency-dependent).
- **C2 friction curve**: as C1 but a hyperbolic sliding-friction curve μ(Δv) = μ∞ + (μs − μ∞) v₀/(v₀ + |Δv|) with
  μs = 1.2, μ∞ = 0.35, v₀ = 0.1 m/s (a standard bowed-string choice, fixed now; Δμ then depends on slip speed).
- **C3 stiffness**: C2 + bending stiffness (stiff-string mode frequencies f_n = n f0 sqrt(1 + B n²), B from EI).
Convergence: the builder chooses the number of modes (≥ 2/β_min) and sample rate, and must show that halving Δt and
doubling the mode count changes F_min by < 10 % at β = 0.03 and 0.12 for C1.

## Measurement analogue (fixed)
Force grid: 24 log-spaced forces 0.005–5 N. Helmholtz label from the bow-point velocity in the window 0.2–0.4 s after
the plateau starts: exactly one stick→slip transition per nominal period in ≥ 90 % of periods and mean period within
±5 % of 1/f0. F_min per β = the lower end of the first run of K = 2 consecutive Helmholtz forces (same K rule as the
data). β grid: 0.02, 0.03, 0.05, 0.08, 0.12, 0.2. Speeds: 0.1 m/s for all configurations; 0.05 and 0.2 m/s for C1 and C2
(speed exponent).

## Readings (fixed)
Q_sim(β) = F_min,sim / F_S(β), F_S with the same Δμ = 0.6 and Z/R_eff = (6/π) Σ_{n≤10} ζ_n/n as in string_damping
(for C0, Z/R = π ζ₁). β exponent b_β,sim = slope of log F_min,sim on log β (all six β).
- **H1 (corner rounding carries the gap)**: C1 has Q_sim(0.1) ≥ 5 → string damping alone lifts F_min by an order of
  magnitude.
- **H0 (it does not)**: C1 has Q_sim(0.1) ≤ 2 → the gap needs friction-law / bow ingredients (C2, C3, or beyond).
- In between: partial.
- Exponent: report b_β,sim for C0–C3 against measured −1.23 / −1.54 and Schelleng −2. "Flattens toward the data" if
  b_β,sim > −1.7 for the configuration that carries the gap.
- Absolute comparison: F_min,sim(β = 0.1, v = 0.1) against measured A T1 level 0.40 N (onset) / 0.27 N (window).
No configuration is tuned to the data. All configurations and both speeds are reported.

## Verification
A second agent tries to refute: reruns C0 and C1 independently, checks modal normalisation (mode mass μL/2,
bow-point admittance, Z recovered from the modal sum), the stick/slip solver, the Helmholtz classifier on raw
waveforms, convergence, and whether the reading follows from the numbers.
