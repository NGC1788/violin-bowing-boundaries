# Simulation ablation v2 — maintenance test with a corner-exact waveguide (pre-registered)

Claude, 2026-10-21 ≈00:12 KST, before any v2 code exists. Post hoc w.r.t. the measured boundaries.

## Why a v2
v1 (`../PLAN.md`) failed its own validation: under the ramp-from-rest protocol the nearly lossless C0 string locks into
multi-slip motion and never gives a K = 2 Helmholtz run (builder and verifier agree; `../RESULT.md`, `../verify/VERIFY.md`).
By the v1 rule nothing was interpreted. The verifier showed that *maintenance* of an existing Helmholtz motion in a
corner-exact digital waveguide with the stick-if-possible Coulomb rule reproduces Schelleng (F_min/F_S 2.1–2.5,
b_β = −1.95). Seen before this plan: v1 C1 numbers from the band-limited modal solver under the ramp (Q 9–60, rule-dependent,
unconverged) — declared uninformative by the verifier. No v2 C1 number exists.

## Model (fixed)
Verifier's waveguide (`../verify/vsim.py`, McIntyre–Woodhouse, integer delays, D = fs/f0 samples per period, A T1
string: T 145.31 N, μ 7.721 g/m, f0 97.8 Hz), Coulomb friction μs 1.2 / μd 0.6, **stick-if-possible rule**, point bow,
v = 0.1 m/s constant, **ideal Helmholtz start** (mid-stick, friction μd F), run 0.5 s, label window 0.3–0.5 s with the
v1 classifier (one stick→slip per nominal period in ≥ 90 % of periods, mean period within ±5 %). Bridge loop lossless;
all loss lumped in the nut-side loop:
- **C0n**: frequency-independent loss, per-period gain exp(−2π ζ₁) (ζ_n = ζ₁/n).
- **C1n**: measured frequency-dependent loss: linear-phase FIR (delay K subtracted from the nut delay) with per-period
  magnitude exp(−2π n ζ_n) at f = n f0, ζ_n from `../../string_damping/damping_digitized.json` (A T1; interpolated in
  n; power law beyond mode 17, as in v1). Realised-vs-target magnitude error reported for n ≤ 100.
- **C1n-flat** (sensitivity, reported, not the reading): ζ_n held at the mode-17 value beyond 17.
Force grid: v1 extended grid (32 forces, 0.00045–5 N). F_min = lower end of the first K = 2 run of Helmholtz labels.
β: 0.02, 0.03, 0.05, 0.08, 0.12, 0.2 (exponent) and 0.1 (reading). D = 1000; convergence D = 2000 at β 0.03 and 0.12
for C1n (pass if F_min changes < 10 %, i.e. ≤ 1 grid step).
Sanity: slip fraction of Helmholtz-labelled runs reported (physical ≈ β).

## Readings (fixed)
- Validation: C0n F_min(0.1) / F_S(C0) in [1, 4] and b_β in −2 ± 0.3; else nothing is interpreted.
- r = F_min,C1n(0.1) / F_min,C0n(0.1). **r ≥ 3**: frequency-dependent (corner-rounding) loss lifts the maintenance
  minimum substantially. **r ≤ 1.5**: it does not. Otherwise partial.
- Q = F_min,C1n(0.1) / F_S(C1) with F_S(C1) = 0.0199 N, against measured A T1 Q 20 (onset) / 14 (window).
- b_β(C1n) against −2 (Schelleng) and −1.23 / −1.54 (measured).
Caveat fixed in advance: this is a maintenance test; the measured boundary is reached from a ramp (establishment +
maintenance), so agreement or disagreement in level is not decisive about establishment.
