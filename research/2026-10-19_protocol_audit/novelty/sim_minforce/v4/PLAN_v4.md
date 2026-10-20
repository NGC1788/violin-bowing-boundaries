# Simulation v4 — smooth measured damping, minimum-phase loss filter (pre-registered)

Claude, 2026-10-21 (before any v4 code; the commit time is the record). Post hoc w.r.t. the measured boundaries.

## What is kept from v3 (validated)
Corner-exact waveguide (`../verify/vsim.py` + `../v3/vsim3.py`), A T1 string, Coulomb μs 1.2 / μd 0.6, stick-if-possible,
ideal Helmholtz start (maintenance test), v = 0.1 m/s, 0.5 s run, label window 0.3–0.5 s, amplitude-checked label
(v1 classifier AND slip fraction in [0.5β, 2β] AND median slip |Δv| ≥ 0.5 v/β), 32-force grid 0.00045–5 N, K = 2 rule,
D = 4000, all loss on the bridge side. v3 C0b (measured ζ₁ = 2.88e-4, frequency independent) passed: F_min/F_S 2.08,
b_β −1.90.

## What changes (why v3 C1 failed: bumpy digitized ζ_n and a linear-phase FIR that could not fit)
- **Smooth damping target**: power law ζ(n) = e^a n^p fitted (least squares in log) to all digitized A T1 modes
  (a = −8.528, p = 0.658; this gives ζ(1) = 1.98e-4, 31 % below the measured ζ₁). Per-period loss target
  ℓ(n) = 2π n ζ(n), capped at 20.
- **Minimum-phase FIR** from that magnitude (real-cepstrum method, FFT 2^18 points), bridge delay shortened by the
  rounded phase delay at f0. Lengths tried in this fixed order, the first passing the gate is used: 64, 256, 1024 taps;
  if none passes, a one-pole IIR loss filter fitted to ℓ(n) for n ≤ 100; if that fails too, nothing is interpreted.
- **Gate** (on loss, not on magnitude): realised per-period loss −log|H(n)| within ±10 % of ℓ(n) for n = 1..30 and
  ±30 % for n = 31..100.

## Configurations
- **C0f** (new baseline with the same fundamental loss as the fit): ζ_n = ζ(1)/n, frequency independent. Must pass the
  same validation as C0b: F_min/F_S(ζ(1)) in [1, 4] at β 0.1 and b_β in −2 ± 0.3.
- **C1s** (reading): smooth measured damping as above.
- **C1s-low** (sensitivity only): power law fitted to modes ≤ 11 only.
β 0.02/0.03/0.05/0.08/0.12/0.2 (+0.1); C1s also at v = 0.05 and 0.2 (β 0.1); convergence D = 8000 at β 0.03 and 0.12 for
C1s with the filter redesigned (pass if F_min moves ≤ 1 grid step).

## Readings (fixed)
- r = F_min(C1s)/F_min(C0f) at β 0.1: **r ≥ 3** frequency-dependent loss lifts the maintenance minimum substantially;
  **r ≤ 1.5** it does not; otherwise partial. (Also reported against v3 C0b.)
- Q = F_min(C1s)/0.0199 N against measured Q 20 (onset) / 14 (window) at β 0.1.
- b_β(C1s) over the six β against −2 (Schelleng) and measured −1.23 (onset) / −1.54 (window); b_v(C1s) against 1.
- Caveats fixed now: maintenance test only (the data come from a ramp); one string (A T1); loss lumped at the bridge.
