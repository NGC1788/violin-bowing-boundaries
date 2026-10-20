# Simulation v3 — maintenance test, loss on the bridge side (pre-registered)

Claude, 2026-10-21 (written before any v3 code; commit time is the record). Post hoc w.r.t. the measured boundaries.

## Changes from v2 (which failed validation)
- All loss lumped in the **bridge-side** loop (Schelleng's resistive-bridge picture); nut loop lossless.
- D = 4000 samples per period (fs ≈ 391 kHz) so that the bridge loop (βD ≥ 80 samples) can hold a linear-phase FIR
  (K = 64, delay subtracted from the bridge delay). FIR quality gate: max |log(realised/target)| ≤ 0.1 for n ≤ 30,
  reported for n ≤ 100.
- Label = v1 classifier AND amplitude checks: slip fraction in [0.5β, 2β] and median |Δv| during slip ≥ 0.5 v/β.
- Seen before this plan: the verifier's bridge-lumped C0 maintenance gave F_min/F_S 1.24–3.0 (so C0b is expected to pass);
  v2's uninterpreted nut-lumped C1n numbers.

## Configurations
- **C0b** validation: per-period bridge gain exp(−2π ζ₁) (frequency independent).
- **C1b**: FIR with per-period magnitude exp(−2π n ζ_n), measured ζ_n (A T1), power law beyond mode 17.
- **C1b-flat** (sensitivity): ζ_n constant beyond mode 17.
Everything else as v2: A T1 string, Coulomb μs 1.2 / μd 0.6, stick-if-possible, ideal Helmholtz start, v = 0.1 m/s,
0.5 s run, label window 0.3–0.5 s, 32-force grid 0.00045–5 N, K = 2 rule, β 0.02/0.03/0.05/0.08/0.12/0.2 (+0.1).
Extra: C1b at v = 0.05 and 0.2 (β = 0.1) for the speed exponent; convergence D = 8000 (K = 128) at β 0.03 and 0.12
(pass if F_min moves ≤ 1 grid step).

## Readings (fixed, same as v2)
- Validation: C0b F_min(0.1)/F_S(C0) in [1, 4] and b_β in −2 ± 0.3, else nothing is interpreted.
- r = F_min,C1b(0.1)/F_min,C0b(0.1): r ≥ 3 → frequency-dependent loss lifts the maintenance minimum substantially;
  r ≤ 1.5 → it does not; otherwise partial.
- Q = F_min,C1b(0.1)/F_S(C1) (F_S(C1) = 0.0199 N) against measured A T1 Q 20 (onset) / 14 (window); b_β(C1b) against
  −2 and measured −1.23 / −1.54; b_v(C1b) against 1.
- Caveat fixed now: maintenance test only; the measured boundary is reached from a ramp.
