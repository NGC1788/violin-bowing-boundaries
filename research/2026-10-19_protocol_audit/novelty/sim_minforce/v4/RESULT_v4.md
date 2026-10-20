# Simulation v4 — result (run once, `run_v4.log`, `results_v4.json`; Claude, 2026-10-21 ≈05:40 KST)

## Gates (pre-registered) — both passed
- C0f validation (frequency-independent loss with the fitted ζ(1) = 1.98e-4): F_min/F_S = 2.24 at β 0.1, b_β = −2.05.
- Loss filter: minimum-phase FIR, 64 and 256 taps failed, **1024 taps passed** (fundamental loss within 5.4 %,
  n 31–100 within 0.003 %). Phase delay at f0 58 samples, subtracted from the bridge delay.

## Readings (pre-registered)
- **r = F_min(C1s)/F_min(C0f) at β 0.1 = 27 → "substantial (r ≥ 3)"**: with the rig's measured, frequency-dependent
  string damping, the maintenance minimum rises by more than an order of magnitude.
- **Absolute level**: C1s F_min(β 0.1, v 0.1) = 0.335 N, Q = 16.8, against measured 0.40 N (onset) / 0.27 N (window),
  Q 20 / 14. The simulated level falls inside the measured range.
- **β dependence: opposite to the data.** C1s F_min = 3.70 / 1.11 / 0.335 / 0.075 / 0.017 N at β 0.05 / 0.08 / 0.1 /
  0.12 / 0.2 and no sustainable Helmholtz motion up to 5 N at β 0.02 and 0.03 → b_β ≈ −4.1 (four β), against Schelleng
  −2 and measured −1.23 (onset) / −1.54 (window). The extra loss raises F_min far more at small β.
- Speed exponent b_v ≈ 1.08 (0.075 / 0.335 / 0.335 N at 0.05 / 0.1 / 0.2 m/s; grid-limited and uneven).
- Convergence D = 8000 (2048 taps): same F_min at β 0.12, still none at 0.03 → pass.

## Caveats found when inspecting the labels (after the readings)
- **Patchy Helmholtz region.** Label strings over the 32 forces: β 0.1 `...H..HH....HH..` (single H labels well below
  0.335 N), β 0.12 `HHH....HHHHH`. The K = 2 rule gives a defined number, but F_min is fragile to one label.
- **Sensitivity to the damping extrapolation**: the pre-declared sensitivity C1s-low (power law fitted to modes ≤ 11,
  p = 0.43 instead of 0.66) gives F_min(β 0.1) = 2.03 N (6× higher), b_β ≈ −3.6. Modes above 17 are extrapolated.
- Maintenance test from an ideal start, point bow, no torsion, no bow-hair compliance, Coulomb friction, loss lumped
  at the bridge, one string (A T1). Not yet independently verified.

## Reading in words
String damping that grows with frequency (rounding the Helmholtz corner) is enough to lift the maintenance minimum to
the measured level at mid-β, as Schoonderwaldt et al. suspected — but in this model it makes the minimum force fall
much too steeply with β (≈ β^−4 instead of the measured ≈ β^−1.2 to β^−1.5). So it cannot be the whole explanation; the
β dependence needs another ingredient (candidates: finite bow width, torsion, bow-hair compliance, or the ramp/
establishment protocol of the measurement).
