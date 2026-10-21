# Simulation v8 — does the "damping steepens the β dependence" claim depend on the Helmholtz label? (pre-registered)

Claude, 2026-10-21 (before any v8 code; the commit time is the record).

## Why
The one claim that held across v4 (Coulomb) and v6 (smooth friction) is that the measured frequency-dependent damping
makes the maintenance minimum fall far more steeply with β (β^−4.1, β^−6.5). Both used the v3 amplitude-checked label.
In v6 many low-force runs at small β passed the period rule but failed only the amplitude check (`c`). If those are
sustained (weak, rounded-corner) Helmholtz motion, counting them would flatten or even reverse the β dependence.

## Model (fixed)
v6 strings unchanged (smooth friction): H1m (measured damping) and H0m (frequency-independent loss); v4 Coulomb C1s as
secondary. Ideal start, run 1.5 s, label window 1.3–1.5 s (later than v6, so start transients have had longer).
Three labels on the same runs:
- **L1** period rule only (one stick→slip per nominal period in ≥ 90 % of periods, mean period within ±5 %).
- **L2** L1 + amplitude check (the v3–v7 label).
- **L3** L1 + stationarity: RMS of the bow-point relative velocity in 1.4–1.5 s ≥ 0.95 × its RMS in 1.0–1.1 s
  (free decay of the fundamental alone would give ≈ 0.93 over 0.4 s; a sustained motion ≈ 1).
β 0.02/0.03/0.05/0.08/0.12/0.2 (+0.1), v 0.1, 32-force grid, K = 2.

## Readings (fixed)
- The claim **survives** if b_β(H1m) ≤ −3.0 under both L2 and L3; it is **label-dependent** if one of L2/L3 gives
  b_β > −2.5; otherwise "weakened". L1 is reported but not used for the reading (it accepts decaying motion).
- Control: H0m must give b_β in −2 ± 0.3 under L2 and L3 (else nothing is interpreted).
- C1s (Coulomb) reported under the same labels.
