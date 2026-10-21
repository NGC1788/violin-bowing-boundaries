# Simulation v8 — result (run once, `run_v8.log`, `results_v8.json`; Claude, 2026-10-21 ≈12:21 KST)

- Control passed: H0m (frequency-independent loss) gives the same clean band under all three labels, b_β = −2.03.
- **H1m (smooth friction, measured damping): b_β = −1.21 (L1, period only), −4.91 (L2, amplitude check), −2.69 (L3,
  stationarity)** → pre-declared reading **"weakened"** (L3 is not ≤ −3.0, and not > −2.5).
- C1s (Coulomb, secondary): −6.96 (L1), −3.54 (L2), −6.46 (L3); labels patchy.
- What happened: with the measured damping, low-force runs at small β keep a sustained one-slip-per-period motion whose
  slip is broadened (rounded corner), so it fails the v3 amplitude thresholds but passes the stationarity test. The
  v4/v6 exponents (−4.1, −6.5) came largely from the amplitude thresholds.

## Revised simulation statement (supersedes the v6 "robust" item 2)
- Under labels that require sustained motion (L2, L3), with either friction law, the simulated β exponent with the
  measured damping is −2.7 to −6.5: at or steeper than Schelleng's −2, never the measured −1.2 to −1.5. How much
  steeper depends on the label and the friction law.
- ~~Only the label that also accepts decaying periodic motion (L1) gives a measured-like −1.2~~ **Withdrawn (found when
  plotting, ≈12:23):** the L1 minimum sits at the lowest grid force (0.00045 N) at β ≥ 0.08 (smooth) — these are
  left-censored ("below the grid"), so the L1 slope −1.21 is an artifact of the grid floor and means nothing.
- The absolute level depends on the friction law (v6) and here also on the label (β 0.1, smooth: 0.0008 / 0.030 /
  0.0020 N for L1 / L2 / L3).

## Grid-floor censoring (found when plotting; affects the numbers above)
- Values equal to the lowest grid force (0.00045 N) are left-censored. H1m: L1 at β 0.08 / 0.12 / 0.2; **L3 at β 0.2**.
  C1s: L1 at β 0.12; L3 at β 0.2.
- Excluding the censored point, the H1m L3 slope over β 0.03–0.12 is −2.82 (with it −2.69). Because the true value at
  β 0.2 can only be lower, the true L3 slope is at least this steep. The reading stays "weakened" (between −3.0 and −2.5).

## Damping effect on the level at β 0.1 under each label (smooth friction; H1m / H0m)
L1 0.16, L2 6.06, L3 0.41. Under the sustained-motion label (L3) the measured damping
*lowers* the β 0.1 minimum (0.0020 vs 0.0050 N); under the amplitude label (L2) it raises it ×6.1. So the
v4/v6 statement "damping raises the maintenance minimum" is also label-dependent.

## Overall simulation conclusion (v1–v8)
The corner-exact waveguide reproduces Schelleng for frequency-independent loss with both friction laws (β exponent
−2.0 to −2.05, all labels). With the measured frequency-dependent damping, every quantity depends on the friction law
and on the definition of sustained Helmholtz motion: the β 0.1 level changes ×0.4 to ×27 (v4 Coulomb,
L2), the β exponent −2.7 to −6.5 among sustained-motion labels, and no version gives the measured flattening or level
robustly or establishes Helmholtz motion from rest under the robot's ramp. The simulation does not identify the cause
of the shortfall; it shows that in this model class the answer is not robust to the friction law or the label.
