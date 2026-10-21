# Label-threshold sensitivity — result (run once; Claude, 2026-10-21 ≈12:29 KST)

## Gate
Base variant rebuilt from the feature files reproduces every stored per-session β exponent **exactly** (max |diff| 0.000
on both axes). The sweep is the paper pipeline with one threshold changed.

## Pre-registered sweep (`label_sweep.py`, `results.json`)
Mean of per-session minimum-force β exponents (window axis, 14 sessions | onset axis, 11), paired shift from base:
- base −1.532 | −1.236.
- Periodicity threshold 0.7–0.9: shifts ≤ +0.011 | +0.021 (labels changed ≤ 1.5 %) → no effect.
- Flyback fraction 0.3 / 0.4 / 0.6 / 0.7: +0.169 / +0.066 / −0.095 / −0.171 | +0.112 / +0.049 / −0.044 / −0.058
  (labels changed 2.6–9 %). Stricter single-flyback rule → flatter; looser → steeper.
- Fixed amplitude floor 1× / 2×: +1.396 / +2.237 | +1.168 / +1.867 (labels changed 12–27 %).
- **S = 2.41 (window), 1.93 (onset) → pre-declared reading "label-dependent (S ≥ 0.3)"**, entirely because of the fixed
  amplitude floor.

## Post hoc (`posthoc_amplitude.py`, `posthoc_amplitude.json`; decided after the sweep)
- Why the floor dominates: it removes 0–6 % of base-Helmholtz windows in the smallest-β tercile but 42–58 % in the
  largest-β tercile in every session (Helmholtz amplitude scales with v/β), and preferentially at low force within a
  column (corr −0.34 to −0.61). The classifier's own source comment warns that a fixed floor "would erase weak but
  regular oscillation at large beta and low speed". The floor variant therefore changes the β axis itself.
- Relative amplitude criterion (the analogue of the simulation's amplitude check: amplitude ≥ r × the median of
  base-Helmholtz windows in the same column): r 0.5 → +0.014 | +0.007; r 0.7 → +0.164 | +0.102.
- Among the variants that do not remove windows by β (periodicity, flyback, relative amplitude): range of means
  0.34 (window, −1.70 to −1.36) and 0.17 (onset, −1.30 to −1.12) — "label-dependent" on the window axis and "moderate" on
  the onset axis by the pre-declared bands. No such variant brings the mean near −2 (steepest −1.70).

## Reading in words
The measured law is moderately sensitive to how strictly "one flyback per period" is applied (about ±0.17 in the β
exponent, the same size as the force-reference-time effect), insensitive to the periodicity threshold, and very
sensitive to a fixed amplitude floor, which removes Helmholtz windows by β and is excluded by the classifier's design.
Under every β-neutral label the measured minimum force falls more gently than Schelleng's β^−2. This is far more robust
than the simulations (where sustained-motion labels moved the exponent from −2.7 to −4.9).
