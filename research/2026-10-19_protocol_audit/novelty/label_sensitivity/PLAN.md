# Label-threshold sensitivity of the measured minimum-force law (pre-registered)

Claude, 2026-10-21 (before any code; the commit time is the record). Post hoc w.r.t. the boundaries (seen); no
variant below has been computed before.

## Why
The simulations (`novelty/sim_minforce/v8`) showed that, with realistic damping, the minimum force and its β exponent
depend strongly on how "Helmholtz motion" is defined (β exponent −2.7 vs −4.9 for two sustained-motion labels). Is the
measured law equally label-dependent? The measured label is `regime_map.classify` on per-window features
(timeres/features): Helmholtz = periodicity ≥ 0.8, not subharmonic (f0 ≥ 0.75 × 98.08 Hz), and exactly one flyback per
period (second-tallest flyback < 0.5 × tallest); the amplitude floor does not enter once periodicity ≥ 0.8.

## Design (fixed)
- Sessions: A1, A2, B1, C1, D1, A3, A4, D2, A_T2, B_T2s1, C_T2, D_T2, B_T2r1, B_T2r2 (window-force axis, 14);
  the 11 with ramp metadata also on the onset-force axis (as `novelty/lower_law/onset_axis.py`). B_T2 sample 2 is left
  out (different 4-speed design and parser).
- Rebuild each session's t = 0.2 s labels from the feature files exactly as `final/data.py` (same columns, same
  reference f0), then the same boundary extraction (`compare.intervals`, K = 2) and per-session interval-censored fit
  log F_min = a + b_v log v + b_β log β (`lower_law.fit`).
- **Gate**: the base variant must reproduce the stored per-session β exponents (window axis: lower_law/results.json;
  onset axis: onset_axis.json) within 0.02 for every session; otherwise the sweep is reported only relative to its own
  base and labelled "not identical to the paper pipeline".
- Variants, one threshold at a time (11 in total, all reported):
  base (p 0.8, f 0.5, no amplitude requirement); periodicity p ∈ {0.7, 0.75, 0.85, 0.9}; flyback fraction
  f ∈ {0.3, 0.4, 0.6, 0.7} (Helmholtz iff ≥ 1 flyback and second-tallest < f × tallest); amplitude required
  (amplitude_std ≥ 1 × or 2 × the applied floor 0.406, in addition to periodicity).

## Readings (fixed)
- Per variant: mean of the per-session β exponents (and SD), and the paired mean shift from base with a 95 % t-interval.
- S = max − min of the variant means, per axis. **S < 0.1 → "measured law robust to label thresholds"; 0.1 ≤ S < 0.3 →
  "moderate (comparable to the force-reference-time effect, 0.21–0.31)"; S ≥ 0.3 → "label-dependent, like the
  simulation"**.
- Also reported: share of t = 0.2 s labels changed by each variant; number of sessions whose exponent moves toward −2.
