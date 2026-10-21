# Threshold-tilt principle — generalisation tests (pre-registered)

Claude, 2026-10-21 (before any code for these tests; the commit time is the record). Same data, base label, strength
rule (15 % of base-Helmholtz windows removed per speed folder; 30 % reported), and fitting as `PLAN.md`.

## G1. A third feature, as an upper bound: second_flyback_ratio (the classifier's own flyback criterion)
- Criterion: second_flyback_ratio ≤ K · β^a, a ∈ {−1.0, −0.5, 0, 0.5, 1.0}; the top 15 % of log(sfr) − a log β among
  base-Helmholtz windows with sfr > 0 are removed (windows with sfr = 0 are never removed).
- b_sfr = within-folder slope of log sfr on log β among base-Helmholtz windows with sfr > 0, computed first.
- Predictions (upper bound): minimum-force Δ(a) **decreases** with a (Spearman ρ ≤ −0.9 on the mean; per session
  ρ ≤ −0.7 in ≥ 11 of 14); zero crossing within ±0.25 of mean b_sfr. Reading "generalises" if both hold; "direction
  only" if only the first; "does not generalise" otherwise.
- Consistency check (not confirmatory; the outcome was seen): the earlier flyback-fraction sweep (f 0.3 → 0.7 moved the
  exponent +0.17 → −0.17) is explained by the principle iff b_sfr > 0 (a constant threshold, a = 0, on a feature that
  grows with β removes large-β windows when tightened and adds them when loosened).

## G2. The maximum-force boundary, same amplitude criterion (lower bound, a ∈ {−1.5 … 0} as PLAN.md)
- Upper boundary per session: interval-censored fit log F_max = c + d_v log v + d_β log β of the `upper` intervals of
  `compare.intervals` (K = 2), window-force axis (14 sessions); onset axis reported.
- Prediction: removing Helmholtz windows at small β lowers the small-β maximum force, i.e. **flattens** the (negative)
  F_max law — the opposite sign to the minimum force. So Δ_max(a) **decreases** with a (ρ ≤ −0.9 on the mean; per
  session ρ ≤ −0.7 in ≥ 11 of 14), with the zero crossing within ±0.25 of the same amplitude b (−0.92 mean).
- Reading "generalises to the maximum force" if both hold; "direction only"; "does not generalise".
