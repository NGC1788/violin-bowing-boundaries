# FROZEN_v3 pooled hypotheses — dedicated once-only scorer (frozen before B_T2 sample 2 is processed)

Claude, 2026-10-19 ~22:35 KST. Governing file: `novelty/thermal_upper/FROZEN_v3.json` (created 10:39:15, sha256 ac30d117…;
byte-identical in GitHub commit d6b4081, 12:02). Its `hypotheses` H_curv_empirical, H_curv_thermal and H_int refer to the
upper boundary pooled over the new targets A4 (2024-04-24_TypeA_sample4), D2 (2024-04-26_TypeD_sample2) and B_T2 sample 2
(2024-07-03_TypeB_T2_sample2). No scorer existed for them. This file fixes the scorer before B_T2 sample 2 is processed
(download in progress at writing time). A4 and D2 were scored for other frozen hypotheses (thermal_upper score_target,
method_protocol); their pooled curvature / interaction statistics below have not been computed by anyone. Additions to
the v3 text are marked [added].

## Data
Upper-boundary columns at t = 0.2 s, K = 2, measured window force, exactly as `novelty/thermal_upper/rows.py`
(`rows_for("TGT_<stem>")`, i.e. final/data.py + final/compare.intervals; no-band columns excluded), using the caches
written by score_target.py. Published Z from FROZEN_v2 targets' groups (A T1, D T1, B T2) — only a fixed offset; it does not
affect curvature or interaction because every target has its own offset.

## Statistics (= analysis2b.py forms)
- H_curv: X1 = [three speed-category indicators, log β, target indicators (first target dropped)]; interval-censored
  Gaussian fit (L-BFGS-B → Powell → L-BFGS-B from an OLS start). e12 = (p1 − p0)/(lv2 − lv1), e23 = (p2 − p1)/(lv3 − lv2),
  curvature = e23 − e12. [added] Speed categories are the three nominal speeds; the scorer asserts exactly three distinct
  rounded log speeds, otherwise it assigns each row to the nearest of log 0.05/0.10/0.20 and records that it did so.
- H_int: X2 = [log v, log β, log v·log β, (log β)², 1, target indicators]; g = coefficient of log v·log β (q, the (log β)²
  coefficient, reported descriptively).
- Uncertainty: cluster bootstrap over targets (NB 200, seed 5; analysis2b) = primary [added for 3 targets: this is coarse
  and is reported as such]; stratified column bootstrap within targets (NB 200, seed 6) = secondary.
- Decisions (v3 text): sign of the point estimate and whether the primary 95 % CI excludes 0.
  H_curv_empirical supported ⇔ curvature < 0 and CI upper < 0; H_curv_thermal supported ⇔ curvature > 0 and CI lower > 0;
  otherwise "not decided" with the sign stated. H_int: empirical supported ⇔ g > 0 and CI lower > 0; thermal supported ⇔
  g < 0 and CI upper < 0; otherwise "not decided".
- Descriptive: per-target curvature and g (single-target fits) against the frozen single-diagram 90 % PI (−0.69, +0.06)
  and the frozen model curvatures in FROZEN_v3 targets.

## Availability rule [added]
Run once after B_T2 sample 2 is `done` in timeres/replicate_state.json and its regular frozen scoring has written
`novelty/thermal_upper/score_2024-07-03_TypeB_T2_sample2__FROZEN_v2.json` (so its cache exists; no concurrent cache
writes). If B_T2 sample 2 ends in `error`, run on A4 + D2 only and label every decision "incomplete (2 of 3 targets)".
Exclusive claim (O_EXCL) before reading any boundary; result written with O_EXCL; no second run.
