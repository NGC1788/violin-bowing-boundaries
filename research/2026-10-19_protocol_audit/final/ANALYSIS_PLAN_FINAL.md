# Final analysis plan — frozen before scoring (Claude, 2026-10-18 20:40 KST)

Question (R&E title): measure and predict the bow-string playability boundary; compare a physics model with a data-driven model.

## Data
- Training specimens: A1 (2024-03-25_TypeA_sample1), A2 (2024-04-19_TypeA_sample2) — type A, tension T1.
- Test specimens (never used for fitting except in the explicit few-shot step): B1 (2024-04-10_TypeB_sample1), C1 (2024-04-12_TypeC_sample1), D1 (2024-04-15_TypeD_sample1), B_T2 rep_1 and rep_2 (Part 6, 2024-10-23_variability_setup). A_T2 and later archives are reserved as a fresh confirmation set (not scored here).
- Windows: 0.2 s, start t = 0.2 s after the plateau start (primary); t = 0.0 and 0.3 s as sensitivity. Labels: fixed classifier (repo regime_map.classify, A2 thresholds, reference f0 98.08 Hz) — the same for every specimen. Force: measured window mean.

## Physical measurement per specimen (no boundary information used)
- Z from the Helmholtz flyback law: Z = median(flyback_height * beta / (2 v)) over Helmholtz windows with t >= 0.1 s.
- f0: median f0_hz of Helmholtz windows.

## Targets
- Lower boundary per (speed, beta column): first K=2 Helmholtz run in current-force order -> interval (left/right censoring as in verify_astra_boundaries.py).
- Upper boundary per column: last K=2 Helmholtz run -> interval to the next force; right-censored if Helmholtz reaches the top force; columns without any run are excluded from the upper target.

## Models (all fitted on A1+A2 only)
- P-Sch: Schelleng forms with measured Z: F_min = c_L Z^2 v / beta^2, F_max = c_U Z v / beta (one constant each).
- P-Beigi: F_min = c Z v / beta.
- P-free: log F = c + a log v + b log beta + g log Z — g cannot be learned from type A alone (A1/A2 Z nearly equal), so g is fixed to the Schelleng value (2 lower, 1 upper); a and b free.
- D-logit: logistic regression P(H) on (log F, log v, log beta, their squares and pairwise products, window start t); boundaries = P = 0.5 crossings along force in each column.
- D-gbm: histogram gradient boosting on the same inputs.
- (No Z input for D models: with two type-A specimens they cannot learn a Z effect; this asymmetry is the point of the comparison and is stated.)

## Scores on each test specimen
1. Lower and upper boundary: median |log(F_pred / F_obs_mid)| over finite-interval columns, and the share of columns whose prediction falls inside the observed interval widened by one grid step; censored columns counted as hits when the prediction lies on the censored side.
2. Helmholtz-region IoU on the force x beta grid per speed.
3. Window log-loss of P(H) (D models; physics models through a fitted logistic slope on training data).
4. Few-shot: k = 5, 10, 20, 40 strokes of the test specimen chosen (a) at random (b) as a fixed coarse grid; physics models refit only their constant(s); D-logit refits its intercept and the log F coefficient; 20 random repeats; same score 1.
Primary comparison: score 1 (lower + upper) zero-shot and at k = 10. No further model changes after the first scoring run; any later change is reported as post hoc.
