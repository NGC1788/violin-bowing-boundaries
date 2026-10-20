# Does a label-free load transform predict the window-axis minimum-force exponents? (post hoc; plan first)

Claude, 2026-10-20 (new session). Builds on novelty/lower_law and verify_lower_law (onset-axis exponents, window-axis
exponents for the same sessions).

## Idea (written before computing)
If the window force is a smooth label-free transform of the onset force,
    log F_win = α + (1 + c)·log F_on + d_v·log v + d_β·log β   (+ noise),
then a boundary F_min,on ∝ v^{b_v} β^{b_β} on the onset axis maps to the window axis with exponents
    b_v,win ≈ (1 + c)·b_v,on + d_v,   b_β,win ≈ (1 + c)·b_β,on + d_β.
c > 0 means relatively larger load loss at low force (seen in posthoc_hf Fig. e). The transform coefficients are fitted
per session WITHOUT labels, on strokes whose onset force lies in the lower-boundary range (0.05–1.5 N), t = 0.2 s window,
0.2-s window length. Prediction uses the session's own onset-axis exponents (labels) and its own label-free transform.

## Test (fixed)
Sessions with ramp metadata: A1, A2, C1, D1, A_T2, B_T2s1, C_T2, D_T2, A3, A4, D2 (window-axis D_T2 exponent from the
cached D_T2 rows). Compare predicted vs observed window-axis exponents per session:
- mean absolute error and correlation across sessions, for b_β and b_v;
- whether the predicted mean shift (window − onset) matches the observed +0.21 (β) and ≈+0.08 (v);
- whether the between-session SD of the predicted window exponents is about twice that of the onset exponents.
"Mechanism supported" if |mean predicted shift − observed| < 0.07 in β AND the per-session correlation of predicted
vs observed window β exponents is > 0.7. Otherwise "not supported". Reported whatever it shows.

## RESULTS (results.json, baseline_check.json; post hoc)
Label-free transform per session (strokes with onset 0.05–1.5 N): 1 + c = 1.12…1.63 (window force ∝ onset^(1+c)),
d_β = +0.11…+0.26, d_v = −0.01…−0.55. Ten sessions with both axes (D_T2 has no window-axis fit).
- Pre-declared rule: predicted mean β shift (window − onset) −0.33 vs observed −0.24 (miss 0.10 > 0.07) → **"not supported"**,
  although predicted vs observed window β exponents correlate r = 0.95.
- That correlation is mostly free: onset and window exponents already correlate r = 0.96. The informative check (after
  the rule): the predicted per-session SHIFT tracks the observed shift, r = 0.78 (β) and 0.70 (speed); MAE 0.114 vs 0.148
  for a "same shift for everyone" baseline (β), 0.048 vs 0.059 (speed); predicted speed shift +0.116 vs observed +0.105;
  between-session SD predicted 0.39 vs observed 0.36 (onset 0.21). A single scale factor (leave-one-out) of 0.70 would fit
  the β shift — the transform overstates it by ≈40 %.
Reading: the label-free force transform (relative load loss larger at low force) accounts for which sessions' exponents
change most between the two force coordinates and for the doubled scatter, but overstates the mean β steepening; it is a
partial, post hoc protocol explanation, not a confirmed mechanism.
