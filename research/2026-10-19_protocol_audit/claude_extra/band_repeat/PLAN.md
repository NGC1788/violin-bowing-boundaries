# Limited validation of ONE method candidate: probabilistic band extraction (frozen before running)

Claude, 2026-10-19 ~21:55 KST. Candidate = Astra's integrated single-band model (`astra_20261019/observation_model/
band_probe.posterior(y, integrated=True)`, priors Beta(18,2) inside / Beta(1,49) outside, uniform over the 1,326
band pairs) used as a boundary EXTRACTOR, unchanged. Comparators with the same information (one column's 50
force-sorted binary labels): K = 2 run rule (this project's convention) and K = 3 run rule (the dataset paper's).
Change-point estimation is prior art (Fearnhead 2006); nothing here claims a new estimator.

## Why this test
There is no ground-truth boundary in the measured data. The only label-free external check available without new
data is test–retest agreement between the two same-day repeats of one set-up, B_T2r1 and B_T2r2 (Part 6 rep_1 /
rep_2; different strokes, same string and day). Both are development data already used in many analyses — this is
NOT an unseen test and cannot support a general claim.

## Fixed procedure
- t = 0.2 s windows, measured window force, columns matched by (speed, β column index), 120 pairs.
- Boundary location in log force: for an index boundary b between sorted strokes b−1 and b, the midpoint
  ½(log f[b−1] + log f[b]); b = 0 or b = n → censored, excluded from the agreement metric.
  K-rules: lower b = first run start, upper b = last run end + 1. Band model: posterior medians of i and j.
- Primary metric, per boundary (lower, upper): median |log F_r1 − log F_r2| over the columns where ALL THREE
  methods give an uncensored boundary in BOTH repeats (common set; N reported). Paired bootstrap over β-column
  positions (paired across speeds), 2,000 resamples, seed 20261019, for the difference BAND − K2 in that median.
- Secondary (reported, not decisive): each method on its own finite set; median signed BAND − K2 offset within
  each repeat (bias check: a stable but shifted extractor is not "better"); K3 vs K2.
- Decision rule: "consistent improvement" only if BAND's median |Δ| is lower than K2's for BOTH boundaries AND the
  95 % CI of BAND − K2 excludes 0 (favouring BAND) for at least one boundary. Otherwise the method direction ends
  here ("no consistent improvement"), and nothing about the band model goes into the paper's results.
- One run. No other priors, K values, windows or diagrams are tried afterwards.
