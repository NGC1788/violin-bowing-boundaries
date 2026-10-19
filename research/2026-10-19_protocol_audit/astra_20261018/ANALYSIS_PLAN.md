# 2026-10-18: protocol-history diagnostic, specified before this run

Owner: Astra. The 2026-10-17 A2 matched-time exponents are descriptive associations.
They do not isolate settling: measured bow force decreases with travel, and ramp
acceleration depends on the design. A1 and A2 have already been inspected by both
collaborators. Neither is a fresh confirmation set. No stationary-law, hysteresis,
critical-slowing, or bifurcation-type claim will be inferred from this exercise.

## Fixed comparison

- Join each stroke's original metadata, ramp summary, and 50 ms block data by
  full folder/trial; reject duplicates, missing keys and inconsistent beta.
- Reuse 0.2 s feature windows at plateau-relative starts 0, 0.1, 0.2, 0.3 s.
  All three speeds must contribute all four windows for every included stroke.
  Keep negative/near-zero force readings via an asinh transform; do not discard
  difficult contact-loss cases using a positive-force filter.
- Fixed waveform surrogate: periodicity >=0.8, one flyback, no subharmonic
  relative to 98.08 Hz, same classifier semantics. These are operational labels,
  not independently adjudicated physical ground truth.
- Baseline inputs: measured current force, beta, measured speed.
- Clock baseline additionally sees elapsed plateau time.
- History model additionally sees measured onset force, reported acceleration,
  and measured travel to the window center (interpolated from 50 ms blocks;
  all contributing samples are available by the decision window's end).
- Compare history ablations removing one of those three extra variables.
- Fixed learners: degree-two logistic regression (C=1) and histogram gradient
  boosting (200 iterations,15 leaves,learning_rate=.07,min_leaf=30,L2=1).
  No hyperparameter selection from target specimen performance.
- Fit A1→A2 and A2→A1. Each stroke's four windows stay in the same partition.
  Primary metric is mean log loss; also Brier, IoU and balanced accuracy, broken
  down by speed and observation time. Compare against the clock baseline.
- Position-cluster paired bootstrap (40 sorted beta columns, all speeds and
  windows together), 2000 draws. These are conditional stability intervals,
  not population uncertainty across specimens. Report both transfer directions.
- Inspect acceleration/travel/onset-force support and deterministic design
  associations before attempting causal coefficients. Predictive improvement
  cannot establish hysteresis or isolate a force-history effect.

B1 and subsequent specimens remain unscored by this script. Any final model,
feature/calibration policy and aggregate endpoint must be frozen before they are
used for confirmation. This diagnostic itself is not claimed as a novel method.
