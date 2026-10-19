# Prospective window-state prediction — development protocol, 2026-10-18

Written before fitting this benchmark. A1 and A2 are already exposed development
specimens, not untouched validation. B1/C1 and Part 6 outcomes are not accessed.

## Question and estimand

At 0.20 s after the supplied constant-speed plateau landmark, predict the
classifier's label for a **future 0.20 s observation window**. Primary target is
[0.25, 0.45] s; sensitivity target is [0.30, 0.50] s. This is not instantaneous
physical state at the target start, survival, a causal effect, or control success.
The target windows overlap each other; they are not independent replications.

The landmark was extracted retrospectively from the full speed trace. A1/A2
metadata show that its maximum speed equals the nominal commanded speed, but a
fully online landmark detector has not been validated. Claim only prospective
inputs relative to this externally supplied landmark.

## Inputs and coverage

- Snapshot: controls and bridge features from [0, 0.20] s, including its fixed
  observer label. History additionally contains 0.10 s windows [0, .10],
  [.05, .15], [.10, .20]. All end by the prediction origin.
- Future force, future bridge features, future duration, onset/acceleration
  metadata, trial identifiers and specimen identifiers are not model inputs.
- Finite force is kept even if zero/negative; asinh(force/0.1) avoids filtering.
- Fixed observer: periodicity .8, one flyback, reference 98.08 Hz, amplitude floor
  .4063327834010124. It is a reproducible proxy, not independent ground truth.
  The amplitude floor was historically derived using A2; it is a fixed exposed
  observer, not a nuisance estimate learned from A1 alone. A2 is development.
- Require exactly one complete row per requested window for every acquired
  stroke. Fail on duplicate/missing windows rather than select surviving trials.
- Both directions A1→A2 and A2→A1; whole specimens, hence whole strokes, stay on
  one side. Each has 6,000 strokes (2,000/speed). No target tuning/calibration.

## Fixed comparators, fitted separately for each horizon

1. Persistence with fixed probabilities .02/.98 (descriptive baseline).
2. Source-calibrated persistence: two-state transition table with Jeffreys .5
   pseudocounts; pooled and by nominal speed.
3. Controls-only HGB, snapshot HGB, history HGB.
4. Snapshot/history polynomial logistic regression (degree 2, standardized,
   C=1, max_iter=3000).
5. Separate entry/retention HGB heads, same history inputs as pooled history HGB.
   This is a standard comparator, not a proposed novel method.

HGB: 200 iterations, 15 leaves, learning rate .07, min leaf 30, L2=1, no early
stopping, seed 20261018. Fixed transforms and missingness flags, no target fit.
No hyperparameter search in this run. All failures and all results are retained.
Every fitted estimator and every prediction is saved before computing scores;
prefit and prescore manifests record source, protocol and artifact hashes.

## Evaluation and interpretation

Primary log loss; Brier, IoU and balanced accuracy secondary. Report each speed
and each previous-state group, including entry/exit counts and calibration.
Compare history vs snapshot, both vs speed-conditioned Markov, and two-head vs
pooled history. Paired 2,000-resample position-cluster bootstrap keeps the same
position across speeds together. Its interval is a conditional sensitivity
analysis on these two specimens, not population uncertainty or independent
session replication. Do not select a winning horizon after results.

Any future method selection uses only A1/A2, followed by a separately recorded
freeze before a single B1/C1 confirmation. More sophisticated fitting alone is
not evidence of novelty. A negative result is an acceptable output of this test.
