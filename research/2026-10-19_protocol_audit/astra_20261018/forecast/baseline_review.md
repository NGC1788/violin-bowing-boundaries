# Independent forecast design and baseline review — 2026-10-18

Status: pre-result review. I inspected A1/A2 time/control columns and feature-extraction source, not B1/C1/Part6 outcomes; I did not fit a forecast model or choose a setting using forecast target scores. Root owns benchmark implementation. This file records independent design checks and objections, not a new-method claim.

## Timing and complete support

The proposed common landmark is 0.20 s after the cached plateau onset. The past feature supports are `[0,0.20)`, `[0,0.10)`, `[0.05,0.15)`, and `[0.10,0.20)` seconds. Two separate target summaries cover `[0.25,0.45)` and `[0.30,0.50)`. All four past windows end at or before the landmark; both target windows begin afterwards, with 0.05 and 0.10 s raw-sample gaps.

Read-only Arrow loading of **folder, trial, window_s, t, plateau_s only** independently found:

| Archive | Speed | Strokes with every proposed past window | Target [0.25,0.45) | Target [0.30,0.50) | If target were extended to [0.35,0.55) |
|---|---:|---:|---:|---:|---:|
| A1 | 0.05 m/s | 2,000 | 2,000 | 2,000 | 2,000 |
| A1 | 0.10 m/s | 2,000 | 2,000 | 2,000 | 2,000 |
| A1 | 0.20 m/s | 2,000 | 2,000 | 2,000 | **657** |
| A2 | 0.05 m/s | 2,000 | 2,000 | 2,000 | 2,000 |
| A2 | 0.10 m/s | 2,000 | 2,000 | 2,000 | 2,000 |
| A2 | 0.20 m/s | 2,000 | 2,000 | 2,000 | **657** |

The fast-speed plateau range is 0.520–0.990 s in both archives. An apparently modest later target would select only 657/2,000 fast strokes, a protocol-dependent subset. Keep the two complete-support targets as specified. Each has 6,000 distinct strokes per archive. Their raw target samples overlap heavily, so they are correlated sensitivity horizons, not independent replication.

`timeres/replicate.py::windows_of` computes each feature from the half-open raw slice `sig[start:start+width]`. The primary feature/target raw slices therefore do not overlap. Overlap among past features is allowed and does not create target leakage; it creates correlated inputs. Report the target **interval** and label availability at its end, not merely a “50 ms forecast,” which would imply a point state 50 ms after the landmark.

Important qualification: plateau onset is defined retrospectively with a smoothed speed trace and 98% of its whole-stroke maximum. Thus this is a **landmark-aligned benchmark**, not yet a validated fully online plateau detector. A declared known commanded speed can eventually support an online alignment rule, but the current cache alone must not be described that way. Never add plateau duration, final force, future block samples, whole-stroke extrema, or number of available future windows as predictors.

## Fixed operational target and source isolation

- Primary outcome is the same 0.20 s waveform classifier at a fixed 98.08 Hz reference and fixed amplitude floor 0.4063327834; the target is an operational proxy, not independently measured contact truth.
- Those fixed constants have historical A2 provenance and were adopted before this forecast experiment. “Fixed, historically exposed classifier” is accurate; “learned solely from A1” is not. They do not create per-example future waveform overlap, but they prevent describing A2 as untouched confirmation.
- The 0.10 s classifier has a different periodicity bias. A later 0.10 s-target analysis must be a separate labelled sensitivity experiment, not pooled as extra outcomes for the 0.20 s endpoint.
- Source model selection, scaling, missing-value imputation, state calibration, thresholds, and any feature selection must happen inside the same five blocked-beta source folds. Freeze both directions and both target intervals before target scoring. A1/A2 remain development data.
- All windows/horizons of a stroke belong to one fold. Beta-column folds should hold out the same positions across all three speeds; record IDs and fold support. Do not partition rows independently.
- Keep zero/negative measured-force strokes. A fixed asinh force transform preserves those cases. Removing them based on the future target force would directly select outcomes using unavailable information.

## Baselines that make a gain meaningful

| Baseline | Required implementation and purpose |
|---|---|
| Hard persistence | Predict the last 0.20 s H decision. Its accuracy/IoU is useful, but raw 0/1 log loss is infinite on errors; do not silently give it an arbitrary favorable clipping constant. |
| Source-calibrated persistence / two-state Markov | Estimate `P(H_target=1 | H_past)` from source only, with a fixed prespecified shrinkage rule (e.g. Jeffreys 1/2 pseudo-counts). This is the serious probability baseline; a method must beat it, not merely hard persistence. |
| Source-calibrated state × speed Markov | Condition additionally on the three known speeds, shrink sparse cells, and keep horizon-specific probabilities. This tests whether a proposed model only learns different switching prevalence at each speed. |
| Snapshot logistic/poly2 and HGB | Same last-window controls and waveform features, including current H state where used by the candidate. These separate information gains from architecture gains. |
| History logistic/poly2 and HGB | Exact same full past information as the candidate. Every transformation and stopping/regularization choice is source-only. HGB should have a reasonable prespecified small complexity grid, not an intentionally weak fit. |
| State-separated HGB | Fit separate conditional future classifiers for H-past and non-H-past using the same available inputs, with a source-only smoothed constant fallback for a one-class branch. This is essential against an entry/loss architecture. |
| State-interaction logistic | Include current-state × every non-state feature interaction, or fit separate source-state logistic models. A pooled logistic lacking those interactions is not a matched test of a two-head predictor. |

For a binary current state, a two-head rule

`P(H_future=1 | x,H_past) = (1−H_past)·a(x) + H_past·(1−b(x))`

is ordinary state-gated classification. Calling `a` an entry hazard and `b` an exit hazard does not create a continuous-time hazard model or novelty. A finite target window can contain multiple physical transitions and is itself only a proxy summary. If source-domain regularization or physical monotonic constraints are introduced, compare against the unconstrained same-information state-separated model. Declining force and censored force bands do not by themselves justify a monotone relation in every coordinate.

Small recurrent networks are not compulsory for this bounded initial benchmark with only four summary windows. If a broad sequence-model superiority claim is made later, a suitably source-tuned compact recurrent/temporal baseline becomes relevant. This benchmark should first answer whether the available short past contains useful transferable information beyond calibrated persistence and strong tabular models.

## Evaluation and interpretation safeguards

Use prespecified probability log loss as the primary endpoint, with Brier score and fixed-threshold IoU secondary. Report both transfer directions, every speed, both target intervals, and source OOF scores. A larger IoU alone must not rescue a failed primary probability claim. Do not choose the “best baseline” from target scores; choose the principal comparator by source OOF loss, while still displaying every fixed comparator.

Report transition-conditional losses and event counts as diagnostics. Pooled accuracy can be high because most strokes keep their state. A branch with few actual transitions is especially vulnerable to noisy estimates, and calibrated state probabilities can dominate an overconfident expressive learner.

Resample complete beta columns with all their strokes, speeds and retained horizons together for paired uncertainty summaries. Such intervals are conditional sensitivity analyses on two exposed specimens, not population uncertainty across independent instruments. The two directions reuse A1/A2, and the two overlapping targets reuse raw samples. No significance claim should count these as four independent experiments.

No future measured force may be a forecasting input. An extrapolated future force computed entirely from past windows is allowed if that rule is source-specified and is labelled a prediction; directly substituting the actual target-window force is not. Attack metadata and travel interpolation must have explicit raw support ending by the origin—an interpolation using the next block can quietly extend beyond the landmark.

## Latest Claude maintenance-check logs: implications for the claim ledger

Read-only inspected `claude_extra/wf_20261018/skeptic_maintenance/common.py`, `matched_time.py`, `nulls.py`, and completed A2 logs plus descriptive/sensitivity logs. These observations are independent of forecast evaluation.

1. **The main low/entry ratio is structurally one-sided.** In `entry_low`, the “low” is the minimum force over eligible H-run endpoints whose candidate set includes that same stroke's first entry endpoint. Therefore `low_i <= entry_i` by construction. For the matched stroke set and a positive denominator, quantile monotonicity gives `q10(low) / q10(entry) <= 1`. Observing that direction, even in every column, cannot on its own establish hysteresis. Strict inequality is easily produced by decreasing force along a persistent labelled trajectory.
2. **The force-decline-only diagnostic almost reproduces the statistic.** The sensitivity log replaces the low force with the end-of-plateau force assuming H is retained: A2 slow-speed median ratio is about 0.64 versus observed 0.65, with across-column correlation about 1.00; the other reported groups also nearly coincide. This supports calling the ratio a descriptive decline/persistence summary, not an identified maintenance threshold.
3. **Time-matched summaries still have unequal opportunities.** `matched_time.py` uses one entry force, the minimum of maintained forces within a bin, and the maximum of not-yet-entered forces. These are different order statistics, with different observation counts and selection. `nulls.py`'s MT statistic pools multiple maintained windows against one entry per stroke, weighting long follow-up. Neither comparison creates matched current-control histories or a causal threshold contrast.
4. **Monte Carlo null tails are conditional adequacy diagnostics.** Marginal logistic models and persistence settings are estimated/selected using the same observed strokes; simulated tail fractions do not propagate fitted-parameter uncertainty or model-selection uncertainty. The copula can contain latent stroke heterogeneity and temporal persistence without explicit force-history hysteresis. Showing that it reproduces a statistic is a valuable counterexample to specificity, not proof that the physical process lacks hysteresis. Conversely rejecting one fitted null does not identify hysteresis.
5. **Positive controls change the generating distribution.** The hysteresis delta shifts the latent stay criterion and can change marginal prevalence and transition rates. A statistic separating those examples is not automatically a test with controlled power/type-I error for the measured problem. Printed zeros in finite Monte Carlo tail counts are not exact zero probabilities.
6. **Keep physical claims suspended.** The logs themselves include late entries below the proposed low-force quantiles, sparse exit events, and strong session dependence. These undermine treating those quantiles as universal establishment/maintenance boundaries. A history-dependent predictor, if successful, would still show predictive information, not a causal hysteresis effect or a specific bifurcation.

The useful new phase is a complete-support, nonoverlapping, same-information forecast benchmark. A real improvement there would be evidence of transferable prediction. Neither this audit nor merely adding a two-state architecture establishes high methodological novelty.

## Implementation review status

At this review's creation, the forecast implementation file had not yet been written; only `pipeline_review.md` existed in the forecast folder. The above timing counts were computed independently from existing feature metadata. Concrete code review findings will be appended after the root implementation is available, without changing its files or inspecting held-out new-specimen scores.

## Initial implementation review (before fits)

Read `benchmark.py` and `PROTOCOL.md` after root created them. No forecast fit or target-score computation was run by this reviewer. Findings were sent to root before fitting.

**Confirmed clean paths:** the four explicit past row selectors are used to construct X; the target rows are accessed separately for y. Future force/bridge features, final duration, attack metadata and IDs are absent from X. The two-state and speed-state Markov tables use source outcomes only. Snapshot and history HGB share the same stated configuration; the two-head HGB receives the same full history representation as pooled HGB. Keeping all finite forces avoids outcome-dependent contact filtering. The fixed primary times match the independent complete-support audit above.

**Changes requested before execution:**

1. **Runtime error in paired loss.** `loss_vector` negated a boolean target array with `-y`, which NumPy rejects. Independently reproduced with a two-element synthetic input: `TypeError: The numpy boolean negative ... is not supported`. Cast y to floating point before BCE. Use one clipping convention for both headline log loss and paired losses; the initial code used sklearn's machine-epsilon convention for one and 1e-15 for the other.
2. **Freeze phase did not match the intended sequencing.** The initial main loop fitted one model, immediately computed/printed target scores, and continued to the next model. Although the current predictor function does not consume target labels, this does not create the promised all-model freeze before score exposure. Separate fitting/prediction-manifest creation from evaluation and persist all fitted models/source membership or a replayable freeze artifact before any reported target metric.
3. **Acquisition coverage check was only relative to loaded records.** `window_indices` catches one missing requested window, but dropping all windows of a stroke shrinks `data['first']` and can pass as complete support. Assert the protocol's 6,000 expected acquired strokes, 2,000 per speed, and complete trial inventory (or verify against the independent acquisition manifest). This is distinct from the already verified current A1/A2 files.
4. **Empty target-state branch.** The two-head loop should skip a branch with no target rows; otherwise a two-class fitted sklearn estimator can reject an empty `predict_proba` array. The source one-class/empty-branch Jeffreys fallback is already explicit.
5. **Historical classifier provenance.** PROTOCOL should state that amplitude floor 0.4063327834010124 is inherited from the historically exposed A2 acquisition audit. A fixed observer is legitimate for this development benchmark; it must not be represented as a calibration learned solely from the training direction. The binary H decision's current implementation is in fact independent of this amplitude floor once periodicity passes the H threshold; changing that behavior would require a new target definition.
6. **Reproducibility metadata.** Record imported audit/grid/classification helper hashes and sklearn/numpy versions along with benchmark/protocol/input hashes, and save stable folder/trial IDs or their hashes. The former code only hashed the top-level benchmark and protocol. Preserve partial-run failure records and refuse overwriting an existing freeze, not merely an already completed results JSON.

The written protocol now uses fixed hyperparameters and **no hyperparameter search**. This is a valid bounded baseline experiment; the earlier recommendation to place all tuning inside blocked source folds applies if/when tuning is added. Do not invent a source-CV claim for this fixed run. The fixed .02/.98 persistence is clearly marked descriptive; source-calibrated Markov remains the meaningful probability comparator.

For future extensions, strengthen the timeline validator to require the stated minimum 0.05 s gap explicitly; the initial code only rejects `target_start < origin`, which permits an adjacent zero-gap target if constants are later changed. Adjacent half-open windows do not share samples, but they would be a different forecast protocol.
