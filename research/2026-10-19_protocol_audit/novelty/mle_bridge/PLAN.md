# MLE bridge protocol — 2026-10-21

This is post-hoc method development on already exposed cello data. The following comparisons are fixed before scoring certificate deletions. They do not reopen frozen tests or provide new confirmation.

## Question and estimand

Does an exact minimum fixed obstruction for all-column, finite-interval compatibility also identify observations that determine the original interval-censored Gaussian MLE? The original model pools all three speeds in each diagram: log F = c + a log v + b log beta + Normal(0,sigma²). Keep all 120 columns, including left-censored and no-run/right-censored columns; use the measured-window force, t=.2, K=2 extraction and 1 mN clamp unchanged. The earlier compatibility epsilon is NOT an MLE parameter.

## Inputs and frozen selection

Use the existing joint_rule/input_joint.json.gz (13 diagrams, 1,560 columns), exact chosen pairs and triples in joint_rule/theory/obstruction_certificates.json, prior arbitrary feasibility, and saved stable baseline MLEs from criterion_identifiability/response/results.json. Analyze all 77 incompatible folder–slope cases, separately report 52 arbitrary-edit-feasible classifier-specific cases and 25 geometry-infeasible cases. Never choose a certificate based on its MLE influence. The two target beta slopes remain -2 and -1.2.

## Primary experiment

1. Reconstruct all original intervals and reproduce all 13 saved stable full-data fits. Record censor counts, hashes, optimizer status, gradient, rank, sigma bounds and likelihood.
2. For each previously chosen fixed certificate, drop only its two or three exact column IDs from the original 120-row fit and refit the same model. Report signed/absolute change in beta and speed slopes, sigma, and distance of beta slope to target p.
3. Fit a fixed-beta-p version with nuisance intercept, speed slope and sigma profiled. Report descriptive T=2(NLL_fixed-NLL_free), before/after deletion. T is NOT a calibrated hypothesis test, p-value or replacement for the earlier bootstrap confidence interval. Comparisons use likelihoods on the SAME remaining observations.
4. Enumerate ALL same-cardinality deletions drawn from the same speed folder's baseline eligible columns. Compare the fixed certificate's absolute beta change, movement toward target and reduction of T with that finite control population; ties and percentile definition are explicit. Also report controls with log-beta span within ±20% of the certificate span, or unavailable if none. These ranks are descriptive, not randomization-test p-values. Cache repeated fits. No adaptive larger search to obtain a desired result.
5. Mechanism: decompose the change in T into removed observations' full-fit log-likelihood contrast and differential nuisance refitting gain. Numerically verify the identity. A geometric conflict need not be a large likelihood contributor.

## Secondary checks

- Delete the same beta-index blocks across all three speeds (6 or 9 rows), rather than only the selected folder, and compare effect signs/magnitudes. These are separate interventions, not the same certificate or independent replications.
- Apply exactly two previously saved coherent threshold witnesses to ALL 78,000 windows: (u=.8,v=.33218555281080747), which restores the p=-2 hard rule at epsilon=.2 with boundary loss elsewhere, and (u=.8,v=.49326392706532574), the no-loss A4 near-cutoff witness. Rebuild all 120 MLE intervals, preserving one-sided censoring; record label/censoring changes and all 13 refits. No threshold optimization against MLE outcomes. This directly checks whether the useful hard-rule examples also change the original MLE.
- Synthetic counterexample with geometrically conflicting finite intervals but free-MLE beta exactly at the fixed target; rank/scale degeneracy checks show why fitting the 2–3 selected rows alone is invalid.

## Numerical integrity and interpretation

Use log-CDF interval mass, standardized design, analytic gradients and bounded log sigma [-9,3.5] for ALL optimization stages (including any fallback). Require full rank, finite objective, successful fit, gradient tolerance 2e-3 and no sigma bound hit. Check fixed-model NLL >= free-model NLL within 1e-7. Retry/flag numerical failures explicitly; never silently drop failures or replace intervals. Compare against existing stable reference and an independently implemented likelihood/refit oracle. Record every failed control and do not call incomplete enumeration exact.

Enumerations may be sequential or use at most two workers to respect the laptop. No raw data download, pipeline restart, external write or GitHub push. Existing source outputs remain immutable. New code/results live under novelty/mle_bridge. Any deviation is timestamped and justified before the dependent calculation.

## Novelty boundary

Minimum unsatisfiable subsets, likelihood-ratio diagnostics, case-deletion influence, AMIP, and multiverse analysis already exist. Potential contribution is a rigorously tested bridge (or demonstrated lack of implication) between waveform-rule obstruction certificates and censored physical-law inference, with exact finite control enumeration and reproducible censoring-aware likelihood attribution. No universal new theorem, causal physics result, classifier-accuracy claim or violin measurement claim is assumed.
