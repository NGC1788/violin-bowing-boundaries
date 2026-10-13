# 2026-10-13 executable development protocol

This document is recorded before the new tabular benchmark is run. It is a local protocol, not a registered preregistration. Existing A/B/C/D sample1 data and past outcomes have already been inspected; all their new comparisons are **retrospective development**.

## Question and candidate

Can retaining the shared, global nature of uncertain signal-classification thresholds improve transfer of an entire bowed-string response set, compared with a fixed rule or independent per-cell uncertainty? Inputs are only beta, measured normal force, measured speed. Outputs are provisional bridge-signal rule memberships, not independently verified physical Helmholtz truth.

Candidate: a topology-free RBF classifier in fixed physical similarity coordinates `(log(beta/.1), log(F/1N) - log(v/.1m/s))`. Its training objective is a smooth minimum of soft intersection-over-union over 15 globally coherent observation maps (flyback fractions .3,.4,.5,.6,.7; periodicity .75,.8,.85; subharmonic factor .75). One threshold pair applies to the whole map. Classifier pitch reference is estimated from the original A/.1 training panel only. Physical force-speed covariance is a declared prior, not learned from a single-speed sample. This is a **method hypothesis**, not an established novelty claim. Dimensionless learning, adversarial label learning, DRO and IoU losses are prior techniques.

The rectangular envelope of these maps allows impossible independent per-cell threshold choices. Its hard-IoU extrema are exact sensitivity bounds for that larger class, not truth coverage or confidence intervals. A loss built from its continuous lower bound has zero derivative on ambiguous cells; it must be compared against masked-label learning rather than advertised as resolving those cells.

## Comparisons fixed before results

- Constant predictions, raw-coordinate logistic regression, RBF-SVM, extra trees and histogram gradient boosting.
- Covariant-coordinate logistic regression, RBF-SVM and extra trees.
- Same RBF basis: nominal BCE, family-mean BCE, consensus-only BCE, nominal soft IoU, rectangular-envelope lower-bound objective, coherent-family minimax IoU.
- No-covariance ablation for the candidate; force-speed exponent sensitivity (0,.5,1,1.5) as diagnostics, never chosen on target performance.
- Existing calibrated simulator predictions are historical **target-adapted and numerically defective diagnostics**, excluded from fair main method ranking. Their prior numbers are not erased.

## Splits, model selection and metrics

Primary source is A at .1 m/s; remaining eleven specimen/speed panels are retrospective transfer evaluations. Use all valid positive-force plateau rows and retain ambiguous rows in the family evaluation. No waveform-derived feature enters the predictor. Five blocked beta-column folds within the training panel select regularization/length scale; no target-panel label enters fitting or selection. Retain both nominal-IoU-selected and coherent-family-minimum-selected baselines when relevant.

Report each panel, specimen averages, macro panel scores and worst panel. Primary diagnostic is minimum IoU across the coherent family per panel; also report default-rule IoU, exact rectangular bounds, family mean/max, and cell counts. A separate global-family minimum uses the **same** rule across all test panels. Per-panel minima allow different nuisance settings and are labeled accordingly. Report probabilistic Brier error only against nominal weak labels. Do not call it physical calibration.

Every model fit, hyperparameter, split identity, training basis and source-file checksum must be saved. Repeat bootstrap over whole beta columns only for conditional spatial variability, if used; it does not create extra independent string specimens. Four historical specimens are insufficient for strong population-generalization claims. Results remain reportable if the new method loses.

## Prospective lockbox

Public archive `2024-04-19_TypeA_sample2.7z`, Zenodo17782542, MD5 ec520fecf6c5011ef8491b335edc1cee, is being downloaded. Its waveform values and labels have not been inspected in this work. Do not open its label outcomes until final algorithm/hyperparameters, fixed observation family, eligibility and evaluation script hashes are frozen in a manifest. A repeat of type A is an unseen specimen/setup test, not unseen instrument-class generalization. If this archive was used in earlier unrecorded work, the lockbox claim must be withdrawn.

## Physical and observational validity

Correct the waveguide reflection recurrence and require passivity before new physical sweeps. Preserve old numerical artifacts as historical. Test synthetic counterexamples (sine, triangle, smoothed sawtooth, multiple flybacks, noise) to establish what the observable label can and cannot identify. Do not equate a single bridge-force flyback with directly measured bow-contact stick/slip.

No own recordings, sensors, purchases or server access are part of this phase. All compute uses the existing Mac with bounded memory/parallelism. Novelty and publication suitability remain open scientific questions.

## Separate additive protocol — 2026-10-13 09:02:25 UTC

After the development comparisons and matched width ablation were completed, but while A2 still existed only as an incomplete download with no decoded outputs, a **separate** candidate evaluation was frozen in `transport_lockbox_spec.json`. Its canonical SHA256 is `787d7839871519816f39d710ae1947dd17548c160b9dee5a3b64635a13186377`. It does not change the original fourteen estimators, original primary comparison, observation family, preprocessing or evaluation specification above.

The addendum embeds twelve models: source A at each of the three two-speed pairs, with full affine flow, matched k=0, learned uniform exponent and raw 3D quadratic. The primary pair is (.05,.10) m/s. Primary endpoint is the equal mean over the three A2 speed panels of each panel's minimum IoU over the existing fifteen observer rules. Primary differences are flow minus k=0 and flow minus the source-CV-selected simple model (learned exponent for this source pair). The other pairs and every model/panel remain secondary and must all be reported. This does not rank the two-speed models against the original one-speed models as if they had equal information.

The new evaluator waits for successful original A2 completion and checks the original models/code/specifications, preprocessing provenance, output CSV hash and a whole-archive verification timestamp **after** this addendum freeze. It performs no fitting or adaptation. The freeze includes source IDs, original selection records, embedded coefficients, implementation hashes and the observed undecoded filesystem state. This local chronological evidence is not public preregistration or proof of all prior human/external exposure history. If earlier target exposure is discovered, the prospective interpretation must be withdrawn.

No A2 outcome is present when this addendum is recorded. Completion of the original test will provide evidence about the original fourteen models; completion of the separate addendum can then provide new-specimen evidence about this candidate. Neither is validation on an assembled violin.
