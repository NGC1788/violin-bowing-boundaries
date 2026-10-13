# Independent implementation and evidence review

2026-10-13. Read-only review of `scripts/set_envelope.py`, `scripts/benchmark_sets.py`, and the relevant historical classifier. No production code was changed for this review.

## Gradient and observation-family checks

The analytic `family_iou` gradient matches the negative normalized soft-minimum objective: each operator's negative IoU derivative is weighted by its softmax adversary weight. The leading sign is correct, as are the derivative of the union denominator and the logistic chain rule.

An independent `scipy.optimize.check_grad` check captured the actual objective closure by temporarily wrapping `minimize`, without editing the implementation. With seed 2718, 24 rows, three label maps and ten ambiguous rows, the BCE, nominal IoU, coherent family IoU and rectangular envelope objectives were checked at zero and random coefficients. At finite-difference step `1e-6`, maximum absolute L2 gradient discrepancy was `2.296e-7`; all four small fits converged. This is a local numerical derivative check, not evidence that the research model generalizes.

The saved ratio of the second-largest flyback to the largest supports reapplying the relative-amplitude threshold for the **binary Helmholtz question**, with fixed folding and hysteresis. It does not recover the number of weaker flybacks or permit changing the original signal preprocessing. The code's statement that it does not recover raw signals is appropriate. The benchmark asserts that the central family member exactly reproduces the fixed nominal binary labels; that assertion is essential.

The envelope-overlap component alone has zero derivative on ambiguous points, but the implementation adds `0.1 * nominal_BCE`. Therefore the complete `robust_iou` objective does receive ambiguous-point gradients, **exclusively from the nominal anchor**. It must be described as envelope-overlap plus a nominal-label anchor, not as a pure lower-bound optimizer. The same nominal anchor appears in the coherent method, which makes the loss ablation fairer but introduces a design choice that needs documenting.

The objective uses a normalized exponential sum. Its negative is a smooth minimum that can exceed the true family minimum; evaluation correctly computes the exact finite-family IoUs separately. Coherent pooled scoring and macro-averaged per-panel minima are both output and have different meanings. The main manuscript should not collapse them into a single unnamed “worst-case” number.

Guard improvements that do not change valid-data mathematics: explicitly reject non-finite lower/upper arrays, an empty operator family, non-finite/non-positive invalid weights in `family_ious`, and malformed classifier parameter ranges. Optimization failures must be counted and inspected before interpreting a winning model; retaining a status field is necessary but does not itself resolve a failed fit. These are defensive checks, not presently demonstrated corruption of the inspected benchmark inputs.

## If the coherent method only matches a conventional learner

The correct conclusion is that **physical covariance is sufficient at this scale, and the additional coherent-family training did not establish incremental predictive value**. One may still find that coherent-family evaluation is much less pessimistic than a pointwise envelope; that is a property of the uncertainty representation, not proof that the new learner is better.

A publishable domain study remains possible if it contributes a reproducible public benchmark, exposes previously unrecognized numerical or validation defects, separates observation-rule sensitivity from physical error, and demonstrates out-of-sample behavior on an untouched specimen. That is a different paper from a new general-purpose learning algorithm, and acceptance cannot be inferred in advance. Do not search across alternative metrics, force ranges or classifier thresholds until a desired winner appears.

The strongest negative-result formulation would be: “Apparent gains from robust regime-set learning are explained primarily by a physically motivated coordinate prior; maintaining globally coherent observation uncertainty improves evaluation fidelity but does not necessarily improve prediction.” This is contingent on actual completed results, not a finding asserted by this review.

If the coherent method improves only its own optimized score while worsening nominal or other-family scores, report the tradeoff and inspect whether source selection or the nominal anchor explains it. If it improves every declared measure on the fresh specimen, that supports a useful method; it still does not prove correctness of all physical labels.

## Compact official data search

The official Zenodo JSON metadata were read on 2026-10-13 for all eight parts of the steady bowed-response collection, plus the two linked earlier experimental datasets. Only metadata were downloaded in this check; no archive payload or author contact was used.

| Official record | Listed released files |
|---|---|
| [Part 1, 17749111](https://zenodo.org/records/17749111) | Three `.7z` files, approximately 14–15 GB each. |
| [Part 2, 17782542](https://zenodo.org/records/17782542) | Three `.7z` files. `2024-04-19_TypeA_sample2.7z` is exactly **14,775,691,753 bytes**; the other files are D sample 1 and A sample 3. |
| [Part 3, 17782552](https://zenodo.org/records/17782552) | Three `.7z` files. The smallest is the A sample 1 cold/warm archive, 8,768,070,823 bytes. |
| [Part 4, 17782565](https://zenodo.org/records/17782565) | Three T2 `.7z` files, approximately 14–15 GB each. |
| [Part 5, 17782538](https://zenodo.org/records/17782538) | Two T2 `.7z` files, approximately 13.6 and 14.6 GB. |
| [Part 6, 17795326](https://zenodo.org/records/17795326) | One setup-variability `.7z`, 28,218,620,427 bytes. |
| [Part 7, 17822016](https://zenodo.org/records/17822016) | One high-force `.7z`, 29,207,549,757 bytes. |
| [Part 8, 17822037](https://zenodo.org/records/17822037) | One high-force `.7z`, 27,081,200,760 bytes. |
| [Transient dataset, 10946413](https://zenodo.org/records/10946413) | One `zenodo.zip`, 21,950,506,374 bytes. Its task is transient attack, not the current steady map. |
| [Static/dynamic bridge-force dataset, 10619541](https://zenodo.org/records/10619541) | One `Database-Examination_of_bridge_force.zip`, 1,310,765,535 bytes. This is a separate experiment, not a compressed substitute for A sample 2. |

No standalone compact author labels, processed feature cache, or small sample-2 supplement was listed in these official record file inventories. The [IWK project page](https://www.mdw.ac.at/iwk/the-bowed-string/) and [funded-project output list](https://www.fwf.ac.at/en/research-radar/10.55776/P34852) point to these records and the separate MATLAB physical-model code. A targeted search of the main paper title and author/Schelleng combinations did not locate a verified alternative compact release. This is **not a proof that none exists**, and the archive payloads were not exhaustively inspected in this search.

The current A sample 2 download remains the strongest already chosen prospective confirmation. Switching to a smaller different dataset would change the evaluation question. If network time becomes the bottleneck, preserve resumable transfer and the frozen protocol; do not substitute digitized published figures or an already inspected backup while retaining the “fresh sample” claim.
