# Adversarial interpretation of the first exact run-fragility results

Date: 2026-10-21. Reviewer: Codex `run_novelty`. Read `PLAN.md`, `results.json` summaries, primary records, eligibility records and stored witness summaries. No re-fitting, label changes, empirical reruns or shared-document edits. All arithmetic below is descriptive of the existing output. This is an interpretation review, not an independent verification of algorithmic correctness.

## Verdict

**There is a useful methods result, but the present empirical result is narrower than “a few errors overturn the physical law.”** The strongest finding is that one must separate (i) whether a slope is representable on the retained finite force grid, (ii) the exact edit distance to an operational all-column compatibility condition, and (iii) how extensively those edits reorganize the β boundaries. The implementation can make this distinction precisely. The current results do not validate either slope as a physical law and do not reverse the previous MLE inference.

The primary all-column criterion excludes both slopes in all 39 folders before any edits. Accordingly, the actual primary-data application exercises the **restoration** direction only. The opposite direction can be supported by synthetic tests and the ε=.4 sensitivity, where some baseline conditions are compatible, but should not be advertised as a primary-data discovery.

## Numbers that may be reported together

Primary condition: K=2, ε=.2 natural log-force, both-direction edits, 39 speed folders from 13 diagrams. Originally finite-positive boundary columns are fixed before editing.

| Quantity | p=−2 | p=−1.2 |
|---|---:|---:|
| Originally compatible folders | 0/39 | 0/39 |
| Folders with a finite restoration | 14/39 | 39/39 |
| Finite-grid infeasible folders | 25/39 | 0/39 |
| Restoration cost, among feasible folders | 2–20; median 13.5 | 5–163; median 26 |
| Median edited fraction of eligible windows | 0.884% | 1.68% |
| Median fraction of eligible β columns changed | **38.32%** | **48.28%** |
| Changed-column fraction range | 6.90–53.85% | 20–100% |

The two cost medians above are **not an apples-to-apples slope comparison**, because their feasible populations differ. The same 14 folders feasible for both slopes have costs:

| Folder | Speed (m/s) | Eligible β columns | Cost at −2 | Cost at −1.2 |
|---|---:|---:|---:|---:|
| A1 | .20 | 32 | 17 | 9 |
| A2 | .20 | 31 | 19 | 14 |
| B1 | .20 | 28 | 16 | 20 |
| C1 | .20 | 32 | 11 | 30 |
| D1 | .20 | 28 | 9 | 26 |
| A3 | .20 | 30 | 20 | 18 |
| A4 | .20 | 29 | 2 | 25 |
| D2 | .20 | 29 | 11 | 17 |
| A_T2 | .20 | 31 | 15 | 39 |
| B_T2s1 | .20 | 30 | 12 | 24 |
| C_T2 | .05 | 18 | 16 | 11 |
| B_T2r1 | .05 | 13 | 3 | 5 |
| B_T2r1 | .10 | 24 | 5 | 20 |
| B_T2r1 | .20 | 26 | 15 | 33 |

On that paired subset, the medians are **13.5 versus 20**, and −2 costs less in 10/14 folders; the median paired difference `cost(−1.2) − cost(−2)` is 9. This is an interesting warning that edit distance is not a ranking of physical truth or the same ranking as a fitted average slope. It is **not evidence that the classical slope is superior**: 11/14 folders are at .20 m/s, and every feasible −2 case has only 13–32 retained columns rather than the original 40. The subset is selected for favorable representability of the steeper line, and includes several correlated measurements. A sign test or confidence claim on 14 “independent replications” would be inappropriate.

## Why the small percentages can mislead

Each retained β column contributes 50 force-ranked windows, but the first K-run boundary is determined by a small subset of them. A pair of newly accepted H labels can create an early run, leaving many later labels untouched while moving the extracted boundary a long way. Conversely, destroying an established band may require many deletions. The denominator of all windows counts many positions irrelevant to the current boundary.

Thus “less than 1% of labels” is mathematically correct for the median feasible −2 case, but materially incomplete without **38% of β boundaries changed**. For −1.2, the median changes almost half the eligible columns. These are sparse label edits distributed across a large part of the measured curve, not necessarily one or two isolated mistakes.

The largest −1.2 cases are A4 at .05 m/s (163 labels in 19/21 columns), D2 at .05 m/s (109 in 32/35), C1 at .05 m/s (100 in 21/23), and A3 at .05 m/s (95 in 28/35). The full range must accompany the median. The result is heterogeneous; it does not support calling every map fragile.

For further physical interpretation, a reader would need to inspect whether the required changes are near the classifier thresholds, whether signals support them, and whether one coherent decision rule could produce them. Independent arbitrary Hamming changes are a worst-case set. The witness gives candidate points to audit; it is not evidence that these points were misclassified. Existing label-noise certification and fragility methods also require this distinction; see the primary sources in `README.md`.

## Is the compatibility predicate too strict?

It is a legitimate operational question, but it is stricter and different from estimating a power-law mean with residual scatter. It asks one line to come within a declared tolerance of **every** selected boundary. A single discordant column can make the verdict negative. Real measurement variation, genuinely irregular boundaries, and model mismatch all contribute; not all incompatibility should be attributed to classification.

The sensitivity output makes this dependence explicit:

| ε | Baseline compatible −2 | Baseline compatible −1.2 | Finite restorations −2 | Finite restorations −1.2 |
|---|---:|---:|---:|---:|
| 0 | 0 | 0 | 12 | 36 |
| .1 | 0 | 0 | 13 | 38 |
| .2 | 0 | 0 | 14 | 39 |
| .4 | 4 | 1 | 20 | 39 |

Restoration medians also change: −2 gives 41.5, 25, 13.5, 4 and −1.2 gives 52.5, 36.5, 26, 10 across these ε values. These medians condition on changing feasible subsets and therefore are not a clean paired trend estimate, though each fixed instance must have nonincreasing restoration cost as tolerance grows.

As an additional descriptive arithmetic check, the stored baseline intersections at ε=.2 imply the minimum unchanged-label tolerance via `ε_required = .2 + (max_lower − min_upper)/2`. Its median across the same 39 folders is .95063 for −2 (multiplicative factor 2.587) and .83221 for −1.2 (factor 2.298). This is a worst-column approximation error, **not** a fitted residual standard deviation or uncertainty estimate. It shows how far the all-column criterion is from a permissive mean-law fit. It does not justify tuning ε to those values after seeing them.

Recommended interpretation: “Under a fixed operational tolerance, every selected map needs at least one classification change to admit either reference slope. The exact restoration cost quantifies sensitivity of this stringent predicate.” Avoid: “Both physical laws are rejected with certainty” or “the old statistical rejection reverses after C labels.”

## Selection and perturbation restrictions matter

At K=2, 1,132/1,560 original columns are eligible; 428 (27.44%) are omitted: 315 left-censored and 113 without an original K-run. Eligibility is fixed, which prevents optimizing away inconvenient columns, but the resulting question concerns the originally resolved subset. Strong coverage differences already exist before any editing.

The restriction to finite edited boundaries usefully prevents trivial all-censored solutions. However, the 25 infeasible −2 cases mean only that no finite-supported assignment in the current grid can express that line within ε=.2. They cannot be counted as robust empirical confirmations of a different law. They may motivate better experimental force coverage, but no new experiment is being inferred from this audit.

Direction restrictions have substantial effects. At ε=.2, addition-only restoration is possible in 8/39 folders for −2 and 19/39 for −1.2; deletion-only restoration is possible in 4/39 and 37/39, respectively. These are different target spaces, so compare per-folder costs where both directions are feasible rather than comparing unpaired medians. The pattern demonstrates the asymmetry of constructing an early K-run and removing existing ones.

K sensitivity also changes the eligible population: at ε=.2, both-direction finite restorations are 19, 14, 13 for −2 at K=1,2,3, and 39,39,38 for −1.2. This is a joint change in boundary definition and complete-case membership, not an isolated test of run length.

## Publishable meaning and honest novelty

The method is a promising component of a focused computational acoustics/methods paper because it makes an otherwise informal objection testable: which part of a force-law comparison comes from finite sampling and which part from the classifier-derived boundary? Its exact witnesses, numerical validation, direction-specific costs and explicit impossible cases are stronger than simply showing several arbitrary thresholds.

The current data support **a structured sensitivity audit**. They do not yet support a universal acoustic correction, a robust learner, a new β law, or a claim that all observed exponent differences are classifier artifacts. The algorithmic ingredients are established soft-regular/shortest-path machinery, interval intersection, and fragility analysis. Novelty is the scientifically useful composition and its demonstrated consequences, not inventing those ingredients.

A defensible result sentence is:

> “For the declared first-run boundary rule, the exact edit distance separates label sensitivity from finite-grid infeasibility. Small fractions of changed windows can reorganize a large fraction of the β boundaries, and the cost ranking of two reference slopes need not agree with their descriptive fitted-slope ranking.”

The last clause should be called a conceptual possibility or linked explicitly to the earlier descriptive fits; this review did not recompute those fits. On the new output alone, report the paired cost ranking without implying a verified cross-analysis association.

The clearest figure would show, for every folder, **three quantities together**: feasibility status, edit count divided by eligible windows, and changed β-column fraction. A worked witness should overlay original and edited first-run brackets. Showing only a histogram of tiny label percentages would invite a justified reviewer objection.

The next useful extension would constrain edits through a coherent feature-based decision rule or restrict them using independently justified ambiguity evidence, then compare that restricted cost with the arbitrary-edit lower bound. That extension is not required to claim the present exact audit, and no such evidence or implementation is claimed here. It would address the most important remaining scientific gap more directly than merely adding more unrestricted sweeps.
