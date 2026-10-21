# Independent interpretation of coherent-threshold results

2026-10-21. Reviewer: Codex `coherent_theory`. Inspected `results.json`, `diagnostics.json`, the preceding arbitrary-edit references and the structural API verification report. This review did not recompute empirical profiles or change classifications. It addresses meaning and scope; independent implementation verification is recorded separately in `verification/`.

## Assessment

**The new results materially narrow the interpretation of the earlier small arbitrary-edit costs.** Under the fixed K=2, ε=.2 operational predicate, only one of 39 folders can be restored for each slope by any of the six declared one-parameter rules. Most unrestricted witnesses are therefore not evidence that these consistent classifier changes would restore the law comparison. The coherent successes themselves have substantial collateral changes and make original finite boundaries unresolved elsewhere.

This is stronger and more informative than showing several ad hoc threshold settings. Complete state enumeration supplies an exact attainability statement within each declared family. It is still not a certificate over all coherent classifiers, a calibrated error model, a reversal of the old MLE inference, or a demonstration that either physical law is true.

## Main results at ε=.2

All original folders were incompatible under this predicate. Each row below applies one scalar threshold to the whole 78,000-window fixture. Two extended rows enlarge a previous feature range and are not independent replications.

| Rule family | Domain | Folders individually attainable at p=−2 | Maximum simultaneously attainable at p=−2 | Individually attainable / simultaneous at p=−1.2 |
|---|---|---:|---:|---:|
| Periodicity floor | [.7,.9] | 0 | 0 | 0 / 0 |
| Second-flyback-ratio ceiling | [.3,.7] | 1 | 1 | 0 / 0 |
| Additional amplitude floor | [0,+∞], stress test | 0 | 0 | 1 / 1 |
| Additional flyback-height floor | [0,+∞], stress test | 0 | 0 | 1 / 1 |
| Periodicity floor, extended | [0,1] | 0 | 0 | 0 / 0 |
| Ratio ceiling, extended | [0,1] | 1 | 1 | 0 / 0 |

The sole −2 folder is B_T2r1/v2 (0.10 m/s). The sole −1.2 folder is B_T2r1/v1 (0.05 m/s), and both stress gates rescue that same folder. Thus these are two speed folders from the same setup, not broad independent replication. For −2 the arbitrary finite-grid relaxation was feasible in 14/39 folders; only 1/14 is attained by these rules. For −1.2 it was feasible in 39/39; only 1/39 is attained here. The 25 geometry-infeasible −2 folders were impossible before any classifier restriction, so they must not be presented as additional evidence about threshold robustness.

### Cost and collateral effects at the minimum-cost successful pattern

| Target and rule | Eligible labels changed in target folder | Any-edit lower bound | Deletion-only lower bound | Changed eligible β columns in target | All 78,000 labels changed | Other folders losing a finite eligible boundary |
|---|---:|---:|---:|---:|---:|---:|
| −2, B_T2r1/v2, ratio | 44/1,200 (3.67%) | 5 | 6 | 21/24 (87.5%) | 3,200 (4.10%) | 9 |
| −1.2, B_T2r1/v1, amplitude | 32/650 (4.92%) | 5 | 6 | 6/13 (46.15%) | 12,021 (15.41%) | 23 |
| −1.2, B_T2r1/v1, height | 26/650 (4.00%) | 5 | 6 | 6/13 (46.15%) | 11,492 (14.73%) | 22 |

The changed-column counts refer to eligible columns with at least one altered label; they do not independently count how many first boundary locations moved. All three successful changes are deletions of baseline H. The ratio optimum is below the baseline .5 ceiling and therefore also has a deletion-only comparison. The direction-matched excess costs are 38, 26 and 20, respectively. Dividing the within-target cost by the deletion-only minimum gives approximately 7.33, 5.33 and 4.33; these descriptive ratios are not statistical effect sizes. The both-direction comparison is valid as a lower bound but also includes a relaxation of edit direction.

Do not divide global cost by local arbitrary cost and label the resulting ratio the price of classifier coherence: it mixes within-folder restrictions with changes elsewhere. The target folder has 88 total changed windows for the ratio rule, 856 for amplitude and 844 for height. Thus changes outside the target folder are 3,112, 11,165 and 10,648. Within the amplitude/height target folder, many changes occur among the originally excluded columns: 824 and 818, respectively. Reporting only 32 or 26 edits would hide most of that rule's effect, even inside the selected folder.

The minimum-cost patterns make 10, 162 and 150 originally eligible columns invalid across the complete fixture, respectively. “Invalid” has a precise analysis meaning: at least one formerly finite-positive first-run boundary becomes unresolved or otherwise inadmissible. It is **not** a measured classifier error, false negative, corrupted signal or physical damage. A different rule could be physically more correct and nevertheless remove an operational boundary; there are no independent truth labels here to decide that.

### Success is an interval, not a single lucky threshold

The primary ratio rule succeeds on `[.3, .33218555281080747]`; its extended range gives `(.11403913787396773, .33218555281080747]`. The amplitude success interval is `(.4705620071075642, .4821003906504866]`, and the height interval is `(1.6047788383136012, 1.6559146572905932]`. All have positive width. They are unions of many event cells, and not all thresholds inside a success region have the same edit cost.

The minimum-cost amplitude and height witnesses are immediately above their open lower endpoints. The JSON stores full-precision valid representatives. A shortened decimal written as exactly the lower endpoint can fail to recreate the same labels; reports should use saved witnesses rather than implying a rounded number reproduces the exact minimum. The broad success regions nevertheless show that compatibility itself is not a floating-point singleton artifact.

The lower ratio threshold is a substantial change from .5, and the absolute amplitude/height gates are uncalibrated stress tests. One should not describe these successes as verified plausible corrections or small routine classifier tuning.

## The structural diagnostic explains part of the gap

The record-envelope implementation was specified before this phase's results and independently checked against direct classification: 14,560 fixtures and 94,024 threshold comparisons passed. The empirical diagnostic counts attainable finite first-run positions per originally eligible column, before asking whether all columns can share a threshold and an intercept.

| Family | Attainable finite starts / all finite-force candidate starts, median across columns |
|---|---:|
| Periodicity [.7,.9] | 2.08% |
| Ratio [.3,.7] | 6.25% |
| Amplitude stress | 20.83% |
| Height stress | 18.75% |
| Periodicity [0,1] | 6.25% |
| Ratio [0,1] | 12.50% |

For the primary periodicity range, 1,112 of 1,132 eligible columns (98.23%) have only one reachable finite first boundary. The profile also records no invalid boundaries throughout that range, so those columns retain their original operational boundary despite possible changes to other labels. This helps explain why changing many interior classifications can leave the decision unchanged.

These percentages are fractions of a finite state set, not probabilities that a classifier change is correct. They are not an attribution of the entire coherence gap: a boundary state can be individually reachable while its required threshold conflicts with another column. The shared-threshold sweep enforces that second restriction. The method should distinguish the within-column record restriction from the between-column common-parameter restriction.

## A stronger no-boundary-loss requirement is a separate post hoc diagnostic

After seeing the collateral boundary losses, the root analysis asked whether **all 1,132 original eligible boundaries across all 39 folders** could remain finite-positive while any target was restored. At ε=.2, no threshold in any of the six families does this. That is an exact additional result over the stored profiles, and the diagnostic is correctly labeled post hoc.

It must not replace the primary question silently. The primary definition required valid boundaries within a restored folder and reported losses elsewhere; the new condition additionally forbids losses everywhere. It does not establish physical robustness or the correctness of the baseline classifier. A suitable wording is: “Every attainable primary restoration in the examined families entails loss of at least one originally resolved boundary elsewhere.”

## ε sensitivity prevents a blanket robustness claim

At ε=0 and .1, no folder is attainable under any of the six rules for either slope. At ε=.4, the original labels already admit −2 in four folders and −1.2 in one. Compatibility counts at .4 include these unchanged baseline cases and must not all be called reversals.

| Family | At ε=.4: p=−2 individually attainable / maximum simultaneous | At ε=.4: p=−1.2 individually attainable / maximum simultaneous |
|---|---:|---:|
| Periodicity primary | 4 / 4 | 1 / 1 |
| Ratio primary | 6 / 5 | 4 / 4 |
| Amplitude stress | 5 / 5 | 3 / 2 |
| Height stress | 5 / 4 | 3 / 1 |
| Periodicity extended | 4 / 4 | 2 / 2 |
| Ratio extended | 6 / 5 | 4 / 4 |

This supplies a concrete common-parameter conflict. For −2 at ε=.4, six folders can each be made compatible by the ratio rule, but no one threshold makes all six compatible simultaneously. The minimum-cost best common threshold retains the original four and adds A4/v3, with only 150 global edits and no boundary loss. Thus “a coherent reversal always requires boundary losses” is false outside the primary ε=.2 condition.

Likewise, the height rule's maximum count at ε=.4 can equal the baseline count even though individual *different* folders are attainable. Counting only total maximum coverage can hide exchanges of which folders are compatible. The local success sets and folder identities in the stored results must remain part of the report. The stricter all-boundaries-valid diagnostic at ε=.4 yields five −2 compatible folders under the ratio rule; other families do not improve on their baseline counts under that restriction.

## Scientific contribution and limits

A defensible result is:

> “Sparse arbitrary relabeling and coherent classifier changes give different sensitivity assessments. At the declared primary tolerance, the force-grid relaxation admits either reference slope in many more folders than any examined shared scalar waveform rule. The few coherent restorations incur much larger within-folder costs and affect observations and finite boundaries elsewhere.”

The exact record-envelope intervals add an interpretable explanation of which counterfactual boundaries a scalar rule can reach. Event-complete profiling, independent oracle checks, fixed eligibility and all-data collateral accounting make the comparison auditable. This is useful methodological evidence about the specific first-run extraction pipeline.

The contribution remains a specialized computational audit. Threshold sweeps, multiverse analysis and restricted fragility have prior art, as detailed in `README.md`. Novelty should not be claimed for sorting threshold scores or the general idea that coherent perturbations differ from arbitrary edits. The present evidence does not establish a new minimum-force law, supervised accuracy improvement, violin measurement result, causal β effect, or external predictive gain. It also does not explain all earlier MLE exponent differences: that older inference is a different downstream target.

The strongest paper framing is therefore to expose **the gap between adversarial label sensitivity and sensitivity attainable by explicit waveform rules**, rather than to announce that the physical law has been rescued or conclusively rejected. Future claims of plausible correction would need independent waveform adjudication or validated rule calibration; selecting a threshold because it makes a favored force law fit is not such validation.
