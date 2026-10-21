# Joint waveform-rule sensitivity: plan before new empirical results

2026-10-21. Post hoc development on already exposed public cello data. This plan fixes the new two-threshold analysis before computing its costs or law compatibility. It is not independent confirmation or public preregistration. Existing raw data, labels, frozen scores, pipelines and GitHub remote remain unchanged.

## Scientific question and rule

Does the apparent restriction found when moving a single classifier threshold survive simultaneous changes to the two existing waveform gates? Use ONE pair (u,v) for every one of the 78,000 windows:

    H = fixed OR (physical_guard AND periodicity >= u AND second_flyback_ratio < v).

The physical guard preserves all other original classifier conditions, including periodicity >= the original max_aperiodic floor. Export it from the existing cached measurements; do not infer it by combining the two old single-feature guards, which lose jointly recoverable observations. A nonfinite value in either tested feature locks the original label. Baseline is (.8,.5); the CLOSED domain is u in [.7,.9], v in [.3,.7]. No beta, force, speed, session ID or folder-specific normalization enters the classifier. All hypothetical changes are separate outputs.

Use the same originally finite-positive K=2 first-run columns (1,132/1,560) and all-column law compatibility decision as the previous two studies. Slopes -2 and -1.2, epsilon .2 primary; sensitivity 0,.1,.4. Folder intercepts are free and different; classifier thresholds are shared. A target folder with any invalid original eligible boundary cannot be compatible. Excluded columns still count toward all-data label changes. This does not reverse the original MLE test or establish true-motion labels.

## Exact method and deliverables

1. Enumerate every induced two-dimensional threshold cell, respecting ties and strict/weak endpoints. For each K-run, activation requires u <= its minimum periodicity and v > its maximum ratio (with fixed and rejected observations treated explicitly). The first-run region excludes all prior activation regions. Per-folder grids may omit irrelevant events; assemble the global arrangement by exact index mapping. This is a finite exhaustive method, not coarse grid search.
2. Retain all local cells, costs, invalid counts, shifted lower/upper intercept endpoints, success sets and identities. Success cells provide actual thresholds and an intercept witness. Failure cells provide one invalid column, or the two columns attaining the conflicting lower and upper intercept limits. These are locally checkable obstruction certificates; the full cell arrangement certifies domain coverage. Treat this as a composition of established threshold/interval geometry, not a new general range-searching algorithm.
3. Global primary choice: maximize compatible folders, then minimize all-window edits, then minimize squared distance from baseline normalized by the two domain widths, then increasing u-cell/v-cell index. Also report exact per-folder oracle minima separately. Never combine different per-folder thresholds as a global rule.
4. Also compute the same choice while preserving all 1,132 originally eligible boundaries everywhere. This condition is declared before joint results (it was added after results in the scalar study). Report budget frontiers at 1%,5%,10%,15%,30%,100% all-window edits for both policies. Preserve failed cases and all original/baseline compatible identities.
5. Compare the full family with its two baseline-coordinate slices and with previous unrestricted arbitrary label-edit lower bounds. A genuine interaction is a condition reachable by a joint pair but by neither one-dimensional slice; record it even if global boundary preservation fails. Whole-data versus within-target costs remain distinct.
6. Reduce dependence on a chosen epsilon: for each folder find the minimum nonnegative tolerance required by any valid joint rule, epsilon_required=max(0,(max lower - min upper)/2), also under global boundary preservation. These are per-folder oracle minima, not one pair simultaneously optimal for all folders. The absolute force uncertainty is not calibrated by these values.

## Validation and interpretation

Baseline labels must match all 78,000 prior labels and both baseline slices must reproduce prior scalar profiles on matching cells. An independent direct-label oracle must verify all fields on exhaustive small synthetic cases, including joint-only success, conflicting local optima, ties, domain endpoints, empty guards, fixed labels and K1–3. Validate real local profiles independently, at least through full column-local grid precomputation and index mapping; directly reapply every reported global optimum. Verify obstruction witnesses and arbitrary-cost lower bounds. A standalone compact input and copy-only reproduction must regenerate outputs without raw archives.

The intended contribution is a composable, exact feasible-region audit: waveform gates -> first-run reachability -> cross-column law constraints -> collateral losses, with success and failure witnesses. Two-dimensional sweeping itself is established. Better predictive accuracy, a new physical law, all possible classifiers, and violin measurements are outside the claim. Report the actual degree of novelty after checking primary prior literature; an outcome that weakens our previous interpretation is equally valid.
