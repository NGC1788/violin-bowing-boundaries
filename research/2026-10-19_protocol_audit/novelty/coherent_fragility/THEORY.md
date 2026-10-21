# Exact profiling under one shared classification threshold

This method asks a narrower and more physically interpretable question than arbitrary label-edit fragility: **can one predeclared scalar classifier threshold, applied identically to every waveform window, change the operational boundary-law verdict?** Its guarantee is conditional on the fixed features, fixed gates, threshold family and domain, original eligible columns, first-run rule, and law tolerance. It is neither a guarantee of correct physical labels nor a reversal certificate for a pre-existing MLE confidence interval.

## 1. Classifier family and fixed analysis population

For window `i`, let `x_i` be its already computed feature, `g_i` its immutable eligibility gate for the mutable classifier branch, and `f_i` its immutable positive branch. The branches are disjoint. The two supported classifier families are

- `H_i(t) = f_i OR (g_i AND x_i >= t)`;
- `H_i(t) = f_i OR (g_i AND x_i < t)`.

A single `t` is applied across **all supplied windows, columns, and folders**. NaN features are permitted only when the mutable gate is false. Neither waveform features nor gate definitions are optimized by the solver. The input baseline labels are kept separately for edit counting; the caller must verify that its designated baseline threshold reproduces them.

The caller supplies one immutable Boolean `eligible` per force-ordered column. Every included column must originally have a finite positive nonzero-width first-K-run bracket. Originally excluded columns remain excluded at every threshold but still contribute to the all-window edit count. An included column becoming left-censored, losing its run, or acquiring a nonpositive or zero-width bracket is **invalid**, not a newly dropped observation. Invalidity prevents the folder from being declared compatible. A folder with no eligible columns is also never declared supportive; its bounds are NaN. This prevents a vacuous compatibility result achieved by deleting every constraint.

## 2. Exact threshold atoms, including ties

The domain `[d_lo, d_hi]` is closed; endpoints may be infinite. All mutable feature values are finite. In both supported families, the label change at score `v` happens **strictly after** `t = v`:

- `x >= t` remains positive at equality and becomes negative immediately above it;
- `x < t` remains negative at equality and becomes positive immediately above it.

Let `v_1 < ... < v_m` be the distinct guarded scores satisfying `d_lo <= v_j < d_hi`. The complete threshold partition is

`[d_lo, v_1], (v_1, v_2], ..., (v_m, d_hi]`.

When `v_1 = d_lo`, the first cell is the necessary singleton `[d_lo, d_lo]`. With no internal event, the partition is simply `[d_lo, d_hi]`. The right endpoint of every cell is a valid representative, including infinite domain endpoints.

**Completeness proof.** A guarded comparison can change only after its score; no comparison changes inside one listed cell. Thus all windows have constant labels within the cell. Conversely, a threshold in the domain lies in exactly one listed cell. Hence evaluating one representative per cell enumerates every reachable label vector. Tied scores must flip simultaneously: changing only part of a tie would create a classifier state that no scalar threshold can realize. Scores at the domain's upper endpoint never flip inside the closed domain.

There is no threshold grid approximation, random search, local optimization, or assumed monotonicity of the final compatibility verdict.

## 3. Exact boundary and operational law decision

Within each fixed eligible column, windows are already ordered by measured force. If the first K consecutive positives start at `s`, the finite bracket is

`[log(F[s-1]), log(F[s])]`, with `s > 0` and `0 < F[s-1] < F[s]`.

This is a run across **different force-ordered strokes**, not a time series within one stroke. The solver recomputes that first run after the coherent threshold changes the labels.

For slope `p`, column bow position `beta_j`, and common intercept `a`, the candidate law is `log F = a + p log beta`. Before the tolerance expansion, each column allows intercept interval

`I_j = [log F[j,s-1] - p log beta_j, log F[j,s] - p log beta_j]`.

A folder's intersection has endpoints

`L(t,p) = max_j lower(I_j)`, `U(t,p) = min_j upper(I_j)`.

The operational verdict at tolerance `epsilon` is

`all included brackets valid AND L(t,p)-epsilon <= U(t,p)+epsilon`.

Each folder has its own intercept, while every folder shares the classifier threshold. The tolerance is a user-defined log-force tube, not an estimated confidence interval or calibrated measurement error. Results depend on the all-column predicate, which can be stricter than an average-fit criterion.

**Decision exactness proof.** For closed one-dimensional intervals, their intersection is nonempty exactly when their maximum lower endpoint does not exceed their minimum upper endpoint. The expanded intervals have maximum lower endpoint `L-epsilon` and minimum upper endpoint `U+epsilon`. Consequently the returned predicate is equivalent to existence of a common intercept for all fixed eligible columns, subject to the explicit validity requirement.

Floating-point implementation uses the displayed shifted-endpoint comparison directly. It does not replace it by `L-U <= 2*epsilon`, whose rounding can differ at equality. Logarithms and endpoint calculations are float64; the exactness claim concerns complete combinatorial enumeration on those supplied numerical values, not symbolic real arithmetic.

## 4. Minimum-change and shared-threshold consequences

The profile stores, for every threshold atom and folder:

- Hamming edits among all windows;
- edits among eligible windows;
- number of changed eligible columns and all changed columns;
- number and causes of newly invalid eligible brackets;
- every fixed-slope intercept intersection and tolerance verdict.

Therefore minimizing an edit count over all atoms satisfying a requested verdict is an **exact finite optimization** for this classifier family. The returned attaining atom gives a realizable witness. If no admissible atom satisfies the verdict, the result certifies absence within this scalar family and domain, **not robustness against all plausible classifier changes**.

Per-folder minimum thresholds can differ and must not be represented as one universal replacement classifier. A genuinely shared analysis evaluates all folder verdicts at the **same atom**—for example, maximizing the number of compatible folders and, among ties, minimizing all-window edits. Because this is selection on exposed data, an optimum is a sensitivity witness, not a validated classifier recommendation or a new confirmatory experiment.

For a matched fixed analysis population, the minimum eligible edits attainable under this coherent family cannot be smaller than the unconstrained arbitrary-label minimum over the same admissible target verdict. The coherent label vectors form a subset of all possible label vectors. This lower-bound relationship is a useful implementation and interpretation check; it does not make the arbitrary-label minimizer realizable by a coherent criterion.

## 5. Algorithm and complexity

The implementation sorts the guarded in-domain scores once and groups exact ties. It builds the labels at the closed lower endpoint, then sweeps event levels. At each event it changes all tied labels atomically, recomputes only affected columns' first run and edit counts, and refreshes only affected folders' intersections. Previously unaffected folder results are copied to the next profile row.

Let `N` be all windows, `M` distinct in-domain guarded scores, `C` columns, `G` folders, `S` slopes, `E` tolerances, and `n_max` and `c_max` the largest windows-per-column and columns-per-folder. Score sorting is `O(N log N)`. The simple dirty-column/dirty-folder implementation is bounded by `O(N n_max + N c_max S)` refresh work. Producing the full requested profile additionally costs `O(M G S E)` time and memory of that order; this is output-size cost, not a hidden search. In this dataset columns have 50 force-ordered windows, folders have at most 40 columns, and `G=39`. No waveform arrays are retained in the profile.

The first-run scan, score arrangement sweep, and interval-intersection identity are established algorithmic ingredients. The methodological contribution being evaluated is their combination into a **coherent-classifier sensitivity certificate for run-defined empirical physical boundaries**, with a fixed analysis population and explicit invalidity accounting. General claims of a new threshold-search algorithm or first-ever robustness certificate would be unsupported.
