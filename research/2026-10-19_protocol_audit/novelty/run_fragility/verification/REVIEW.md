# Independent algorithm verification

Date: 2026-10-21. Scope: code and synthetic cases only; no empirical-result
selection, downloads, or shared-paper changes. The verification implementation
does not use the production dynamic program or endpoint sweep to compute its
oracle answers.

## First-run edit costs: passed

`exhaustive.py` enumerates every binary source and target string of lengths
0–9. It computes ordinary Hamming distances and discards forbidden edits for
each mode (`both`, `delete`, `add`). Literal slice checks identify the first
run for K=1,2,3,4. The resulting minimum costs are compared against **every**
possible first-run location and the no-run state.

- 12,276 source / K / mode configurations.
- 92,238 endpoint or no-run cost comparisons.
- 55,132 finite optimal witnesses reconstructed and checked.
- 37,106 impossible targets checked for explicit infeasibility.
- 4,194,300 source-target-mode comparisons in the enumeration.
- The public `first_run_start` helper is also compared with literal slices.

Witness checks include the exact edit count, allowed edit direction, first
run location, binary output, feasibility flag, and reported changed indices.
An impossible target's returned unchanged labels are **not** treated as a
witness.

`validation_cases.py` additionally checks 20 invalid input cases, accepted
binary float/bool/integer arrays, deterministic tie handling, and that supplied
arrays are not mutated. All pass.

## Global law compatibility and incompatibility: passed

`global_oracle.py` enumerates the complete Cartesian product of candidate
intervals—one chosen interval per column. An assignment is compatible exactly
when `max(lower endpoints) <= min(upper endpoints)`; rejection uses the strict
opposite inequality. This oracle does not reproduce the production sweep.

- 3,016 solver/fixture cases, covering both restoration and rejection.
- 218,318 explicitly enumerated candidate combinations.
- 2,286 feasible optima with validated objective and chosen states.
- Closed endpoint-only intersections, zero-width intervals, unbounded
  intervals, equal-cost states, impossible states, and empty input included.

## Entire label-to-law pipeline: passed

`end_to_end.py` enumerates literal target labels for three short force grids,
derives finite first-run brackets directly, and enumerates every cross-column
target-label choice. Its minimum is compared with the production DP costs
followed by the global optimizer.

- 540 cases; 107,742 target-label combinations.
- Lengths 3–5, K=1–3, all three edit modes.
- Slopes -2, -1.2, 0, +1; log-force tolerances 0, .1, .2, .4.
- Both compatibility restoration and exclusion are tested.
- Reconstructed production witnesses are checked against the original labels
  and their actual intervals; the total edit count and verdict must agree.

These tests cover the declared **finite-positive first-run bracket** analysis:
the target may not replace a column by a no-run or left-censored state.

## Algorithm inspection

The local DP stores the cheapest run-free prefix by trailing-one length. A
first run at position s requires the prefix to end in zero (or be empty),
followed by K ones; its suffix is unconstrained. This justifies the local
factorization rather than assuming that deleting the first two H labels is
always optimal.

The global acceptance sweep evaluates endpoint cells with both ending and
starting closed intervals active. For rejection in one dimension, an empty
intersection implies two different columns with `lower_i > upper_j`. Pair
increments above each column's cheapest state therefore suffice for an exact
rejection minimum. This would not extend directly to multiple law parameters
or higher-dimensional regions.

## Numerical edge audit: fixed and passed

An initially observed extreme-float edge case (`[-inf, -1e308]`) returned an
infinite intercept because a representative point overflowed. The root owner
fixed the sweep to prefer a closed endpoint on an equal-cost tie. Both extreme
regressions (`-1e308` and the most negative finite float) now return finite
intercepts. The 16 malformed interval/cost cases are rejected. The entire
global oracle and end-to-end oracle were rerun on the fixed implementation
and still pass. This was an API edge; it did not affect the bounded log-force
inputs used in this study.

## Exported empirical inputs: passed

Independent identity/order checks found 13 session tags, 39 speed folders,
1,560 beta columns, and 78,000 unique `(session, folder, trial)` windows.
Every folder has 40 columns and every column contains 50 labels. There are
no duplicate force values within a column and no duplicate window identities.
The exported force order is increasing. There are 510 nonpositive force
windows in the complete export; they are not silently clamped.

For K=2 the declared filter retains 1,132 columns / 56,600 windows and excludes
315 originally left-censored columns and 113 no-run columns. None of the
retained original lower endpoints is below the old 1 mN clamp. The raw-force
bracket and legacy clamped bracket therefore agree at these original K=2
endpoints. This does not assert equivalence for every edited candidate.

`input_audit.json` records additional source hashes for the classifier
threshold file, column-assignment implementation, and grouping/loader code.
The frozen compressed input was not changed. Classifier correctness and raw
signal extraction were not revalidated by this audit.

## Empirical witness reapplication: passed

`empirical_witness_audit.py` uses no production DP or global optimizer. It
starts from frozen input labels, applies each saved change by column, trial,
and force rank, then computes the first run and common-intercept intersection
using literal slices and elementary arithmetic.

- All 234 primary K=2, epsilon=.2 directional records checked.
- 355 feasible witnesses reapplied, including 4,901 explicitly listed label
  changes; 113 infeasible entries correctly contain no witness.
- Before/after labels, edit directions, window identity, force, beta, changed
  column count, total cost, boundary start, and final verdict all agree.
- All 53 unrestricted primary decision changes are restorations. Every one
  has a **positive-width** feasible intercept intersection; the smallest width
  is 0.0001192546 in log-force units. Across all 121 feasible directional
  restoration witnesses there are also zero endpoint-only intersections.

The algorithm intentionally defines closed intervals. A literal monotone
threshold bracket could instead be open at its lower endpoint. Because the
primary optimal restoration witnesses have an interior point, the same edit
counts remain attainable under that stricter lower-endpoint convention.
This is a check of the primary restoration minima, not a blanket assertion
about every tolerance or hypothetical dataset.

## Interpretation and limits

1. The certificate is exact for the stated discrete edit problem and the
   supplied floating-point intervals. It is **not** a minimum edit count for
   reversing the original interval-MLE p-value, confidence interval, or fitted
   exponent. No original statistical test has been rerun by this verifier.
2. It concerns a fixed exponent with a freely chosen common intercept and a
   declared tolerance. Changing the force axis, tolerance, run length,
   permitted edit directions, or allowable censoring states changes the
   question and may change the minimum.
3. An edit witness is an adversarial sensitivity example, not evidence that
   the edited labels are correct. A small count does not establish that the
   target physical law is true or that that many actual errors occurred.
4. A support-infeasible target is not evidence of unlimited robustness. It
   may simply be unreachable on the finite measured force grid under the
   declared admissible states.
5. The core accepts an ordered label sequence. Empirical force sorting,
   duplicate-force handling, source selection, and label construction belong
   to the caller and require a separate data audit.
6. Minimum-change sensitivity, finite-state dynamic programming, and interval
   intersection are established ideas. These checks establish correctness of
   this specialized composition; they do not establish first-in-literature
   novelty.

## Post hoc grid-support certificate: formula reviewed

`diagnose_results.py` computes the following separate diagnostic after the
primary costs were available. For column j, take the union of **all** allowed
finite candidate intercept intervals before adding tolerance. The code first
asserts that adjacent candidate intervals overlap or touch, so this union is
one closed interval `[L_j, U_j]`. With unrestricted label edits every such
first-run location is attainable. Symmetric tolerance epsilon expands the
union to `[L_j - epsilon, U_j + epsilon]`. A common intercept exists precisely
when

`max_j L_j - epsilon <= min_j U_j + epsilon`.

Consequently, the minimum nonnegative tolerance permitted by the grid alone
is exactly `max(0, (max_j L_j - min_j U_j) / 2)` for the declared closed-interval
problem. This is a one-dimensional interval-intersection identity, not a new
general mathematical result. The implementation matches this expression and
records the columns attaining the two extrema. Its primary feasibility flags
are checked in the production diagnostic against all 78 folder/exponent
conditions. The saved diagnostics have 25/39 grid obstructions at epsilon=.2
for p=-2 and 0/39 for p=-1.2.

Scope: this ignores the number of edits, assumes independent unrestricted
two-direction label changes, uses the fixed selected columns, and requires
each candidate union to be contiguous. It is not a directional-edit
certificate, a budget-constrained certificate, an independent confirmation,
or a proof about the complete unselected measurement design. The formula
would need revision for disconnected candidate unions or shared restrictions
on edits across columns. Under open lower endpoints the same expression is
an infimum, and equality need not be attainable; the implemented conclusion
uses the explicitly declared closed semantics.

## Compact-copy audit portability

`input_audit.py` now accepts `--external-dependencies optional|required|skip`.
The default is `optional`; missing external provenance files are reported.
The original research audit was rerun with `required`, preserving all seven
dependency hashes. `skip` performs no external provenance reads. All modes
still require the frozen compact input and every numerical, identity, force
order, duplicate, and eligibility check.

A fresh temporary compact copy passed both default-optional and skip modes,
with every scientific field matching the required original audit. Only four
provenance fields may differ across these runs: `dependency_sha256`,
`threshold_summary_candidates`, `external_dependency_mode`, and
`missing_external_dependencies`. See `input_audit_portability.json`.

## Reproduction

Run, with the project's Python environment:

```sh
python verification/exhaustive.py
python verification/validation_cases.py
python verification/global_oracle.py
python verification/end_to_end.py
python verification/global_validation.py
python verification/input_audit.py --external-dependencies required
python verification/empirical_witness_audit.py
```

Machine-readable counts and elapsed times are saved in the corresponding
`*_results.json` files. Runtime is a few seconds for the full local oracle and
under one second for each global oracle on this laptop; no GPU is needed.
