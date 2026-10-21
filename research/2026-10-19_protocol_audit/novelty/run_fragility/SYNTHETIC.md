# Synthetic behavior check for exact operational fragility

## Fixed plan, recorded before computation — 2026-10-21

This is a bounded software and interpretation validation, not a benchmark
for detecting the physically true law. It uses no empirical labels or
previous empirical fragility costs. The root `PLAN.md` defines the same
first-K-run boundary, finite force intervals, closed-interval compatibility,
and distinction between decision reversal and restoration.

All parameters below are fixed before running `synthetic.py`:

- Seed: 20261021, NumPy default_rng.
- Six beta values geometrically spaced from 0.025 to 0.2.
- Forty-eight force values geometrically spaced from 0.01 to 20 N, shared
  by all columns. The numerical unit supplies a convenient scale only.
- Two generating lower boundaries: `F_min = 0.4*(beta/0.1)**p_true`, with
  `p_true=-2` and `p_true=-1.2`.
- Clean success bands: `F_min <= F <= 4*F_min`. Only the operational first
  run is the target; recovering the full binary band is a different task.
- K=2, log-force tolerance epsilon=0.2. Target-law slopes are both -2 and
  -1.2. Intercept is unrestricted and shared across the six columns.
- For each generating slope: one clean case; four cases with three
  distinct uniformly sampled label flips; four cases deleting the first
  success in two randomly selected distinct columns; four cases deleting
  a four-label block starting at the clean first success in one randomly
  selected column. Total: 26 cases. No outcome-driven case replacement.
- Report the exact cost to restore the clean **first-run vector**, the
  number of labels actually perturbed, and the two law-restoration and
  exclusion costs under both/delete/add permissions. Decision reversal
  means switching the current compatibility verdict; restoration may be
  zero and is never relabeled as a zero-cost reversal.
- Apply all returned witnesses to the original perturbed labels and
  verify their actual Hamming distance, first-run vector, directional
  permissions and target verdict.
- Also use a deliberately narrow common force grid: 12 points from 0.4
  to 0.6, with original first run at index 4 in every column. Query both
  slopes with the same beta range, K and epsilon. This support-only case
  must distinguish an unrepresentable shape from infinite robustness.
  All finite start states are considered with zero cost to check grid
  support independently of label costs and flip direction.

The independent global and end-to-end exhaustive verifications passed
before this experiment: 3,016 global fixtures and 540 full-label cases.
The synthetic study will use `core.py` and `global_profile.py` unchanged.

## Interpretation rules fixed before computation

The cost of restoring the original first-run vector cannot exceed the
known injected Hamming changes in both-direction mode: undoing all noise
is an admissible witness. It can be smaller or zero because labels far
from the first run need not change the operational boundary. This is
desirable target specificity, not successful recovery of every label.

The true generating slope must be compatible in the clean condition.
Arbitrary noise can remove that compatibility, and restoring it need not
recover the original labels or boundaries. Conversely, an alternative
slope can become compatible after a finite number of edits even when it
did not generate the labels. These facts demonstrate precisely what the
metric measures and what it cannot identify.

A support-infeasible target must be labeled `grid_support_infeasible`.
It will not be reported as an infinitely robust scientific conclusion.

## Results

The fixed computation completed without altering any parameter or
replacing any case. All six columns remained eligible in all 26 cases.
There were 156 law/mode comparisons, and **286 feasible global witnesses**
were reconstructed and checked directly against the perturbed labels.
The remaining 26 requested global outcomes were impossible under the
delete-only restriction; they were correctly distinguished from lack of
force-grid support. Raw results and individual witnesses are in
`synthetic_results.json`.

The following table reports both-direction edits. Ranges are the minimum
and maximum across the fixed cases, not confidence intervals.

| Perturbation | Cases | Injected changes | Cost to restore original first-run vector | Generating law remains compatible | Cost to restore generating-law compatibility | Cost to make alternative law compatible |
|---|---:|---:|---:|---:|---:|---:|
| Clean | 2 | 0 | 0 | 2/2 | 0 | 5–6 |
| Random sparse | 8 | 3 | 0–1 | 8/8 | 0 | 5–6 |
| Two boundary-adjacent deletions | 8 | 2 | 2 | 8/8 | 0 | 5–6 |
| Four-label boundary burst | 8 | 4 | 2 | 0/8 | 1–2 | 4–6 |

### What the controlled examples establish

1. **Restoration and reversal are different.** In both clean examples the
   generating law is already compatible, so its restoration cost is zero.
   Its compatibility verdict takes **two** edits to reverse. For the
   excluded alternative slope, restoring compatibility is the reversal:
   it takes six edits when the generator is -2 and five when it is -1.2.
2. **The operational target ignores irrelevant label errors.** Seven of
   eight sparse-noise examples require zero edits to recover the original
   first-run vector even though all contain three altered labels; one
   requires one edit. Yet their generating-law reversal cost is one,
   versus two without noise. Existing isolated errors can therefore leave
   the present conclusion unchanged while reducing its distance to a
   different conclusion. This is not full-label denoising performance.
3. **A burst can reject the generating law without changing it.** The
   fixed four-label deletion excludes the true generating slope in all
   eight burst examples. The exact compatibility-restoration cost is only
   one or two edits. Restoring the original first run always takes two
   edits: recreating its first two successes is sufficient even if the
   other two deleted labels remain zeros. Neither operation recovers all
   four labels, and neither is claimed to do so.
4. **Low reversal counts need not select the true law.** These examples
   retain the known generating slope only because that slope was supplied
   by the simulation. An empirical reversal count by itself does not
   identify which candidate law is physically correct. The alternative
   law becomes compatible after four to six changes in these examples.

The generating-law reversal cost is 2 for clean cases, 1 for all sparse
cases, 1–2 for boundary-adjacent cases, and 1–2 for burst cases. Its
direction changes: the first three groups reverse compatibility into
exclusion, whereas bursts reverse exclusion into compatibility.

Every exact clean-boundary restoration cost was no larger than the known
injected Hamming changes. Every generating-law restoration cost was no
larger than the cost of recovering the original first-run vector. Both
are required feasibility inequalities, and both held in all 26 cases.

### Unsupported force-grid example

On the narrow force grid, neither slope can be made compatible under any
binary relabeling that preserves finite first-run intervals. The reported
status for both is **`grid_support_infeasible`**, with a null numerical
reversal count rather than an infinite scientific-robustness score.

This result also has a direct geometric explanation. Across the beta
range, the proposed log-force changes are approximately 4.159 for slope
-2 and 2.495 for slope -1.2. The entire admissible boundary range, including
the two tolerance margins, spans only 0.769 log-force units. No shared
intercept can fit either shape within that grid. This is a limitation of
measurement support, independent of which labels were supplied.

### Scope

This is a small, pre-specified demonstration of exact cost behavior on
known constructed labels. It verifies interpretability and concrete
certificates, not physical validity, superior accuracy, a calibrated error
probability, or generalization to real instruments. The stronger software
correctness evidence is the independent exhaustive oracle in
`verification/`; these 26 cases complement rather than replace it.
