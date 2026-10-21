# Exact editing costs for the first consecutive-success boundary

Date: 2026-10-21. This is an algorithmic statement about an operational
classification rule, not a physical law. No empirical data were used to
derive or implement this part.

## Object being certified

Let `x[0:n]` be binary labels in a **fixed order of increasing force**.
For an integer `k >= 1`, define

\[
 R_k(y)=\min\{s: y_s=y_{s+1}=\cdots=y_{s+k-1}=1\}.
\]

If the set is empty, write `R_k(y) = None`. The output is the **start**, not
the end, of the first length-`k` run. Runs may overlap. The sequence need
not be monotone in force. An existing later success island therefore does
not justify treating all earlier labels as successes.

The optimization for a target `s` is

\[
 C_s(x)=\min_{y:\,R_k(y)=s}\sum_{i=0}^{n-1}\mathbf 1\{y_i\ne x_i\},
\]

subject to one of three edit sets:

- `both`: either label direction is permitted;
- `delete`: only `1 -> 0` is permitted;
- `add`: only `0 -> 1` is permitted.

An impossible target has cost infinity. A separate optimization `C_none`
requires that no length-`k` run remains. The minimum is a **count of
changed labels**, not a force displacement or a probability.

If multiple records have the same force, the caller must supply the fixed
ordering or aggregation rule used by the actual boundary extractor. This
module does not invent a tie rule, interpolate missing measurements, or
convert missing labels into zeros.

## Exact dynamic program

Let `c_i(b)` be the cost of writing bit `b` at position `i`: zero for
retaining the original bit, one for an allowed change, and infinity for a
forbidden change. Define `D_t(r)` as the minimum cost for a prefix of length
`t` that contains **no** length-`k` run and ends with exactly `r`
consecutive ones, where `0 <= r < k`.

Initialize `D_0(0)=0` and `D_0(r)=infinity` for `r>0`. Then

\[
 \begin{aligned}
 D_{t+1}(0)&=\min_{0\le r<k} D_t(r)+c_t(0),\\
 D_{t+1}(r)&=D_t(r-1)+c_t(1),\qquad 1\le r<k.
 \end{aligned}
\]

There is deliberately no transition from state `k-1` on writing a one:
that transition would complete a forbidden run. Consequently,

\[
 C_{\mathrm{none}}=\min_{0\le r<k}D_n(r).
\]

To make the first run start at `s`, the preceding prefix must be run-free
and, if it is nonempty, must end in **zero**. Otherwise the newly forced
ones would finish a run starting before `s`. Thus

\[
 \boxed{C_s=D_s(0)+\sum_{i=s}^{s+k-1}c_i(1)},\qquad 0\le s\le n-k.
\]

The empty prefix at `s=0` uses `D_0(0)=0`. All labels after position
`s+k-1` can be left unchanged: they cannot change the first run, and
changing them cannot reduce a nonnegative Hamming objective.

### Proof of optimality

The recurrence is exhaustive: every run-free nonempty prefix ends either
in a zero appended to one of the run-free states, or in a one appended to
the unique preceding trailing-run state. These alternatives are disjoint.
Induction over prefix length therefore gives their exact minimum costs.

For a sequence with first run start `s`, its prefix has no earlier run;
for `s>0`, the preceding bit must be zero. The cost of that prefix is at
least `D_s(0)`, and writing the next `k` ones costs at least the displayed
sum. This proves a lower bound on every feasible sequence. Conversely,
an optimal state-zero prefix, followed by those `k` ones and the unchanged
suffix, is feasible and attains the bound. No earlier run can lie wholly
in the prefix or cross its final zero. This proves equality.

This proof also covers `k=1`: the only run-free state is zero, so every
position before the first success must be zero. When `n<k`, no run can
exist, the no-run cost is zero, and there are no finite start positions
to request.

### Runtime, exactness and witnesses

The dynamic program has `k` states and `n` stages. All start costs can be
read during one pass, so time is `O(n*k)` and memory is `O(n+k)` including
the output. The `n<k` case returns without allocating `k` states.

The forced-window cost uses separate cumulative sums of finite costs and
forbidden writes. It never subtracts infinity from infinity.

Witness reconstruction stores only the selected predecessor of state
zero at each stage. A positive state has its unique predecessor one
smaller, so a full `n*k` predecessor table is unnecessary. Reconstruction
uses `O(n+k)` memory and `O(n*k)` time. Ties choose the smaller state for
determinism; this is not a claim that the minimum-edit witness is unique.

All finite objectives are integer sums. The implementation stores the
cost table as floating point only to accommodate infinity; there is no
continuous optimization, tolerance-based optimality declaration, or
simulation in this calculation. It also verifies that each reconstructed
witness changes exactly the returned number of labels.

## API and example

Implemented in `core.py`:

```python
table = first_run_costs(labels, k=2, mode="both")
# table["costs"][s], table["no_run_cost"], table["n"], table["k"], table["mode"]

witness = witness_first_run(labels, start=s, k=2, mode="both")
# witness["labels"], witness["flipped_indices"], witness["cost"],
# witness["start"], witness["feasible"]

start = first_run_start(labels, k=2)  # integer start or None
```

`start=None` requests a no-run witness. An impossible witness has
`feasible=False`, `cost=infinity`, an unchanged copy of the source labels,
and an empty list of flip indices. **Those unchanged labels are not a
witness for the requested impossible target.** Callers must inspect
`feasible`.

Inputs must be one-dimensional Boolean or real numeric sequences
containing only zero and one. Empty sequences are allowed. `k` must be a
positive integer, and a non-None `start` must be an integer between zero
and `n-k` inclusive. Input arrays are not mutated.

For `x=[1,0,1,1,0,1]` and `k=2`, the observed first run starts at index 2:

| Target first start | 0 | 1 | 2 | 3 | 4 | No run |
|---|---:|---:|---:|---:|---:|---:|
| Both directions | 1 | 2 | 0 | 2 | 2 | 1 |
| Delete only | impossible | impossible | 0 | impossible | impossible | 1 |
| Add only | 1 | impossible | 0 | impossible | impossible | impossible |

For example, making the first run start at index 1 cannot merely fill the
zero at index 1: that would create an earlier run at index 0. The leading
one must also be removed, which costs two edits and is forbidden in the
add-only mode.

## Composition and limits of interpretation

When columns have independent edit permissions and their Hamming costs
add, the sum of these per-column costs is an exact cost for any specified
vector of first-run starts. If a global law-compatibility constraint uses
only that vector, optimizing over feasible target vectors preserves
exactness. The parent analysis must separately define and verify that
constraint, its treatment of censored intervals, and its numerical
solution. This module by itself does **not** certify a fitted exponent,
likelihood-ratio test, significance level, or confidence interval.

Important distinctions:

1. **Operational robustness, not truth.** A small edit count means the
   operational result is fragile to admissible changes. It does not show
   that those labels are wrong, or that the revised law is physically true.
2. **Independent flips versus a realizable classifier.** Arbitrary flips
   may not be achievable by one coherent threshold change. The minimum
   over this larger edit set is a lower bound on the minimum under stricter
   classifier-realizability constraints. Directional modes constrain only
   flip direction, not monotonicity in a measured feature.
3. **A no-run output is not a physical assertion.** This algorithm only
   says that the operational run rule no longer finds a band. Any later
   mapping to right censoring, an absent band, or missing information must
   be justified outside this module.
4. **First boundary only.** The first-run cost does not preserve the last
   run, upper boundary, band width, total number of successes, or internal
   topology unless those are added explicitly to the optimization.
5. **Novelty must be scoped.** Dynamic programming over a finite-state
   sequence constraint is a standard construction. The defensible
   contribution, if supported by the complete analysis, is its exact and
   auditable composition with this operational boundary rule and a
   specified scientific conclusion, rather than inventing dynamic
   programming or generic classification fragility.

The independent exhaustive oracle and its validation records are owned
by the separate verification task, under `verification/`.
