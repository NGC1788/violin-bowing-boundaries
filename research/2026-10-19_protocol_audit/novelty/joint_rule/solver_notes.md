# Exact joint-rule profiler

`solver.py` contains only finite-array calculations. It does not load results,
select preferred thresholds, perform downloads, or alter eligibility.

## Contract

`analyze_group(columns, u_domain=(0.7,0.9), v_domain=(0.3,0.7),
slopes=(-2,-1.2), epsilons=(0,0.1,0.2,0.4), k=2)` accepts one folder/group.
Its rule is exactly

\[
H_i(u,v)=f_i\lor[g_i\land(P_i\ge u)\land(R_i<v)].
\]

The guard must encode only the other frozen classifier gates; it cannot contain
either old threshold gate. `fixed` and `guard` are disjoint. Scores are finite
on guarded windows; unguarded scores can be NaN. Baseline labels determine
the editing cost and original eligibility; they are not recomputed internally.
The caller should independently check that the original threshold pair produces
all baseline labels. IDs, force ordering, fixed eligibility, and other gates
must be exported without using compatibility results.

The score dimensions are not physical controls. The profile changes a global
measurement/decision rule, not the experiment's bow force or position.

## Complete finite partition

Both predicates change immediately *after* their score. At `u=P`, the
periodicity condition still passes; at `v=R`, the ratio condition still fails.
Thus both axis partitions use right-closed atoms. The initial atom is left
closed. If a score equals the lower domain endpoint, the initial atom is the
singleton at that endpoint. Scores equal to the upper endpoint create no
additional atom inside the closed domain. Equal scores form one event.

An axis event can be pruned only if the other predicate can never pass anywhere
in the whole rectangle:

* A periodicity event requires `ratio < v_hi`.
* A ratio event requires `periodicity >= u_lo`.

These strict/non-strict endpoints are deliberate. Pruning by the original
threshold pair would incorrectly miss joint-only changes. All retained axis
events are guarded and lie in `[axis_lo, axis_hi)`. The Cartesian product of
their atoms covers the entire closed parameter rectangle. Within every
rectangle each predicate, label, first-run boundary, and cost is constant.
The upper-right corner is a representative with the correct labels.

The number of rectangles is an enumeration count, **not necessarily a count of
distinct label vectors**: inactive intersections can repeat a label pattern.
It is also not a number of independent scientific observations.

## Boundary validity and compatibility

The first start `s` of K consecutive H labels is found in force order. It gives
the bracket `[force[s-1], force[s]]` only if `s > 0` and
`0 < force[s-1] < force[s]`. A missing run, a run starting at zero, or a
nonpositive/tied bracket is invalid. These causes are counted separately.
Any invalid originally eligible column invalidates the group's compatibility;
disappearing boundaries cannot turn into support. No eligible columns also
means no compatibility.

For slope p, every valid column j supplies

\[
L_j=\log F_{j,lo}-p\log\beta_j,\qquad
U_j=\log F_{j,hi}-p\log\beta_j.
\]

The group endpoints are `L=max(L_j)` and `U=min(U_j)`. The code evaluates the
explicit floating-point predicate `L-epsilon <= U+epsilon`, not an algebraic
rearrangement that might round differently at equality. Math-library float64
logs are used; the exactness guarantee concerns the exhaustive combinatorial
search, not symbolic exact real-number arithmetic.

Each valid rectangle includes the eligible-column indices attaining L and U.
When incompatibility holds, those two columns alone prove it under that
rectangle's labels. Each invalid rectangle records the first invalid eligible
column, giving a one-column obstruction. Tie witnesses use input-column order.
Witnesses are -1 when not applicable. The output includes eligible column IDs
so every certificate can be replayed directly from the input fixture.

## Arrays and global aggregation

* Eight cost/validity count arrays: `[n_u, n_v]`.
* `lower`, `upper`, `gap`, `lower_witness`, `upper_witness`:
  `[n_u, n_v, n_slopes]`.
* `invalid_witness`: `[n_u, n_v]`.
* `compatible`: `[n_u, n_v, n_slopes, n_epsilons]`.
* `u_atoms`, `v_atoms`: endpoint arrays, closure flags, representatives.
* `metadata`: total/eligible window and column counts, IDs, eligible indices.

`threshold_atoms(scores, domain)` constructs axis atoms independently.
`map_atoms(local_atoms, global_representatives)` maps a finer shared grid to a
local row/column using `searchsorted(local_hi, representative, side='left')`.
For global aggregation the union of all group axis events must be formed.
Each global pair maps to exactly one local pair per group. All costs and
compatibility indicators are then evaluated at that **same pair**. The union
of per-group feasible rules is only an oracle relaxation; individual group
optima cannot be added as if they were simultaneously attainable.

No baseline-distance or threshold-tie rule is built into the profiler. The
calling protocol must define those conventions before looking at costs.
Similarly, adding a no-invalid-column constraint after viewing results is a
separate sensitivity analysis unless it was specified in advance.

## Resource strategy and implementation validation

For each u row, labels for all v representatives and all padded columns are
computed by broadcasting. K-run starts, editing costs, and interval bounds are
then reduced with NumPy. Padding is always non-H. Local outputs are retained;
the label tensor is not retained across u. Memory is of order
`n_v*n_columns*max_column_length + n_u*n_v*(metrics + slopes*epsilons)`.
Global aggregation can proceed one shared-u row at a time without a
`global_u * global_v * groups` tensor.

A synthetic 40-column, 50-window-per-column smoke benchmark with seed 41
enumerated 470 by 389 local rectangles in about 0.62 seconds on the task
machine and returned zero baseline changes at (0.8,0.5). This benchmark says
nothing about empirical law compatibility. Independent validation is owned by
the separate `verification/` implementation; its reports, not this smoke
benchmark, establish replay agreement.

The search and interval-intersection arguments are elementary. The intended
contribution is their auditable combination for a physically motivated
measurement rule, jointly attainable boundaries, and collateral losses. It is
not a claim to have invented general two-dimensional threshold sweeps.
