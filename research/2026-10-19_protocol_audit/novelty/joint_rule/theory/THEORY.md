# Exact shared two-threshold boundary certificates

2026-10-21. Derivation prepared before inspecting the empirical two-threshold results. This document defines a mathematical object, not an empirical result. Features, physical gates, K=2, the original eligible columns, force ordering, law slopes, and tolerance are fixed. The two thresholds are shared by all observations.

## 1. Classifier and domain

For force-ordered observation i, use

`H_i(u,v) = fixed_i OR (guard_i AND x_i >= u AND y_i < v)`.

Here x is periodicity, y is second-flyback ratio, and the guard contains all unchanged physical gates. The declared primary domain is the closed rectangle D=[.7,.9]×[.3,.7]; the baseline is (.8,.5). Fixed and guard are disjoint. Missing features must never be silently promoted to valid data; the export must lock or reject them under a predeclared policy and reproduce every baseline label exactly.

**Export pitfall:** taking the union of the old one-feature guards is not sufficient. An observation with x<.8 and y>=.5 can fail both old guards but become H when u is lowered and v raised. Recover the immutable physical guard from the original feature cache. Do not infer it from previous H labels.

## 2. Every K-run is an anchored rectangle

For possible first-run start s, let W_s={s,…,s+K−1}. If any member is permanently non-H, the K-run is impossible. Otherwise ignore permanently H observations and define

`P_s = min_{i in W_s, mutable} x_i`,
`Q_s = max_{i in W_s, mutable} y_i`,

with min(empty)=+∞ and max(empty)=−∞. The run is active exactly on

`R_s = D ∩ {u <= P_s, v > Q_s}`.

This follows by distributing the conjunction over the K comparisons. A permanently H block activates everywhere; an impossible block nowhere. No independence assumption or monotonicity of the final scientific verdict is required.

The first run starts at s on precisely

`F_s = R_s \ union_{r<s} R_r`.

This subtraction is essential: mere existence of a later run does not define the operational first boundary. R_s can be nonempty while F_s is empty because earlier starts cover it. The F_s sets are disjoint. Together with the no-run region they partition D.

## 3. Skyline/record-envelope decomposition

For a fixed u, only starts with P_r>=u are active in the u direction. Let

`M_s(u) = min {Q_r : r<s and P_r>=u}`,

with min(empty)=+∞. Then

`(u,v) in F_s iff u<=P_s and Q_s<v<=M_s(u)`.

Thus a reachable first start must set a strict new record low in Q among the u-active starts. This is the two-dimensional counterpart of the one-dimensional record-envelope condition. It is a standard dominance/envelope construction, not a claimed new general skyline algorithm.

M_s is piecewise constant in u and can change only strictly after one of the P_r. Partition the closed u domain into [a,e_1],(e_1,e_2],…,(e_m,b], using all distinct in-domain P_r<b. Retain a singleton [a,a] when an event equals the lower endpoint. On each slab, evaluate u at its right endpoint, scan starts in force order, and maintain the smallest active Q so far. Whenever Q_s is strictly smaller, emit the rectangle given by the u slab and `(Q_s,M_s]` clipped to the closed v domain. Update M_s even if its emitted v interval is empty. This gives a disjoint, exact first-start decomposition, including ties and endpoint closures.

The remaining v interval on that slab, `v<=min_active Q`, is the no-run region. Start zero remains a separate left-censored state. Validity of positive force endpoints and nonzero force-bracket width is checked afterwards; a region cannot become scientifically admissible merely because it has a first run.

For n observations, at most n−K+1 runs and n−K+2 u slabs are needed. The straightforward scan costs O(nK+n²) and produces O(n²) rectangles per column. It avoids the potentially much larger arrangement of all individual observation feature pairs for boundary feasibility. Combining rectangles across columns/folders can still have large output complexity; no general linear-time global guarantee is asserted.

## 4. Composing the scientific predicate

For fixed slope p and a valid first-run start s_j in eligible column j, the permissible log-intercept interval is

`I_j=[log F[j,s_j−1]−p log beta_j−epsilon, log F[j,s_j]−p log beta_j+epsilon]`.

On any common refinement cell where every column's first-run state is fixed, all boundaries are fixed. That cell is compatible if all original eligible columns remain valid and max_j lower(I_j)<=min_j upper(I_j). The intercept is folder-specific; the same (u,v) is used in all folders. This is exactly the existing all-column predicate, not the MLE or a statistical confidence test.

Computationally, one can compose local column rectangles into a folder boundary map and then intersect or overlay folder maps in the common (u,v) plane. A per-folder optimum does not imply one common pair realizes all folder optima. If only the boundary verdict is required, features that change labels without changing a first run need not refine the boundary map. **For exact minimum edit cost they must still be counted:** a boundary-constant rectangle can contain many label vectors and edit counts. A representative sufficient for feasibility is not automatically a minimum-cost witness.

## 5. Short local obstructions and a complete cover certificate

Every incompatible constant-boundary cell has one of two checkable obstructions:

1. One fixed eligible column is invalid: no run, left-censored start, nonpositive bracket, or tied force endpoints.
2. Every column is valid, but two columns j and k satisfy lower(I_j)>upper(I_k). One may choose j attaining the largest lower endpoint and k attaining the smallest upper endpoint.

The second statement follows from the closed-interval intersection identity. It does not require enumerating all subsets or a generic SAT solver. The pair is sufficient at that cell. It need not be unique or minimum in any global sense, and the pair may change as thresholds move.

To certify **no shared threshold pair anywhere in D** can satisfy the target predicate, supply a rectangle cover of D and, on each rectangle, one invalid-column or separating-pair witness valid throughout it. A verifier checks the rectangle cover, recomputes the witness columns' first-run states, and checks the inequality. This turns an opaque solver failure into a finite, checkable explanation. It is not generally enough to provide one obstructing pair at the baseline: the pair must be valid on the entire claimed region.

Conversely a success certificate consists of a threshold pair (or nonempty region), all eligible first starts, the nonempty intercept intersection, and directly recomputed labels/costs. Neither certificate establishes physical truth or classifier accuracy.

## 6. More informative extension than another sensitivity plot

The most useful output is a **composable feasible-region certificate**: each folder contributes the exact set of common rules that preserve all eligible boundaries and make the declared law compatible. Set intersection answers whether one rule can support several folders simultaneously; an arrangement maximum answers how many can be supported. Comparing this to the union of separately optimized folders isolates incoherent pooling. An explicit no-boundary-loss condition measures collateral identifiability rather than calling missing boundaries successful.

Optional Pareto summaries should keep objectives distinct: number of target folders restored, number of invalid original boundaries, edits within the target eligible subset, and edits elsewhere. They are sensitivity summaries, not probabilities over plausible classifiers. The feasible set has no calibrated prior measure, and Euclidean distances in differently scaled threshold coordinates have no physical interpretation without a declared normalization.

The narrow contribution candidate is exact propagation from a shared waveform classification rule, through first-run boundary extraction, to a physical-law compatibility decision with explicit losses of identifiability and local obstruction witnesses. General threshold optimization, box geometry, multiverse analysis, and counterexample certificates all have prior literature. Two dimensions alone do not supply high novelty. A useful paper must show the practical conclusion that the richer rule family changes or strengthens, explain why, and preserve the scope limitations.

## 7. Required checks

Test direct classifications against the rectangle decomposition for all cells on small fixtures, including tied x/y, equal domain endpoints, empty/always-H runs, missing locked values, and no-run/left-censoring. Verify the baseline pair reproduces the original labels. Check both scalar slices exactly: v=.5 must reproduce the old periodicity sweep and u=.8 the old ratio sweep. Synthetic cases must include a successful interior point with no success on either baseline slice, proving the implementation genuinely allows simultaneous changes. Independently replay all selected empirical witnesses and validate any full-domain obstruction cover. Exactness is finite combinatorial completeness for supplied float64 features, not symbolic real-arithmetic certainty.

## 8. Required tolerance as a graded diagnostic

For a valid folder at (u,v), write L and U for the unexpanded maximum lower and minimum upper intercept bounds. The mathematically minimum symmetric log-force tolerance is

`delta_g(u,v)=max(0,(L−U)/2)`.

If any original eligible column is invalid, define delta_g=+∞. Minimizing delta over D reports how far the best realizable rule is from the candidate law without relying on one chosen epsilon. Distinguish this value from a calibrated physical error; it remains the slack required by this operational all-column predicate. In float64, check the selected value against the original shifted-endpoint inequality, because evaluating the half-gap can differ by an ULP at equality.

For a prespecified collection of folders, `min_(u,v) max_g delta_g(u,v)` is the smallest common tolerance allowing one common rule to reconcile them all. It is at least `max_g min_(u,v) delta_g(u,v)`. The difference isolates shared-rule conflict beyond each folder's individual mismatch. Both optimizations require the same original eligible columns and invalidity policy. The general minimax inequality is standard; its domain-specific use is a potentially useful diagnostic, not a new minimax theorem.

## Independent structural implementation

`rectangle_oracle.py` implements the anchored-rectangle/prefix-envelope construction using only the Python standard library. It does not import the production solver or read empirical files. `test_rectangle_oracle.py` checks it against literal pointwise classifications, using small exhaustive fixtures, tied scores, closed single-point domains, neighboring floating-point probes, and seeded longer cases. This is an additional structural check, not a substitute for independently verifying the empirical solver and obstruction certificates.
