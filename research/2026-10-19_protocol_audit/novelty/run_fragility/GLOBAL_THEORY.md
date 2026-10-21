# Exact global decision distances for first-run boundary laws

This note proves the composition implemented by `global_profile.py`. Local first-run costs and witness reconstruction are proved in `THEORY.md`. This is a specialized use of known regular-language correction and one-dimensional interval geometry; no new generic dynamic-programming or Helly theorem is claimed.

## Problem

For fixed eligible column j and admissible first-run start s, let C_js be the exact allowed Hamming-edit cost, and [l_js,u_js] its finite log-force bracket. With x_j=log beta_j, fixed slope p, and declared tolerance epsilon, define the **closed** intercept interval

`I_js = [l_js − p*x_j − epsilon, u_js − p*x_j + epsilon]`.

Choose one state per column. Compatibility means that the chosen intercept intervals have a common point. Columns, measured forces, force order and beta are fixed. Edits affect labels independently between columns. Impossible local states have infinite cost and are omitted. We do not allow a column to disappear or turn into a censored/no-run state in the empirical primary analysis.

## Restoration theorem

Set `g_j(a)=min_{s: a in I_js} C_js`, with an empty minimum equal to infinity. The exact distance to compatibility is

`C = min_a sum_j g_j(a)`.

**Lower bound.** Any compatible edited dataset has an intercept a and one local start s_j per column. Its edits in column j cost at least C_j,s_j, which is at least g_j(a). Thus its total cost is at least the displayed minimum.

**Attainment.** At a minimizing a, select a cheapest covering state in each column and reconstruct its exact local witness. The columns are disjoint sets of labels, so these witnesses can be combined. The common a certifies compatibility, and their costs sum to C.

Each g_j is constant in every open cell between sorted candidate endpoints. Its endpoint value can differ because intervals are closed. Hence an exhaustive endpoint-and-cell sweep suffices; no numerical intercept grid or local optimizer is needed. For a point event, intervals starting and ending there must both be active. The implementation keeps a minimum-cost active-state heap per column and the sum of their minima. With M total states and J columns, a conservative bound is O(M log M + M log J) time and O(M+J) storage (heap logarithms are bounded by log M). Final state selection is O(M).

## Exclusion theorem

Let `m_j=min_s C_js`. The exact distance to incompatibility is

`D = min_{i!=j, r,s: lower(I_ir)>upper(I_js)} [C_ir+C_js + sum_{h not in {i,j}} m_h]`.

**Lower bound.** A finite family of intervals has empty intersection exactly when its largest lower endpoint exceeds its smallest upper endpoint. Because individual intervals are nonempty, these two extremes belong to different columns. Any incompatible assignment therefore includes such a disjoint pair and pays at least its local costs plus all other column minima.

**Attainment.** Combine an optimal disjoint pair with minimum-cost states in other columns and their local witnesses. The pair alone makes the intersection empty; other columns cannot restore it. This attains the bound. The implemented sweep sorts lower and upper endpoints and searches for the cheapest eligible upper endpoint from a different column, using costs relative to m_j. Strict inequality is essential: touching closed intervals are compatible. Sorting and heap updates give O(M log M) time and O(M+J) storage.

If the original verdict is incompatible, the reversal distance is C; otherwise it is D. A finite distance d certifies that no allowed edit of fewer than d labels reverses that verdict. For d>0, d−1 is the integer robustness radius for this **specified decision and admissible state set**. No confidence level or physical error model enters this deterministic statement.

## Force-grid support diagnostic (post hoc)

Ignoring costs and edit-direction restrictions, each column's admissible finite first-run brackets form a contiguous union in the present strictly ordered positive force grid. The starts can range from the first positive finite adjacent bracket to n−K. Adjacent brackets touch. Let

`L_j = min_s l_js − p*x_j`, `U_j = max_s u_js − p*x_j`.

The union of candidate intercept intervals at tolerance epsilon is then `[L_j−epsilon,U_j+epsilon]`. At least one assignment is compatible if and only if

`max_j L_j − epsilon <= min_j U_j + epsilon`.

Therefore the least tolerance permitting any label assignment is

`epsilon_grid = max(0, (max_j L_j − min_j U_j)/2)`.

The two columns attaining these extremes certify infeasibility below epsilon_grid. This diagnostic was added after observing the empirical infeasible cases; it was not a new preregistered hypothesis. `diagnose_results.py` checks contiguity before applying the formula and agrees with all 78 primary both-direction geometry decisions. **For disconnected unions this envelope formula is only necessary, not sufficient.** Add-only/delete-only constraints may also remove candidate states and break contiguity; those use the full solver.

## Boundaries of the guarantee

The proof assumes additive independent label edit costs, fixed slope, and one shared free intercept. A shared feature-classifier threshold couples edits across columns and gives a different problem. Because its permitted label patterns form a subset, our arbitrary-edit distance lower-bounds its distance, but an arbitrary witness need not be implementable by that classifier. More free law parameters would also require a different global geometry.

The brackets describe an operational first-run boundary, not a certified physical transition. The all-column tolerance decision is not an interval-MLE confidence interval or p-value. All primary feasible restoration witnesses have an interior common intercept, so their minima survive an open-lower endpoint interpretation; this does not imply universal endpoint-convention equivalence. Exact costs are combinatorial; interval endpoints and comparisons use ordinary floating-point measured inputs. See `verification/REVIEW.md` for independent exhaustive tests and numerical edge checks.
