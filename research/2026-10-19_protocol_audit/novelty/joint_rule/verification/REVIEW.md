# Independent computational verification of the joint waveform rule

2026-10-21. This audit checks the finite declared classifier family and the derived compatibility decision. It is **not** an independent physical replication, a validation of true Helmholtz labels, or a check that a preferred threshold is scientifically calibrated. All empirical observations were previously exposed.

## Independence and coverage

`oracle.py` imports no production code. It classifies every observation directly as `fixed OR (guard AND periodicity >= u AND ratio < v)`, searches K-runs with Python `all`, takes the preceding/current force bracket, and forms the shifted interval intersection. This differs from the production row/event sweep.

`exhaustive.py` compares this literal implementation with the production solver at feature endpoints, between-feature probes, next-representable values and every production cell representative. It covers K = 1, 2, 3; tied scores; malformed force brackets reached after threshold changes; fixed labels; absent guards; and groups with no eligible columns. Three analytic examples specifically establish that the test suite can detect a joint-only rescue, conflicting folder-local optima and collateral loss of another original boundary. Thus the real-data absence of a joint-only rescue is not hard-coded into the test.

The empirical oracle builds separate literal Cartesian tables for each column, maps them onto the complete folder arrangement, and aggregates them. It checks every stored count, bound, decision and obstruction witness. Finally `global_optima.py` checks that **no feasible cell has a better objective than each reported witness**, including the full deterministic tie break; it does not call the production sorting/selection functions.

## Completed checks

| Check | Coverage | Result |
|---|---:|---|
| Small-fixture literal solver check | 755 fixtures, 90,140 threshold pairs, 5,841 solver cells, 4,492,535 scalar comparisons, 25,944 minima, 464,745 certificate checks | PASS |
| Input rejection and endpoint regression | 30 malformed group inputs; 6 atom/mapping errors; 4 endpoint/empty-guard fixtures; signed/unsigned NumPy integer K | PASS |
| Compact-fixture identity and frozen baseline | 1,560 columns / 78,000 windows, including 1,132 eligible columns / 56,600 eligible windows | PASS |
| Both original scalar slices | 62,394 per-column event/probe patterns; 3,119,700 label comparisons against the prior compact fixture | PASS |
| Complete empirical local profiles and certificates | 39 groups, 597,003 cells, 16,184,036 stored scalars | PASS |
| Independent column-local empirical construction | 52,357 literal local patterns / 2,617,850 window classifications | PASS |
| Every distinct reported optimum/tolerance threshold and law | 63 direct whole-dataset replays / 4,914,000 window classifications | PASS |
| Complete global dominance audit | 22,282,064 cells; 96 budget optima, 1,248 folder/mode optima (including infeasible/null cases), 156 minimum-tolerance optima | PASS |

These counts describe computational verification work, **not independent scientific samples**. Tolerance comparisons use the explicitly specified float64 expressions `L - epsilon <= U + epsilon`. “Exact” describes exhaustive enumeration of the finite classification arrangement, not arbitrary-precision real arithmetic.

The frozen baseline has no changed labels. The exporter supplies 105 observations that are accessible only by jointly relaxing the original two gates, and the input audit confirms they were retained. Missing feature values lock 821 windows to their original labels; none silently enter the varied guard.

## Interpretation checked against the outputs

At the primary epsilon = 0.2, the enlarged two-feature family reaches the same one beta^-2 folder as the ratio-only family; it reaches no beta^-1.2 folder. Under preservation of all 1,132 original finite boundaries, neither slope gains a compatible folder. At epsilon = 0.4, beta^-2 can retain the four baseline-compatible folders and add one with 150 whole-dataset edits and no original boundary loss. These are scoped family/decision results, not evidence that arbitrary classifiers cannot reverse a physical conclusion.

The absence of a joint-only empirical rescue is a valid negative finding. The computational advance is the exact feasible-region and obstruction audit plus its independent certificate checks. Existing threshold-search and interval geometry methods should be acknowledged; this audit alone does not establish a broad algorithmic novelty claim.

## Re-run

Run `exhaustive.py`, `validation_cases.py`, `fixture_audit.py`, `real_profiles.py`, and `global_optima.py` with the project Python. The latter two require completed `profiles/`, `results.json`, and `global_atoms.npz`. The input audit uses `prior_input_features.json.gz` when supplied, otherwise the existing sibling coherent-fragility fixture. Scripts only read production inputs/profiles and write their own verification reports. Each report records hashes of the files it checked. No raw archive, download, pipeline or Git remote is used.

## Arbitrary-edit lower-bound comparison

`lower_bounds.py` independently reclassifies the seven distinct classifier pairs among all feasible local-mode and global-budget witnesses. It decomposes additions and deletions separately for the target's eligible columns, the entire target folder, and the whole dataset. All **33 feasible local-mode witnesses** and **96 global-budget witnesses** pass. There are 126 matched unrestricted (`both`) lower-bound checks, 126 applicable deletion-only checks, and 26 addition-only checks (the latter are zero-edit witnesses and therefore qualify for both directions).

At the primary beta^-2 rescue, the target has **44 eligible deletions and no additions**, versus a previously certified arbitrary minimum of **5 unrestricted changes or 6 deletions**. The full target folder has 88 deletions; the complete dataset has 3,200 deletions, including 2,272 within original eligible columns. These denominators are distinct. No whole-dataset cost is presented as the price relative to a target-only arbitrary bound.

The prior arbitrary costs are imported from the earlier independently verified artifact; this check establishes consistency and scope matching rather than rerunning that earlier dynamic program. Run `lower_bounds.py` with `prior_arbitrary_reference.json` present. The compact baseline audit was rerun with the copied `prior_input_features.json.gz`, so the new artifact has no sibling-directory dependency when reproduced.

## Continuous-tolerance and rounding check

The threshold at which a boundary-preserving rescue first becomes possible is close to the chosen primary tolerance. The independently replayed A4 v3 witness uses `u = 0.8`, `v = 0.49326392706532574`; it changes **one target-folder label and 150 whole-dataset labels**, preserving all 1,132 eligible boundaries. Its minimum tolerance is **0.2055524847973138**, versus 0.40916524926568076 at the original classifier for that folder. The literal comparison fails at 0.205552 and succeeds at 0.205553. Thus absence of a rescue at 0.2 must not be portrayed as large physical robustness.

`tolerance_cutoff_checks.py` checks all 156 reported tolerance witnesses. It also bisects the ordered positive binary64 bit patterns to identify the first representable epsilon satisfying `L-epsilon <= U+epsilon`. For this A4 witness, the minimum is 0.2055524847973138 and the immediately preceding float, 0.20555248479731378, fails. All reported closed-form epsilon values pass the literal float predicate in this dataset. The numerical epsilon remains a **log-force compatibility tolerance, not independently calibrated force uncertainty**.

## Minimum fixed-subset obstruction certificates (post hoc explanation)

After seeing the joint-family results, the theory analysis searched for one fixed set of columns that blocks the law throughout the entire threshold rectangle. `obstruction_verify.py` independently checks **all 33,728 pairs** and the **5,246 triples** in the two cases where no pair suffices. It uses literal column-local classifier tables, not the theory rectangle construction. It also checks all **77 selected certificates**, containing **1,660 rectangles and 3,350 boundary claims**, on a separate arrangement of raw feature events and rectangle endpoints (75,062 probes, including strict-edge successors).

The minimum-cardinality conclusions pass: among the 38 incompatible beta^-2 folders, 37 admit a fixed obstructing pair and one requires a triple; among the 39 incompatible beta^-1.2 folders, 38 admit a pair and one requires a triple. A one-column global obstruction is excluded because each original eligible column has a valid force bracket at the baseline threshold pair. Triple minimality follows from the exhaustive absence of any obstructing pair.

The audit separately checks whether each pair/triple rules out the law numerically or through a lost boundary on each cell, and verifies the prior arbitrary-geometry stratification. In particular, the **25 beta^-2 folders already impossible under arbitrary labels remain separate**. The more informative restriction-of-rule comparison is the other 14: one is jointly feasible, 12 have a fixed obstructing pair, and one needs a triple. The 39 beta^-1.2 folders all had an arbitrary-label restoration, so their fixed-subset obstructions diagnose the declared coherent family rather than the already known force-grid impossibility.

These certificates are compact, locally inspectable explanations for family-specific incompatibility. They are not independently new physical evidence, and the post hoc subset search is explicitly distinguished from the earlier frozen joint protocol.
