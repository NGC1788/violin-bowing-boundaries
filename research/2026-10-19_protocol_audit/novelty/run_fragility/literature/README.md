# Exact K-run law-compatibility editing: prior art and interpretation

Date: 2026-10-21. Author: Codex `run_novelty`. Scope: bounded primary-source review and mathematical interpretation; no empirical results inspected or generated. This supplements `../../criterion_identifiability/literature/README.md`.

## Finding

**Minimum Hamming correction to a regular language, exact fragility indices, and label-corruption certificates already exist.** The credible contribution here is a reproducible, structure-specific reduction from first-K-run labels to finite force-boundary intervals and then to exact compatibility with a specified power-law slope. The application supplies an auditable witness and a lower bound for a scientifically relevant, discontinuous analysis pipeline. It does not introduce dynamic programming or fragility analysis in general.

The bounded search did not locate the same combination of K-run force boundaries and profiled power-law compatibility. This is not proof of worldwide priority. A generic-algorithm novelty claim would require substantially more than the present construction.

## Primary sources closest to the construction

### 1. Regular-language correction already solves the column subproblem

Robert A. Wagner, “Order-n correction for regular languages,” *Communications of the ACM* 17(5), 265–268 (1974), DOI [10.1145/360980.360995](https://doi.org/10.1145/360980.360995). [Full author article in a university archive](https://www2.cs.sfu.ca/~anoop/courses/CMPT-755-Fall-2004/p265-wagner.pdf).

The article formulates minimum-edit correction to a finite-automaton language, gives a state-based dynamic program and reconstructs an optimal correction by tracing stored choices backwards. Insertions and deletions are permitted there; our force grid remains fixed and only substitutions are allowed. Thus finite-state dynamic programming and witness reconstruction are established machinery, not new contributions here.

Willem Jan van Hoeve, Gilles Pesant and Louis-Martin Rousseau, “On Global Warming (Softening Global Constraints),” [author full text, arXiv:cs/0408023](https://arxiv.org/html/cs/0408023), §4.1–4.2, especially Theorem 4.1. The expanded journal article is “On global warming: Flow-based soft global constraints,” *Journal of Heuristics* 12, 347–373 (2006), DOI [10.1007/s10732-006-6550-4](https://doi.org/10.1007/s10732-006-6550-4).

This is even closer: the authors explicitly minimize Hamming distance to a regular language through a shortest path in a layered directed graph. Theorem 4.1 states the exact minimum-cost correspondence; weighted transition penalties are also discussed. A first-K-run constraint is a particularly small regular language. Our column calculation should therefore be described as a specialized soft-regular dynamic program. The additional coupling through force intervals and a common regression intercept is the part specific to this research question.

### 2. Exact and restricted fragility indices are already available

Benjamin R. Baer, Mario Gaudino, Mary Charlson, Stephen E. Fremes and Martin T. Wells, “Fragility indices for only sufficiently likely modifications,” *PNAS* 118(49), e2105254118 (2021), DOI [10.1073/pnas.2105254118](https://www.pnas.org/doi/10.1073/pnas.2105254118). [Author institutional record](https://research-portal.st-andrews.ac.uk/en/publications/fragility-indices-for-only-sufficiently-likely-modifications/).

The paper gives an exact algorithm for incidence fragility indices, critiques one-direction heuristic calculations, and extends the fragility-index concept beyond binary outcomes. It also restricts changes by their plausibility. Thus neither “exact minimum number of changes” nor “allow only plausible changes” is by itself new. Our target is a deterministic compatibility predicate on extracted boundary intervals, rather than a clinical significance test. Without a calibrated label-error model, arbitrary Hamming edits must not be represented as plausible mistakes or their probability.

### 3. Certificates against label changes are not a new general concept

Elan Rosenfeld, Ezra Winston, Pradeep Ravikumar and Zico Kolter, “Certified Robustness to Label-Flipping Attacks via Randomized Smoothing,” *ICML*, PMLR 119:8230–8241 (2020). [Official paper page](https://proceedings.mlr.press/v119/rosenfeld20b.html).

This work certifies individual predictions against bounded training-label changes. Our certificate concerns an explicit downstream boundary/law analysis, with no learned predictor retraining. A claim to be the first label-flip robustness certificate would be false; the distinct object being certified must be stated.

Tamara Broderick, Ryan Giordano and Rachael Meager, “An Automatic Finite-Sample Robustness Metric: When Can Dropping a Little Data Make a Big Difference?”, [arXiv:2011.14999v5](https://arxiv.org/abs/2011.14999) (2023 revision).

AMIP approximates influential data removal using an influence function and supplies finite-sample bounds. It already motivates sensitivity of scientific conclusions to small changes. The K-run extraction is discontinuous; exact discrete editing avoids a local influence approximation for this specified target. This is a distinction of structure and guarantee, not proof that the application is generally stronger than AMIP.

### 4. The bowed-string literature already warns that analysis choices move exponents

Homayoon Beigi and Grace Conneely, “Learned Bow Control on a Measured Bowed-String Model: a Revised Minimum-Bow-Force Law, a Recurrent Controller, and the Domain of a Supervision Ceiling,” [arXiv:2609.14990v3](https://arxiv.org/html/2609.14990v3), §II-H (2026).

The authors explicitly describe multiple H/non-H crossings during a force scan. They report that time-window conventions move an A-string bow-position exponent from −1.14 to −1.58. They also discuss physical restrictions on regime classification. Our audit cannot claim to discover analysis dependence for the first time. What remains different is computing the smallest discrete classification change needed for a prespecified slope to become compatible with all selected boundary intervals, together with an explicit witness.

The slope −2 is the classical minimum-force reference in their equation (7). **−1.2 is our already-seen empirical reference, not their exact proposed law:** their alternative law has a β exponent of −1. A comparison with −1.2 must be called descriptive rather than an independent test of that paper.

## Precisely what the proposed calculation establishes

For eligible column i, let x_i = log β_i. For every admissible first-run start j, let c_ij be the minimum allowed Hamming edits giving that first K-run, and let [L_ij,U_ij] be its finite log-force boundary bracket. With fixed slope p and nonnegative log-force tolerance ε, the admissible intercept interval is

`I_ij(p, ε) = [L_ij − p x_i − ε, U_ij − p x_i + ε]`.

For an intercept a, column i costs

`g_i(a) = min { c_ij : a belongs to I_ij(p, ε) }`,

with minimum of an empty set equal to infinity. Independence of edits between columns then gives the exact profiled compatibility-restoration distance

`C(p, ε) = min_a sum_i g_i(a)`.

This reduction is our mathematical formulation, not a theorem copied from the preceding references. Every g_i is piecewise constant with changes only at candidate interval endpoints. Examining endpoints and the open cells between consecutive endpoints is sufficient. A continuous intercept profile therefore needs no arbitrarily fine intercept grid. A reconstructed minimum-cost sequence per column supplies a witness; exhaustive small-instance agreement and endpoint cases test implementation correctness.

### Opposite direction: the exact distance to incompatibility

The updated implementation plan includes the opposite decision direction as well. For a selected candidate interval in every column, joint compatibility fails exactly when the largest lower endpoint exceeds the smallest upper endpoint. Those extrema belong to two different columns when every individual interval is nonempty. This is the elementary one-dimensional case of the classical Helly intersection property, not a new geometric theorem.

Let `m_i = min_j c_ij`, and write candidate intercept endpoints as `l_ij, u_ij`. The exact distance to incompatibility is therefore

`D(p, ε) = min_{i ≠ k, r, s : l_ir > u_ks} [c_ir + c_ks + sum_{h not in {i,k}} m_h]`.

Proof: every incompatible assignment has such a disjoint pair and incurs at least the displayed pair cost plus the independent minima in other columns. Conversely, any such pair can be joined with those other minimum-cost assignments; the pair already makes the whole intersection empty. Thus the lower bound is attained. For the planned originally finite columns, leaving the labels unchanged is admissible, so every m_i=0. An endpoint sweep can find the cheapest disjoint pair while excluding states from the same column. If there is no admissible disjoint pair, the opposite target is impossible within the declared support and edit restrictions; that is not universal physical robustness.

The baseline verdict determines which distance constitutes a reversal: use C when initially incompatible, and D when initially compatible. Both quantities may still be reported explicitly. They certify a declared all-interval compatibility decision, not the old MLE confidence-interval decision.

The theorem certifies the specified finite analysis problem. It does not establish that any edited label is physically right, that the resulting bracket contains a physical transition, or that the slope is a physical law. A first run is an operational boundary; short H islands and nonmonotonic regimes remain possible.

## Interpretation requirements and mathematical pitfalls

1. **Choose the distance using the baseline verdict.** C is the distance *to compatibility*, while D is the distance *to incompatibility*. If the original labels already satisfy the rule, C=0 means no restoration is needed; it does not mean zero changes reverse a conclusion. Report D for reversal in that case. A report that only implemented the intercept profile would establish C, not D; the separate disjoint-pair computation and its tests are required.

2. **This changes the decision rule.** The old interval-normal MLE confidence interval and the new “all selected brackets intersect a law tube” criterion are different. A certificate for the latter does not certify reversal of the former, even when both are called law rejection. Keep old frozen scores and inferential claims unchanged.

3. **Tolerance is a declared resolution allowance.** ε=.2 in natural log-force corresponds to a multiplicative tube approximately [0.819,1.221]. It is not a 95% interval or a calibrated measurement uncertainty. ε=0,.1,.4 sensitivities are useful, but choosing ε afterwards to make a law pass would invalidate its intended interpretation.

4. **Finite support prevents trivial escape.** If left-censored states are allowed for every column, changing at most the first K labels per column to H creates an all-left-censored map. Then an arbitrarily negative intercept makes any finite slope compatible. Likewise, destroying every K-run can permit an arbitrarily high intercept under a no-band-as-right-censored convention. These constructions reveal information loss, not evidence for either law. Fixing original finite-positive columns and requiring edited finite-positive boundaries is a defensible primary estimand.

5. **Selection still limits the population.** The finite-positive subset is selected by the original labels. Report the number and β span of omitted columns and each reason, and keep membership fixed during optimization. A conclusion then concerns those originally resolved columns, not the full playability map. Do not drop newly difficult columns to obtain a smaller cost.

6. **Geometry can be infeasible without saying anything about physical truth.** At a fixed slope and tolerance, the measured force range may offer no admissible finite boundary combination. Separate “no finite-grid solution” from a numerical failure and from a large edit cost. Do not describe this as infinite physical robustness. An unrestricted-label geometry check distinguishes this from impossibility caused by an add-only/delete-only restriction.

7. **Model sensitivity remains.** A common intercept within each speed folder avoids using cross-session speed contrasts to determine p, but β-linked acquisition order, grid coverage, threshold artifacts, and physical model inadequacy remain. There is no causal identification of β from the calculation.

8. **Check endpoints and ties.** State whether force brackets are closed or half-open. At ε=0 a singleton intersection can be optimal. A sweep that evaluates only midpoints can miss it. Duplicated forces and zero-width brackets need an explicit convention; do not silently invent a positive interval width. Exact means combinatorially exhaustive under declared floating-point comparison rules, not symbolic exact arithmetic on measured reals.

9. **State the perturbation set.** Both-direction, add-H-only, and delete-H-only costs answer different questions. Each exact minimum should include label count, changed-column count, H additions/deletions, and an auditable witness listing original data identifiers and force ranks. Unit Hamming weights do not say all errors are equally likely. Weighted costs remain a possible extension but are known soft-regular machinery, not new in themselves.

10. **A restricted robustness radius is valid.** For an initially incompatible case with finite C>0, no permitted change of fewer than C labels restores compatibility; equivalently C−1 is a certified exclusion radius for this target and admissible state set. Analogously D−1 certifies a currently compatible verdict. These statements are deterministic and conditional. They are not population confidence levels or classification-accuracy estimates.

11. **Do not equate count with practical robustness.** A single label near the first run can move a boundary far, while many interior labels can have no effect. Give force displacement and β coverage alongside totals. Counts from overlapping conditions or multiple tilts are not independent replicates. Separate folder-level descriptive comparisons from any across-string statistical test.

## Recommended contribution statement

“We quantify the dependence of operational bowed-string force-law comparisons on regime classification. A soft-regular dynamic program gives the exact minimum label-edit cost for each admissible first-run boundary. A continuous intercept profile and a disjoint-interval-pair calculation then compute the minimum edits required to reverse compatibility with a specified power-law slope in either direction. The computation returns an explicit witness and separates label sensitivity from finite-grid infeasibility.”

This is a defensible **specialized computational audit**, with meaningful value if it distinguishes robust differences from fragile classification artifacts in the actual maps. At present it does not support a claim of a new physical minimum-force law, a new general robust-learning algorithm, or first-ever exact fragility computation. Successful implementation and small-instance verification are prerequisites; this literature note alone establishes none of the empirical findings.

## Search and access record

Searched combinations of minimum Hamming correction, regular language / soft regular constraint, exact/generalized fragility index, label-flipping certificates, power-law label changes, and interval stabbing. Opened full primary articles for Wagner, van Hoeve et al. and Beigi, and official/author pages for Rosenfeld and AMIP. PNAS full text opened once; a subsequent find request redirected to a cookie page, so the publisher abstract and author institutional record were used for the summarized generalization. The preceding literature note covers interval censoring and partial identification; those established topics were not presented as novel here. No repository/process changes or empirical runs were made.
