# Final scientific/editorial review of MLE bridge synthesis

2026-10-21. Reviewed the top-level README against production summary/selected/baseline results, coherent-rule replay, whole-family results and review, theory, independent verification records, and both rendered PNG figures. No new fits were run and no source outputs were edited. This reviewer implemented production and therefore this is **editorial/scientific cross-checking, not an additional independent numerical implementation**; numerical independence is provided by `../verification/`.

## Verdict

The substantive interpretation passes: minimum uniform geometric obstructions do not generally identify observations dominating the original full-data MLE. The text accurately preserves the different estimands, original censoring, overlapping cases, post-hoc status, family-selection caveat, and lack of physical/causal or calibrated hypothesis-test claims. Figures are readable, untruncated and correctly distinguish a single-speed hard compatibility calculation from all-three-speed likelihood results. The finite control rank's diagonal is explicitly a visual reference, not a null test.

Three small wording/numerical corrections were sent to the integrating agent before publication:

1. Section 1 says “모든 다른 삭제 조합”, but the reference distribution explicitly **includes the chosen certificate itself**. Prefer “선택 집합 자신을 포함한 모든 삭제 조합”. This makes the reported exact tie/rank convention visible.
2. “모집단·판정 대상이 다르다” is better as “분석 대상 관측 집합·판정 대상이 다르다”. The difference is an observed eligibility/censoring set, not a newly sampled statistical population.
3. Section 5's “NLL 차이 4.60e−11 이하” is microscopically below the saved maximum `4.604316927725449e-11`. Use `4.61e−11 이하` or “약 4.60e−11”. This does not affect any result.

A useful optional clarification in the fixed-scale likelihood bound is `g>0` and “the two columns share one speed and the same residual distribution after subtracting the candidate law.” The linked theory already states these assumptions correctly. It is a bound on **absolute fixed-model NLL**, never on T or a p-value.

## Counts and quantifiers checked

- 13 diagrams ×120 rows =1,560; 1,132 finite,315 left-censored,113 no-run/right-censored. The original force sorting, first K=2 run and1mN clamp remain in production; zero-width intervals are rejected, not silently widened.
- 77 previously fixed cases separate into52 classifier-specific and25 force-grid-infeasible cases. The selected experiment's52-case median absolute beta effect .00613, directions17/35, rank.542, span-matched rank.347, top5% count2, and T directions37/15 agree with `production/summary.json`.
- The complete family contains7,742 memberships =7,718 pairs+24 triples. The classifier-specific family has3,599 memberships;12 cases have every member move beta away,6 have every member increase T, and31 contain both T directions. These are different quantifiers and are reported correctly.
- Both exact-minimum-three cases contain24 triples total:19 A1/v3/p=-1.2 and5 B_T2r1/v1/p=-2. All24 effects have the stated unfavorable directions. This supports a genuine failure of universal transfer, without claiming that no pair/certificate can ever influence MLE.
- The best-T family selection is an explicitly outcome-aware explanation choice. Its46/52 positive cases do not prove enrichment, predictive superiority or calibrated selection performance. The README states this limitation.
- A4 shared-rule witness: full-data beta−1.41687275→−1.40459447; p=-2 T32.216078→34.944025; two intervals change, including one left→finite. Finite85→86,left27→26,right8→8. Attributing the full MLE change to only the one originally eligible target-folder boundary would be wrong; the README explicitly avoids that.
- Whole-rule maximum absolute beta changes .0154731 for150 label changes and .286897 for3,200 label changes agree with the saved replay. Original eligible-boundary losses are reported separately and not discarded from MLE.
- Verification counts779 independent optimizations =425 baseline/selected/block/rule fits+354 control refits;60,234 independently evaluated objectives;22,110 exhaustive control subsets;462 rank comparisons. These are computational checks, not experimental samples.

## Theory and novelty interpretation checked

The symmetric finite-interval construction is a valid exact counterexample: the free fit admits the fixed target slope as a global optimum despite hard incompatibility; T=0 follows from the symmetry/log-concavity argument rather than numerical rounding. The text should not infer unique identification, but currently does not require it.

The Gaussian interval likelihood's canonical `(eta=theta/sigma,tau=1/sigma)` convexity is established prior art and is presented as verification machinery. Numerical parameter agreement/gradients do not become rigorous finite objective-gap certificates. The README explicitly distinguishes exhaustive discrete enumeration from numerical optimization.

The deletion decomposition is an algebraic identity, not a new attribution theorem. Its direct term is already multiplied by2; the compact Korean formula is consistent with the two numeric terms−25.245 and−18.987. Neither term is a causal share or probability.

Generic case-deletion analysis, minimal inconsistent sets, interval MLE, and likelihood ratios are not claimed as inventions. The narrow contribution is the verified domain-specific bridge and demonstrated separation between rule-uniform waveform/boundary certificates and the original censored law fit. This remains a defensible methods component of an acoustics study; it does not yet establish new physics, violin generalization, or a predictive benchmark improvement.

## Figure review

`fig_mle_influence.png`: quantities, effect directions, the extreme B-repeat2 case and52-case denominator match the text. The right-panel ranks are of absolute beta changes among same-folder size-matched controls; they are not T ranks. Both the axis label and prose agree. Span-matched ranks are table-only and are not misrepresented as plotted.

`fig_geometric_vs_mle.png`: zero-based bars show the favorable hard-tolerance change and the simultaneous unfavorable beta-distance/T changes. Panel A explicitly uses one speed and originally finite columns; B/C use all three speeds and120 censored rows. The heading labels150 as a global count. No significance marker, confidence interval or causal interpretation is introduced.
