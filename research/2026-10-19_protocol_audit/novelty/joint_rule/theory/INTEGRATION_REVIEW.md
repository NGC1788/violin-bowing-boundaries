# Final interpretation and integration review

2026-10-21. Read-only review of `README.md`, the abstract/new §3.14/limitations of `paper/draft_v2.md`, and §10.14 of `final/최종보고서.md`, compared with the saved obstruction and tolerance outputs. Only this review file was written. No source, scientific result, profile or shared document was changed.

## Verdict

**The integrated scientific interpretation is consistent with the verified outputs. No blocking numerical or logical error was found.** The new section appropriately describes a retrospective, classifier-family-specific computational result rather than physical truth or a new independently validated prediction method. A stale verification-status sentence and a few local clarity corrections remain below.

## Checks passed

- **52 versus 77:** The 25 p=−2 failures already impossible on the force grid are explicitly separated. The classifier-specific comparison consists of 13 p=−2 plus 39 p=−1.2 failures, with 50 minimum fixed pairs and two minimum triples. Across all failures the totals are 75 pairs plus two triples = 77.
- **Pure numerical obstructions:** The integrated statement says a boundary-loss-free certificate **exists** for 50/52 classifier-specific failures. This correctly refers to some qualifying fixed subset, not necessarily the lexicographically selected pair. The stored lexicographic choices are pure numerical in 51 of all 75 pair cases; existence of some pure numerical pair holds in 73 of 75. Both selected triples are pure numerical. These different counts are not confused in the integrated documents.
- **Pointwise versus uniform:** §3.14 and the Korean sections distinguish a pair that can depend on the threshold from one fixed pair over the entire domain. The two triples serve as measured counterexamples to exchanging these quantifiers. There is no unsupported universal cardinality-three claim.
- **Minimum cardinality:** Baseline feasibility excludes singleton obstructions. Exhaustive pair checks justify minimum three in the two exceptional cases. The quantity is explicitly β-column count, not label edits, specimens, or causal factors.
- **Cells versus evidence:** 597,003 local cells and 22,282,064 common cells are identified as arrangement cells, neither independent observations nor necessarily distinct label vectors.
- **No-loss tolerance:** The A4 v3 threshold (.8,.49326392706532574), one target-label change, 150 global changes, no invalid original eligible boundary, baseline δ=.40916524926568076 and minimum δ=.2055524847973138 agree with `verification/tolerance_cutoff_checks.json`. Its next lower float fails. The manuscript correctly avoids calling the ε=.2 result strong physical robustness and does not interpret δ as calibrated sensor uncertainty.
- **Scope and errors:** The paper limitations and artifact README explicitly state that the fixed subsets identify incompatible constraints, not erroneous observations or physical causes. No diagnosis of which column is wrong follows from a certificate.
- **Verification completion:** `verification/obstruction_verify.json` has `passed: true`, all 33,728 pairs, 5,246 triples, 1,660 certificate rectangles and 75,062 coverage probes. `reproduction/reproduction_report.json` reports `status: passed`, 162.208 seconds, original-source integrity and all scientific comparisons passed.
- **Novelty:** The paper acknowledges prior threshold/rectangle/constraint-explanation methods, geometric multiverse work and the absence of physical ground truth. It does not claim a new generic optimization primitive, a new physical law, or an ICML-level novelty guarantee. The claimed candidate contribution is appropriately the exact composition and interpretable downstream diagnostic.

## Corrections to apply

1. **Stale status, definite correction.** `theory/obstruction_summary.md:44` still ends with “별도의 검증 에이전트가 … 전체 주장 재검증을 진행한다.” Verification has now passed. Suggested replacement: “별도 독립 검증도 완료됐다. 생산 및 구조 oracle을 호출하지 않고 모든 쌍·삼중 조합과 77개 인증의 1,660개 사각형·75,062개 영역 및 끝점 검사를 다시 확인해 통과했다. [독립 검증](../verification/REVIEW.md).” The preceding 2,391,827/8,265 counts are the construction-side checks and can remain; do not replace them as though they were the same verifier.

2. **Tolerance coordinate clarity.** `paper/draft_v2.md:236` defines L,U as “shifted intercept endpoints.” This is numerically correct when “shifted” means subtracting p log β, but could be read as already including ε. Replace with “the largest lower and smallest upper intercept endpoints **before epsilon expansion**.” The formula then cannot be mistaken for double-expanding the tolerance.

3. **No-loss protocol timing clarity.** `paper/draft_v2.md:275` says “Its no-boundary-loss finding is a post hoc condition at epsilon=.2.” The §3.13 no-loss diagnostic was added post hoc, whereas the same condition was included in the local before-cost plan for §3.14. Suggested wording: “The no-boundary-loss condition was added post hoc in §3.13 and included in the local before-cost plan for §3.14; both analyses reuse exposed data. Section 3.14 locates a no-loss transition already at approximately .205553.” This preserves the overall retrospective status while accurately distinguishing the local protocol history.

4. **p=−1.2 minimum wording.** `README.md:53` gives .227326/.290378 after a paragraph about per-folder minima. Add “폴더들 중 가장 작은” before “최소 δ” to make clear that these are the minimum across folders, not the minimax tolerance jointly reconciling every folder. The preceding text already states that folder minima can have different optimizing threshold pairs; this is a small clarification rather than an incorrect result.

5. **Standalone Korean errors caveat.** §10.14 is accurate, but unlike the artifact README and paper limitation it does not explicitly say the minimal subset identifies no erroneous observation. Add after its scientific-scope sentence: “이 집합에 포함된 열이 잘못 측정되었거나 잘못 분류되었다는 뜻은 아니며, 물리적 원인도 특정하지 않는다.” This matters if that section is read alone.

6. **Editorial punctuation.** In `paper/draft_v2.md:22`, “(§3.12) a coherent …” is missing a comma after §3.12. No scientific effect.

## Nonblocking submission consideration

The closest direct joint-threshold prior art, Herbold, Grabowski & Waack (2011), is documented in `theory/LITERATURE.md` but absent from the paper's references. The current novelty wording already admits joint threshold optimization is established, so this is not an overclaim. When shortening the paper for submission, consider citing that direct prior work alongside the broader robustness and constraint-explanation references. Do not expand the claim to the first geometric multiverse method; the 2026 preprint is already acknowledged.

The remaining scientific limitation is substantive rather than a document defect: a small exact obstruction set under an uncalibrated operational predicate does not establish which measurement is correct, how to improve classifier accuracy, or performance on new strings. The present integration states those limits clearly.

## Root resolution

All listed wording corrections were applied before the final snapshot. The missing comma had already been corrected. Code, numerical results and reproduction records were not changed.
