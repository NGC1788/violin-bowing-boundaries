# Integration review — v2.10

2026-10-21. Read-only review of the paper abstract and §3.15, Korean report §10.15, and the MLE bridge README against the theory, family outputs and independent verification records. No new fitting or simulation. The reported family counts, both triple conclusions, the A4 censoring qualification and the bounded novelty claims agree with the artifacts. Only the following corrections are needed.

1. **State the correct convex object.** In README §5, replace “검열 정규 우도는 볼록 문제가 된다” with “검열 정규 **음의 로그우도 최소화는 볼록 문제가 된다**.” The probability likelihood itself is not what the proof says is convex; log likelihood is concave and its negative is convex in the inverse-scale variables. The English paper's “inverse-scale convex parameterization” is acceptable.

2. **Avoid a universal absence-of-influence implication.** In the README opening, replace “작은 모순 집합이 기존 MLE의 결론을 지배한다는 주장은 성립하지 않았다” with “작은 모순 집합이 기존 MLE의 결론도 **자동으로 지배한다고 볼 수는 없었다**.” In Korean §10.15, similarly write “최소 모순 집합이 원래 MLE를 기각시키거나 지배하는 최소 관측 집합이라고 **자동으로 해석할 수는 없었다**.” Some selected deletions have large effects; the demonstrated result is lack of a guaranteed correspondence, not that every geometric certificate has no statistical relevance. A more precise English §3.15 title would be “Geometric certificates do not imply likelihood rejection or large deletion influence.” The abstract already has the appropriate scope.

3. **Correct the numerical upper bound.** README §5 says the independent NLL difference is “4.60e−11 이하.” `verification/control_audit.json` records `4.604316927725449e-11`, which is slightly larger. Use “4.61e−11 이하” or “약 4.60e−11”. This is only a rounding issue.

4. **Make the rank population explicit.** README §1 says “모든 다른 삭제 조합,” but every reference population includes the selected certificate itself, as declared in the plans and rank implementation. Use “모든 삭제 조합(선택 집합 자신 포함)” to avoid an off-by-one interpretation. The paper's “all same-cardinality deletions” is already consistent.

5. **Update section scope pointers.** The paper's data paragraph and limitations still say §§3.11–3.14 use the exposed 13-diagram cohort and have no unused confirmation set. Extend those references through §3.15, which reuses the same cohort. This prevents the new MLE bridge from appearing outside the explicit retrospective-data limitation. No additional physical experiment or independent validation set was added.
