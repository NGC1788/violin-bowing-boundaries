# Final integration review: draft v2.8 and Korean report

2026-10-21. Reviewed the coherent-fragility README; English abstract, §3.13, discussion, limitations and final internal notes; Korean abstract, §10.13 and surrounding status paragraphs. No shared documents, code or results edited.

## Verdict

The newly integrated results are materially consistent with the stored analyses and preserve the necessary scope limits. The 5/6/44 distinction, target-versus-global costs, uncalibrated stress families, same-setup limitation, ε=.2-specific no-loss result and ε=.4 counterexample are correctly stated. The large validation counts are explicitly distinguished from independent scientific samples. No new physical-law or classifier-accuracy overclaim was found in the new abstract or §3.13/§10.13.

## Corrections before freezing the snapshot

1. **Stale untested statement, Korean §10.12 (`final/최종보고서.md`, currently line 583).** “하나의 일관된 특징 판정 규칙으로 그 증거를 만들 수 있는지는 검증하지 않았다” is now contradicted by §10.13. Replace the paragraph opening with: “이 절은 임의의 독립 라벨 변경을 허용한다. 특정 일관된 특징 판정 규칙의 도달 가능성은 다음 10.13절에서 별도로 검증했다. 따라서 이 절의 비용은 그 제한 문제에 대한 하한이며 실제 오분류 개수는 아니다.” This preserves the distinction between unrestricted and coherent problems without presenting completed work as pending.

2. **Citation specificity, English §3.13 final paragraph.** “Multiverse and specification-curve analysis ... [25]” cites only Steegen et al.'s multiverse paper. Either change this to “Multiverse analysis already addresses processing choices [25]” or also cite Simonsohn, Simmons & Nelson (2020), *Specification curve analysis*, DOI 10.1038/s41562-020-0912-z. The latter source is already documented in the literature memo. This is attribution precision, not a change in the result.

3. **Updated cohort clarity, Korean abstract / §2.** The first paragraph still describes the research as 10 diagrams/60,000 strokes; the newly central exact analyses use 13/78,000. Mark 10/60,000 explicitly as the *initial* development cohort and add a sentence such as: “10.11–10.13절의 후속 분석은 기존 자료에 A3·A4·D2를 포함한 13개 도표·78,000개 창을 사용하며, D_T2는 제외한다.” The existing §10.12 denominator is correct, but the opening summary and table currently leave this scope transition implicit. Do not add the two sample counts together as independent new observations.

4. **Changed columns versus changed boundaries, README result paragraph.** “24개 β 열 중 21개가 바뀌고” counts columns containing at least one changed label; it is not a separately computed count of moved boundary locations. Prefer “24개 대상 β 열 중 21개에서 적어도 하나의 라벨이 바뀌고”. The English §3.13 wording already correctly says label changes across 21/24 columns.

## Minor inherited status wording

Korean §2 still says D2/B_T2s2/high-force exposure must be judged by comparing freeze and processing times. The opening status and §11 correctly say all applicable frozen targets have been scored. If the chronology audit is complete, point the §2 sentence to the actual §10.3–10.4-2 exposure accounts rather than retaining a future obligation. If it is still an author-level provenance audit, label it that way; do not imply the analyses have not run.

English §3.12 is not stale: it says arbitrary witnesses *need not* be realizable, which remains true. The new discussion and internal “still to do” list correctly distinguish completed scalar-rule verification from unsearched classifier families and independent physical validation.

## Root resolution

All four requested corrections applied before final snapshot. The 13-diagram scope was checked directly against all group identifiers in input_features.json.gz and also made explicit in English §2. The minor inherited chronology audit sentence remains an author-level provenance check; no experimental completion is delayed by it. No scientific output or verified code was changed.
