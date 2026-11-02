# 고정12개 donor-deletion 학습 결과 — 2026-11-03

**12개 학습을 완료했지만 사전 성능 기준과 제안 연산의 비교 기준을 통과하지 못했다.** 이 batch는 COMPLETE_STOP_THIS_BATCH이며 재시드·설정 수정·재동결·재실행하지 않는다.

| 방법 | 사람 macro log loss ↓ | Markov 대비 개선 | 최대 사람 loss ↓ | donor 삭제 risk ↓ |
|---|---:|---:|---:|---:|
| markov | 0.542081 | +0.0000% | 0.705669 | 해당 없음 |
| summary_hgb | 0.581722 | -7.3127% | 0.848416 | 해당 없음 |
| joint_aux | 0.541763 | +0.0586% | 0.862105 | 해당 없음 |
| mdn_aux | 0.570751 | -5.2889% | 0.838973 | 해당 없음 |
| retrieval_erm | 0.534838 | +1.3361% | 0.763597 | 0.554216 |
| retrieval_random_delete | 0.534964 | +1.3128% | 0.765998 | 0.554347 |
| retrieval_adversarial_delete | 0.535767 | +1.1648% | 0.741879 | 0.554240 |
| retrieval_query_group_dro | 0.533974 | +1.4956% | 0.772502 | 0.553419 |

일반 검색은1.3361%, query-group-DRO는1.4956% 낮은 손실을 보였다. 작은 탐색적 개선은 관찰됐지만 고정2% 기준 미달이며 승격 후보는0개다. 네 검색 방법은 모두 fold0에서 Markov보다 나빴고 fold1·2에서는 좋았다. Source34명은 과거에 노출된 자료여서 독립 확증이 아니다.

제안한 최악 donor 삭제는 일반 검색보다0.1736%, 무작위 삭제보다0.1499%, query-group-DRO보다0.3358% 손실이 높았다. 전자 둘보다 나은 fold는1개, query-DRO보다 나은 fold는0개다. 평균 donor 삭제 risk도 일반 검색0.554216 대 제안0.554240으로 개선되지 않았다. 최대 사람 loss는 일반 검색보다 낮지만 Markov보다 높았으며, 이를 사후에 새로운 성공 목표로 바꾸지 않는다. **제공자 삭제 연산의 추가 이득과 높은 방법론 novelty는 입증되지 않았다.**

## 수행·검증·비용

- 원train34명1442개/invalid15개를유지했다. 같은73derivedprefix/encoder/generator/aux/초기공통parameter/minibatch/memory/prior로4목적×3personfold를각40epoch·seed0·CPUfloat64에서한번학습했다. 기존Markov/HGB/joint_aux/mdn_aux는원봉인예측재사용이며새참조fit0이다. Parametrichead와memory자원·330parameter차이는남으며과거head를새로재학습한matchedcontrol로부르지않는다.
- 실제worker는82.473초였다. 각job은약6.48–7.18초로180초,전체1800초한도안이었다. 실제새학습예산3batch중2개사용·1개남음,standalone수치감사2/2소진이다.
- Parent는12job의예측/모든donor반사실예측/가중치/trace/stats/memory/영수증72개를고정하고OUTPUT_SEAL뒤held사건을평가했다. Held최악donor선택은봉인뒤보조risk집계뿐이며발행예측에는쓰지않았다. 새전처리/문턱/기억/학습에는innertrain만사용했고olddev/test정답을열거나hash하지않았다.
- [독립 재계산 검토](EXECUTION_REVIEW.json), [원 결과](RESULTS.json), [출력 봉인](OUTPUT_SEAL.json)을함께읽는다. 코드·집계·해시만공개하며실제기억·개인별자료·예측·가중치·trace는local/에보존한다.

## 실패에서 확인한 것과 확인하지 못한 것

새네검색모델은공동pattern확률로100%논리일관성을유지했지만일관성이성능우위를뜻하지않는다. 모든heldquery/donor/event셀에서raw positive-support소실비율은0이었다. 따라서이전Gaussian모델의표현불가능성과똑같은문제였다고설명할근거는없다. 각fold에서약20–22명의유효donor를사용해기억은이미넓게분산됐다. 제안은특히fold0에서최대donor질량을.08466에서.07305로낮췄지만그fold의주손실은.55682에서.55993으로높아졌다. 이는분산증가만으로정확도개선이따르지않았다는서술적결과이며인과기전의증명은아니다.

일반검색의개선이학습된거리/공통aux/경험적분포/제공자수보정중무엇때문인지는이비교만으로분해하지못한다. Epoch내online곡선도변화중인모델의평균이며종료가중치의train성능이나수렴증거가아니다. 낮은평균loss와더높은최악사람loss를함께보존한다.

## 다음 한 작업

[NEXT_POSTMORTEM_GATE.md](NEXT_POSTMORTEM_GATE.md)의저장산출물기반진단만수행한다. 새fit/forward/설정탐색/다운로드/추가수치감사없이같은사람가중치로개선과악화의기여·donor집중·rawsupport·기존곡선을검토한다. 이adversarial삭제batch를구조하지않고마지막학습예산은별도가설이정당화될때만쓴다. 근거가없으면학습을더돌리지않고성과/한계를정리한다.
