# 고정12개 실제 학습 결과 — STOP

**제안 null-conditioning의 예측 개선 가설은 고정 기준을 통과하지 못했다.** 12/12개를 한 번 실행했고 승격0개다. null은 Markov보다0.1994% 낮은 손실이지만2% screen에 못 미치고, 같은 구조·감독의 무작위 방향 대조보다0.4779% 나빴다. 해당 추가 단계의 예측 보존은 확인됐으나 이를 성능 성공으로 바꾸지 않는다.

## 원래 주 점수

원train34명1442개·기존3personfold·invalid15개를 포함한 사람 macro9query log loss다. 낮을수록 좋다. 개선율은 Markov 대비 상대 손실 감소이며 음수는 악화다. 과거 reference는 새로 적합하지 않고 봉인된 예측을 재사용했다.

| 방법 | Log loss | Markov 대비 개선 | Brier | 최악 사람 loss |
|---|---:|---:|---:|---:|
| markov | 0.542080926 | +0.0000% | 0.181943112 | 0.705668740 |
| summary_hgb | 0.581721835 | -7.3127% | 0.190000474 | 0.848416212 |
| main_only | 0.541589277 | +0.0907% | 0.179636755 | 0.836682662 |
| ordinary_matched | 0.546119981 | -0.7451% | 0.181584123 | 0.820291573 |
| random_matched | 0.538426814 | +0.6741% | 0.178728177 | 0.835519647 |
| null_conditioned | 0.541000216 | +0.1994% | 0.179528472 | 0.835894854 |
| retrieval_query_group_dro | 0.533973795 | +1.4956% | 0.179137598 | 0.772501643 |

과거 joint_aux0.541763256, MDN0.570751189와 다른 retrieval arm도 [원 결과 JSON](RESULTS.json)에 모두 남겼다. 위 표의 과거 최선 queryDRO1.4956% 개선도 고정2% screen 미달이며 새 방법 성공이 아니다.

## 불리한 분할과 matched 비교

| 방법 | fold0 | fold1 | fold2 |
|---|---:|---:|---:|
| markov | 0.542957283 | 0.586568430 | 0.496637397 |
| main_only | 0.583924281 | 0.583223059 | 0.453771854 |
| ordinary_matched | 0.580073655 | 0.585569725 | 0.469629866 |
| random_matched | 0.583378947 | 0.574274446 | 0.453540493 |
| null_conditioned | 0.582464032 | 0.576483614 | 0.460283565 |

새 네 arm 모두 fold0에서 Markov보다 나빴다. 제안 null은 main_only보다 전체0.1088%/2fold, ordinary보다0.9375%/2fold 좋았으나 random보다 전체0.4779% 나쁘고 이긴 fold도1개뿐이었다. 원래2%/최소2fold screen과 세 matched arm 모두에 대한 추가 이득 조건은 유지했으며 결과 후 완화하지 않았다. 최악 사람 loss도 null0.835895 대 Markov0.705669로 높다. 두 최대값이 같은 사람이라는 뜻은 아니다.

## 연산은 작동했는가

실제 학습에서 null 추가 단계의 최대9query 확률 변화는9.796e-16, centered-readout projector의 최대 상대 residual은2.746e-15였다. Null의 최대 adapter condition은3.823으로 고정 실패 한도1e8 이내였다. 일반/무작위 step은 각각 최대8.042e-5/5.364e-5의 확률 변화를 보였다. 추가 연산의 구분과 수치 보존은 작동했으며, 학습 이득은 별개의 실패였다.

새 네 arm과 Markov의 기록된 coherence는 모두1이다. 따라서 coherence나 해당 step 보존만으로 제안법의 성능 우월성을 주장할 수 없다. Fine-step 평균 norm은 arm 자신의 counterfactual norm에 맞췄고, 달라진 학습 경로 사이의 실제 norm 완전 동일성은 주장하지 않는다.

## 실행·누수·비용

- 12/12개 완료, 실패/timeout0개. 결과 기록 직전 worker시간 54.059636초, ledger 최종 비용 54.085116초. 최종 기록 비용 때문에 두 값은 약0.025초 다르며 둘 다1800초 이내다.
- 네 모델7583parameter, 같은 초기값/73prefix/보조 감독/주 목적/배치 순서.40epoch/batch64/seed0/CPUfloat64, 개별180초 한도를 지켰다.
- 참조 모델 새fit0, 새 원음 처리/다운로드/standalone 수치 감사0. 코드37개·입력8개·개별 산출물60개를 봉인했다. 전12개 출력 봉인 뒤에만 held 사건을 채점했다.
- 모든 문턱/통계/label bin은 innertrain만 사용했다. 미래 mask/guard/center/target은 inference 인자가 아니다. Olddev/test 정답은 hash도 열지 않았다.
- Worker는 source 전체 라벨을 읽어 검증한다. Trainer의 train-label/held-prefix 분리는 함수 수준이며 OS 보안 격리가 아니다. Source는 반복 노출된 탐색이므로 독립 확증이 아니다.
- 독립 점수/봉인 검토는 EXECUTION_REVIEW.json에 별도로 기록한다. 개인/행별 예측·정답·가중치·trace·점수는 local/에 보존하며 공개하지 않는다.

## 의미와 한계

현재 예측을 바꾸지 않는 보조 이동을 구현할 수 있다는 것과 이후 학습 성능이 좋아진다는 것은 다르다는 반증 결과다. 새로운 정보·함수 용량을 추가하지 않는 factorization에서, 이 고정 감독/optimizer/분할 조건의 null 제약은 무작위 방향보다 유리하지 않았다. 왜 무작위 방향이 더 좋았는지, fine label 자체가 필요한지, 다른 optimizer에서의 효과는 이 네 arm으로 식별되지 않는다. 실패를 특정 원인의 인과 증명이나 모든 null-space/AI 방법의 불가능성으로 확대하지 않는다.

투영·재매개화·conditional auxiliary 자체는 기존 원리다. 높은 방법론 novelty, 독립적 예측 개선, 바이올린 음질·연주 향상은 입증하지 못했다. 마지막 실제 학습 batch를 사용했으므로 같은 batch를 구조하거나 학습 예산을 늘리지 않고 자동 학습을 일시중지한다.
