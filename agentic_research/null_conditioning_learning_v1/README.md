# 함수보존 conditioning: 실행 전 준비

**실행 완료 — STOP:** 아래 준비를 마친 뒤 고정12개 실제 학습과 독립 재계산까지 완료했다. null은 Markov보다0.1994% 낮은 loss지만2% screen 미달이며 random보다0.4779% 나빴다. 추가 step 예측 보존만 확인됐고 제안 성능 가설은 기각이다. [결과](RESULTS.md), [독립 검토](EXECUTION_REVIEW.json), [이번 탐색 종료 기록](CLOSURE.md). 마지막 학습 batch까지 사용했으므로 아래 준비·실행 명령은 이력이며 재실행하지 않는다.

AI 학습 방법 후보의 반증 실험이다. 미래 변동량을 prefix에서 예측하며 바이올린은 적용 자료다. [서면 가설/직접 선행/한계](../research_reframe_v1/README.md)를 먼저 읽는다. 선형 null-space 투영은 기존 원리이며 높은 novelty나 실제 성능 개선은 아직 없다.

## 같은 모델에서 추가 step만 비교

73차원 prefix→32 GELU→32 GELU의 encoder u를 사용한다. Coarse head는 `A(Wu)+b`이고 W는 bias 없는 identity 초기화32×32 선형 층이다. 공통 path/mask generator는 **u**에서 분기하며 fine head는 Wu에서65개 logits를 낸다. 네 arm 모두7583개 parameter와 같은 초기값/학습 label/minibatch/주 손실을 사용한다.

주 AdamW step은 기존9query BCE+0.1 common auxiliary이며 fine head를 optimizer에서 제외한다. 그 뒤 encoder/readout/generator를 고정한 fine SGD step을 적용한다. 학습 label은 train-only octile의64개 cell+invalid이고 실제 coarse fibre 안에서 conditional CE를 계산한다. 경계의 `<=` 규칙과 원래 quartile 값을 유지한다.

| arm | adapter의 추가 변화량 |
|---|---|
| main_only | 0, fine head만 같은 감독으로 학습 |
| ordinary_matched | 일반 SGD 제안을 자신의 null 제안 norm에 맞춤 |
| random_matched | 같은 rank의 무작위 방향 투영을 자신의 null 제안 norm에 맞춤 |
| null_conditioned | 고정 centered readout nullspace에 실제 SGD 제안 투영 |

Fine SGD는 lr.001·weight.1, momentum/decay/추가 clipping0이다. Main AdamW는 lr.001·decay.001·clip5,40epoch/batch64/seed0/CPUfloat64를 사용한다. 수치 설정은 실제 자료/fit을 보기 전에 제안대로 정했으며 설정 탐색을 하지 않는다. Rank cutoff1e-10, relative projector residual1e-8, 추가 null-step 확률 drift1e-10, W norm1e4/condition1e8이 실패 sentinel이다. 실패한 W를 재정규화해 구조하지 않는다.

Null 추가 step의 확률 보존은 해당 단계에만 해당한다. W는 입력 정보나 함수 용량을 추가하지 않는다. 효과가 있다면 이후 최적화 경로의 효과이며 일반화 보장은 없다. Norm matching은 각 arm 내부 규칙이다. 학습 후 arm끼리 실제 step norm이 동일하다는 주장은 하지 않는다. Random 방향은 label과 독립인 고정 RNG로 만들되 rank는 해당 readout에 조건부다.

## 누수·기록·비용

원train34명1442개만 사용한다. 기존3personfold, invalid15개를 유지한다. 모든 문턱/표준화/label bin은 innertrain에만 적합한다. 추론 API는 prefix만 받으며 현재 future mask/guard/center/target/fine label을 인자로 받지 않는다. 실제 음원 처리나 새 다운로드는 없다. Olddev/test 정답은 hash도 열지 않는다.

모든 새 모델은 같은 fine/path/mask 감독을 받는다. 과거 Markov/HGB/joint_aux/MDN/retrieval 예측은 sealed reference로만 재사용하고 새 refit0이다. 과거 reference는 추가 fine 감독이나 새 adapter가 없는 경우가 있어 matched causal control은 새 네 arm이다. HGB60clip8 대 신경망73derived 표현 차이, 과거 joint/MDN4414 대 새7583 parameter 차이를 공개한다. 새 모델의 inference memory는0이고 fine head는 배포 시 제거할 수 있다. W는 coarse readout에 합칠 수 있지만 wall-clock 가속을 측정한 것은 아니다.

주/fine 단계 시간, inference 시간, artifact 크기, epoch별 main 손실과 fine step drift/norm/rank/conditioning을 private로 저장한다. Online 평균은 endpoint/convergence 증거가 아니다. Parent가12개 모든 prediction/weight/trace/stat/receipt를 고정한 뒤에만 held 사건을 채점한다. 개인/행별 산출물과 가중치는 `local/` 밖에 공개하지 않는다.

## 승인·실행 경계

합성 기능시험과 별도 소스 검토 뒤에만 `freeze`로8개 allowlist 입력을 확인할 수 있다. 그 과정은 source 준비이며 실제fit0이다. 전체동결 후 별도 검토자가 `FREEZE_REVIEW.json`을 승인해야 `run`이 ledger 시작에 도달한다. 새 실제 학습은 다음 실행 단위이며 준비 중에는 `run`을 호출하지 않는다.

```bash
.venv-mac/bin/python -m pytest --import-mode=importlib agentic_research/null_conditioning_learning_v1/test_model.py agentic_research/null_conditioning_learning_v1/test_runner.py -q
```

마지막 실제 학습은 네 arm×3fold12job 한 번뿐이다. 개별180초/전체1800초, 첫 실패·timeout STOP, 실패도 소비하며 재시드/epoch/bin/cutoff/보조 가중치 변경으로 구조하지 않는다. 주 score는 사람 macro9query log loss다. Markov/HGB 대비2%·최소2fold screen에 더해 null이 세 matched arm 각각보다 전체와 최소2fold 좋아야 연산의 추가 이득 후보로 인정한다. Fine label 필요성·OGD/HNC 일반 우월성은 이 네 arm으로 식별되지 않는다. Source는 반복 노출된 탐색이므로 성공해도 독립 확증이 아니다.
