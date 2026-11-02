# 정확 count 기반 AI 학습 방법: 코드 준비 단계

이 폴더는 미래 관측 누락을 주변화하는 신경망의 **학습 준비 코드**다. 실제 원자료·행별 정답을 열거나 학습·예측 평가하지 않았다. 기존 beam 제안은 실패로 보존하며 다시 실행하지 않는다. 물리 법칙 발견이나 음질 평가가 주 기여가 아니다.

`model.py`의 신경망은73차원 prefix 입력에서 두 잠재 regime, 미래 최초 관측 확률, 두 구간의 마스크 전이, 구간별 contrast 및 두 scale을 출력한다. 정확 batched count DP와 기존 비중심 chi-square 미분 연산을 연결해9개 미래 사건 확률을 계산한다. 실제 미래 마스크를 추론 입력으로 받지 않는다.

네 모델은 같은 encoder와 보조 generator, 같은 미래 mask/경로 contrast 정답을 받는다. 자유 joint10패턴 head, 직접 lognormal mixture head, exact count head, conditional-variance moment approximation head다. 자유 head의 추가 파라미터는 숨기지 않는다. MDN의 미래 유효 확률도 실제 guard가 아니라 같은 mask generator의 DP에서 예측한다.

## 이번 준비에서 추가한 것

- Batch DP와 Gaussian contrast·scale을 통과하는 미분. 기존 수치 감사의 mask-logit 미분만 확인한 범위를 넘어 **이 학습 코드의 기능**을 검사한다.
- [미래 레벨을 버린 정규화된 contrast likelihood](INVARIANT_OBJECTIVE.md). 중심화한 프레임을 독립 표본처럼 계산하지 않고 자유도 \(n-1\)를 사용한다. Invalid pitch도 mask/power 감독에 남는다.
- Inner-train-only 표준화·문턱·prefix 상태와 unclipped moment anchor. 미래 레벨의 녹음 간 분산이 noise 단위를 지배하지 않도록 pooled within-record contrast scale을 사용한다.
- 추론과 정답 변환을 분리한 API. `forward(prefix, thresholds)`에는 실제 미래 경로·mask·guard가 없다.

Gaussian contrast와 nuisance 제거도 고전 원리다. 기능시험 통과는 높은 신규성, 수치 가속, 실제 사람 예측 성능을 입증하지 않는다.50ms/10ms 겹치는 프레임의 조건부 iid 가정은 여전히 한계다.

## 현재 허용된 기능 시험

```bash
.venv-mac/bin/python -m pytest --import-mode=importlib -q agentic_research/exact_count_learning_v1/test_model.py
```

이 시험은 합성 입력만 사용하며 optimizer step이 없다. 별도의 대규모 수치 감사나 설정 탐색이 아니다. 완료한 과거 감사2회를 재실행하지 않는다.

`CONTRACT_PROPOSAL.json`은 아직 실행 계약이 아니다. `READINESS_FREEZE.json`도 `execution_authorized=false`인 코드 준비 봉인이다. 현재 이 폴더에는 실제 자료 loader나 학습 실행기가 없다. 다음 작업은 [실행기 관문](NEXT_RUNNER_GATE.md)이며, 실제 자료와 fitting은 별도 전체 실행 동결 이후다.
