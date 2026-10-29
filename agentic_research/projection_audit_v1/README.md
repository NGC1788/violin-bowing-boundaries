# 고전 확률 보정의 수치 수렴 감사

**실행 완료:** [256건 결과와 한계](RESULTS.md). 사전 gap 조건은 50/256건만 통과했으므로 참조 solver 전체를 충분히 수렴했다고 주장하지 않는다.

이번 단계는 새 모델을 학습하거나 개발 정답으로 점수를 다시 매기는 실험이 아니다. v3의 근사 KL 보정이 수렴 부족 때문에 원예측으로 돌아갔는지 확인하는 작은 수치 감사다. 기존 v3 소스·규약·결과는 변경하지 않는다.

## 결과를 보기 전에 정한 범위

HGB seed 0과 독립 신경망 seed 0/1/2에서 **같은 ID 64개씩, 256개 예측 벡터**를 검사한다. ID와 고정 문자열의 SHA256 순서로 선택한다. 라벨·오류·그룹·기존 보정 성공 여부는 선택에 쓰지 않는다. NPZ에서 읽는 배열은 `ids`, `probabilities`, `repaired_probabilities`뿐이며, 관측 정답과 행별 성능 지표에는 접근하지 않는다.

기존 64회 반복 알고리즘의 후보와 저장된 최종 보정 예측을 분리한다. 후자는 원예측 fallback일 수 있으므로 비일관적이면 최적화 목적함수의 feasible upper bound로 취급하지 않는다.

## 참조 계산

허용 출력 행렬을 V, 그 확률 질량을 w라 하면 p = wV, w ≥ 0, Σw = 1이다. 목표는 `mean KL(Bernoulli(p_j) || Bernoulli(clip(q_j)))` 최소화다. `q`의 score clipping은 기존 평가기와 같은 1e-6이다.

SLSQP를 두 개의 고정 시작점에서 실행한다. 최대 1,000회, ftol 1e-13, 각 질량 하한 1e-12다. 각 후보의 질량을 하한·정규화로 정리한 뒤 실제 p로 목적함수와 gradient를 다시 계산한다. 정리 전 위반과 최대 변경량을 기록한다. 최종 gradient 계산에서는 p를 clip하지 않는다.

전체 simplex 위의 Frank–Wolfe gap `g·w − min(g)`는 볼록성 아래 목적함수 오차의 상한이다. 질량 하한을 둔 최적화 결과도 전체 simplex 기준 gap으로 검사한다. float64 gap ≤ 1e-8, feasible 오차 ≤ 1e-10, 가능한 열 가지 출력 모두에서 계산된 clipped loss 변화 ≤ 0일 때만 최종 보정을 채택한다. 아니면 원예측을 반환한다.

이는 **수치적으로 확인한 오차와 열 가지 출력의 유한 검사**이며 정확 산술·interval arithmetic 인증이 아니다. `solver.success`만으로 정확 최적해라고 부르지 않는다. 기존의 고전 최적화 기법이며 새로운 학습 방법이나 정리가 아니다.

## 재현과 기록

`protocol.json`, Python 소스·테스트, 기존 의존 소스 및 입력 예측 파일 해시를 `freeze_manifest.json`에 고정한 뒤 한 번 실행한다. 기존 12개 모델의 예측 파일은 원 연구의 로컬 `forecast_v3/runs/`에 보존되어 있다. GitHub에는 그 전체 예측 파일을 포함하지 않았으므로 다른 머신에서는 먼저 v3를 재현하고 별도 경로·새 감사 버전에서 입력 해시를 고정해야 한다. 공개 준비 NPZ만으로 이 감사 명령이 즉시 실행되는 것은 아니다.

```bash
.venv-mac/bin/python -m pytest agentic_research/projection_audit_v1 -q
# 동결·실행은 원 머신에서 이미 한 번 수행한 뒤에는 반복하지 않는다.
.venv-mac/bin/python agentic_research/projection_audit_v1/run_audit.py freeze
.venv-mac/bin/python agentic_research/projection_audit_v1/run_audit.py run
```

동일 디렉터리의 두 번째 실행은 `run_receipt.json`으로 거부한다. 600초 제한, 2026-11-06 00:00 KST 종료일을 적용한다. 한 시작점이 실패하면 해당 오류를 기록하고 다른 시작점을 시도한다. 둘 다 실패하면 실패 ID·이전 완료 행을 보존하고 전체 감사를 중지한다. 좋은 행만 남기지 않는다.

참조 계산은 행별 두 시작점이고 기존 계산은 작은 batch이므로 실행 시간 비율을 실서비스 처리량 비교로 과장하지 않는다. 참조 보정이 좋아져도 관측 정답을 읽지 않았으므로 신경망 사이의 성능 순위가 바뀌었다고 말할 수 없다. 기존 후보에 대해 더 강한 보정 비교가 필요하다는 정도만 판단한다.
