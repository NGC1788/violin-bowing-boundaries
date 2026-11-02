# 고정12개 AI 예측 비교: 실행기와 전체 계약

이 폴더는 준비한 네 신경망을 기존 세 source 참여자 fold에서 비교하는 실행기다. AI·정보가 주 연구이며 바이올린은 검증 자료다. 원본 학습 코드는 수정하지 않는다. 높은 novelty나 성능 개선은 아직 입증되지 않았다.

이번 준비 단위는 합성 loader/controller 검사와 별도 소스 검토 뒤, 원train34명1442개 자료와 기존 source HGB 예측의 동일성을 확인해 **전체 실행 동결까지** 완료하는 범위다. 이 단위에서 실제 학습하지 않는다. 동결이 준비되면 다음 실행 단위에서 아래 명령을 한 번 실행한다.

```bash
cd /Users/jeongjihoon/violin-bowing-boundaries
.venv-mac/bin/python -m agentic_research.exact_count_runner_v1.runner check
.venv-mac/bin/python -m agentic_research.exact_count_runner_v1.runner run
```

이미 `local/state.json`이나 결과가 있으면 다시 실행하지 않는다. 실행 중이면 실제 process/worker lock과 상태를 확인한다. 현재 버전은 실패 후 resume/rescue를 지원하지 않는다. `freeze`는 이번 준비 단위에서 완료하며 반복하지 않는다. 동결된 코드·계약·입력을 바꾸어 실패를 만회하지 않는다.

## 비교와 누수 통제

- `joint_aux`, `mdn_aux`, `exact_count`, `moment_count`×3fold=12neural jobs.40epochs, batch64, seed0, AdamW1e-3, weight decay1e-3, auxiliary0.1, CPUfloat64를 고정했다.
- 모든 신경망에 같은 raw60에서 파생한73prefix정보, 같은 encoder 초기 파라미터, minibatch 순서, 미래 mask/contrast 보조감독을 준다. 자유 head의330추가 파라미터를 공개한다.
- 임계값·정규화·보조 path scale은 innertrain만 맞춘다. 실제 미래 mask/center/guard는 추론 입력이 아니다. Invalid15개를 유지한다.
- Source Markov는 평가 단계의 고정된3회 analytic count fit이다. 자식별 전처리에서는 이를 적합하지 않는다. HGB는 이전 source 예측3개를 재사용하고 새로 적합하지 않는다. 이 기준선 작업과 모든 입출력·검증·평가는1800초 worker 시간 안에 포함한다.
- 과거 HGB는60개 요약의 clip8 표현이고 신경망은73개 파생 표현이다. 동일 raw 정보에서 나온 것이지만 표현과 학습 감독은 다르다. 그래서 동일 감독의 joint/MDN 대조도 필수다. Markov/HGB 점수는 기존 source 결과와1e-12 이내로 재현되어야 한다.
- 각 child 예측/가중치/trace/전처리/receipt를 parent가 해시로 고정한다. **12개 모두의 parent output seal을 만든 뒤에만** held event 라벨을 구성해 점수를 계산한다. 끝에서도 다시 봉인을 확인한다.

각 fold의 규칙 문턱은 그 fold의 inner train 분위수다. 따라서 pooled 점수는 이 cross-fitting 절차에 대한 점수이며 하나의 보편적인 고정 문턱 문제의 점수가 아니다.34명을 독립 프레임 수로 부풀리지 않고 참여자 macro로 비교한다. Gaussian 경로의 조건부 iid 가정도 실제 겹치는50ms/10ms 기술자에서 검증되지 않았다. 코드의 정확성, 정합성이나 보조 likelihood 개선을 실제 미래 예측 개선으로 대체하지 않는다.

## 예산과 실패

시작 시 shared budget lock으로 batch와 job을 먼저 등록한다. 최대12job, job180초, worker1800초,2026-11-06 00:00KST를 넘지 않는다. Parent가 child lifetime을 소유하고 pending alarm·grace interruption에서도 종료·회수한다. 잠재적인2초 정리 여유는 성공 예산의 추가 허용이 아니며 초과한 job은 실패로 남긴다. 첫 실패나 timeout은 전체 batch를 종료하고 불완전 결과를 승격하지 않는다.

학습 곡선은 batch마다 모델이 변하는 **online within-epoch 가중 평균**이다. Epoch 종료 시 고정한 모델의 정확한 training loss라고 해석하지 않는다. 마지막/부분 가중치와 모든 trace는 private에 남긴다. 실제 큰 batch 비용·최적화 수렴은 이번 준비에서 시험하지 않았다.

실제 원음/행별 특징·정답·개인별 점수·예측·가중치·토큰·CLI 로그는 공개하지 않는다. 공개하는 것은 코드, 집계와 파일 해시다. 공개 결과도 역사적으로 노출된 source 내부의 탐색이며 독립 확인시험이 아니다. 모든 head와 불리한 fold를 보고한다. Free joint/MDN만 좋아진 것을 exact decoder의 승리로 바꾸어 해석하지 않는다.
