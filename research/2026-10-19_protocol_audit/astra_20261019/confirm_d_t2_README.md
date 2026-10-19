# D_T2 두 단계 확인 실행기 — 실행 전 준비본

2026-10-19 작성. **실제 D_T2 feature·cache·archive는 열지 않았고, 확인 채점은 실행하지 않았다.** 네트워크·다운로드·프로세스 제어도 하지 않았다. 원래 `final/CONFIRM_PLAN_D_T2.md`와 동결 계수·receipt는 변경하지 않았다.

`confirm_d_t2.py`는 사용자가 이후 자료 처리를 승인하고 기존 단일 파이프라인에서 체크섬·공간·feature 생성을 확인한 뒤 사용할 실행기다. 다운로드나 원시 신호 처리를 대신하지 않는다.

## 단계와 파일

1. `freeze-controls`: 명시적으로 전달된 feature CSV(.gz 가능)에서 folder/trial/window_s/t/force/speed/beta/error **제어 열만 파싱**한다. H·브리지 특징은 요청하지 않는다. 3개 속도, 속도별 trial 1~2000, 네 시점의 완전한 격자, 누락·중복·오류, 40×50 β 열을 검사한다. 동결된 dec5_pl 기술자와 예측값을 계산한 뒤 controls receipt와 별도 `.seal.json`을 독점 생성한다.
2. `score`: 원래 동결 문서/계수/source hash, 추가 classifier 의존성, 실행기, 입력 SHA256, controls receipt seal을 재검증한다. `.score_claim.json`을 독점 생성한 **뒤에만** 브리지 특징 4개를 읽어 동결 classifier의 H 판정을 계산한다. 같은 receipt로 출력 이름을 바꿔 재채점할 수 없다. 성공·검증 불가·예외를 최초 result 파일에 남긴다.

CSV 전체의 SHA256은 불투명한 원시 바이트를 읽어 입력 동일성을 확인한다. 이는 outcome 열의 파싱·통계·라벨 계산과 다르다. 제어 열만의 parsed hash도 별도로 저장한다. outcome은 새로 계산하며 이미 저장된 H 열이 있더라도 읽지 않는다.

모든 파일은 `open('x')`로 새로 생성하며 기존 결과를 덮어쓰지 않는다. score claim을 지우고 다시 실행해 실패를 숨기면 안 된다. outcome 접근 후 실패했다면 해당 실패 기록을 보존한 상태에서 별도의 명시적 수정 계획을 먼저 작성해야 한다.

## 향후 사용 형식 — 지금 실행하지 않음

아래 `FEATURES`에는 기존 파이프라인이 검증한 해당 표본 feature 경로를 직접 지정한다. 입력을 자동 탐색하거나 다운로드하지 않는다. 출력 부모 디렉터리는 미리 존재해야 한다.

```bash
cd "$HOME/violin-rne"
FEATURES="/검증된/feature/파일.csv.gz"
PYTHON="$HOME/violin-bowing-boundaries/.venv-mac/bin/python"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PYTHON" astra_20261019/confirm_d_t2.py freeze-controls \
  --features "$FEATURES" --out astra_20261019/D_T2_controls_frozen.json
```

첫 receipt의 inventory·descriptor·외삽 여부·예측값과 seal 저장이 끝난 다음에만 별도 명령으로 채점한다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PYTHON" astra_20261019/confirm_d_t2.py score \
  --controls-receipt astra_20261019/D_T2_controls_frozen.json \
  --out astra_20261019/D_T2_confirm_result.json
```

## 원 계획에 미명시되어 관측 전에 정한 실행 사양

- 원 동결 hash는 `data.py`에 연결된 실제 classifier·A2 threshold summary·β columns 함수까지 전이적으로 포함하지 않았다. controls 단계의 추가 receipt에 세 파일과 실행기 SHA256, 실제 source threshold를 묶어 두 단계에서 검증한다. A2 summary가 하나로 결정되지 않으면 멈춘다. 이것은 원 계획을 바꾸는 재동결이 아니라 실행 의존성을 명시한 부록이다.
- `compare.py`, `data.py`, 다른 연구 스크립트를 통째로 import하지 않는다. import 시 CLI 파싱, 전역 경로 변경, 데이터 로드 부작용을 피하려고 hash가 검증된 파일에서 필요한 `columns`, `from_columns`, `classify` 함수 AST만 추출한다. top-level 코드가 실행되지 않는 회귀 테스트가 있다.
- lower level은 동결한 A1+A2 reference에 대한 offset이다. strict upper는 동결한 전체 Z 포함 식에 대한 residual offset이며 주 오차는 그 절댓값이다. no-Z 비교는 같은 상수에서 log Z만 빼므로 residual에 log Z를 더해 계산한다. empirical secondary에는 reference의 `c`까지 포함한다.
- 수준 likelihood는 기존 normal interval likelihood와 동일하고 극단 꼬리에서 안정적인 log-CDF 차분으로 계산한다. 기존 `confirm_A_T2.offset_fit`과 같은 **한 번의** Nelder–Mead 초기값 `[0, log(.5)]` 및 scipy 기본 종료 설정을 쓴다. 결과를 보고 시작값을 교체하지 않는다. `log sigma` ±50은 부동소수점 overflow 방지용 수치 guard이며 ±49 접근도 유한 수준의 성공으로 인정하지 않는다.
- 모두 한쪽 검열, 관측 가능한 구간 없음, 0폭/역전 구간, 모든 residual interval의 공통 교집합에 따른 sigma→0/수준 미식별, 최적화 실패는 `검증 불가`다. 임의 finite level이나 3.0 penalty를 대입하지 않는다.
- β 순서 40개를 같은 ordinal 위치끼리 세 속도에서 함께 묶는다. 40개 cluster를 복원 추출해 **2,000회, seed=20261019**를 쓴다. CLI에서 횟수/seed를 바꿀 옵션은 없다.
- bootstrap은 **동결 예측에 조건부**인 관측 수준 민감도다. 학습 계수·descriptor 추정·새 세션 불확실성은 포함하지 않는다. 이 해석을 결과에 명시한다. 실패 재표집 수·이유를 공개하고 성공한 재표집의 percentile 95% 구간만 표시한다. 실패를 빼고 얻은 구간을 무조건 coverage 보장으로 부르지 않는다.
- 판정은 원 계획의 0.23/0.12 실용 허용오차·대조군 margin만 사용한다. bootstrap 구간이나 경험적 secondary로 주 분석 실패를 뒤집지 않는다. 외삽 descriptor도 숨기지 않고 별도 표기한다.
- 미검출 아래 경계의 오른쪽 검열 규약, 1e-3 force clamp, K=2 다른 긋기 이웃 규칙은 원 계획대로 유지한다. β-힘 띠가 전혀 없을 때 물리적 경계가 높은 힘 쪽에 있다는 증명은 아니며 미검출 수를 따로 기록한다.
- K=3, t=.1/.3, 공칭 힘 축 보조 분석은 이 실행기에서 **아직 구현하지 않았다**. 따라서 전체 연구 계획이 모두 실행 준비됐다고 주장하지 않는다. 이 항목은 별도 보조 실행기가 필요하며 결과에 `secondary_variants_deferred`로 남긴다.

## 검증

실제 D_T2 자료 없이 synthetic-only 테스트 **20개 통과**. 함수 선택 import 부작용, extreme-tail likelihood, 한쪽 검열/σ-collapse 미식별, 고정 단일 optimizer 시작점/실패, 2,000개 cluster draw의 세 속도 짝 유지/seed 재현, frozen 식과 판정, outcome 미요청, 누락·중복·오류, hash 변조, receipt 변조, 독점 생성, 단일 사용 score claim, synthetic 6,000-stroke 두 단계 처리와 실패 기록을 확인했다.

end-to-end synthetic test에서만 bootstrap 2회를 내부 호출했고, 실제 CLI의 2,000회 채점은 실행하지 않았다. 실제 feature의 형식·결측/검열 분포·모든 bootstrap 최적화 수렴은 자료 접근 전에는 검증할 수 없다. 해당 단계가 실패하면 실패 기록을 보존하고 `검증 불가`로 보고해야 한다.
