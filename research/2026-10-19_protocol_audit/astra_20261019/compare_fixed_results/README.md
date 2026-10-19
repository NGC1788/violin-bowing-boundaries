# 기존 경계 비교 코드의 사후 수정 검증 (2026-10-19)

A1+A2로만 학습·slope 교정하고 기존에 이미 본 B1, C1, D1, B_T2r1/r2를 t=0.2 s에서 재평가했다. D_T2는 읽지 않았다. 이 결과는 새 확증 실험이나 성능 우위의 증거가 아니다.

## 수정 내용

- 확률모델의 0.5 band는 각 열의 실제 측정 force 최솟값~최댓값 안에서만 추출했다. support 끝까지 양성이면 경계값을 만들어 넣지 않고 좌/우 검열로 기록했다. 여러 양성 구간은 기존처럼 hull로 묶되 개수를 남겼다.
- NaN·빈 band를 slope 계산에 섞지 않는다. 빈 band는 측정 범위에서 p=0, invalid band는 제외 수를 공개한다. source loss가 일정하면 slope 미식별로 표시한다. 기존의 임의 3.0 log-error 대입과 비유한 경계 자동 적중을 없앴다.
- 오차는 유한 관측구간과 point prediction이 함께 있는 조건부 값이다. 전체 적중 수·실패 수·관측 검열·예측 검열 수를 별도로 공개한다. 검열구간이 겹치는 것은 point 적중으로 세지 않는다.
- 공개 Z 버전과 target H 파형에서 유도한 과거 Z 버전을 분리했다. 후자는 엄밀한 무표적 정보 예측으로 부를 수 없는 진단용이다.
- Triv-5-finite는 confirm_A_T2.py의 동일 속도·source별 β 최근접 5개 유한구간 중점 정의다. Triv-5-legacy-edge-diagnostic은 robustness의 β 이웃 ±2와 검열 edge 대입 정의를 재현한다. 서로 다른 baseline이다.
- 기존 few-shot 결과는 재사용하지 않았다. 비유한 경계의 임의 대입과 교정에 쓴 stroke의 평가 중복을 함께 수정해야 한다.

## 평균 조건부 경계 오차와 평가 가능 수

오차는 표본별 median 다섯 개의 평균, N은 다섯 표본의 평가 가능한 열 수 합계(최대 600)다. 서로 다른 N의 숫자만 비교해 우위를 주장하면 안 된다.

| 모델 | 아래 오차 | 아래 N | 위 오차 | 위 N | 예측 검열 L/U | 빈/invalid band |
|---|---:|---:|---:|---:|---:|---:|
| P-Sch-publishedZ | 0.6468 | 376 | 0.1949 | 259 | 0/0 | 0 |
| P-Beigi-publishedZ | 0.3063 | 376 | 0.1949 | 259 | 0/0 | 0 |
| P-free-publishedZ | 0.2679 | 376 | 0.1552 | 259 | 0/0 | 0 |
| P-Sch-targetDerivedZ-diagnostic | 0.5251 | 376 | 0.2544 | 259 | 0/0 | 0 |
| P-Beigi-targetDerivedZ-diagnostic | 0.3231 | 376 | 0.2544 | 259 | 0/0 | 0 |
| P-free-targetDerivedZ-diagnostic | 0.3217 | 376 | 0.2018 | 259 | 0/0 | 0 |
| D-logit | 0.3067 | 363 | 0.2580 | 181 | 52/333 | 68 |
| D-gbm | 0.2878 | 364 | 0.1692 | 241 | 50/279 | 43 |
| Triv-5-finite | 0.2687 | 376 | 0.1882 | 259 | 0/0 | 0 |
| Triv-5-legacy-edge-diagnostic | 0.2663 | 374 | 0.1759 | 259 | 0/0 | 35 |

## 남아 있는 한계

기존 K=2 연속 H 셀 규칙, 단일 band hull 가정, 관측 force 1e-3 clamp, 정규 interval likelihood, 4개 시간창으로 데이터 모델 학습/한 시점으로 physics 경계 학습을 유지한 코드 수정이다. H band가 전혀 없는 열의 아래 경계를 오른쪽 검열로 놓는 기존 convention은 물리적으로 유일하게 정당화되지 않으며 `no_detected_observed_band`로 별도 집계했다. likelihood의 새로운 타당성 증명은 아니다.

GBM의 기존 내부 무작위 행 validation 및 같은 stroke의 여러 시간창 의존성도 유지했다. 별도 모델 선택·튜닝을 하지 않았다. 연산 optimizer success와 finite 결과를 확인했고 target scoring 전에 source parameters·slope·예측 파일을 저장했지만, 이 다섯 표본의 과거 노출은 사라지지 않는다.

검열된 예측을 버리면 조건부 오차가 좋아질 수 있다. 그래서 전체 failure/coverage, 전체 관측 대비 point-hit, 모든 모델이 point를 낸 공통 열의 오차까지 results.json에 함께 기록했다. 공통 열 역시 선택된 subset이므로 일반 성능 주장으로 확대하지 않는다.

현 데이터는 제어된 steady-state 최소 힘을 직접 측정한 것이 아니다. force가 진행 중 변하고 speed와 session이 얽히므로 이 비교가 물리적 분기·CSD·보편적인 최소 힘 법칙을 확립하지 않는다.

파일: `results.json`(모든 열·적중·검열·실패 수), `predictions.json`, `native_probabilities.npz`, `frozen_before_scoring.json`(코드/캐시 SHA256, source-only slope losses, optimizer 진단).

## 실제 검증 결과

23개 합성 회귀 테스트를 통과했다. 12개 physics optimization 모두 수렴했고 경고는 없었다. 과거 Z physics 계수는 원래 파일과 1e-12 이내로 같으며, D 모델의 native IoU·log-loss도 1e-14 이내로 같다(실제 최대 LL 차이 1.7e-16). 경계 처리만 수정한 것을 확인했다. 500개 집계 항등식과 support 밖 유한 D 경계가 없는지도 검사했다.

source slope는 D-logit 2.75, D-gbm 3.75이다. source의 빈 band 28열/17열은 고정 p=0으로 정상 반영되었다. Triv-5-legacy-edge는 source 14개 invalid band(700창)를 제외한 사실을 기록했다.

다섯 test에서 실제 H band가 검출되지 않은 열은 46/600이다. D-logit은 빈 band 68/600, 아래 support 검열 52/600, 위 support 검열 333/600이며, D-gbm은 각각 43/600, 50/600, 279/600이다. Triv-5-legacy-edge는 invalid band 35/600이다. 빈 band는 측정 support 안에 양성 예측이 없다는 뜻이며 support 밖 물리적 경계 위치를 알려주지 않는다.

다음 공통 열 결과도 추정 모델 모두가 두드러진 측정 범위에서 point를 낸 열만 골랐으므로 선택 편향이 남는다. 표본별 median 평균을 다시 표기했다.

| 모델 | 공통 아래 오차 | 공통 아래 N | 공통 위 오차 | 공통 위 N |
|---|---:|---:|---:|---:|
| P-Sch-publishedZ | 0.6274 | 356 | 0.1877 | 181 |
| P-Beigi-publishedZ | 0.3017 | 356 | 0.1877 | 181 |
| P-free-publishedZ | 0.2680 | 356 | 0.1804 | 181 |
| P-Sch-targetDerivedZ-diagnostic | 0.5160 | 356 | 0.2506 | 181 |
| P-Beigi-targetDerivedZ-diagnostic | 0.3386 | 356 | 0.2506 | 181 |
| P-free-targetDerivedZ-diagnostic | 0.3220 | 356 | 0.2221 | 181 |
| D-logit | 0.3044 | 356 | 0.2580 | 181 |
| D-gbm | 0.2870 | 356 | 0.1762 | 181 |
| Triv-5-finite | 0.2683 | 356 | 0.2036 | 181 |
| Triv-5-legacy-edge-diagnostic | 0.2630 | 356 | 0.2099 | 181 |

따라서 현재 산출물은 비교 코드의 신뢰도를 높이는 수정이다. 새로운 방법론, 보편적 physics 법칙, 또는 통계적으로 검증된 모델 우위가 확립됐다는 결론은 내리지 않는다.
