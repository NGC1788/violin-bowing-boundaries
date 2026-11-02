# 활현 AI 연구: 재현 가능한 미래 예측과 에이전트 루프

**새요청에따라넓은탐색재개·24추가학습완료:** [discovery_v1](discovery_v1/RESULTS.md), [discovery_v2](discovery_v2/RESULTS.md). 원래source34명만쓴표현/감독/attention조합탐색이며olddev/test정답은재사용하지않았다. 신경망은HGB보다좋았지만Markov를못넘었고novelty는미입증이다. 새 [경로분포후보](discovery_v2/NEXT_EXPLORATION.md)와 [가설은행](discovery_v1/IDEAS.md)을준비했다. 종료된보정분기를다시돌리는것이아니며각완료batch와부정결과를보존한다.

**조건부 보정 방향도 검증 후 종료:** [가능성 감사와최종판정](correction_feasibility_v1/DECISION.md). 사후개발한사람제외보정에서단순전역혼합은Markov보다2.62%낮은손실을보였지만상태별혼합은전역보다1.10%나빴다. 유효한위험인증·독립확증·단순기법을넘는신규성도미확보여서이후속은STOP이다. 추가모델탐색없이기존성과와실패를보존한다.

**후속 원인 분석 완료:** 저장모델의train/dev복원으로일반화격차와확신오류손실집중을확인했다. 새학습·시험재평가없이 [진단](violin_diagnosis_v1/CAUSE_ANALYSIS.md), [직접선행과새방법후보의가능조건](violin_diagnosis_v1/NEXT_METHOD.md)을정리했다. 높은방법론신규성과실제성능개선은아직미입증이며,추가학습에앞서통계적타당성부터검토한다.

**공개 바이올린 `violin_raw_v1`의 고정 12개 실험과 시험 평가까지 완료했다.** 제안 offset 신경망의 시험 log loss는 0.664192로, 개발 자료에서 선택한 Markov 기준선 0.552413보다 20.23% 높았다. 성능 개선과 높은 방법론 신규성은 입증되지 않았다. [바이올린 전체 결과](violin_raw_v1/RESULTS.md), [검증 기록](violin_raw_v1/FINAL_AUDIT.json)을 보존하고 캠페인은 STOP, 자동 루프는 PAUSED로 닫았다.

기존 첼로 `forecast_v3` 고정 12개도 완료·STOP이다. 공동 신경망은 보정 HGB보다 손실이 3.335% 낮았지만 독립 신경망보다 0.185% 높았다. 정확 질의 v1/v2와 보정 solver 감사도 종료했다. 아래 실행 예시와 계획은 재현·이력 설명이며 완료한 연구를 자동으로 반복하라는 지시가 아니다.

## 바로 시작

- **다른 컴퓨터·Claude·Codex·기타 AI:** [실행 안내](START_HERE.md)
- **AI에게 그대로 주는 문서:** [공통 인계 프롬프트](HANDOFF_ANY_AI.md)
- **실제 결과와 다음 작업:** [실행 현황](docs/EXECUTION_STATUS.md), [작업 큐](docs/WORK_QUEUE.md)
- **12개 실험과 독립 재계산:** [결과](published_results/forecast_20261030_v3/RESULTS.md), [독립 검토](published_results/forecast_20261030_v3/FORECAST_REVIEW.md)
- **입력 데이터:** [약9MB 재현 패키지](benchmarks/forecast_v3/README.md), [출처·CC-BY-4.0](benchmarks/forecast_v3/DATA_CREDITS.md)

```bash
# 현재 Mac의 완료 상태 확인 — 이 캠페인은 STOP을 유지한다.
bash agentic_research/scripts/run_forecast.sh status --campaign forecast_20261030_v3
```

새 머신은 `START_HERE.md`에 따라 새 캠페인을 초기화·동결한다. 예측 실험은 CPU에서 동작하며sudo·GPU가 필요하지 않다. LLM 없이 고정된 후보를 실행하거나, 다른 챗봇이 작성한 JSON을 수동으로 받아 진행할 수도 있다.

## 무엇을 검증하는가

54,000개 A/B자료로 학습하고24,000개 C/D자료에서,0.20초까지의60개 특징으로0.25–0.45초의 관측 규칙을 예측한다. 단순 persistence·source Markov·HGB와 독립/공동 신경망을 같은 입력에서 비교한다.9개 판정 규칙에 가능한10개 공동 패턴을 표현하는 모형과 기존 확률 coherence 보정도 비교한다.

이것은 물리적 참 상태·음질 점수가 아니라 고정 관측 규칙의 미래 출력이다. plateau landmark의 전체 신호 의존성 때문에 엄격한 실시간 예측으로 주장하지 않는다. 기존 첼로 데이터는 연구자가 이미 본 자료이며, 새 분할로 독립 확인시험이 되지 않는다.

- [동결된 실험 계약](forecast_v3/configs/forecast_v3.json)
- [데이터·시간 누수 감사](forecast_v3/DATA_AUDIT.md)
- [가장 가까운 선행과 확률 보정의 한계](forecast_v3/COHERENT_REPAIR_REVIEW.md)
- [최신 기술 표적 문헌 검토](docs/LITERATURE_REVIEW.md)
- [실제로 받은 바이올린 자료의 감사](forecast_v3/violin_data_audit/README.md)

## 자동 루프 구조

```mermaid
flowchart LR
    A[자료·누수 감사] --> B[가설·비교군·예산 고정]
    B --> C[테스트·코드와 입력 동결]
    C --> D[제안 AI]
    D --> E[별도 검토 AI]
    E --> F[해시·예산·중복 검사]
    F --> G[단일 Python worker]
    G --> H[봉인된 점수·반박 결과]
    H --> D
    H --> I[예산·후보 소진 또는 실패 시 종료]
```

현재 설정은12개 실행, 최대worker1800초, 최대18제안라운드,2026-11-06 00:00 KST까지다. LLM의 승인으로 평가 규약을 바꿀 수 없다. Claude·Codex·수동 JSON 어댑터는 같은 후보 해시와 request_sha를 사용한다. 동일 OS 사용자를 신뢰하는 로컬 무결성 장치이며 완전한 보안 sandbox는 아니다.

사용자의 앱에는30분 간격 반복 작업이 별도로 있다. 다른 컴퓨터에 GitHub를 복제했다고 그 앱 작업까지 복제되지는 않는다. 수동 종료는 캠페인STOP과 앱 자동화 일시중지를 함께 한다. 잠금은 캠페인별이므로 서로 다른 캠페인을 같은 자원에서 무분별하게 병렬 실행하지 않는다.

## 보존한 실패와 연구 이력

- [전체480열의 개선 상한5.83%](docs/FULL480_RESULTS.md)
- [독립 수학 감사](docs/independent_audit/SCIENTIFIC_AUDIT_20261030.md)
- [참고한 Loop-Research-for-MaskedKD의 코드 감사](docs/REFERENCE_LOOP_AUDIT.md)
- [v1/v2의 과거 실행 설명](PILOT_V1_V2_MANUAL.md)

기존 모델을 조합하거나 에이전트를 많이 호출했다는 사실만으로 새 방법론이나 높은 성능이 입증되지는 않는다. 실제 점수·강한 기준선·직접 선행·다른 자료에서의 적용 결과를 함께 평가한다.
