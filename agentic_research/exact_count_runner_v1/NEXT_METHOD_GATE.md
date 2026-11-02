# 다음 한 작업: 별도 AI 방법론 설계 검토

현재 고정12학습·결과/원예측 검토·사후 용량 진단은 완료했다. 같은 exact_count_runner_v1 또는 frozen learning 원본을 수정·재실행·재동결하지 않는다. `RESULTS.md`, `EXECUTION_REVIEW.json`, `CAPACITY_DIAGNOSIS.json`을 먼저 읽는다.

## 다음 단위의 범위

별도 `method_redesign_v1/`에서 **서면 방법론 선택 한 단위**를 수행한다. 새 fitting/forward/원음·NPZ·개인별 정답 접근/다운로드/standalone 수치 감사는 0이다. 공개 집계와 기존 코드·직접 선행 원문만 사용한다. planning/worker lock·Git·실제 state·선점을 확인한 뒤 작업한다.

1. 실패에서 확인된 정보와 확인되지 않은 설명을 구분한다. exact는 정규화/고정 σ하한/clip의 표현·기울기 결함이고, moment 실패의 원인은 아직 확정되지 않았다. 단순 하한 변경·더 긴 학습·seed 교체를 새 방법이나 신규성으로 포장하지 않는다.
2. 최대3가설을 비교한다. 원인 기반의 scale-equivariant 조건부 잔차/경로 학습 가설을 하나의 **미채택 후보**로 검토할 수 있지만 단순 scale-family MDN/기존 residual forecaster와 다른 학습 원리가 있는지 먼저 반박한다. 다른 축의 학습 목표/표현 가설도 허용한다. 한 분기의 실패가 모든 탐색 불가능성이라는 결론은 금지한다.
3. 후보당 가장 가까운 직접 선행 최대2개(총6개 이내)를 primary source로 확인한다. 유명 기술 이름의 조합보다 수식·학습 정보·불변성·확률 출력·계산비에서 어떤 차이가 있는지 표로 비교한다. 오래된 원리와 최신 선행 모두 존중하고 검색 범위/남은 불확실성을 기록한다.
4. 유망한 **최대1개**만 다음 준비 후보로 고른다. 핵심 학습 연산, 추론 입력, 손실, 새로운 정보 이득 가설, 최소1개 명확한 반박 대조를 명시한다. prefix만의 변환·innertrain-only 통계·invalid 유지·같은 미래 감독을 받는 directjoint/MDN·강한 Markov/HGB 대조를 확보한다. 목표 문턱의 표현 가능성과 clip dead-gradient 문제를 학습 전에 확인하는 규칙을 제시한다. 이 규칙 자체는 기존 용량 진단이며 novelty 주장이 아니다.
5. 별도 검토를 받아 `JUSTIFIED_TO_PREPARE` 또는 `NO_JUSTIFIED_CANDIDATE`를 남긴다. 전자는 다음 **별도** 코드/누수 기능시험·계약 동결 준비로 넘어가며 이번 단위에서 학습하지 않는다. 후자는 다른 과거 캠페인을 되살리거나 끝없는 준비를 반복하지 말고 남은 가설·한계를 정리해 자동화를 일시중지한다.

## 새 실행이 정당해진 경우의 이후 경계

최대2actualbatch×12job만 남았다. batch1800초/개별180초, 시작·실패도 ledger에 소비한다. 별도 가설/공정 대조/기능시험/독립 검토/전체 입력·환경·계약 동결 이후에만 새 batch를 한 번 실행한다. source34명은 노출된 탐색이고 olddev/test 정답hash도 금지다. 같은 parameter floor 수정 시험을 근거 없이 새 novelty 캠페인으로 위장하지 않는다. 성능 상승 가능성이 있어도 새 학습법의 차이가 없으면 그 차이를 정직하게 기록한다.

## 완료물

`method_redesign_v1/DECISION.md`(실패→후보→선행 대비 차이→반박 조건→채택/기각), `PRIOR_REVIEW.md`, 검토 기록. 결과·실패·다음 한 작업을 최신 WORK_QUEUE/EXECUTION_STATUS에 남긴다. 새 가중치/데이터 다운로드·추가 결제 없이 진행한다. 기존 예산/2026-11-06 00:00 KST 종료일을 늘리지 않는다.
