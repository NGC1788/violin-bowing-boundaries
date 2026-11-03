# 다음 한 단위: 동결한12개 실제 학습

이 문서는 준비 완료·전체동결·FREEZE_REVIEW 승인 뒤에만 적용한다. 그 전에는 실제 run을 호출하지 않는다. 같은 캠페인의 state/결과가 이미 있으면 재실행하지 않는다.

1. 최신 WORK_QUEUE/EXECUTION_STATUS, 실제 process/planning·worker lock, Git, `local/state.json`, 공유 budget ledger를 확인한다. 다른 작업자가 실행 중이면 중복 실행하지 않는다.
2. `.venv-mac/bin/python -m agentic_research.null_conditioning_learning_v1.runner check`로 소스/환경/8개 입력/규약/검토 봉인을 확인한다. `execution_review()`도 실제 FREEZE hash와 APPROVE_FIXED_EXECUTION/blocker0을 확인해야 한다.
3. 새 state/기존 결과가 없고 남은 마지막 batch가 미사용이면 같은 interpreter로 `...runner run`을 **한 번만** 실행한다. 네 arm×source3fold12job,40epoch/batch64/seed0/CPUfloat64, job180초/worker1800초 고정이다. 참조 모델 새fit0이다.
4. 시작과 첫 실패/timeout도 atomic ledger에 소비한다. RUNNING은 실제 process와 lock을 확인해 관찰만 한다. Orphan/중단도 소모된 시도이며 임의 초기화하지 않는다. 첫 실패면 불완전 비교를 승격하지 않고 batch STOP한다.
5. Parent가12개 전 산출물60개를 봉인한 뒤에만 held 사건 score를 계산한다. 완료/실패 뒤 원예측/영수증/가중치/trace/입력·코드 봉인·사람 macro 점수와 재사용 reference를 독립 검토한다. 개인·행별 결과/예측/가중치/trace는 private다.
6. 기존2%/최소2fold screen과 세 matched arm 대비 추가 이득을 유지한다. 보존 시험이나 다른 arm 개선을 null 방법의 예측 성공으로 바꾸지 않는다. 불리한 fold/사람·Brier·outcome/사건별·비용·수치 실패를 보고한다. 역사적으로 노출된 source의 탐색이며 독립 확증/높은 novelty는 별개다.
7. 마지막 batch가 완료/실패하면 학습 예산은 소진된다. 자동화를 일시중지하고 결과/한계/실패를 정리한다. 저장 산출물 독립 검토는 가능하나 추가 fit·standalone 수치 감사·설정 변경 구조는 하지 않는다. 2026-11-06 00:00KST 종료일을 늘리지 않는다.
