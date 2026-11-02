# 다음 한 단위: 실행기·누수 검증·전체 실행 계약 준비

이전 수치 감사는2/2회 소진·STOP이다. 이번 코드 준비 승인도 실제 학습 실행의 승인이 아니다. 다음은 별도 `exact_count_runner_v1`에서 학습 실행기·source-only 허용 입력·계약·소스 검토를 완성하는 단위다. **그 단위에서도 코드 검토 전 실제 자료를 열거나 fitting하지 않는다.**

1. 현재 프로세스·잠금·선점·Git과 공유 ledger를 확인하고 새로운 실행기 작업을 선점한다. `exact_count_learning_v1/READINESS_FREEZE.json`의 소스·의존성·검토·시험을 확인한다. 기존 준비 파일과 코드는 수정하지 않는다.
2. `CONTRACT_PROPOSAL.json`의 네 방법×기존 세 source 참여자 fold12job,40epochs/batch64/seed0/AdamW1e-3/weight decay1e-3/aux0.1 등 제안을 실제 실행 계약으로 검토한다. 결과를 보고 설정을 고르는 것이 아니며, 변경이 필요하면 fitting 전에 이유·대조·새 소스를 문서화해 별도 검토한다.
3. 입력 allowlist는 원train34명1442개만 포함한다. 별도 준비의 prefix 입력·source 메타데이터·future label 파일의 봉인·source ID순서·fold provenance를 확인하도록 코드를 작성한다. olddev/test 정답은hash도 열지 않는다. 실제 입력 내용/해시는 실행기 코드 검토가 통과한 뒤 이 단위의 새 동결 단계에서만 읽는다. 모델 입력과 미래 정답을 분리한다.
4. 각 fold의 모든 문턱·스케일·마스크/보조 prior·정규화는 inner train에만 맞춘다. Future-path centering은 label 변환이며 추론 입력이 아니다. Prefix 특징은 같은 raw60개와 같은 파생73정보를 모든 신경망에 준다. Synthetic loader fixture에서 held label poisoning·행 재정렬·participant overlap·누락/중복ID·오염된prefix와 private-output 경로를 fail closed로 검증한다.
5. 동일 future contrast/mask 보조감독, 같은 encoder 초기화·minibatch 순서·학습 epoch를 보장하고 네 head의 추가 파라미터 및 비용을 기록한다. 참여자 macro 학습 가중치와 평가지표의 분모를 고정한다. Invalid15개도 포함한다. Markov/HGB 이전 source 점수와 동일한 분할·문턱을 재현하는 기준선을 명시한다. 다른 신경망이 추가 정보를 받은 것으로 보일 경우 원raw60에서 파생된 정보임을 검증하고 강한 기준선 접근 범위도 공개한다.
6. 실제 시도를 시작하기 전에 shared budget lock으로 batch와 job을 원자적으로 등록한다. 시작한 실패·timeout도 소진한다. 최대12job/180초 per job/1800초 batch, 전체 남은3batch 및2026-11-06 00:00KST를 지킨다. 같은 batch 구조·재동결·재실행은 금지다. 앞으로 Gaussian/network 학습 속도를 작은 기능시험 시간으로 추정하거나 보장하지 않는다.
7. 모든 가중치·학습 곡선·예측·원점수·실패는 비공개로 봉인한다. 공개는 코드·문서·집계·파일해시만 한다. 좋은 fold/epoch/seed만 골라 공개하지 않는다.9개 규칙별 BCE를 사람별로 평균한 뒤 사람 macro를 주 점수로 하고 강한 baseline 대비2%/3fold중2fold 개선 screen을 사전에 고정한다. Source 노출 탐색임을 유지한다.
8. 실행기·검증·전체계약의 별도 검토가 통과하면 이 단위의 후반에서 허용된 source-only 입력/소스/환경/계약 동결까지 완료한다. 이를 `FROZEN_READY_FOR_BOUNDED_TRAINING`으로 기록하고 다음 실행 단위에서 고정 비교를 진행한다. 사용자가 이미 허용한 연구 범위이며 새 사용자 확인을 요청하는 단계가 아니다. 실패하면 명확한 근거를 남기고 이 학습 후보를 중단한다. 준비 작업을 무한 확장하지 않는다.

이 단위는 새로운 데이터셋이나 물리/음악 해석으로 주 목표를 바꾸지 않는다. 주요 가설은 **같은 학습 감독 아래 구조적 count head가 직접 확률/밀도 head보다 예측에 유리한가**다. 아직 novelty와 개선은 미입증이다.
