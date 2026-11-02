# 고정12개 실행과 test1회 평가 계약

2026-11-02. 데이터 코드와 `protocol.json`의 이전 동결은 유지한다. 새 `execution_freeze_manifest.json`에 준비자료4개 hash·준비 영수증·데이터 동결·모델/지표/실행기/시험·별도 검토·라이브러리 버전을 결합한다. 이는 같은 OS 사용자를 신뢰하는 무결성 장치이며 보안 sandbox나 외부의 실행 입회가 아니다.

## 접근 순서

1. `freeze`: 현재 코드 snapshot에 대한 별도AI 실행 검토가 approve이고 blocker가 없어야 한다. 기존 freeze/campaign은 덮어쓰지 않는다.
2. `train`: 입력과 **train_labels.npz**만 배열로 연다. dev/test 정답은 byte hash로 신원만 확인하고 배열을 해석하지 않는다. 모든12개 설정은 protocol 순서 그대로다. 평가 입력도 prefix에서 만든60개X와prefix state뿐이다. 전처리는train에만 적합한다.
3. 모델마다 전체dev/test 예측과 원래 행 index를 저장한다. 모든12개 예측·영수증을 해시로 봉인한다. 원시 예측과 모델 가중치는 `local/`에 보존하며 공개하지 않는다.
4. `evaluate`: 예측 봉인을 검증하고 dev 정답만 읽어 frequency/Markov/HGB 중 고정 tie 순서로 비교군을 고른다. 모든 후보의 dev 점수와 비교군·예측 hash를 selection.json에 봉인한다.
5. 봉인을 다시 확인하고 `test_opened.json`을 먼저 만든 뒤 test 정답을 한 번 연다. label과행순서가 맞아야 한다. test 열람 뒤 실패해도 재열람/새 seed/재훈련으로 보완하지 않는다. 결과/결과 hash를 보존하고 EVALUATED·STOP으로 닫는다.

## 비용과 실패

12시도, 누적worker1800초, 개별600초,2026-11-06 00:00 KST. 각fit 호출에 남은 개별/누적/종료시각 중 최소 시간의 timer를 둔다. worker초는 모델학습·예측 호출의 실제 wall time이며 코드 검토·합성시험·준비·hash검사는 별도로 포함하지 않는다. signal은 OS/런타임 스케줄링 정밀도에 제한되므로 초과를 숨기지 않는다. 실패/시간초과/사용자중단은상태와 시도를 남기고종료한다. RUNNING 흔적이 있는 중단 캠페인도 새fit으로 이어서 다시 계산하지 않는다. hash불일치는 test열람 전에차단한다.

## 점수와 주장

각참여자의9규칙/녹음평균을구한후참여자를같은비중으로평균한다.신경망은3seed **손실의평균**이다.평균예측의손실로바꾸지않는다. pairedbootstrap은같은참여자위치의손실차이를5000회재표집한다.9명을창/규칙/seed수로부풀리지않는다.전체점수와유효/무효미래별coverage·손실,cohort별손실,일관성,Brier,계산비를남긴다.

주후보offset은dev로미리고른단순기준선,plainjoint,independent에각각2%개선해야운영상screen을통과한다.95%bootstrap구간은탐색적불확실성으로제시하며모집단보장이나교육효과로해석하지않는다.부정적결과를좋은seed/하위집단/지표로바꾸지않는다.최신source문헌에서확인했듯이offset·공동head·coherence자체의방법론novelty는주장하지않는다.
