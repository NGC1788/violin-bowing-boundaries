# 공개 바이올린 미래 예측: 동결 전 규약

2026-11-02. 기계 규약은 `protocol.json`이다. 아직 학습하거나 실제 미래 목표를 계산하지 않았다. Data preparation freeze와 나중의 training/evaluation freeze를 구분한다. 전자는 준비 코드와 분할·선택·목표를 고정하고, 후자는 준비 입력 및 완성된 실행기·평가기를 추가로 고정한다. 앞선 동결을 덮어쓰거나 좋은 점수가 나올 때까지 다시 만들지 않는다.

## 질문과 의미

처음1초의 원음 지표로0.25초 간격 뒤0.5초 구간의 pitch/power 변동 규칙을 예측할 수 있는지, source Markov prior를 추가한 작은 공동 신경망이 강한 단순 모형과 독립/공동 신경망을 실제로 개선하는지 확인한다. Markov offset과 공동 확률 head는 기존 방법이다. 높은 방법론 신규성이 확보됐다는 주장은 하지 않는다. 동결된12개 비교가 추가 이득을 보이지 않으면 같은 자료에서 자동 탐색을 늘리지 않는다.

## 자료와 시간축의 한계

Blanco·Tassani·Ramirez 원논문 §2.4는 세션 오디오 녹음을, §3.2는 trial 전체 pitch/power 표준편차와 YIN을 기술한다. 파일을 만드는 clipping/onset 절차의 인과성은 확인하지 못했다. 따라서 원음 구간을 분리한 계산의 미래 참조를 검사하되 **recording-relative forecast**로 한정하고 실제 연주 시작 시각·실시간 조기경보·청각적 음질의 검증으로 부르지 않는다. [원논문](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2021.648479/full)

CG18/FG18/EG15명,2168녹음은 목표 열람 전 메타데이터로 정했다.34/8/9명 분할은 이미 고정했다. 무음/무주기 미래를 좋은 안정성으로 채우거나 표본에서 제외하지 않는다.9명 test로 얻는 모집단 불확실성은 크다. 녹음/9규칙/3seed를 추가 참여자로 세지 않는다.

## 누수와 평가 고정

구간을 먼저 자르고 frame 안에서만 계산한다. 전처리·문턱은 train에서 정한다. features 파일에는 미래 목표/guard/라벨을 넣지 않고 train/dev/test 정답 파일을 별도로 둔다. 실제 데이터의 해시 선택8건에서도 미래를NaN으로 치환해 입력의 bitwise 불변성을 검사한다. 모든 예측을 저장·봉인하고 dev에서 비교 기준선을 고정한 뒤 test를 한 번 평가한다.

각 신경망군의 주 점수는3seed의 **손실 평균**이다. 앞선 개념 검토에 있던 평균 예측 ensemble과 구분하여 실제 목표/점수 열람 전에 확정했다. 단일 seed로 훈련되는 알고리즘의 평균 결과를 비교하는 선택이며, ensemble 점수를 계산한다면 별도 보조 결과다. comparator는 dev의 frequency/Markov/HGB 중 최선이며 tie 순서도 고정한다. offset의 실용 개선 후보 기준은 이 comparator와 독립/공동 MLP 각각에2% 이상 낮은 log loss다. 유효/무효 미래별 성능·coverage, coherence와 계산비도 공개한다.

paired participant bootstrap5000회,seed20261102. 좋은 하위집단이나 seed를 골라 전체 실패를 성공으로 바꾸지 않는다. Conditional CDF·10패턴·일관성 보정 자체가 기존 기술임을 유지한다. [직접 CDF 선행](https://proceedings.mlr.press/v124/chilinski20a.html)

예산은12개CPU학습,누적1800초,개별600초,2026-11-06 00:00 KST까지다. 데이터 생성·LLM 검토 시간은 학습 worker 예산과 별도로 기록한다. 실패도 시도로 포함하고 같은ID로 몰래 재시도하지 않는다.
