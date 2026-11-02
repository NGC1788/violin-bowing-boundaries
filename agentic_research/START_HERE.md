# 다른 AI·다른 컴퓨터에서 이어가는 방법

**최신(2026-11-02):** [source 미래 경로 준비](path_source_gate_v1/RESULTS.md)를 1회 완료했다. 기존 정답 재구성과 시간 누수 검사는 통과했지만, inner-train 마스크 사전은 한 분할의 미지 유효 패턴7.22% 및 다른 분할의67종으로 고정 운영 기준을 실패했다. 새 학습0건이다. 다음은 [미래 마스크 모형의 서면 가능성 검토](path_source_gate_v1/NEXT_METHOD_REVIEW.md)이며 준비·기존 수치 감사를 반복하지 않는다. 아래 대기 설명은 이전 이력이다.

**최신:** [공동변동량decoder의수치감사](discovery_v3_math/RESULTS.md)까지완료·PASS다. 해당감사는재실행하지않는다. 다음은 [별도source경로자료gate](discovery_v3_math/NEXT_DATA_GATE.md)이고실제자료준비/학습은아직안했다. 이론/구현통과를실제바이올린성능개선과혼동하지않는다.

**최신(2026-11-02): 사용자의새요청으로넓은탐색을재개했다.** [첫새12학습](discovery_v1/RESULTS.md), [두번째12학습](discovery_v2/RESULTS.md), [다음가설](discovery_v2/NEXT_EXPLORATION.md), [추가루프예산](discovery_v2/LOOP_BUDGET.json)을읽는다. 기존STOP을연구전체종료로해석하지않는다.24추가학습에서도Markov개선은미확보이며,새가설은아직검증된novel방법이아니다. 완료batch는보존하고다음은별도label-free수치프로토타입이다.

**최종 판정(2026-11-02):** [보정 가능성 감사](correction_feasibility_v1/DECISION.md)까지완료했고조건부보정·인증후속도STOP이다. 전역혼합의탐색적개선은있지만조건부장치의추가이점·유효인증·높은novelty가없어더돌리지않는다. 기존시험과결과를보존하며등록된다음실험은없다. 아래미완료수학검토안내는완료이력이다.

**최신 후속:** [신경망 실패 진단](violin_diagnosis_v1/CAUSE_ANALYSIS.md)과 [후속 방법론의 가능 조건](violin_diagnosis_v1/NEXT_METHOD.md)을정리했다. 새학습없이저장모델의train/dev만검증했고,높은novelty·독립성능개선은미확보다. 현재다음작업은수학적가능조건·직접선행검토이며기존캠페인실행이아니다. 완료캠페인STOP·자동화PAUSED유지.

**최신 상태(2026-11-02):** 바이올린 raw 고정 12개 학습·시험 평가·최종 감사가 완료됐고 개선 screen은 실패했다. [RESULTS.md](violin_raw_v1/RESULTS.md), [WORK_QUEUE.md](docs/WORK_QUEUE.md) 최상단을 읽는다. 캠페인은 STOP, 자동화는 PAUSED다. 아래의 준비/학습 대기 설명과 실행 예시를 현재의 미완료 작업으로 해석하지 않는다. source·입력·점수·부정적 결과를 보존하며 새 방향에는 별도 근거와 계약이 필요하다.

**현재 다음 작업(2026-10-31):** [CLAUDE_NEXT.md](CLAUDE_NEXT.md), [실행 현황](docs/EXECUTION_STATUS.md), [raw 특징 감사](violin_raw_v1/RAW_FEATURE_AUDIT.md)를 먼저 읽는다. 아래 v3 명령은 완료된 첼로 비교의 이식/재현 설명이다. 현재 연구의 진행 명령이 아니며 raw 바이올린 실험을 실행하지 않는다. 보정 solver 보완도 이미 종료했다.

현재 연구는 **초기 신호로 미래의 여러 관측 규칙 출력을 예측하는 AI 연구**다. 구현, 테스트, 실행, 신규성 주장을 구분한다. 최신 실제 결과는 `docs/EXECUTION_STATUS.md`를 먼저 읽는다.

공개 브랜치: `research/agentic-forecast-20261030`

저장소: https://github.com/NGC1788/violin-bowing-boundaries

## 1. 새 컴퓨터에서 준비

Python3.11 이상이 이미 설치된 환경에서 실행한다. 관리자 권한이나 GPU가 필요하지 않다. 최초 Python/CLI 설치 자체를 이 스크립트가 수행하지는 않는다.

```bash
git clone --branch research/agentic-forecast-20261030 \
  https://github.com/NGC1788/violin-bowing-boundaries.git violin-agentic-research
cd violin-agentic-research
python3 -m venv .venv-loop
.venv-loop/bin/python -m pip install -r agentic_research/requirements.txt
export VIOLIN_LOOP_PYTHON="$PWD/.venv-loop/bin/python"
"$VIOLIN_LOOP_PYTHON" -m pytest agentic_research/forecast_v3/tests -q
```

`requirements.txt`는 이 Mac에서 검증한 버전이다. 새 OS에서 설치·테스트가 실패하면 그 오류부터 해결하고 실험을 시작한다. 별도 바이올린 MAT 감사에만 필요한 `mat-io` 등은 일반 예측 실험의 의존성이 아니다.

기본 실험에 필요한 **78,000행 준비 자료 약9MB**가 `benchmarks/forecast_v3/`에 있다. 수백GB 원자료를 다시 받을 필요 없이 같은 실험을 재현할 수 있다. 원자료의 CC-BY-4.0 저작자·출처·변경 내역은 그 폴더의 `DATA_CREDITS.md`에 있다. 모델 가중치나 계정 토큰은 요구하지 않는다.

## 2. 이 머신의 캠페인 만들기

기존 Mac의 외부 잠금을 복사하거나 덮어쓰지 않는다. 새 경로에서 새 이름으로 입력·코드·규약을 동결한다.

```bash
bash agentic_research/scripts/run_forecast.sh init --campaign local_forecast_v3 \
  --data agentic_research/benchmarks/forecast_v3/dataset.npz \
  --protocol agentic_research/forecast_v3/configs/forecast_v3.json
bash agentic_research/scripts/run_forecast.sh freeze --campaign local_forecast_v3
```

계약의 종료일은 **2026-11-06 00:00 KST**다. 날짜가 지나면 기존 계약으로 실행하지 않는다. 과거 결과를 보존하고, 이유를 기록한 새 계약과 캠페인이 필요하다. `freeze` 실패를 잠금 파일 삭제로 우회하지 않는다.

## 3. Claude·Codex로 자동 라운드

각 CLI는 해당 사용자의 로그인으로 실행된다. Claude는 이 Mac에서 실제 구조화 응답 호출을 확인했다. 비밀 키를 저장소에 넣지 않는다. CLI가 없는 경우 아래 수동 JSON 모드를 사용한다.

```bash
# Claude 제안자 + Codex 검토자, 한 단위 실행
bash agentic_research/scripts/run_forecast.sh tick \
  --campaign local_forecast_v3 \
  --proposer-backend claude --reviewer-backend codex

# 같은 조건에서 남은 고정 비교를 유한하게 이어서 실행
bash agentic_research/scripts/run_forecast.sh tick \
  --campaign local_forecast_v3 \
  --proposer-backend claude --reviewer-backend codex --max-rounds 18
```

둘 모두 Claude로 하려면 두 옵션을 `claude`, 둘 모두 Codex로 하려면 `codex`로 지정한다. 별도 호출이 통계적으로 독립된 검토자를 뜻하지는 않는다. LLM은 동결된12개 설정 중 실행 순서를 고르고, Python 평가기가 학습·계산한다. 새 Python을 즉석 생성하여 평가기 권한으로 실행하는 구조는 아니다.

한 캠페인은 최대12개 작업, worker누적1800초, 작업당600초, 제안라운드18회다. 실패도 기록하고 예산에 포함한다. 단일 프로세스·내용 해시·설정 해시·검증 결과가 실제 실행을 통제한다. 학습 실패가 나면 좋은 숫자를 얻기 위해 다른 seed로 교체하지 않는다.

## 4. Gemini·다른 챗봇·수동 JSON으로 진행

공통 인계문서는 [`HANDOFF_ANY_AI.md`](HANDOFF_ANY_AI.md)다. 파일 접근이 가능한 AI에게 저장소와 이 문서를 준다. 파일 접근이 없는 챗봇에는 생성된 프롬프트·schema를 붙여넣고 JSON 응답만 저장한다.

```bash
mkdir -p manual_replies
bash agentic_research/scripts/run_forecast.sh agent \
  --campaign local_forecast_v3 --execute \
  --proposer-backend manual --reviewer-backend manual \
  --manual-dir "$PWD/manual_replies"
```

첫 실행은 `awaiting_manual_input`과 round ID를 반환한다. `forecast_v3/state/<campaign>/agent_rounds/<round>/`의 `proposer_prompt.txt`와 `proposer_schema.json`을 AI에게 주고 응답을 `manual_replies/proposer_response.json`으로 저장한다. **request_sha를 schema 그대로 반환해야 한다.**

```bash
# ROUND_ID에는 방금 반환된 실제 ID를 넣는다.
bash agentic_research/scripts/run_forecast.sh agent \
  --campaign local_forecast_v3 --execute \
  --proposer-backend manual --reviewer-backend manual \
  --manual-dir "$PWD/manual_replies" --resume-round ROUND_ID
```

이제 생성된 `reviewer_prompt.txt`·schema를 별도 AI/호출에 주고 `manual_replies/reviewer_response.json`으로 저장한 뒤 같은 round ID로 재개한다. 검토 응답은 후보 해시와 request_sha 모두에 묶인다. 재개가 같은 라운드를 두 번 과금하지는 않지만 사용하는 외부 AI의 요금·사용량은 해당 서비스 설정을 따른다. 다음 라운드의 프롬프트가 나오기 전에 이전 응답을 재사용하지 않는다.

## 5. LLM 없이 고정 실험 실행

```bash
bash agentic_research/scripts/run_forecast.sh enqueue --campaign local_forecast_v3 \
  --config agentic_research/forecast_v3/candidates/hgb_history_s0.json
bash agentic_research/scripts/run_forecast.sh run --campaign local_forecast_v3 \
  --max-jobs 1 --max-seconds 600
```

평가기 자체는 LLM이나 인터넷 없이 실행할 수 있다. 모델 성능이 LLM 문장에 의존하지 않게 하기 위한 분리다.

## 6. 상태·중지·결과

```bash
bash agentic_research/scripts/run_forecast.sh status --campaign local_forecast_v3
bash agentic_research/scripts/run_forecast.sh report --campaign local_forecast_v3
bash agentic_research/scripts/run_forecast.sh stop --campaign local_forecast_v3
```

status의 `integrity_ok`, `stopped`, `safety_halt`, `counts`, `exhaustion_reason`을 읽는다. 프로그램 종료 코드0만으로 과학적 성공이나 무결성 통과를 판단하지 않는다. 각 run의 `worker_result.json`은 봉인된 점수다. `predictions.npz`는 별도 원예측이며, 외부 분석에서 사용하려면 추가 파일 해시도 보존한다.

이 사용자의 Codex 앱에는30분 간격 heartbeat가 별도로 있다. **GitHub clone만으로 그 앱 자동화까지 복제되지는 않는다.** 다른 AI에서는 위 유한 CLI 루프를 직접 실행하거나 그 서비스가 제공하는 반복 작업에 같은 명령을 등록한다. 현재 Mac의 앱 작업까지 중지하려면 앱의 **활현 AI 연구 루프**도 일시중지한다.

## 7. 수정할 때

소스·입력·평가 규약·후보가 바뀌면 이전 캠페인의 무결성이 깨져야 정상이다. 이전 결과를 보존하고 이유·반박 조건·테스트를 먼저 작성한 다음 새 엔진/버전으로 옮긴다. 기존 파일을 수정하고 manifest를 다시 쓰는 방식은 금지다. GitHub 업데이트는 사용자의 요청 범위에서만 하고 원자료·토큰·원시 CLI 로그를 공개하지 않는다.

현재 전체 정확질의 과제(v1/v2)는 종료했다. 재개할 연구는 미래 예측과 별도 바이올린 적용 가능성 검증이다. 세부 작업·실패 기준·남은 한계는 `docs/WORK_QUEUE.md`, `forecast_v3/PROTOCOL_DRAFT.md`, `forecast_v3/DATA_AUDIT.md`, `forecast_v3/violin_data_audit/README.md`를 읽는다.

공식 CLI 근거: [Claude 비대화형·구조화 출력](https://code.claude.com/docs/en/headless), [Codex 비대화형 실행](https://learn.chatgpt.com/docs/non-interactive-mode).
