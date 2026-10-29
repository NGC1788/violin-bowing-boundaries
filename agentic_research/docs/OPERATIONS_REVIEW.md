# 실행·자동화 운영 독립 점검 — 2026-10-30

검사 시각: 2026-10-30 01:02 KST. 동결된 v1/v2 소스, 규약, 입력, 결과를 수정하지 않았다. 실제 상태 명령, 외부 동결 manifest, 봉인된 headroom 기록, SQLite 작업 수, 프로세스와 잠금, 실행 스크립트 및 앱 자동화 설정을 읽었다. 이 문서는 운영 검토이며 신규 성능 결과가 아니다.

## 판정

**v2는 정상적으로 종료되어 있다.** 전체 480열에서 원래 10% 질의 절감 목표가 수학적 하한 때문에 불가능해, 기준선 5개만 실행하고 학습 및 LLM 호출 전에 멈췄다. 이제 `research_tick.sh`를 반복하는 것만으로 미래 예측 연구까지 자동 실행되지는 않는다. 후속 연구는 앱의 자동화 에이전트가 작업 큐를 읽고 별도 버전을 준비하는 단계다.

| 검사 항목 | 실제 확인 결과 |
|---|---|
| v1 | `integrity_ok=true`, `stopped=true`, 완료 6개 |
| v2 | `integrity_ok=true`, `stopped=true`, 완료 5개, queued/running 없음 |
| v2 평가 규약 | 480열, 9개 규칙, 시드 0의 비학습 기준선 5개 |
| 가장 좋은 기준선 | 입력 없는 source-prevalence + entropy, 평균 16.125개 질의 |
| 정확 공동 최소 인증서 | 평균 15.1854166667개; 전체 정답을 먼저 아는 하한 |
| 상대 개선 상한 | `1 − 15.1854166667 / 16.125 = 5.826873385%` |
| 개선 가능성 관문 | 봉인 완료, `passed=false`; 요구 개선은 10% |
| v2 LLM 라운드 | 0개 (`agent_rounds/*/round.json` 없음) |
| v2 worker 누적 시간 | 33.5587초; 앱·LLM 대기까지 포함한 전체 시간은 아님 |
| 실행 중인 엔진 프로세스 | 없음 |
| 잠금 | stage/worker 잠금 비어 있음; planner 잠금은 아직 생성되지 않음 |
| 안전 위반 | `safety_halt=null`; 이번 종료는 오류가 아니라 목적의 개선 불가능성 때문 |

v2 manifest: `e29bd38e63f796ffbbc0817f003c8f6ab5d73b7f913ba3be75d55174bc3168b5`.
headroom seal: `c9af8f9668947cabcbfd51b29030e3e6a78a0e7879bff38ece7e1d32701bda92`.
외부 manifest의 47개 파일 목록에 `stage.py`, `planner.py`, `harness/headroom.py`, 테스트와 후보 JSON이 들어 있음을 확인했다.

## 사용자가 실행할 명령

현재 상태를 확인하는 명령은 실제 실행해 정상 반환을 확인했다.

```bash
cd "$HOME/violin-bowing-boundaries"
bash agentic_research/scripts/run_v2.sh status --campaign development480_20261030_v2
```

`integrity_ok=true`, `stopped=true`, `headroom.passed=false`가 현재의 올바른 상태다. `status`는 오류 상태도 JSON으로 보여 주기 위해 종료코드 0을 반환할 수 있다. 따라서 종료코드만으로 무결성이 정상이라고 판단하면 안 된다.

```bash
bash agentic_research/scripts/run_v2.sh status --campaign development480_20261030_v2 |
  python3 -c 'import json,sys; s=json.load(sys.stdin); print(json.dumps({k:s.get(k) for k in ("integrity_ok","integrity_error","stopped","headroom","safety_halt")},ensure_ascii=False,indent=2)); sys.exit(0 if s["integrity_ok"] and not s.get("safety_halt") else 2)'
```

아래 명령은 **v2 한 단계만** 호출한다. 현재는 STOP 때문에 추가 실험 없이 종료한다. 후속 v3를 생성·동결·학습시키는 일반 목적 명령이 아니다.

```bash
bash agentic_research/scripts/research_tick.sh
```

코드 오류가 발생했을 때는 자동 resume/retry/refreeze를 붙이지 말고 상태와 원본 로그를 남긴다.

```bash
if bash agentic_research/scripts/research_tick.sh; then
  bash agentic_research/scripts/run_v2.sh status --campaign development480_20261030_v2
else
  code=$?
  printf '연구 단계 실패(exit=%s). 자동 재개하지 말고 상태와 로그를 확인하세요.\n' "$code" >&2
  bash agentic_research/scripts/run_v2.sh status --campaign development480_20261030_v2
fi
```

완전히 쉬려면 캠페인 중지와 앱의 **활현 AI 연구 루프** 반복 작업 일시중지를 함께 적용한다. 캠페인 STOP만으로 앱이 후속 연구 문서를 준비하는 것까지 막지는 않는다.

```bash
bash agentic_research/scripts/run_v2.sh stop --campaign development480_20261030_v2
```

현재 두 캠페인에는 resume를 쓰지 않는다. v2에서 STOP을 해제하더라도 실패한 headroom 관문은 그대로여서 학습 enqueue/run/LLM은 막힌다. v1에는 이 후속 관문이 없으므로 과거 README의 resume 예제를 그대로 실행하면 안 된다.

## 자동화의 실제 범위

앱 도구의 조회와 `~/.codex/automations/ai/automation.toml` 읽기로 다음을 확인했다. 설정은 변경하지 않았다.

- 이름 **활현 AI 연구 루프**, ID `ai`, heartbeat, ACTIVE, 30분 간격이다.
- 실제 지침은 먼저 `AGENTS.md`, `WORK_QUEUE.md`, `EXECUTION_STATUS.md`와 상태를 확인하게 되어 있다.
- v2가 진행 중인 경우에만 tick을 호출한다. 관문 실패 후에는 미래 예측 프로토콜 준비와 공개 바이올린 자료의 적합성 확인으로 제한한다.
- 새 연구의 감사·누수 검사·독립 검토·규약 동결을 요구하고, 자동 GitHub push·삭제·결제를 금지한다.
- 2026-11-06 00:00 KST 또는 작업 큐 종료 시 스스로 일시중지하도록 되어 있다.

**기계적으로 강제되는 부분:** v2 작업 잠금, 후보 계약, 소스/입력 해시, 작업 수·worker 시간·날짜, 봉인된 개선 가능성 관문, 판정 검증 실패 시 영구 중단이다. stage/planner/worker의 잠금은 각각 동일 캠페인 안에서 중복 소유를 막는다. 서로 다른 캠페인까지 전역적으로 worker 하나만 허용하는 잠금은 아니다.

**아직 에이전트 지침에 의존하는 부분:** WORK_QUEUE에서 다음 연구 방향을 고르는 일, v3 감사 문서를 실제로 완성하는 일, 새 버전의 검토 통과를 판단하는 일, 앱 자동화 자체를 종료일에 일시중지하는 일이다. 앱이 활성 상태라는 것과 무인 연구 전체가 구현·검증됐다는 것은 다르다. 노트북이나 앱이 꺼져 있으면 예약 시각의 수행도 보장되지 않는다.

검사 시점의 `forecast_v3`는 작업 큐의 후속 설계 대상이며 학습 완료·동결 완료 결과가 없다. 다른 작업자가 작성 중인 규칙 패턴 모듈은 모델 학습, 데이터 시간 누수 감사, 평가 파이프라인의 완료 증거가 아니다. v3의 최대 12작업·1800초·사전 리뷰 규정은 해당 새 실행기가 구현되기 전까지 문서 규칙이다.

## 문서에서 구분할 항목

다음은 검사 시점의 문서 차이다. 담당자가 현황을 갱신 중이므로 고정 결과와 이후 설명을 구분한다.

1. `EXECUTION_STATUS.md`는 검사 시점에 v2를 구현·테스트 중이라고 적고 있었다. 실제로는 동결 후 기준선 5개와 관문 평가가 완료됐다. 새 결과와 “v2 LLM 0회”를 반영해야 한다.
2. README의 뒤쪽은 v1 재현 기록이다. 36작업·12라운드는 v1 규약이고, v2는 **24작업·18라운드**다. 상단 안내 및 일반 `AGENTS.md`의 12라운드 문구와 버전 범위를 명확히 구분해야 한다.
3. `AI_DECISION_GATES.md`의 초기 제안은 무작위 기준선 3시드와 최종 후보 하나를 언급한다. 실제 동결 v2는 기준선 시드0 5개, 관문 통과 시에만 등록 학습 비교 5종×3시드다. 실행 증거는 실제 규약을 따른다. 설계안 문장을 완료된 실행으로 인용하면 안 된다.
4. 검증된 v2 소스의 테스트 결과는 **63개 통과 + 하위 검증 13개 통과**다. v1의 29개 + 9개와 섞지 않는다. 이번 운영 감사에서 다시 실험하거나 테스트를 추가하지 않았다.
5. `run_v2.sh`와 `research_tick.sh`는 기본 `.venv-mac/bin/python`이 없으면 오류로 멈춘다. v1 `run.sh`의 여러 Python 환경 자동 탐색과 다르다. 다른 머신에서는 `VIOLIN_LOOP_PYTHON`을 검증된 환경의 절대 경로로 지정한다.

## 운영 결론

현재 v2의 중단은 정상이고, 실패한 목표를 계속 최적화하지 않는 관문이 실제로 작동했다. 다음 유의미한 단위는 동일 tick 재실행이 아니라 **별도 미래 예측 연구의 입력·시간축·규칙·분할 감사**다. 그 단계가 검증되기 전에는 새로운 AI 성능 개선, 완성된 v3 자동 학습, 바이올린 일반화 성과를 주장할 근거가 없다.
