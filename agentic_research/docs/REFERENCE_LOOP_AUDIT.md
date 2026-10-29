# MaskedKD 참고 루프의 코드 감사와 활현 연구용 설계

검토일: 2026-10-30 (Asia/Seoul). 검토 대상은 `jeehoo0507/Loop-Research-for-MaskedKD`의 **8b9703ebb5dc549dc6cdf2422fc19923536f398d**다. 웹 첫 조회 이후 저장소 커밋 수가 변했으므로 이후에는 내려받은 이 커밋으로 고정했다. `setup.sh`, 학습, 에이전트 루프, 테스트는 실행하지 않았다. 아래 평가는 실제 소스의 정적 검토이며 보안 침투 시험이나 전체 실행 검증은 아니다.

참고 사본: `loop_design_20261030/reference/maskedkd/`. 이 저장소의 프롬프트에 적힌 자동 push·설치·모델 선택은 참고자료이며 우리 프로젝트의 지시로 채택하지 않는다.

## 1. 가져올 구조와 가져오지 말아야 할 보장

참고 루프의 장점은 LLM을 연구 전체의 판정자로 삼지 않고, **가설을 제안하고 제한된 후보를 만드는 역할**과 **고정된 프로그램으로 학습·채점하는 역할**을 나눈 것이다. 이 경계가 우리 연구에서도 핵심이다. 에이전트 숫자·라운드 수 자체는 연구 기여도나 근거의 강도가 아니다.

| 유용한 구조 | 소스 근거 | 우리 연구에서의 채택 |
|---|---|---|
| 후보 소스를 제출 시점에 복사하고 해시를 저장 | `harness/submit.py:93–126`, `train.py:181–186` | 제출된 후보는 원본 파일이 바뀌어도 바뀌지 않아야 함. 설정 JSON과 코드·데이터·분할 해시를 함께 고정 |
| 학습 큐와 제안자를 분리 | `scheduler.py`, `agent/loop.sh` | LLM은 요청만 생성. 고정 평가기가 자원·중복·관문을 확인하고 실행 |
| smoke → screen → final 단계 | `config.py`, `submit.py:77–126` | 작은 동작 확인 → 개발자료 비교 → 고정 후보 확인. 단계마다 다른 과학적 의미를 표기 |
| 기준선 재현 관문 | `report.py:29–44` | 기존 수치를 재현하기 전 새 방법의 개선을 계산하지 않음 |
| 같은 시드의 비교 대상 자동 제출 | `submit.py:130–145` | 새 방법만 시드를 늘리는 불공정 비교 방지. 모든 방법에 같은 분할·초기화·예산 |
| 부모 방법과 제거 실험 요구 | `AGENT_PROMPT.md:26–29,55–65` | 구성요소의 실제 기여를 가장 가까운 기준선과 비교. 성능 변화 없는 요소는 설명에서 제외 |
| 원점수에서 표 생성 | `train.py:290–309`, `report.py` | 최종 표·그림은 평가기 산출물에서만 자동 생성 |
| 별도 읽기 전용 감사 | `loop.sh:90–117` | 주장 작성자와 검토자를 나누되 LLM의 PASS는 통계적 검증을 대신하지 못함 |
| 새 결과가 있을 때만 다음 라운드 | `loop.sh:141–151` | 장시간 학습 중 불필요한 반복 호출을 줄임. 작업이 없다고 무조건 실험을 추가하지 않음 |
| RNG 상태를 포함한 재시작 자료 | `train.py:204–215,272–275` | 체크포인트 재개 시 동일 후보·분할·환경을 확인하고 중복 실행 방지 |

## 2. 현재 구현에서 확인한 한계

### 2.1 외부 SHA 파일은 보안 격리가 아니다

`harness/lock.py`는 `~/.maskedkd_lock`의 해시와 현재 코드 해시를 비교한다. 경로가 저장소 밖이라는 사실은 좋지만 같은 운영체제 사용자에게 쓰기 권한이 있으면 변경을 막지 못한다. 루프의 `Read,Edit,Write`와 Codex `workspace-write`, 소스 로더의 같은 프로세스 실행도 별개의 격리가 아니다. **정상 동작하는 에이전트의 실수 탐지**와 **적대적인 코드의 접근 차단**을 구분해 설명해야 한다.

검사기는 Python AST에서 일부 import·이름·속성을 금지하지만, torch/numpy 같은 큰 라이브러리 전체를 허용하고 동적으로 만든 모듈을 `exec`한다(`validate.py:31–79,108–118`). 이는 완전한 Python sandbox가 아니다. 정적 검사와 일부 부정행위 테스트가 통과했다고 접근 불가능성을 증명하지는 못한다.

우리 1차 구현에서는 임의 Python 코드를 받지 않는 **선언형 후보 JSON**을 권한다. 평가기에 구현된 모델·손실·전처리·선택 전략과 검증된 범위의 매개변수만 선택한다. 자유 코드 연구가 필요해지면 별도 프로세스·읽기 전용 마운트·네트워크 차단·독립 사용자/컨테이너 경계를 추가하고, 이 격리가 없는 환경에서는 실행 수준을 명확히 제한한다. 같은 사용자 파일 권한만으로 완전 방어를 주장하지 않는다.

### 2.2 test 전용 경계가 프로그램으로 강제되지 않는다

`test_eval.py`의 “HUMAN ONLY, once”는 주석과 에이전트 프롬프트다. 함수는 임의 run 목록을 받아 test를 읽으며 1회 사용권·고정 후보 목록·평가 허가 토큰을 검사하지 않는다(`test_eval.py:20–35`). `data.load_split('test')`도 호출 가능하다. 감사에서 사후 확인하는 구조는 직접적인 접근 차단과 다르다.

우리 자료는 이미 과거 연구에 노출되었다. 지금 그룹을 나누어도 **역사적으로 미사용된 확증 자료**가 되지 않는다. 현재 자료의 그룹 분할 결과는 사후·탐색적 일반화 평가로 적는다. 실제 미사용 외부 자료가 확보될 때만 별도 평가기와 1회 고정 후보 목록을 구성한다. 결과를 다시 본 뒤 다음 방법을 바꾸면 해당 자료는 이후 개발자료로 승격시키고 그 이력을 삭제하지 않는다.

### 2.3 결과 요약에서 서로 다른 방법 버전을 섞을 수 있다

표의 `table()`은 SHA와 설정을 구분하지만, `by_seed()`는 **method 이름·keep·tier**만으로 최신 run을 고른다. `core()`와 `compare()`가 이를 사용하므로 같은 이름의 소스를 수정하거나 `sel_args`를 바꾸면 서로 다른 시드에서 다른 방법 버전을 비교 집합으로 묶을 수 있다(`report.py:112–165`, `compare.py`). 부모 버전도 이름으로 식별한다. `primary()`는 기준선과 후보의 시드 집합을 일치시키지 않고 각 평균을 비교한다(`report.py:185–202`).

대안: 결과를 `(experiment_contract_sha, data_sha, split_sha, model_family, candidate_sha, config_sha, budget, tier)`의 정확한 키로 묶는다. 재실행은 독립 표본이 아니라 같은 seed에 속한 attempt다. 최종 비교에서 seed 목록과 평가 group 목록의 일치를 검사하고 불일치하면 계산을 거부한다. 부모도 이름 대신 후보 해시로 참조한다.

### 2.4 감사 PASS가 특정 산출물에 암호학적으로 연결되지 않는다

`submit.audit_passed()`는 `AUDIT.md`의 블록에 방법 이름과 `VERDICT: PASS` 문자열이 같이 있으면 성공으로 본다(`submit.py:44–52`). 코드 SHA·설정·정확한 run 목록에 묶이지 않는다. stage2 규칙도 같은 source SHA의 seed 완료를 확인하지만 budget·설정 전체 조합을 묶지 않는다(`submit.py:55–74`). 같은 이름을 바꿔 쓰면 과거 PASS의 의미가 흐려질 수 있다.

대안: 감사 결과는 JSON schema로 `claim_id`, `candidate_sha`, `contract_sha`, `run_ids`, `evidence_hashes`, `verdict`, `limitations`, `reviewer`를 갖는다. runner가 원점수와 해시를 확인하고 claim ledger에 상태를 기록한다. 자유 서술 PASS 문자열로 승격하지 않는다. 감사 실패나 API 실패는 요청을 보존한다. 참고 루프는 Codex 감사 호출의 실패를 별도 처리하지 않고 요청을 비우는 경로가 있으므로(`loop.sh:104–116`) 복사하지 않는다.

### 2.5 큐·예산·재개에 운영 취약점이 있다

- 큐와 결과 JSON을 `write_text()`로 바로 써서 중간 종료 시 부분 파일이 남을 수 있다. 원자적 임시 파일 교체 또는 SQLite transaction이 필요하다.
- `start.sh`의 tmux 세션 존재 확인은 편의상 중복 실행을 줄이지만, scheduler 자체에는 단일 인스턴스 잠금이 없다. 직접 두 번 실행하면 같은 큐를 처리할 수 있다.
- 복구된 작업은 PID만 보고 생존을 판단한다. PID 재사용·다른 호스트 프로세스·명령 동일성은 검사하지 않는다.
- `scheduler.py`의 강제 시간 제한은 새로 만든 `Popen` 객체가 있을 때만 적용한다. 복구한 작업에는 같은 강제 제한이 적용되지 않는다.
- 개별 작업 시간 제한은 있지만 전체 캠페인의 최대 실험 수·총 CPU/GPU 시간·실패 횟수·LLM 비용 상한이 없다. 새 프로세스에서 round 변수도 0으로 시작한다.
- 대기열 8개 이하·승격 전 분해 분석 등 중요한 조건 중 일부는 프롬프트에만 있다.
- `start.sh:8`은 tmux 부재 시 sudo 설치를 시도한다. 우리 no-sudo 서버·Mac 환경과 맞지 않는다.
- 자동 pull/rebase 실패를 무시하고, 라운드 종료마다 자동으로 main에 push한다. 우리 프로젝트에는 자동 채택하지 않는다.

### 2.6 반복 선택의 낙관성을 새 시드만으로 해결할 수 없다

새 seed는 초기화 변동을 확인하지만 같은 validation 집합에 대한 반복적인 가설 선택을 없애지 못한다. `core()`는 많은 방법·예산·대조에 각각 t 구간을 만들고 “Significant only”를 표기한다. 시도 횟수를 기록하는 것은 유익하지만, 다중 비교·적응적 선택을 통제하는 통계적 장치는 아니다. 2–3개 seed의 구간은 자료 모집단에 대한 불확실성 구간도 아니다.

우리 루프는 **screen 결과는 선택용**, **고정 후보의 그룹별 평가가 성능 보고용**임을 구분한다. 탐색의 모든 후보·중도 실패·제외 이유를 남긴다. 일반화 단위는 창 78,000개가 아니라 세션/표본 그룹이며, 그 수가 적으므로 결과의 범위를 좁혀 말해야 한다. 독립 확인이 없는 상태에서 p값을 붙여 성공 기준을 만족시키는 방향으로 루프를 돌리면 안 된다.

### 2.7 GPU 유휴를 없애는 것이 과학적 목적이 되어서는 안 된다

`loop.sh:146–148`는 GPU가 놀면 반드시 작업을 제출하라고 지시한다. 불필요한 변형·중복 시도와 점수 사냥을 부를 수 있다. 우리 루프에서는 다음 반박 가능한 가설이 없으면 `NO_JUSTIFIED_NEXT_EXPERIMENT`로 정지한다. 실패한 연구 방향을 명확히 종료하는 것도 자동화의 성공이다.

## 3. 권장 디렉터리와 신뢰 경계

```text
research_loop/
  harness/                 고정 데이터 계약·평가·통계·runner
  tests/                   누수/수치/재개/변조 실패 테스트
  contracts/v001.json      과제·대상·분할·지표·예산·종료 조건
  candidates/<sha>.json    허용 모델/손실/매개변수만 선언
  proposals/<id>.json      가설·근거·부모·예상 실패·반박 실험
  prompts/                 proposer/reviewer/skeptic 문서
  data_manifest.json       입력 파일 해시·열 의미·노출 이력
  state/ledger.sqlite      작업·attempt·예산·감사·누수 사건
  runs/<run_id>/           원점수·예측·타이밍·환경·해시
  reports/                 원점수에서 생성한 상태/결과/그림
  STOP                     새 작업 중단 신호
```

프로젝트 바깥에 고정 계약·평가기 해시를 보관하되, 이는 우발적 변경 탐지라는 한계를 문서화한다. 진정한 독립 test 저장소/서비스는 추후 별도 계정이나 프로세스 경계가 있을 때만 표현한다.

## 4. 실행 계약: 에이전트가 바꿀 수 없는 것

캠페인 시작 전에 다음 항목의 JSON을 canonical serialization하여 SHA-256으로 고정한다.

1. 학습 입력/목표 정의와 금지 입력. 동일 판정 기준을 결정하는 특징을 그대로 주고 그 판정 라벨을 맞히는 것은 새로운 물리 정답 예측이 아님을 명시.
2. 그룹 구성, train/dev/audit 용도, 기존 노출 이력, 가까운 시간창과 같은 긋기의 분할 금지.
3. 모든 전처리/정규화/특징 선택의 train-only fitting.
4. 주 지표 하나, 보조 지표, 성능뿐 아니라 실패율·보류율·계산 비용. 불편한 그룹을 결과에서 빼지 않음.
5. 기준선: 다수/단순 회귀·로지스틱, 강한 tree 기준선, 가장 가까운 기존 방법, 제안법의 핵심 요소 제거.
6. 조건/seed/예산의 비교 동등성. 성공하는 seed만 고르는 것이 아니라 계약에 정한 전체 목록을 사용.
7. 승격 규칙과 최대 승격 후보 수. 후보 추가는 탐색 계약 안에서만 가능.
8. 종료 날짜, 최대 run 수, 작업당 시간, 전체 시간, 재시도, 디스크·메모리 한도.
9. 최종 후보 고정 후 확인 결과를 보고 후보를 바꾸면 새 캠페인으로 기록하고 기존 확인 자료는 개발자료로 표시.
10. 방법론 기여와 결과 판정의 기준. 인기 모델 탑재, LLM의 아이디어 생성, 대규모 탐색 자체를 novelty로 세지 않음.

## 5. 상태 기계와 각 에이전트의 출력

`PROPOSED → VALIDATED → SMOKE → DEVELOPMENT → ABLATION → FROZEN → AUDIT → REPORTED`.

별도 종료 상태는 `REJECTED`, `FAILED`, `TIMEOUT`, `INVALID`, `CANCELLED`, `NO_GAIN`, `BUDGET_EXHAUSTED`다. 모든 상태 전이를 거래 단위로 기록한다.

| 역할 | 읽는 자료 | 쓰는 산출물 | 하면 안 되는 것 |
|---|---|---|---|
| 문헌/가설 제안자 | 개발 요약, 실패 기록, 확인한 1차 문헌 | proposal JSON, candidate JSON | 계약·평가기 수정, 최종 확인 결과 접근, 성능 숫자 작성 |
| 구현/runner | 유효 후보와 고정 계약 | predictions, metrics, runtime, logs | 논문 해석이나 결과를 보고 평가 기준 변경 |
| 반박 검토자 | 가설과 개발 증거 | ablation/null/leakage 검증 요청 | 좋은 결과만 승격하거나 실패 파일 삭제 |
| 감사자 | 고정 후보의 원점수·해시·독립 재계산 | claim verdict JSON | 실패를 새 프롬프트로 PASS로 바꾸기 |
| 보고서 생성기 | 검증된 ledger 조회 | 상태표·증거 링크·제한 포함 요약 | 자유 문장에 근거 없는 숫자 생성 |

LLM이 없어도 제출된 유효 작업은 끝까지 실행돼야 한다. LLM 인증 만료·사용 한도는 `PROPOSER_UNAVAILABLE`이며 계산 worker가 손상되었다는 뜻이 아니다. 실패한 LLM 호출을 무제한 즉시 재시도하지 않는다.

## 6. 권장 초기 자원 계약 (제안값, 실측 후 고정)

현재 M2 16GB 로컬에서는 시작을 작고 유한하게 한다. 예를 들어 첫 캠페인은 **동시 작업 1개, CPU 스레드 2개, 최대 24개 후보, 후보당 10분, 총 4시간, 최대 3개 승격**, 재시도 1회다. 이는 과학적 정답이나 사용자 예산 확인을 대신하는 값이 아니라 안전한 초기 제안이며 root가 실제 작업량을 측정한 뒤 계약에서 확정한다. A5000로 옮길 때도 동일 데이터·분할·평가 계약을 유지하고 계산 예산 변경만 별도 버전으로 기록한다.

종료 조건은 달력 기한뿐 아니라 다음을 포함한다.

- 기준선 재현/입력 무결성 실패 → 새 연구 실험 금지, 복구 기록.
- 계약·데이터·분할 해시 불일치 → 즉시 fail-closed.
- 동일 infrastructure 실패 3회 → 캠페인 정지.
- 사전에 정한 후보 수/벽시계/디스크 한도 도달 → 새 작업 금지.
- 서로 다른 사전 선언 계열 모두에서 의미 있는 개선 없음 → `NO_GAIN` 보고서 생성, 억지 추가 탐색 금지.
- 성능 개선이 있어도 누수·최소 기준선·제거 실험·비용 기준 실패 → 확정 성과로 승격 금지.
- 다음 가설의 근거와 반박 가능성을 작성할 수 없음 → 대기/종료, 유휴를 이유로 실험 생성 금지.

## 7. 최소 구현에서 반드시 검증할 테스트

1. 같은 데이터·계약·후보·seed 요청의 중복 실행 방지.
2. 실행 중 후보 파일을 수정해도 해당 run 입력 SHA가 변하지 않음.
3. 과거 source SHA와 새 SHA의 seed 결과를 섞으면 report가 실패함.
4. 그룹/긋기 중복 또는 train-only 전처리 위반 검출.
5. 알려진 작은 합성 자료에서 기준 점수와 exact algorithm을 독립적으로 확인.
6. NaN/Inf/예측 길이 불일치/없어진 그룹/목표 누락을 실패로 처리.
7. worker 중도 종료 후 동일 attempt를 성공으로 오인하지 않음. PID 재사용과 유실 heartbeat 처리.
8. 두 runner를 켜면 두 번째가 작업 시작 전에 종료함.
9. STOP·총예산·개별 timeout이 새 작업과 실행 중 프로세스에 약속한 방식으로 적용됨.
10. 결과 파일 변조·부분 JSON·harness 변경은 리포트에서 invalid로 표시되고 승격에 쓰이지 않음.
11. 같은 설정을 두 번 실행한 attempt는 새로운 독립 seed로 세지 않음.
12. 감사 API 오류나 schema 불일치로 요청이 PASS 처리되거나 삭제되지 않음.

## 8. 이 프로젝트에서의 최종 판단

참고 저장소는 연구 흐름의 좋은 출발점이지만, 그 안전·확증 설명을 그대로 복사할 만큼 강하게 격리되어 있지는 않다. 우리에게 맞는 방향은 **기존 과학적 감사 결과를 보존하고, 명확한 학습 문제와 강한 기준선·분할·중단 기준을 갖춘 새 ML 캠페인을 만드는 것**이다. 현재 파생 데이터에서 가능한 계산을 먼저 재현하고, 새로운 모델은 정확한 경계/판정 관련 알고리즘과 연결해 그 구성요소가 실제로 필요한지 시험해야 한다. 자료의 정답 한계를 해결하지 못한 채 더 큰 모델을 쓰는 것과, 정당한 정보기술 기여를 만드는 것은 다르다.

## 고정 커밋의 1차 소스 링크

- [저장소](https://github.com/jeehoo0507/Loop-Research-for-MaskedKD/tree/8b9703ebb5dc549dc6cdf2422fc19923536f398d)
- [lock.py](https://github.com/jeehoo0507/Loop-Research-for-MaskedKD/blob/8b9703ebb5dc549dc6cdf2422fc19923536f398d/harness/lock.py)
- [validate.py](https://github.com/jeehoo0507/Loop-Research-for-MaskedKD/blob/8b9703ebb5dc549dc6cdf2422fc19923536f398d/harness/validate.py)
- [submit.py](https://github.com/jeehoo0507/Loop-Research-for-MaskedKD/blob/8b9703ebb5dc549dc6cdf2422fc19923536f398d/harness/submit.py)
- [scheduler.py](https://github.com/jeehoo0507/Loop-Research-for-MaskedKD/blob/8b9703ebb5dc549dc6cdf2422fc19923536f398d/harness/scheduler.py)
- [train.py](https://github.com/jeehoo0507/Loop-Research-for-MaskedKD/blob/8b9703ebb5dc549dc6cdf2422fc19923536f398d/harness/train.py)
- [report.py](https://github.com/jeehoo0507/Loop-Research-for-MaskedKD/blob/8b9703ebb5dc549dc6cdf2422fc19923536f398d/harness/report.py)
- [test_eval.py](https://github.com/jeehoo0507/Loop-Research-for-MaskedKD/blob/8b9703ebb5dc549dc6cdf2422fc19923536f398d/harness/test_eval.py)
- [loop.sh](https://github.com/jeehoo0507/Loop-Research-for-MaskedKD/blob/8b9703ebb5dc549dc6cdf2422fc19923536f398d/agent/loop.sh)
- [start.sh](https://github.com/jeehoo0507/Loop-Research-for-MaskedKD/blob/8b9703ebb5dc549dc6cdf2422fc19923536f398d/agent/start.sh)
- [tests/test_harness.py](https://github.com/jeehoo0507/Loop-Research-for-MaskedKD/blob/8b9703ebb5dc549dc6cdf2422fc19923536f398d/tests/test_harness.py)
