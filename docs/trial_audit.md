# 전체 시행 감사

2026-09-17. 첫 데이터 준비(`TODAY DATA PREPARATION: PASS`)가 끝난 서버에서 RustDesk로 연 **Ubuntu 터미널**에서 실행한다. 추가 설치나 sudo는 필요 없다.

추출된 6,000개 시행의 `whole_N.csv`를 한 번씩 읽어 **파일 구조와 값의 범위**를 확인한다. 신호를 다른 형식으로 복사하지 않고 작은 보고서만 쓰므로 디스크를 거의 쓰지 않는다. 상태 라벨, 단위, 보정, 연구 성능은 확인하지 않는다.

## 실행

코드를 받고 먼저 폴더마다 20개 시행만 확인한다. 몇 초 걸린다.

```bash
cd "$HOME/research/violin-bowing-boundaries"
git pull --ff-only
git log -1 --oneline
bash scripts/run.sh trial-audit --limit 20
bash scripts/run.sh trial-audit --show
```

`TRIAL AUDIT: PASS`면 전체를 실행한다. 몇 분 걸리므로 `nohup`으로 실행한다.

```bash
mkdir -p logs
TASK_LOG="$(mktemp "$PWD/logs/trial-audit-XXXXXXXX.log")"
nohup bash scripts/run.sh trial-audit > "$TASK_LOG" 2>&1 < /dev/null &
printf 'PID: %s\nLog: %s\n' "$!" "$TASK_LOG"
tail -f "$TASK_LOG"
```

`audited 500/6000`처럼 진행 줄이 올라간다. `Ctrl+C`는 로그 보기만 끝내고 작업은 계속된다. `TRIAL AUDIT: PASS` 또는 `FAIL`이 나오면 다음을 실행해 **출력 전체**를 공유한다. 폴더당 6~8줄 정도다.

```bash
bash scripts/run.sh trial-audit --show
```

자세한 값은 `reports/trial_audit/<실행>/trials.csv`(시행당 한 줄)와 `summary.json`에 있다. `reports/`는 Git에 올라가지 않는다.

## 결과 읽기

| 줄 | 의미 |
|---|---|
| `failed`, `orphans` | 읽지 못한 시행과 짝이 없는 `beta`·`timestamp` 파일 수. 0이어야 `PASS` |
| `rows` | 시행 길이. 괄호 안 초는 공개 설명의 50kHz를 가정한 값 |
| `beta` | 시행에 기록된 β의 범위와 서로 다른 값의 개수 |
| `window widths` / `start` | 분석 창 폭과 시작 위치의 분포 |
| `inside c2 plateau` | 분석 창 전체가 2열의 최대값 근처(기본 1% 이내)에 있는 시행 수 |
| `c2 peak levels` | 폴더 안 시행들의 2열 최대값. **폴더 이름이 아니라 이 값으로 속도 조건을 정한다** |
| `c1 window mean` | 창 안 1열 평균의 범위와 서로 다른 수준의 개수 |
| `c4 min step` | 4열 값 사이의 가장 작은 간격. 양자화를 보여주는 참고값 |

`WARNING`은 실패가 아니다. 특히 창이 속도 평탄구간을 벗어난 시행이 있다는 경고가 나오면, 그 시행의 창 안 통계를 **일정 속도 조건으로 취급하지 않는다.**

## 주의

- 열 이름은 공개 설명의 순서(활 힘, 활 속도, 브리지 힘, 너트 힘)를 따를 뿐 이 감사로 검증되지 않는다. 보고서는 `c1`~`c4`로 표기한다.
- 공개 메타데이터의 폴더-속도 대응(1→0.1, 2→0.05, 3→0.2 m/s)은 첫 시행의 실측값과 맞지 않았다. [데이터 계약](data_contract.md) 참조.
- `PASS`는 파일 구조와 값이 읽혔다는 뜻이다. 라벨의 존재, 물리 단위, 시행 독립성, 평가 분할은 이후 단계에서 따로 확인한다.
- 실패한 시행이 있어도 데이터를 지우거나 다시 추출하지 않는다. 오류 내용을 먼저 확인한다.
