# Claude Code 최신 인계 — 2026-10-30

**2026-11-02 재개:** 사용자의 “루프 재개” 요청으로 아래10월31일 중단을 해제했다. 자동화 `ai` ACTIVE. 최신 작업 선점은 `docs/WORK_QUEUE.md` 최상단을 따른다. 데이터 준비/평가 실행기·규약 검토가 다음 단계다.

**같은 날 데이터 준비 완료:** `violin_raw_v1/README.md`, `PREPARED_SUMMARY.json`, `data_freeze_manifest.json`을 읽는다.42개 합성 시험과 Claude 별도 데이터 검토 두 건 후2168녹음×60입력과 분리된 정답 파일을 `local/prepared/`에 생성했다. 실제8녹음의 미래 변조 입력 불변성도 통과했다. 데이터 코드/규약/검토는 동결되어 수정하지 않는다. download/prepare를 다시 실행하지 않는다. 다음은 고정12개 학습·예측 봉인·dev 비교군 선택·test1회 평가 실행기 구현과 별도 검토/추가 동결이다. 아직 모델은 학습하지 않았다. 실제 시험 정답의 분포나 점수는 보고하지 않았다.

**2026-10-31 중단 요청:** 사용자가 “중단했다가 이따 이어하자”라고 요청했고 자동화 `ai`는PAUSED다. 아래 실행 안내를 지금 실행하지 말고 명시적 재개 요청을 기다린다. 원음 다운로드·51명/2168녹음 metadata 감사와34/8/9명 분할, 합성 시험36개가 완료됐다. AG2 미분류 해소는 `violin_raw_v1/data_partition_receipt.json`, 최신 상태는 `docs/EXECUTION_STATUS.md` 최상단을 읽는다. 이번 추가 파일은 미커밋이다. 실제 미래 목표/모델 학습은 아직 없다.

## 2026-10-31 갱신

Codex가 사용자 요청으로 루프를 재개했다. `violin_raw_v1/features.py`와 `test_features.py`가 추가됐고 특징16개+모델3개 합성 시험이 통과했다. [RAW_FEATURE_AUDIT.md](violin_raw_v1/RAW_FEATURE_AUDIT.md)를 읽는다. 원음 다운로드는 기존 조각을 보존하고 재개했으므로 **현재 프로세스·download.lock을 확인한 뒤** 행동한다. 기존의 “파일3개뿐/다운로드 중단” 문장은10월30일 인계 스냅샷이다. 실제 데이터 생성기·평가 실행기·독립 검토·계약 동결과 바이올린 성능 결과는 아직 없다. 새 특징 추출기는 원음의 지원 구간을 먼저 자르고, 사전 메모의 Hann 대신 구간 내부 normalized ACF를 사용한다. 모델의 신규성은 입증되지 않았다.

사용자의 현재 요청은 보완 검증을 끝내고, 공개 바이올린 자료의 미래 예측을 실제로 실행하며, 누수·신규성·실제 성능 개선을 엄격하게 확인하는 것이다. 계획만 제안하지 말고 검증된 실험 결과까지 진행한다. 좋은 성능이나 신규성을 보장하지 않는다.

## 먼저 읽고 실제 상태 확인

작업 폴더는 `/Users/jeongjihoon/violin-bowing-boundaries`, 브랜치는 `research/agentic-forecast-20261030`이다. 기존 로컬 폴더를 사용한다. 최신 미커밋 작업이 있으므로 새 clone으로 대체하거나 reset/clean하지 않는다.

이 파일 → `AGENTS.md` → `docs/EXECUTION_STATUS.md` → `docs/WORK_QUEUE.md` → `projection_audit_v2/RESULTS.md` → `violin_forecast_v1/DATA_AUDIT.md`, `METHOD_REVIEW.md` → `violin_raw_v1/` 실제 소스 순서로 읽는다. 이 파일의 상대 경로는 `agentic_research/` 기준이다. 과거 문서와 충돌하면 실행 영수증과 최신 상태를 대조한다. 인계 파일도 실제 상태 확인을 대신하지 않는다.

시작 시 Git 변경, 실제 프로세스, 파일 잠금, 실행 영수증을 확인한다. Codex의 30분 자동화가 별도로 있으므로 `docs/WORK_QUEUE.md` 상단에 담당 작업·수정 파일·시작 시각·현재 상태를 기록한다. 다른 실행자의 진행 중 작업과 겹치지 않는다. 잠금 파일 존재 자체로 실행 중이라고 단정하거나 잠금을 삭제해 우회하지 않는다. 명령/프로세스 출력에 계정정보가 있으면 공개하지 않는다.

## 확인한 현재 상태

- v1/v2와 `forecast_20261030_v3`는 STOP이다. 기존12개 실험을 다시 돌리지 않는다. 공동 head는 보정 HGB보다3.335% 개선됐지만 독립 head보다0.185% 나빴다. 구조의 추가 이득은 입증되지 않았다.
- `projection_audit_v2`는 마지막 보완 검증까지 완료했다. 합성20개 시험, 동결 후 실제256건 감사1회. FW gap 통과239건, clipping 밖 coherent identity11건, 실제 최적화 실패6건. 모든 입력에 수렴하는 참조 solver는 아니다. **제3 solver 탐색 없이 이 분기를 종료한다.** 동결 코드·규약·결과는 변경하지 않는다.
- 기존 MAT 바이올린 시퀀스는72개 참여자 기록,2,117개 고유 시퀀스였지만 저장 YIN 특징의 양방향 시간 참조가 발견됐다. frame 간격만 벌려 엄격한 미래 누수를 해결했다고 주장할 수 없다. `violin_forecast_v1`은 합성 시험만 했고 실제 데이터 생성·학습·점수 계산은 하지 않았다. 이 MAT 실험을 재개하지 않는다.
- 새 방향은 원음 WAV에서 입력/목표 구간을 **먼저 자르고**, 각 구간 안에서만 음향 특징을 계산하는 `violin_raw_v1`이다. 현재 `download.py`, `models.py`, `test_models.py`만 있다. 특징 추출기, 실제 데이터 생성기, 평가 실행기는 **아직 없다**. 앞선 에이전트 계획을 완성된 파일로 취급하지 않는다.
- 2026-10-30 인계 확인 시 다운로드/학습 프로세스는 없었다. 약390MiB의 다운로드 조각과 약130MiB의 이전 prefix 파일이 외부 저장소에 남아 있다. 조각에는 미완성 파일도 있을 수 있으므로 완료한 다운로드 크기라고 합산하지 않는다. 최종 아카이브와 다운로드 영수증은 아직 없다.
- `violin_raw_v1/test_models.py` 합성 시험은 방금 **3 passed (3.16s)**였다. 실제 바이올린 모델 성능은 아직 없다.
- 세 새 폴더 `projection_audit_v2/`, `violin_forecast_v1/`, `violin_raw_v1/`는 인계 시 미추적 로컬 파일이다. GitHub의 이전 커밋만 보면 이 진행 상황이 누락된다.

## 다음 작업 1: 원음 취득·내용 감사

기존 다운로드 프로세스가 없는지 재확인한 뒤 아래 스크립트로 이어받는다. 조각을 지우거나 두 인스턴스를 시작하지 않는다.

```bash
cd /Users/jeongjihoon/violin-bowing-boundaries
.venv-mac/bin/python agentic_research/violin_raw_v1/download.py
```

출처: https://zenodo.org/records/4553940

- 파일 `Participants_Raw_data.rar`, 2,203,971,621 bytes, MD5 `b7e8fef0b4a3e989319c5c5b38a819c2`, CC-BY-4.0.
- 보관 위치 `~/violin-rne/data/external/blanco2021_raw/`. 16개 연결의 재개 가능한 Range 다운로드다. 마지막 전체 byte count/MD5/SHA256 검증을 반드시 통과해야 한다.
- 원음 시간 참조 문제를 해결하기 위해 최초 파일럿2GiB 한도를 이 다운로드에 한해3GiB로 늘린 이유를 코드 영수증에 기록했다. 무제한 추가 다운로드 허가로 해석하지 않는다.
- 아카이브 경로·타입을 검사한 뒤 실제 바이올린 WAV만 추출한다. 음성/Sing, reference, 설문은 분석하지 않는다. 파일명만 보고 악기를 단정하지 말고 원자료 설명과 대조한다.
- 참여자·그룹·세션·중복·PCM subtype·sample rate·길이를 먼저 감사한다. MAT의72명과 raw 음원의 참여자 구성이 같다고 가정하지 않는다. 파일 형식과 적격성만 확인하는 단계에서 미래 목표/성능을 탐색하지 않는다.

## 다음 작업 2: 실제 데이터 전 테스트·검토·계약 동결

아래는 **검토 중인 설계이지 동결된 계약이 아니다.** 원음 메타데이터와 직접 선행을 확인한 뒤 타당성을 검토하고, 라벨·점수를 보기 전에 최종 규약을 동결한다.

1. 한 녹음당 고정 원점1개. prefix `[0,1.0)`초, gap0.25초, target `[1.25,1.75)`초. 원점은 파일 시작이며 미래 기반 onset/plateau 정렬을 하지 않는다. 길이 부족 제외는 메타데이터로 결정하고 그 선택 편향을 명시한다.
2. native sample rate,50ms frame/10ms hop, 완전히 각 구간 안에 든 frame만 사용. PCM subtype의 고정 스케일, stereo는 샘플별 평균. 전체 녹음 정규화·리샘플링·양방향 필터 금지. **구간을 자른 뒤** 변환·검사해서 미래 NaN/진폭이 입력 처리에 영향을 주지 않게 한다.
3. 구간 내부 RMS, log RMS, 고정 ACF 기반 pitch/periodicity. 100–1600Hz lag 범위, RMS≥1e-4 및 periodicity≥0.5를 voiced 조건 후보로 검토한다. 이 고정 추출기가 옥타브 오류나 노이즈를 만드는지 합성 음원으로 검증한다. 관측 지표를 사람의 음질/기술/헬름홀츠 정답으로 부르지 않는다.
4. 미래 pitch/power 표준편차 두 개를 목표로 삼는다. voiced frame≥50% 및 최소2개인 경우만 pitch 유효. 미래가 무효인 녹음을 제거하지 않고 별도 guard와 전체 거짓9규칙으로 남기며 유효/무효 성능·coverage를 따로 보고한다.
5. 모든 모델에 같은 prefix 특징60개:3채널×3개 suffix창×6통계=54, voiced fraction3, 목표창과 같은 길이의 prefix pitch/power 표준편차2, prefix guard1. 구현·단위·결측 대체를 먼저 명시한다.
6. 참여자 전체를 train/dev/test로 분리한다. cohort는 분할용이며 모델 입력이 아니다. 참여자/파일명/전체길이/미래 결측은 입력 금지. exact PCM 및 prefix 중복을 감사하고 참여자 간 중복이 발견되면 사전 규약대로 묶거나 중단한다. 분할 비율과 seed는 목표를 보지 않고 고정한다.
7. 두 축 .25/.5/.75 분위 문턱은 **train의 유효 목표만** 사용한다. 9개 규칙은 `pitch_std<=a AND power_std<=b AND guard`이다. 가능한 패턴은 suffix×suffix 사각형9개와 전체거짓1개다. 첼로의 다른 방향 패턴을 재사용하지 말고 전수 검사한다.
8. 미래 원음을 임의 변화/NaN으로 바꿔도 prefix가 bitwise 동일한지, held-out 목표 변화가 train 문턱/스케일러에 영향을 주지 않는지, 참가자·창·중복 누수가 없는지 검사한다.

## 다음 작업 3: 유한한 AI 비교와 성능 판정

현재 `models.py`는 train 데이터만 학습 함수에 받는다. Frequency, source Markov, HGB 각1개와 독립/공동/Markov logit-offset 신경망 각3seed를 비교할 준비다. 총12개 CPU 실행, worker누적1800초, 개별600초, 종료일 **2026-11-06 00:00 KST**. 실패도 예산과 결과에 포함한다.

- 모든 신경망은 같은60개입력·backbone·업데이트·9규칙BCE를 사용한다. 참여자별 동일 가중치를 학습과 주 평가에 적용한다.
- 후보 학습 설정: hidden32/GELU,80epoch,batch128,AdamW lr.003/wd1e-4; HGB100iteration. 실제 코드와 대조하고 변경은 동결 전에만 한다.
- Dev에서 Frequency/Markov/HGB 중 강한 비교군을 고정된 tie 순서로 선택하고, 모든 설정·예측을 고정한 뒤 test를 한 번 평가한다. test를 본 뒤 비교군/seed/라벨을 바꾸지 않는다.
- 주 지표는 참여자별9규칙 평균 log loss의 참여자 평균. 세 seed는 평균 손실로 비교하고 별도 독립 표본으로 세지 않는다. paired participant bootstrap, Brier, 일관성, 유효/무효 목표, 참여자별 실패와 계산비를 함께 보고한다.
- 제안 offset의 의미 있는 추가 이득 후보 기준은 선택한 강한 기준선뿐 아니라 plain joint와 독립 MLP에도 최소2% 개선이다. 사전 검토에서 기준을 확정한다. 통과하지 못하면 방법론의 이득 실패를 그대로 보고한다.
- 조건부 CDF/혼합모형/10패턴 head/Markov offset/coherence projection은 기존 기술이다. 이것들을 조합했다는 사실만으로 새로운 방법이라고 하지 않는다. 직접 선행과 필요한 제거 실험을 검토하고 실측 결과로 주장 범위를 제한한다.
- actual runner, dataset builder, features, protocol, tests, 독립 검토가 완성되기 전에는 학습하지 않는다. 기존 `run_forecast.sh tick`은 첼로 v3용이며 새 raw WAV 연구를 자동 구현·실행하는 명령이 아니다.

## 종료·공유

가설→사전 반박 조건→독립 검토→코드/입력/계약 동결→고정 실행→결과/한계 기록 순서로 진행한다. 자료 타당성 실패나 미개선을 새 모델·seed 무한 탐색으로 덮지 않는다.

각 작업 종료 시 `docs/EXECUTION_STATUS.md`, `docs/WORK_QUEUE.md`와 이 인계 파일의 상태를 갱신한다. 완료 작업과 계획을 구분한다. 코드·문서·허가된 준비 자료는 지정 연구 브랜치에 커밋/푸시할 수 있으나 raw WAV/MAT, 계정정보, 개인 설문, 원시CLI로그는 공개하지 않는다. main 병합·원자료 삭제·외부 이메일은 하지 않는다. 새 다운로드 데이터의 공개는 기존 첼로 준비 자료 공개 허가와 구분한다.

사용자에게는 한국어로 실제 수행 결과와 실패를 알린다. 이 파일은 사용자 요청을 구체화한 공동연구 메모이며 이후 사용자 지시를 우선한다.
