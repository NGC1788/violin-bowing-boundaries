# Blanco 2021 바이올린 자료: 실제 다운로드·스키마 감사

2026-10-30. **연속 신호가 실제로 존재하고 Python에서 읽힌다는 것을 확인했다.** 새로운 바이올린 실측 없이 미래 관측 지표 예측의 입력을 만들 수 있다. 다만 현재 허용 범위는 **음향 지표 단독의 frame-index 기반 미래 예측 준비**다. 정확한 초 단위·다중모달 동기화·인간 음질 정답·활 힘 경계의 검증은 완료하지 않았다. 모델을 학습하거나 예측 점수를 계산하지 않았다.

## 확보와 재현

출처는 Blanco, Tassani, Ramirez의 [공식 Zenodo 4554201](https://zenodo.org/records/4554201)이다. 다운로드 전에 공식 API에서 파일 크기·checksum·CC-BY-4.0을 다시 확인했다. 연구나 파생 자료를 배포할 때 원저자·DOI와 변경 내용을 표시한다. 원자료는 Git 저장소 밖 `~/violin-rne/data/external/blanco2021/`에 있다.

- archive: `Sound_and_Motion_Data.rar`, 468,988,201 bytes.
- MD5: `352be95f849ebeeb76a3cc5e761208e7` — 다운로드 전체 일치.
- SHA-256: `dde1717818007970e499ec7c451991a4ab3386dfab865270511fe1e5269dcc55`.
- 주요 MAT: 469,462,149 bytes, SHA-256 `fef94d704500624c8391e1caaf3c20ec764f606ae17f48d81a7e15932db31822`.
- 27개 archive 항목을 먼저 열거하고, MAT·MATLAB 설명 소스·README만 추출했다. 받은 MATLAB 코드는 실행하지 않았다.
- 원자료·추출본·분리된 parser 패키지를 합쳐 약 1.01 GB로, 2 GiB 한도 안이다. 원자료를 삭제하지 않았다.

MATLAB v5의 `classdef` 객체다. 현재 SciPy `whosmat`는 `NoneType` 오류를 내므로 원자료를 읽었다고 가장할 수 없었다. [제작자 mat-io](https://github.com/foreverallama/matio) 1.0.2를 별도 폴더에 설치해 객체 속성을 해석했다. 기존 연구 환경의 NumPy/SciPy와 고정 평가 코드는 바꾸지 않았다. M2에서 전체 해석 약 2.7초였으며, 이는 파일 읽기 시간이고 모델 성능이 아니다.

```bash
cd "$HOME/violin-bowing-boundaries"
audit="agentic_research/forecast_v3/violin_data_audit"
raw="$HOME/violin-rne/data/external/blanco2021"
py="$HOME/violin-bowing-boundaries/.venv-mac/bin/python"

"$py" "$audit/download_blanco.py" --data-root "$raw"
"$py" "$audit/extract_metadata.py" --data-root "$raw"
uv pip install --python "$py" --no-deps --target "$raw/python_packages" \
  -r "$audit/requirements-audit.txt"
"$py" "$audit/audit_mat_schema.py" --data-root "$raw"
```

다른 환경에는 libarchive `bsdtar`의 RAR5 지원, NumPy/SciPy, python-dateutil도 필요하다. Homebrew의 현재 `7zz`에는 RAR 해독기가 없어서 `bsdtar`를 썼다. 임의 MATLAB 코드를 실행하는 경로는 없다. 원자료나 설치된 parser를 Git에 복사하지 않는다.

## 실제 읽은 구조와 수량

| 항목 | 실제 결과 |
|---|---:|
| 원자료의 참가자 객체 | 72: CG 28, FG 29, EG 15 |
| 음향 raw trial cell | 3,161 |
| 음향 3채널 내용이 서로 다른 시퀀스 | **2,117** |
| 중복·alias 초과 cell | **1,044**, 참가자 사이 동일 내용은 0 |
| 채널별 저장 샘플 | 27,589,187 |
| 음향 채널 | `f0_raw_data`, `power_raw_data`, `ap_raw_data` |
| 음향 길이 최소/중앙/최대 | 2,116 / 8,098 / 26,728 frames |
| 같은 cell 안의 음향 채널 길이 불일치 | 0 |
| 음향 NaN/Inf | 0 |
| 동작 trial cell | 3,090 |
| 명시적 timestamp 필드 | 검사한 sound/motion 구조에 없음 |

72명은 원자료 객체 수이며 기존 논문의 제외 기준 적용 뒤 분석 대상 수와 다르다. 이번 후속 연구는 인구통계·설문·건강 관련 값을 입력으로 사용하지 않고 공개 결과에도 내보내지 않았다.

원자료는 `participant → Results_pre/feed/post → Violin_Trials_GS/... → cell당 연속 벡터`로 구성된다. `Voice_Trials_*`는 연구 입력에서 제외했다. 여러 속성이 같은 시퀀스를 참조하므로 3,161개를 독립 시행으로 세면 안 된다. 세 음향 채널 전체를 SHA-256으로 묶었을 때 고유 시퀀스는 2,117개다. 원래 위치와 alias를 보존하며 참가자 안에서 중복을 관리해야 한다.

## 발견한 분석 위험

**1. F0 단위를 Hz라고 읽으면 틀린다.** README는 Hz라고 설명하지만 실제 `f0_raw_data`의 약 95%는 음수다. 인용된 [원래 YIN 구현](https://github.com/lemonzi/matlab/blob/master/yin/yin.m)은 440 Hz 기준 octave 값을 반환한다. 저장 수치와 이 설명은 맞지만, 자료를 만든 extraction 코드는 포함되지 않았으므로 변환 경로 전체를 확인한 것은 아니다. 현재는 `stored_pitch_coordinate`로 명명한다. Hz/cent 변환을 사용할 때는 `Hz=440·2^x`, `cent=1200·x`라는 해석을 명시하고 별도 확인한다.

**2. 시간축이 보존돼 있지 않다.** 논문은 음향 hop 약 0.7 ms, motion 약 86.13 Hz라고 기술한다. 인용된 YIN 기본 hop 32와 44.1 kHz를 가정하면 음향 1,378.125 Hz이고 motion과 대략 16:1이다. 그러나 실제 배열 길이 비는 2.48–92.22, 중앙값 15.78이다. 시작점·종료점·누락 구간의 정렬이 확인되지 않았다. 전체 길이에 맞춰 stretch하면 미래 길이가 prefix에 들어가므로 금지한다. [원논문 방법](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2021.648479/full)

**3. Motion을 즉시 학습에 합치면 안 된다.** 연속 값의 약 66%가 바로 앞 값과 같고, bridge-distance와 speed에는 큰 추적 이상값이 있다. 이는 실제 모든 샘플이 NaN 없이 존재해도 센서가 정상이라는 뜻이 아님을 보여준다. 반복값은 약 30 Hz 측정을 더 촘촘히 기록한 결과와 양립하지만 생성 과정을 확정한 것은 아니다. `bowPosition_raw_data`는 없거나 비어 있다.

**4. raw와 aggregate의 인덱스를 무조건 맞추면 안 된다.** 70개 참가자×조건 항목에서 raw trial 개수와 저장된 `pitch_stab`/`dyn_stab` 또는 motion trial 개수가 다르다. GS-task의 추가 cell 상당수는 앞 cell의 정확한 복사다. 동일 인덱스끼리 계산한 raw 표준편차와 저장 aggregate도 꼬리에서 크게 다르므로, 이를 일치하는 ground truth로 취급하지 않는다. 새로운 미래 목표는 해당 미래 raw 구간에서 직접 계산한다.

## 가능한 CPU 파일럿: 별도 프로토콜 제안, 아직 미동결

추천은 처음부터 다중모달·파운데이션 모델을 붙이는 것이 아니라 **음향 지표 3채널의 미래 변동 예측**이다. 파형이 아니라 이미 추출된 지표이므로 오디오 FM을 직접 적용할 데이터는 아니다.

- 한 고유 시퀀스당 첫 prefix 1,024 frames, 간격 128 frames, target 512 frames를 사용한다. 배열 인덱스는 X=`[0:1024]`, target=`[1152:1664]`다. 모든 감사 대상 시퀀스가 이 길이보다 길다. 이후 여러 원점을 만들더라도 같은 시행의 창은 같은 split에 있어야 한다.
- 단위는 일단 frame으로 고정한다. 위의 YIN rate 가정이라면 약 0.74초 관측, 0.09초 간격, 0.37초 target이지만 정확한 실시간 경고 지연이라고 보고하지 않는다. 인용된 분석 창에는 인접 음향 샘플 의존성이 있으므로 raw extraction의 미래 참조 범위도 해결해야 한다.
- CPU 특징은 prefix 내부 마지막 64/256/1,024 frames의 평균·표준편차·차분·선형 기울기·aperiodicity 분포와 prefix-only 결측/유효 플래그다. 전체 duration, target 통계, 파일명, 참가자 ID, 전체 배열 normalization은 입력 금지다.
- target은 미래 pitch-coordinate 표준편차와 미래 power 표준편차다. 두 지표의 훈련 자료 quantile로 3×3 문턱군을 고정하면 논리적으로 일관된 공동확률 학습을 시험할 수 있다. 임상적/교육적 정상 기준이나 사람의 음질 평점은 아니다. threshold 선택은 train에서만 한다.
- 먼저 persistence·선형 AR/ridge·HGB·작은 causal TCN을 비교한다. 다음에 같은 backbone의 독립 head, 공동 pattern head, 독립 head의 사후 일관성 보정을 비교한다. 기존 첼로와 센서·목표가 다르므로 단순 zero-shot 전이 성공을 주장하지 않는다.
- 후보 분할은 **참가자 전체 단위 49/10/13명**(CG 19/4/5, FG 20/4/5, EG 10/2/3)이다. 이름을 정렬한 뒤 고정 seed로 분할하고 manifest hash를 남긴다. 아직 실제 분할 파일이나 점수를 생성하지 않았다. 모든 phase·trial·alias가 해당 참가자의 한 split에만 속해야 한다. 원자료의 오래된 outlier 삭제를 자동 승계하지 않는다.
- 점수 전에 주 log loss/Brier, 회귀 기준선, 그룹별 손해 한계, 계산 예산, 반복 횟수와 중단 기준을 별도로 동결한다. 72명의 새 음향 자료를 살펴보았다는 사실을 기록하고, 개발 중 전체 test 목표를 반복 열람하지 않는다.

**관문 판단:** 음향 frame-defined 미래 예측의 자료 가용성은 통과. 초 단위 실시간성, 센서 결합, 물리/인간 품질 정답은 보류. 신호의 미래 의존성을 추가 확인하기 전에는 엄격한 causal benchmark라는 표현도 보류한다. 가장 단순한 미래 예측이 기준선을 못 넘으면 이 자료에서 큰 모델 탐색으로 확대하지 않는다.

## 추가로 찾은 원음과 출처 기록

같은 제작자의 [Zenodo 4553940](https://zenodo.org/records/4553940)에는 별도의 `Participants_Raw_data.rar`가 있다. 공식 API로 **2,203,971,621 bytes**, CC-BY-4.0, MD5 `b7e8fef0b4a3e989319c5c5b38a819c2`를 확인했다. 이 파일은 현재 2 GiB 예산을 넘어서 **받지 않았다**. 따라서 “Blanco 연구에 WAV가 전혀 없다”는 표현은 정정해야 하며, 정확한 표현은 “이번 469 MB 패키지에는 WAV가 없다”이다. 원음까지 후속 확보하면 sample rate·duration과 raw descriptor의 대응을 더 엄격하게 확인할 수 있다.

기계 판독 증거는 `download_provenance.json`, `extraction_receipt.json`, `archive_listing.txt`, `schema_audit.json`이다. 원자료, 중간 waveform, 개인 설문 값, parser 설치본은 Git에 포함하지 않는다. 코드·검사 결과만 재현 가능한 형태로 공개한다.
