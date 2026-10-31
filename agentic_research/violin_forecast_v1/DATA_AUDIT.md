# 바이올린 미래 지표 자료 준비: 시간 의존성 관문 보류

2026-10-30. **실제 MAT 목표 자료를 만들거나 저장하지 않았고, 모델을 학습하거나 개발·시험 점수를 계산하지 않았다.** 구현된 것은 재사용 가능한 분할·중복 제거·prefix 특징·학습 측 문턱·판정 패턴 함수와 합성 시험이다. 원음 WAV의 시간 의존성을 먼저 확인한다는 공동연구 결정에 따라 실제 MAT 자료 생성은 중단 상태다.

## 이번에 확인한 정보와 한계

기존 공개 자료 감사는 72개 참가자 객체(CG 28, FG 29, EG 15), 음향 cell 3,161개, 세 채널 내용이 서로 다른 시퀀스 2,117개를 보고했다. 이번 작업에서는 그 수량 감사 문서와 MATLAB **소스 코드**를 읽었으며, 실제 MAT 배열이나 개인 설문·인구통계 값을 추가로 열람하지 않았다.

번들 `participant_results.m`의 음향 속성 설명은 `pitch_stab`과 `dyn_stab`을 각각 저장 pitch/power 벡터의 표준편차로 정의하며 실제로는 instability라는 이름이 맞다고 명시한다. YIN 창 33 ms, hop 0.7 ms라는 설명은 있지만, 실제 추출 호출·인수·수정판·누락값 제거 과정이 포함되어 있지 않다. 저장 aggregate와 raw cell 대응도 기존 감사에서 일부 불일치했다. 그러므로 새 목표는 저장 aggregate를 정답으로 연결하지 않고, 선택한 연속 창에서 직접 계산하도록 설계했다.

인용된 원래 [YIN 구현](https://github.com/lemonzi/matlab/blob/master/yin/yin.m)을 직접 읽었다. 그 코드는 pitch를 `log2(sr/period)-log2(440)`으로 반환하고, 기본 hop은 32, 기본 integration width는 `ceil(sr/minf0)`, 기본 shift는 symmetric이다. 따라서 저장 pitch 좌표를 그대로 Hz라고 부르면 안 된다. 해당 `yin.m`의 확인 SHA256은 `2e2bd713ebec74cc7bded5ec8d0183ad30f700bb134be86af489475f3eb79024`다.

더 중요한 것은 [yink.m](https://github.com/lemonzi/matlab/blob/master/yin/private/yink.m)의 추가 의존성이다. 저역 평활화, 대칭 이동 차분창, 그리고 주변 estimate 중 더 나은 값을 선택하는 단계가 있다. 주변 범위는 `2*round(maxlag/hop)`으로 설정된다. 이 파일의 확인 SHA256은 `9741f9c77dd005514ed28fa12c245b5e50084ef6b4dddb7c88b75190744d258e`다. 따라서 **저장된 descriptor의 창이 겹치지 않는다는 사실만으로 원음의 시간 지지가 겹치지 않음을 증명할 수 없다.** 실제 설정과 원음 대응이 없으므로 128-frame 간격도, 임시로 늘린 512-frame 간격도 엄밀한 causal separation을 보증하지 않는다.

YIN 코드의 `pwr`는 제곱 신호에서 얻는 비음수 선형 power다. 기존 MAT 감사의 저장 power 값은 음수였으므로, 자료 제작 과정에 확인되지 않은 변환이 있다. 로그 power와 양립하지만 dB 변환을 단정하지 않는다. 새 필드명은 `stored_pitch_coordinate_std`, `stored_power_coordinate_std`로 제한한다. 사람의 음질 평가나 교육적 정상 기준이라는 표현도 사용하지 않는다.

## 현재 구현된 인터페이스

`data.py`의 일반 함수는 실제 MAT 없이도 검증할 수 있다.

- `participant_split(participants, seed=20261030)`: cohort 안에서 seed·cohort·익명 배열위치 ID의 SHA256으로 순위를 정한다. CG 19/4/5, FG 20/4/5, EG 10/2/3, 합계 train/dev/test 49/10/13이다. 내용·목표·성능으로 순서를 정하지 않는다.
- `deduplicate(records)`: 세 채널 전체 값·길이·채널 이름을 함께 해시한다. 동일 참가자 안의 정확한 alias는 하나로 합쳐 위치 목록을 남기고, 참가자를 가로지르는 동일 내용은 오류로 중단한다. 내용 해시는 정체성 감사용이며 모델 입력이 아니다.
- `sequence_features(channels)`: `[0:1024]` 안에서만 마지막 64/256/1024 frame의 세 채널 평균·표준편차·10/90분위·평균 절대차분·기울기를 계산한다(54개). 같은 관측 정보를 모든 모델에 주기 위해 마지막 512 frame의 pitch/power 표준편차 두 개를 더하여 **56차원**이다. 학습하지 않은 고정 요약이며 스케일러는 호출자가 train에만 적합한다.
- `sequence_target(channels)`: 임시 MAT 설계는 `[1536:2048]`의 두 표준편차(`ddof=1`)다. prefix `[0:1024]`, gap 512 frames로 변경되었으나 **이 변경은 미검증 descriptor 인과성을 해결했다고 주장하지 않는다**. WAV 설계가 결정되면 새 시간 단위·창 계약이 필요하다.
- `fit_thresholds(targets, split, participant_ids)`: **train 행만** `np.quantile(method='linear')`에 제공하여 각 축의 .25/.5/.75 문턱을 얻는다. 문턱이 중복되면 임의 jitter 없이 실패한다. 적합 참가자 목록·해시를 영수증에 남긴다.
- `labels_and_patterns(targets, thresholds)`: `(pitch_std<=a) & (power_std<=b)`인 아홉 규칙을 계산한다. 양축 모두 문턱이 커질수록 참이 늘어난다. 양성 영역은 suffix×suffix인 9개 비어 있지 않은 사각형과 전체 거짓 1개, **총 10패턴**이다. 기존 첼로 규칙의 반대 방향 축을 그대로 재사용하지 않았다.

`assemble_dataset`은 먼저 참가자 분할을 적용하고 prefix X를 완성한 다음 미래 목표를 계산한다. 반환값은 `X`, `targets`, `prefix_targets`, `thresholds`, `Y9`, `pattern_ids`, `prefix_Y9`, `participant_ids`, `cohorts`, `split`, `sequence_ids`, alias 목록·출처 영수증이다. 원시 prefix 시퀀스는 이번 작은 비교에서 사용하지 않으므로 캐시에 넣지 않는다.

MAT 전용 `build_dataset(data_root, protocol)`는 `data_build_approved=true`와 `timing_scope=retrospective_stored_descriptor` 없이는 소스를 열기 전에 중단한다. **현재 이 승인을 사용해 호출하지 않았다.** 향후 생성물이 생기면 `local/` 아래 비공개 ignored 경로에만 저장한다. 데이터·참가자별 목표·설문은 GitHub에 추가하지 않는다.

## 실제 수행한 시험

```bash
.venv-mac/bin/python -m pytest -q agentic_research/violin_forecast_v1/test_data.py
```

**합성 시험 10개 통과**: 정확한 참가자 분할, 순서 불변성, 미래 변경 시 X의 bitwise 동일성, prefix 변경 시 미래 목표 동일성, held-out 값을 크게 바꾸거나 NaN으로 만들어도 학습 문턱 동일, 참가자 간 split 분리, alias 처리, 참가자 간 내용 복제 거부, 512개 이진 벡터 중 정확한 10패턴, 동등 문턱 처리, 전체 길이 정보 배제, 중복 분위 실패, 비공개 캐시 영수증·손상 검출, 승인 없는 실제 자료 생성 차단을 포함한다.

기울기 계산에서 BLAS dot의 메모리 정렬별 극소 반올림 차이가 변이 시험에 드러났다. 고정된 NumPy 요소별 곱·합으로 바꿔 동일한 prefix의 재계산이 bitwise 동일함을 확인했다. 이는 누설 발견이 아니라 결정론적 구현 정리이며, 결과를 근사 비교로 완화하지 않았다.

현재 관문은 **원음 기반으로 시간 지지가 명확한 특징을 정의할 때까지 실제 실험 보류**다. 원음 도착 뒤의 별도 검토·새 계약 없이 이 준비 함수를 실제 MAT 성능 실험으로 이어가지 않는다.
