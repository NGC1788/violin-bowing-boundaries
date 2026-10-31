# 원음 입력의 시간 지지 검증 — 2026-10-31

상태: **합성 검증 통과, 실제 자료 적합성·독립 검토·실험 계약 동결 전**. 이 파일은 모델 성능 결과가 아니다. 실제 WAV 목표 배열을 만들거나 바이올린 모델을 학습하지 않았다.

## 계산한 것

`features.py`는 파형 배열을 받아 입력 `[0,1)`초와 미래 `[1.25,1.75)`초를 먼저 나눈다. 각 구간을 자른 뒤 dtype 변환, 유한값 검사, 고정 PCM 스케일 변환, stereo 평균을 한다. 전체 녹음의 진폭·길이·미래값으로 정규화하거나 onset을 찾지 않는다. 최소 길이 검사는 자료 적격성 판단이며 예측 특징이 아니다. 이 적격성 조건으로 생기는 선택 편향은 실제 자료 감사에서 별도로 보고해야 한다.

각 구간에 완전히 들어가는50ms frame을10ms 간격으로 둔다. 시작점을 매번 초에서 sample로 반올림하여 누적 hop 오차를 피한다. 입력96개, 미래46개 frame이며, 테스트한8000/16000/22050/44100/48000Hz에서 입력 끝≤1초, 미래 시작≥1.25초, 미래 끝≤1.75초다.

frame 안에서만 평균을 빼고 RMS/dBFS와 FFT autocorrelation을 계산한다. lag별로 겹치는 두 벡터 에너지의 기하평균으로 정규화한다.100–1600Hz에 해당하는 양의 local peak 중 최고 peak의95% 이상인 첫 peak를 선택하고3점 포물선 보간한다. RMS≥1e-4와 periodicity≥0.5를 voiced 조건으로 사용한다. pitch는440Hz 기준 cents, power는20log10(RMS), RMS floor는1e-12다.

앞선 메모의 Hann window 계획 대신 직사각 frame과 overlap-energy normalization을 사용했다. 이는 **실제 음원 목표/성능 열람 전에** 결정한 구현 변경이다. 주기 peak의 위치를 합성 순음과 배음 음원으로 검증했다. ACF 기반 pitch/periodicity는 전통적인 신호처리이며 신규 AI 방법이 아니다. 이 구현을 YIN 또는 McLeod NSDF 원식의 정확한 재현이라고 부르지 않는다. 알려진 pitch, 어택·잡음·배음 구성에 대한 실제 자료 검증은 여전히 필요하다. 강한 고차 배음이나 복수 음원에서의 octave 오류가 해결됐다는 주장도 하지 않는다.

입력은60개다. pitch/power/periodicity 각각 마지막4/16/전체 frame에서6통계(평균, 표준편차,10/90분위, 평균 절대차분, 초당 기울기)를 계산한다. pitch 통계에는 voiced frame만 쓰며 빈 창은0과 별도 voiced fraction으로 나타낸다. 세 창의 voiced fraction3개, 미래와 같은46개 frame 길이의 입력 pitch/power 표준편차2개, prefix guard1개를 더한다. 모든 모델에 동일 입력을 제공할 예정이다.

목표는 미래 voiced pitch 표준편차와 모든 미래 frame의 power 표준편차다. voiced가 절반 이상이고2개 이상이어야 pitch 목표가 유효하다. 그렇지 않으면 pitch는NaN, guard는false로 반환한다. **미래 guard=false 행을 제거하지 않는다.** 데이터 생성기는 이를 전체 거짓9규칙으로 보존하고 유효/무효별 coverage와 성능을 따로 보고해야 한다. 아직 그 생성기는 구현하지 않았다. 학습 문턱·정규화도 이 특징 추출기가 정하지 않는다.

## 실행한 검증

```bash
.venv-mac/bin/python -m pytest agentic_research/violin_raw_v1/test_features.py agentic_research/violin_raw_v1/test_models.py -q
```

**19 passed in2.53s**: 특징16개 사례, 모델3개 시험. 첫 수집에서 상대 import 오류가 발생했고 절대 package import로 고친 후 위 실행을 완료했다. 이를 실험 모델 실패나 실제 데이터 성능으로 세지 않는다.

- 미래/간격 부분을 큰 무작위값·NaN·Inf로 바꾸거나 prefix 뒤를 잘라도 입력 X와 prefix 목표가 bitwise 동일했다.
- 입력을 바꾸거나 목표창 뒤에NaN을 넣어도 미래 목표가 동일했다. 실제 요청 구간 내부의NaN은 오류로 거부했다.
-5개 sample rate의 support 경계를 검증했다.
-196/220/440/880/1318.51Hz 및 배음 음원의 pitch 오차, 알려진 순음 RMS, 고정음과 sweep의 변동 차이를 검사했다.
-PCM_U8/16/24/32, float, stereo의 고정 스케일 일치와 잘못된 subtype/dtype 거부를 검사했다.24-bit는 `scipy.io.wavfile.read`의 left-aligned int32 convention을 입력 계약으로 삼는다. 디코더가 정규화된 float를 반환하면 subtype을 DOUBLE/FLOAT로 전달해야 하며 원래 PCM subtype을 그대로 붙이면 안 된다.
-무음·고정 잡음·미래만 무음인 경우를 제거하지 않고 guard와 별도NaN 목표를 반환했다.
-모델 시험은 train-only 전처리와 held-out 입력을 추가했을 때 기존 예측의 불변성을 검사했다. 실제 참여자 분할·훈련 문턱·held-out 목표 비열람은 아직 실제 데이터 생성 단계에서 확인해야 한다.

## 남은 관문

1. 원음 다운로드 최종 MD5/SHA256 및 아카이브·WAV 헤더 감사.
2. 실제 참여자·반복·내용/prefix 중복·길이·recording origin·악기 구분 확인. MAT의72명/2117시퀀스를 raw 표본 수로 복사하지 않는다.
3. train/dev/test participant split, train-only 문턱, guard 처리와 중복 누수의 합성/실자료 검사.
4. 직접 선행과 강한 기준선/제거 실험에 대한 독립 검토, 별도 계약·코드·입력 동결.
5. 최대12개CPU 실행,1800초 누적/600초 개별, test1회. 성능과 방법론의 추가 이득이 없으면 그대로 종료한다.

학습 결과를 내기 전에 이 관문을 통과해야 한다. 이번 성공은 **구간 밖 값을 특징 계산에 쓰지 않는다는 구현 검증**이며 바이올린 일반화·음질 개선·높은 신규성의 증명이 아니다.
