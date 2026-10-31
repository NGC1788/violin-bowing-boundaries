# 바이올린 미래 지표 예측: 독립 방법·누출 검토

검토일: 2026-10-30. 범위: Blanco 공개 자료의 저장 음향 지표 감사, 원음 전환 및 동결 전 12-job 설계. 이 검토는 모델을 학습하거나 새 target 결과를 열람하지 않았다. 원자료 설문 응답은 분석하지 않았다. 앞선 schema 감사의 집계값과 공개 코드·방법을 근거로 한다.

**후속 결정: 저장 MAT를 주 실험에 쓰는 계획은 학습 전에 보류했다.** 원음 아카이브를 확보해 시간 구간이 명시된 특징을 직접 추출하는 방향으로 전환한다. 아래 §1–2의 MAT 논의는 전환 이유를 보존한 감사 기록이다. 현재 권고는 §7의 raw-WAV 계획이며, 실제 파일 길이·경로 검사가 끝나기 전에는 어떤 프로토콜도 동결 완료로 표시하지 않는다.

## 판단

**조건부 진행 가능**하다. 연구 질문은 “처음 관측한 바이올린 음향 지표로 이후 구간의 pitch-coordinate·power 변동을 확률적으로 예측할 수 있는가; 공동 판정 제약과 간단한 상태 전이 사전분포가 동일 입력의 독립 신경망보다 도움이 되는가?”로 한정한다. 실제 바이올린 자료와 학습 알고리즘을 사용하므로 물리 데이터의 사후 분석에만 머물지는 않는다. 그러나 사용자의 음질·숙련도·교육 효과를 평가하는 연구도, 최소 활 힘 법칙을 검증하는 연구도 아니다.

**가장 중요한 수정은 원음에서 prefix·target을 분리해 직접 특징을 만드는 것**이다. 공개 YIN 구현은 33 ms 분석 창 외에 양방향 이웃 선택을 사용한다. 원래 128-frame gap은 불충분할 수 있고, 검토 중 제안했던 512-frame gap도 제작자의 실제 추출 설정과 전처리를 증명하지 못한다. 따라서 MAT 계획을 실행하지 않고 §7의 원음 계획을 우선한다.

## 1. 시간 지지 구간: 실제 소스에서 확인한 문제

원논문 §3.2와 배포 `participant_results.m`은 YIN의 창 33 ms, hop 0.7 ms를 기록한다. 공개 MAT에는 원래 YIN이 반환하는 `sr`, `hop`, `minf0`, `wsize`, `shift` 등의 설정과 waveform 대응표가 없다. 따라서 시간·단위 문구를 그대로 재현성이 확보된 설정으로 취급하면 안 된다. [Blanco et al., 2021](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2021.648479/full)

인용된 YIN 저장소를 실제로 읽었다. `yin.m` 기본값은 `minf0=30`, `hop=32`, `shift=0`이며, `yink.m`은 저역 평활화, 시간·lag 방향 차분 적분, 이웃 중 가장 좋은 추정치 선택을 수행한다. 이웃 선택 범위는 `2*round(maxlag/hop)`이고 C 구현은 이를 현재 위치의 좌우로 배분한다. 즉 “window 33 ms”만으로 look-ahead 상한을 계산하면 빠진 연산이 있다. [YIN yin.m](https://github.com/lemonzi/matlab/blob/master/yin/yin.m), [yink.m 고정 커밋](https://github.com/lemonzi/matlab/blob/41fb42fa7f76a63a49b0cfc4e1a0844d89ab2895/yin/private/yink.m), [mininrange.c](https://github.com/lemonzi/matlab/blob/master/yin/private/src/mininrange.c)

독립적으로 계산한 기본 설정의 보수적 지지 폭은 다음과 같다. 이는 제작자가 그 설정을 실제 사용했다는 확인이 아니다.

- `sr=44100`, `minf0=30`, `hop=32`라면 `maxlag=1470`, 이웃 반경은 46 frames다.
- 현재 차분 함수는 약 46-frame lag와 46-frame 적분 창에 걸쳐 원음을 읽는다. 선택된 이웃의 차분 함수까지 고려하면 지지 폭의 보수적 envelope는 약 `46 + 45 + 46 + 46 = 183` frames다. 작은 평활화·반올림 경계는 별도다.
- 기존 prefix 마지막 frame 1023과 target 첫 frame 1152의 간격은 129 frames다. 두 지지 구간이 겹칠 가능성을 배제하지 못한다. 공통 출력 정렬 offset은 두 구간의 간격을 늘리지 않는다.
- 권장 새 정의는 prefix `[0:1024]`, gap `[1024:1536]`, target `[1536:2048]`이다. 앞선 감사에서 최소 길이는 2116이므로 별도의 target 길이 선별 없이 적용할 수 있다. 기본 구현 envelope에는 넉넉하지만, 알려지지 않은 전처리까지 해결했다는 뜻은 아니다.

**즉시 가능한 검증과 남은 한계를 구분한다.** 추출된 배열의 미래 구간을 바꿔도 X가 불변인지는 검증할 수 있다. 그러나 이 검사는 이미 만들어진 prefix 지표가 원음의 미래를 사용했는지 검사하지 못한다. 실제 원음과 정확한 추출 코드를 확보한 뒤, 미래 원음을 바꿔 prefix 지표를 재추출하여 불변성을 시험해야 엄격한 causal benchmark가 된다. 현재 source의 기본 계산을 복제하는 것만으로 과거 제작자의 처리 경로를 증명할 수 없다.

## 2. 필요한 데이터 경계와 목표 정의

앞선 수치 감사는 72명, 원시 cell 3161개, 고유 세 채널 시퀀스 2117개를 확인했다. 중복 1044개는 주로 조건별 alias와 마지막 cell 복사이며 참가자 사이의 완전 중복은 발견되지 않았다. 이 수는 새 독립 시행 3161개가 있다는 뜻이 아니다. [기존 감사와 기계 판독 결과](../forecast_v3/violin_data_audit/README.md)

- 세 채널 전체의 동일 길이·동일 값 hash로 중복을 제거하고 alias 목록을 보존한다. 시퀀스 선별·통계 계산에 설문, 숙련도 응답, 원논문의 outlier 선별을 사용하지 않는다. 동일 참가자의 모든 phase·trial·alias를 반드시 한 split에 둔다.
- 49 train / 10 dev / 13 test 참가자를 고정한다. CG·FG·EG 층화는 표본 구성 보존을 위한 split에만 사용한다. participant ID, 그룹명, 조건명, 파일명, 전체 길이를 모델 입력으로 주지 않는다. 이 분할은 새 예측 작업의 분할이며 “이 연구에서 이전에 한 번도 본 적 없는 원자료”라고 부르지 않는다.
- 입력은 prefix 안에서만 계산한 54개 요약과 마지막 512-frame pitch/power 표준편차 2개다. 모든 모델에 같은 정보를 제공한다. 전체 trial 평균으로 중심화하거나 전체 길이로 resample하는 것은 금지한다. scaling·imputation·clip 범위도 train에서만 정한다.
- target은 해당 미래 구간에서 새로 계산한 두 표준편차다. 저장된 `pitch_stab`/`dyn_stab` aggregate와 cell 인덱스를 맞춰 정답으로 쓰지 않는다. 앞선 감사에서 개수·수치 불일치가 있었다.
- 저장 pitch는 음수가 많고 인용된 YIN은 octave-relative-to-440 좌표를 반환한다. 확인 전에는 Hz 또는 cents 오차라고 이름 붙이지 않는다. `stored_pitch_coordinate` 변동이라고 쓴다. power 역시 raw coordinate를 쓴다. 이 좌표의 변화는 추정기 오류·aperiodicity와 섞일 수 있다.
- train의 미래 변동으로 각 축의 3개 quantile 문턱을 한 번 정한다. 이는 교육적으로 정당화된 좋음/나쁨 기준이 아니다. `<`/`≤`와 quantile interpolation을 고정하고 ties를 검사한다. 문턱이 중복되면 임의 jitter로 3개처럼 만들지 않는다. 유효 rule 수를 보고하거나 동결 전에 설계를 수정한다.

두 변동이 각각 문턱보다 작을 사건을 3×3으로 정의하면 4×4 bin 중 한 축이 모든 문턱을 넘은 7개 bin은 같은 all-zero 판정이 된다. 따라서 가능한 9-bit 패턴은 10개다. 이 finite-grid 표현은 9개 문턱의 논리적 정합성을 제공하며, grid 밖 연속 문턱의 확률까지 식별하지 않는다.

## 3. 최대 12 jobs: 비교의 의미와 공정성

추가 architecture 탐색 없이 다음 계획은 적절하다. 모든 고정값·split·feature 순서·target·문턱·seed를 실행 전에 hash로 동결한다.

| 작업 | 수 | 역할 |
|---|---:|---|
| source frequency | 1 | 미래 패턴의 train 빈도. 입력 없이도 가능한 성능 |
| source Markov | 1 | prefix 마지막 512-frame 상태 → 미래 10-pattern 전이표, Dirichlet(1) 평활화 |
| HGB, seed 0 | 1 | 비신경망의 강한 표형 데이터 기준선 |
| 같은 MLP backbone + 독립 9 logits, BCE | 3 seeds | 공동 제약 자체가 도움이 되는지 판단하는 필수 비교 |
| 같은 MLP backbone + 10-pattern head, marginal BCE | 3 seeds | finite-grid 정합성을 구조로 강제하는 대조군 |
| 같은 MLP backbone + Markov offset 10-pattern head, marginal BCE | 3 seeds | 단순 전이 사전분포에 대한 잔차 학습 후보 |
| 합계 | **12** | 실패 seed까지 모두 공개 |

MLP의 hidden=32, 80 epochs, AdamW lr=0.003, batch=128, weight decay=1e-4를 공통 고정하는 것은 이 작은 파일럿에 합리적이다. dev 결과를 본 뒤 epoch나 규제를 바꾸면 같은 시험의 연속으로 취급하지 않는다. 독립/공동 모델은 같은 입력, backbone, batch 순서와 seed를 사용하고 output parameter 수 차이를 기록한다. 세 seed 평균 확률을 사전 지정된 모델군 예측으로 쓰고 best seed를 고르지 않는다.

offset 후보는 다음처럼 명시한다.

\[
\pi_k(x)=\operatorname{softmax}_k\{\log \widehat P_{\rm train}(k\mid s(x))+r_k(x)\}.
\]

`s(x)`는 관측 prefix의 두 변동으로 만든 상태다. 미래 target 상태를 넣지 않는다. 전이표는 train만 사용하며 Dirichlet 평활화 후 모든 확률이 양수여야 한다. 무관측 prefix 상태의 fallback도 train 빈도로 고정한다. joint 모델과 같은 marginal BCE를 사용해야 head 구조 효과를 loss 변경 효과와 섞지 않는다. 이 모델은 **표준 offset/잔차 보정의 적용**이며 새로운 통계 원리로 부르지 않는다.

## 4. 점수보다 먼저 통과해야 할 검사

1. **경계 불변성:** dev/test 미래 target을 전부 교란해도 split, train quantile, scaler, Markov 표, X와 학습 결과가 바뀌지 않아야 한다. 한 시퀀스의 `[1024:]`를 교란하면 그 시퀀스 X가 bitwise 같아야 한다. 해당 시퀀스 target은 바뀌어도 정상이다.
2. **중복·그룹 분리:** 참가자 집합 교집합이 0, sequence hash split 교집합이 0. 중복 cell의 multiplicity가 학습 weight와 test 표본 수를 늘리지 않아야 한다. 각 참가자 trial 수를 공개하되 개인 응답은 공개하지 않는다.
3. **사건 정확성:** train 문턱의 바로 아래·같은 값·바로 위에서 10-pattern 코딩과 9-rule decoding이 직접 조건식과 같아야 한다. all-zero edge도 포함한다. 임의 softmax 질량에 대해 정규화·확률 범위·grid의 2차 직사각형 질량 비음성을 확인한다. 독립 head는 위반을 숨기지 말고 보조 지표로 보고한다.
4. **학습 가능성 및 회계:** synthetic 작은 자료에서 gradient와 loss 감소만 검사한다. 실제 test로 smoke test를 하지 않는다. 출력 smoothing/clipping은 모든 모델의 평가에 동일 적용하고, 실패·비수렴도 12 jobs에 포함한다.

## 5. 평가·중단 기준

주 지표는 participant-macro 9-rule log loss다. 먼저 한 참가자 안의 고유 시행과 9개 rule을 평균하고, 참가자들을 같은 weight로 평균한다. 2117개 행을 독립 표본으로 bootstrap하지 않는다. 13명 test의 불확실성은 크게 남을 수 있다.

dev screen은 offset 후보가 **best(HGB, Markov) 및 plain joint보다 각각 최소 2%** 낮은 loss인지를 확인한다. 정보기술 기여를 주장하려면 독립 MLP와도 별도로 비교해야 한다. 모든 기준선과의 실제 차이를 그대로 표시하고 기준선이 약할 때 단순히 “AI 향상”이라고 결론내리지 않는다. dev gate를 통과하지 못해도 고정된 12개를 최종 진단용으로 한 번만 평가할 수는 있지만, 이를 confirmatory success로 승격시키거나 후속 tuning에 test를 쓰지 않는다.

test에서는 세 seed 평균 예측으로 모델군을 정의하고, 동일 참가자들을 함께 재표집하는 paired participant bootstrap으로 loss 차이 CI를 구한다. seed를 추가 참가자로 세지 않는다. point estimate가 좋아도 CI가 0을 포함하면 우월성 미확정이다. Brier, 각 rule, CG/FG/EG, 계산 시간·메모리·정합성 위반은 보조 지표다. 작은 하위집단 숫자로 교육 효과나 전문가/초보자 구분 성능을 주장하지 않는다.

**구조 기여의 반증 조건:** offset이 Markov만큼이거나 독립 MLP보다 나쁘면 새 제약의 예측상 이득은 없다. plain joint와 offset이 비슷하면 사전분포가 불필요한 것이다. 손실만 좋아지고 결과가 센서/추정기 artifact에 의존하면 해석 범위를 지표 예측으로 제한한다. 평균 성능 향상 없이 정합성 위반만 없애는 경우에는 그 trade-off를 사실대로 보고한다.

## 6. novelty와 실제 의미의 최종 범위

원 연구가 전체 시행의 변동을 계산했다는 사실과, 우리가 관측 prefix로 나중 구간의 불확실성을 예측한다는 것은 질문이 다르다. 따라서 새로운 검증 대상은 있다. 성공한다면 소량의 공개 바이올린 지표에서 미래 변동 예측이 가능한지와 구조적 출력이 실제 도움이 되는지에 관한 재현 가능한 AI 응용 결과가 된다. 이 문서는 “바이올린 미래 변동 예측의 최초 연구”를 증명하지 않는다.

공동 분포에서 논리적으로 일관된 임계값 확률을 얻는 원리와 직접 CDF 학습·질의는 이미 문헌에 있다. 특히 UAI 2020은 조건부 CDF와 tail-area probability를 직접 다룬다. 그러므로 10-pattern head, MDN, conditional flow, CDF를 붙였다는 사실로 방법론 novelty를 주장할 수 없다. [Chilinski & Silva, 2020](https://proceedings.mlr.press/v124/chilinski20a.html)

**현재 판정: 응용 의미는 충분히 시험할 가치가 있고, 방법론 novelty는 낮다.** 신규성이 강한 방법이라고 이름부터 붙이지 않는다. 위 12 jobs에서 matched independent head와 강한 단순 기준선을 이기는 명확한 결과가 있어야 다음 문제를 고민할 근거가 생긴다. 실패하면 이 데이터에서 자동 모델 탐색을 계속하지 않는다. 성공해도 실시간 조기 경보, 청각적 음질, 연주 향상, 물리적 힘 경계로 결론을 확대하지 않는다.

이상은 MAT 계획 검토의 기록이며 후속 결정으로 원음 검사를 먼저 한다. 현재 실행 순서는 §7 마지막에 정의한다. 점수 개선을 위해 dataset·문턱·task를 계속 갈아타는 루프는 금지한다.

## 7. 원음 전환 후 권고 — 현재 적용할 계획

앞 절의 MAT prefix/gap은 실행하지 않는다. 동일 연구의 별도 [Zenodo 원음 아카이브](https://zenodo.org/records/4553940)는 앞선 공식 API 확인상 CC-BY-4.0, 2,203,971,621 bytes였다. 현재 root가 다운로드와 메타데이터 검사를 담당한다. 이 검토는 WAV를 이미 확인했다고 주장하지 않는다. 음향 task는 유지하되, 계산자가 원음 시간 지지 구간을 통제할 수 있게 만드는 전환이다.

### 7.1 입력·구간·추출

- 파일 allowlist를 명시한다. 실제 바이올린 연주인 `pitch.wav` 계열만 쓰고 `Sing`/voice와 reference tone은 제외한다. 이름만으로 불분명한 파일은 임의 해석하지 말고 schema 검사에서 보류한다. 제외 수를 이유별로 공개한다.
- participant 49/10/13 split은 유지하되 누락 참가자나 서로 다른 이름 alias가 나오면 점수 전에 정확한 identity를 해결한다. PCM content hash에 sample rate·channel 해석을 포함하여 완전 중복을 검사한다. WAV header 차이 때문에 같은 PCM이 다른 시행으로 남지 않게 한다.
- 고정 origin은 recording time 0이다. waveform 전체에서 onset을 찾거나 최대 에너지 구간을 정렬하지 않는다. prefix `[0,1.0)`초, gap `[1.0,1.25)`초, target `[1.25,1.75)`초를 우선 제안한다. 1.75초 미만 파일의 비율을 target 값 없이 metadata로 확인하여 적용 가능성을 결정한다. 미래 길이로 모델을 보조하거나 조용히 다양한 origin을 시험하지 않는다.
- native sample rate를 유지한다. PCM subtype에서 정해진 scale로 float 변환하고, 녹음 전체의 peak/RMS를 이용한 normalization은 하지 않는다. 다채널 처리도 미리 고정한다. 기본은 첫 녹음 채널이며, 채널 선택을 target 품질에 따라 바꾸지 않는다.
- 각 구간 내부에 **완전히 들어가는** 50 ms frame, 10 ms hop을 사용한다. frame 양쪽을 주변 구간에서 pad하지 않는다. local mean 제거, frame RMS/log-power, normalized autocorrelation(ACF)의 100–1600 Hz 범위 pitch 후보와 periodicity를 계산한다. 스펙트럼 전처리를 쓸 때도 해당 frame 안에 한정한다. 구간 밖을 읽는 필터·centered-STFT padding·전체 trial detrending은 금지한다.
- 100–1600 Hz는 G3만 가정하지 않는 고정 허용 범위다. 실제 연주 범위를 보장하거나 octave error를 없애지는 않는다. ACF의 정규화 정의, peak 선택, sub-sample 보간, periodicity threshold를 동결한다. 50 ms에서 낮은 주파수는 적은 주기만 관측되므로 추정기의 분산이 target에 포함된다.

### 7.2 무음·무주기성을 정답에서 숨기지 않는다

target에 reliable pitch frame이 거의 없는데 pitch 표준편차를 0으로 두면 무음을 “완벽히 안정”으로 오인한다. 반대로 해당 시행을 전부 제외하면 미래 결과를 보고 표본을 선택한다. 다음 guard 방식을 권한다.

\[
g=\mathbb 1\{\text{target 내 reliable-pitch frame 비율}\ge 0.8\},\qquad
y_{ij}=g\,\mathbb 1\{\sigma_{\log_2 f_0}\le a_i,\;\sigma_{\log power}\le b_j\}.
\]

reliable frame은 고정 periodicity와 절대 PCM RMS floor를 모두 통과해야 한다. 초기 예로 periodicity 0.6, RMS `1e-4`를 제안하지만 이는 음악적 정답이 아니며, 정확한 ACF normalization·PCM 범위를 확인한 뒤 **점수 전에** 한 번 정해야 한다. target에 reliable frame이 2개 미만이면 pitch 표준편차는 missing으로 남기고 `g=0`으로 처리한다. 모든 `g=0` 시행은 all-zero 패턴으로 남긴다. power 변동은 target의 모든 frame에서 계산해 순간 dropout을 숨기지 않는다. 안정 사건을 정의하는 quantile은 train 중 `g=1`의 target에서만 정하고, 그 표본 수와 ties를 보고한다. 유효 표본이 부족하면 파일럿을 중단한다.

이 방식도 완벽한 음질 기준은 아니다. 모델이 미래 pitch·power 변동보다 단순한 무음/발음 실패를 예측해서 이길 수 있다. 따라서 전체 primary 점수 외에 `g=0`, `g=1`의 수와 보조 성능을 함께 공개한다. guard subset은 해석을 위한 보고이며, 잘 나온 subset만 새로운 주 결과로 바꾸지 않는다. prefix의 동일 guard는 **prefix만** 보고 계산하고, Markov 전이표에는 invalid-prefix 상태를 별도 state로 둔다. target guard를 X에 넣지 않는다.

### 7.3 12-job 제약과 주장의 범위

§3의 12-job 비교, 동일 56개 입력, 동일 학습 설정, participant-macro 점수 및 paired bootstrap 원칙을 유지한다. prefix summary가 유효하지 않을 때는 train에서 정한 imputation과 prefix-only 유효성 표시를 동일하게 제공한다. 고정 56개 입력에 mask를 포함하려면 어떤 기존 통계를 대체할지 실행 전에 명시한다. 숨겨진 추가 feature를 offset 모델에만 주지 않는다.

시작 전 필수 검사는 **원음 suffix를 교란하고 특징을 재추출해 prefix가 정확히 불변인지** 확인하는 것이다. source loader가 WAV 전체를 메모리에 읽는 행위 자체보다, feature 함수가 prefix 이후 값을 사용하지 않는다는 실행 검증이 중요하다. prefix를 별도 잘라 넣어 계산한 결과와 전체 파일로부터 prefix slice만 제공한 결과가 같아야 한다. source/test 교란 불변성, duplicate·participant 분리, threshold coding 검사는 그대로 적용한다.

실제 WAV의 길이·채널·sample rate·바이올린/참조음 구분이 해결되고 위 검사가 통과하면, **신호 구간이 분리된 바이올린 미래 음향 지표 예측**으로 기술할 근거가 생긴다. 여전히 ACF의 output은 사람의 지각적 음질 정답이나 실제 pitch trajectory 정답이 아니다. raw 자료에서 알고리즘을 검증하는 것과 교육적 효과를 검증하는 것은 별도다.

현재의 세 단계는 **(1)** 원음 메타데이터와 prefix 재추출 불변성 검사 후 동결, **(2)** 12 jobs 한 번 실행, **(3)** 고정 gate와 불확실성을 보고하여 유지/중단 결정이다. 새로운 method name이나 더 큰 모델 탐색은 이 관문을 대신하지 않는다.
