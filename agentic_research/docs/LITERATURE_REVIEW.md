# AI 방향 재설계와 최신 문헌 검토

기준일: 2026-10-30 (Asia/Seoul). 작성: 독립 문헌 검토 에이전트. 이 문서는 연구 방향과 검증 계획이며 성능 결과가 아니다.

## 1. 현재 연구에 대한 판단

2026-10-24 최종 보고서를 읽었다. 현재 성과는 물리 법칙 발견이 아니라 **파형 판정 → 힘순 K-run 경계 → 법칙 양립 → 통계 추정**을 잇는 정확 민감도 분석이다. 동적계획, 조합 탐색, 최소 모순 집합, 독립 재현은 정보과학에 속한다. 따라서 “정보 기술이 전혀 없다”는 평가는 정확하지 않다. 다만 **학습되는 핵심 알고리즘, 새로운 정보 획득 전략, 미관측 조건의 예측 개선**은 아직 최종 성과로 입증되지 않았다. AI 중심 연구로 소개하기에 부족하다는 지적에는 근거가 있다.

새 모델 이름을 추가하거나 LLM에게 분석을 시키는 것만으로 이 문제가 해결되지 않는다. LLM 자동 연구 루프는 연구 수행 도구이고, 논문의 AI 방법론은 그 루프가 제안·학습·검증하는 별도 알고리즘이어야 한다.

현재 데이터는 13도표·39속도 폴더·78,000창이지만, 독립 단위가 78,000개인 것은 아니다. 모든 자료는 과거 연구에서 노출되었다. 새 train/dev 분할은 **회고적 그룹 평가**이며 새 외부 확인이 아니다. 기존 H는 파형 규칙의 출력이므로 이를 맞힌 정확도를 물리 진동 정답 정확도나 음질 정확도로 바꾸어 말할 수 없다.

## 2. 추천: 인증된 경계 능동 획득

작업 제목 후보: **학습된 관측 선택과 정확 인증을 결합한 구조화 신호 경계의 능동 복원**.

질문은 “어떤 물리 법칙이 맞는가?”에서 “모든 파형을 처리하지 않고도, 학습된 정책으로 필요한 관측을 먼저 선택하여 선언한 경계를 정확히 복원할 수 있는가?”로 바꾼다. 악기는 응용 대상이고, 중심 문제는 **부분 관측에서 구조화된 함수값을 최소 비용으로 알아내는 AI 알고리즘**이다.

한 열에는 힘순 라벨 y_1,…,y_n이 있고 n=50이다. 목표 함수 S_K(y)는 처음 K개의 연속 H가 시작되는 위치이며, 없으면 infinity다. 정책은 현재까지 공개된 라벨·공개 좌표·남은 예산만 보고 다음 관측 위치를 고른다. 미질의 라벨과 파형 특징은 평가기가 숨긴다. 모델은 힘·β·속도에서 H 확률을 배우고, 정확 K-run 확률 전파를 통해 다음 질의가 **경계 불확실성**을 얼마나 줄이는지 계산한다.

### 2.1 AI와 인증의 역할 분리

- AI: Logistic/HGB/작은 MLP로 조건부 prior를 학습한다. 이후 boundary likelihood 학습이나 작은 정책 모형을 비교한다.
- 정확 계산: prior를 K-run 자동기계에 통과시켜 경계 분포와 정보이득을 계산한다. 잘못 학습된 prior는 질의를 낭비할 수 있지만 인증의 참·거짓을 바꾸지 못한다.
- 인증: 아직 모르는 모든 비트를 0으로 채운 벡터와 1로 채운 벡터의 첫 K-run 위치가 같을 때만 정확한 경계를 선언한다. 첫 K-run 위치는 비트를 0→1로 바꾸면 늦어질 수 없으므로 두 극단이 같으면 모든 중간 완성도 같은 값을 갖는다.
- 시간·질의 제한에 걸리면 `UNRESOLVED`를 반환한다. 확률이 높다는 이유로 `CERTIFIED`를 쓰지 않는다.

이 인증은 **고정된 H 규칙과 공개된 힘 좌표**에 대한 증명이다. 실제 운동 상태, 다른 규칙, 미측정 힘의 오차까지 보증하지 않는다. 조건부 독립 Bernoulli prior로 계산한 확률은 근사 모형이며, 결정론적 인증과 구분한다.

### 2.2 중요한 정보 경계

기존 경계는 실측 힘으로 정렬한다. 미질의 관측의 실측 힘을 공개하면 그것을 실제 로봇 측정 절감으로 해석할 수 없다. 첫 구현은 **이미 기록된 데이터에서 힘 좌표는 알려져 있고 파형 판정만 비용을 지불해 공개하는 sparse audit**이다. 이 경우 실측 힘 공개가 설계상 허용되며, 쿼리는 파형 특징 추출·검토 단위다. 실제 계산 시간 절감 여부는 특징 추출 비용까지 측정해야 한다.

새 로봇 실험의 절감을 주장하려면 공칭 설정값만 미리 공개하고, 질의 후 실측 힘을 공개하며, 아직 모르는 힘의 순서까지 다루어야 한다. 그 확장은 현재 파일 재생 실험의 주장에 포함하지 않는다.

### 2.3 선행연구와 구별해야 할 점

이 문제는 stochastic Boolean function evaluation(SBFE), equivalence-class determination(ECD), active feature acquisition(AFA)와 직접 연결된다. EC²는 동등한 결론을 주는 가설들을 하나의 클래스로 묶어 필요한 테스트를 선택한다. ECED는 관련된 잡음·상관 조건을 다룬다. 따라서 “AI가 필요한 관측을 고르고 확실할 때 멈춘다” 자체는 새롭지 않다. [S1–S5]

K-run 자동기계와 미분 가능한 DP도 오래된 계열이다. 후보 novelty는 **비단조 힘순 라벨의 첫-run·검열 출력을 다루는 계산 효율적인 획득 기준**, **분류 규칙 변화에 대한 공통 인증**, **그 구조를 반영한 학습 목적함수** 중 실제로 검증되는 부분에 한정한다. 베이스라인보다 좋은 실증 없이 조합만으로 높은 novelty를 선언하지 않는다. [S6]

### 2.4 강한 기준선과 대조군

1. 전체 50개 공개: 정확성 참조, 절감 없는 상한 비용.
2. 힘순 앞에서부터: 실제 첫 경계가 이른 자료에서 매우 강하다.
3. 짝수/홀수 우선: K=2에서 0을 찾으면 여러 후보를 한꺼번에 배제할 수 있는 구조적 휴리스틱.
4. 균등 간격 및 무작위 순서: 같은 초기 관측과 같은 인증기를 사용한다.
5. 점별 불확실성 최대: 학습된 확률은 같고 획득 기준만 다르다.
6. 학습 없는 0.5 prior + K-run 정보이득: 학습의 실제 추가 가치를 분리한다.
7. 학습 prior + K-run 정보이득: 첫 제안 후보.
8. EC²/decision-region 계열: 작은 n에서 정확 구현, 큰 n에서는 동일한 공개 정보의 샘플 근사를 명시한다. 단순 정보이득이 항상 좋은 정책이라는 보장은 없다. [S1–S3]
9. GP level-set/straddle: 비용이 허용되는 작은 조건에서 비교한다. 단조 이분탐색은 H 띠·미검출 때문에 정당한 일반 기준선이 아니며 단조 스트레스 조건에서만 쓴다. [S7–S9]
10. BCE 학습 대 첫 K-run NLL 학습: 모델 크기·입력·학습 예산을 고정하여 구조화 학습의 기여를 분리한다.

새로운 작은 MLP를 복잡한 모델이라는 이유로 채택하지 않는다. HGB가 가장 좋으면 그 결과를 인정한다. 최신 foundation model은 강한 선택적 기준선이지 논문 기여의 대체물이 아니다.

### 2.5 과학적으로 의미 있는 목표와 중단 기준

주 지표는 **인증까지 필요한 관측 수**이며, 그 정확성은 전수 참조와 100% 일치해야 한다. 평균뿐 아니라 폴더별 중앙값·90백분위·worst group, 예산별 인증률, 미결률, 실제 실행 시간·메모리를 함께 기록한다. 미결 사례를 성공 사례의 평균에서 조용히 빼지 않는다. 정책 overhead가 큰데 1개 관측만 줄었다면 실용적 개선이라 부르지 않는다.

효과 기준은 결과를 보기 전에 고정한다. 예를 들어 강한 비학습 기준선 대비 관측 수 10% 상대 감소를 실용 기준으로 정할 수 있으나, 이는 제안된 운영 기준이며 통계적 법칙이 아니다. 모든 그룹에서 개선을 요구할 수는 없지만 평균 개선이 한 속도 폴더의 큰 이득에만 의존하는지 반드시 공개한다. 그룹 수가 작으므로 반복 seed를 독립 현처럼 세지 않는다.

첫 파일럿에서 학습 prior가 0.5 prior·짝수 우선보다 못하면 네트워크를 계속 키우지 말고 calibration, posterior의 독립 가정, 학습 목표를 진단한다. 정해진 3개 방법군·총 실험 예산을 다 써도 강한 기준선을 못 넘으면 “개선 미입증”으로 종료하거나 아래 대안으로 전환한다. 에이전트가 같은 dev 결과를 무한히 보면서 성능이 나올 때까지 반복하지 않게 한다.

### 2.6 인증의 최소 비용이라는 설명 가능한 하한

실제 첫 시작점 s가 알려진 뒤 계산하는 oracle 하한을 둘 수 있다. 시작점 s의 K개 1은 모두 관측해야 하고, 더 이른 각 길이 K 창에는 적어도 하나의 관측된 실제 0이 있어야 한다. 미검출이라면 모든 길이 K 창을 실제 0으로 덮어야 한다. 이는 선 위의 구간을 허용된 0 위치로 찌르는 최소 interval hitting 문제다. 같은 길이의 창들을 순서대로 처리하고 아직 덮이지 않은 창에서 가장 오른쪽 허용 0을 택하는 교환 논증으로 최소 개수를 얻을 수 있다. 구현 전 작은 n 전수 열거로 검증한다.

이 하한은 정답을 본 뒤 계산하므로 배포 가능한 경쟁 방법이 아니다. `정책 질의 수 / 최소 인증 크기`로 구조적 난이도와 정책 비효율을 구분하는 참고값이다. 일반적인 certificate complexity가 새 개념인 것처럼 주장하지 않는다.

작은 합성 n에서는 실제 최적 적응 정책도 별도로 구할 수 있다. 인증된 부분 관측의 가치는 0, 나머지는 `V(obs)=min_i {c_i + q_i V(obs,i=1)+(1-q_i)V(obs,i=0)}`로 두고 메모이제이션한다. 독립 Bernoulli prior에서만 이 간단한 갱신식을 사용한다. n≤8 정도의 모든 부분 상태를 검사하면 정보이득 휴리스틱과 최적 정책 사이 차이를 직접 볼 수 있다. 이는 현실 50개 문제의 최적성 보장이 아니라 작은 문제에서의 정확성·메커니즘 대조다.

## 3. 두 가지 대안 방향

### B. 학습 안내를 받는 정확 공통규칙 검증

기존 두 문턱 공간의 전수 탐색을 학습된 분기·우선순위와 exact checker로 가속한다. NN은 탐색 순서만 고르고 pruning은 검증 가능한 bound로만 한다. 성공 증거와 실패 증명은 기존 독립 checker로 확인한다. Learning to Branch 및 신경망 검증의 lemma learning은 가까운 선행 계열이다. [S10–S11]

장점은 물리 정답 라벨이 없어도 계산 정답이 명확하다는 것이다. 단점은 기존 `joint_rule/results.json`의 전 영역 실행이 약 91.11초이고, `solver_notes.md`에 국소 470×389 영역 약 0.62초가 기록되어 있어 학습을 넣을 필요부터 증명해야 한다는 것이다. 4–6개 실제 의미 있는 판정 변수를 추가했을 때 생기는 병목을 측정하고, hard-coded analytic pruning·벡터화·캐시와 먼저 비교해야 한다. 인위적으로 느린 baseline을 만들어 가속을 주장하지 않는다.

### C. 프로토콜 조건부 구조화 신호 학습

작은 시계열 인코더 또는 고정된 MOMENT/MuQ 임베딩으로 각 파형의 잠재 상태를 추정하고, K-run 분포를 통해 경계 likelihood를 직접 학습한다. 분류 규칙의 문턱을 입력으로 주거나 규칙군 전체의 손실을 함께 평가한다. 이 방향에는 실제 AI 모델의 학습·표현·전이가 중심에 온다. 그러나 현재 H를 반복 학습하면 규칙 모사일 뿐이며, 파형과 규칙 특징을 입력으로 주고 같은 규칙 라벨을 맞히는 평가는 사실상 답을 제공한다.

정당한 목표는 미래 시각/새 세션의 **운영상 경계** 예측이거나, 새로운 외부 자료의 독립 라벨에 대한 전이다. raw waveform의 존재·용량·라이선스를 먼저 확인하고, 창을 섞은 무작위 분할을 사용하지 않는다. 0.2초 고주파 진동·브리지 힘은 음악 의미 분류 데이터와 다르므로 음악 FM의 우월성을 가정하지 않는다. 초기에 대형 모델 fine-tuning보다 frozen embedding + linear probe, MiniROCKET/작은 CNN/TCN, HGB를 비교한다. [S12–S16]

## 4. 2026-10-30 기준 기술 지도와 채택 판단

이 검토는 직접 관련성이 있는 1차 문헌·공식 저장소를 중심으로 한 **표적 검색**이다. 인터넷 전체·모든 최신 기술을 빠짐없이 조사했다고 주장하지 않는다. 검색엔진의 상대 날짜는 사용하지 않고 arXiv 제출·수정일과 공식 게시일을 확인했다. 논문 초록만 확인한 항목은 이론·벤치마크 수치까지 검증했다고 말하지 않는다.

| 계열 | 확인한 자료 | 현재 연구에서의 용도와 판단 |
|---|---|---|
| Tabular foundation model | TabPFN v2 Nature 2025, 공식 저장소는 TabPFN-3.5 기본값을 안내. 공식 블로그에는 2026-10-28 Thinking 업데이트 | 공개 좌표의 prior 기준선 후보. 3.5 사용권·가중치 크기·local 지원을 먼저 확인. 모델 호출 자체는 novelty 아님. vendor의 SOTA 문구를 독립 검증으로 인용하지 않음. [S12] |
| 일반 시계열 FM | MOMENT, ICML 2024 | 실제 파형이 있을 때 frozen representation 비교. 여러 task 지원이 장점이나 bridge-force 데이터에서 검증 필요. [S13] |
| 다변량 예측 FM | Chronos-2, 2025-10-17 | 시간 방향 forecasting을 할 때 후보. 힘순 서로 다른 stroke는 시간계열이 아니므로 그대로 투입하지 않음. [S14] |
| 음악 SSL | MuQ, 2025-01-02 | 마이크 오디오 표현의 기준선. 현재 bridge-force 특징 테이블에는 바로 쓸 수 없음. [S15] |
| 최신 음악 품질 모델 | MuQ-Eval, 2026-03-24 | frozen encoder + 작은 head라는 현실적인 설계 참고. 생성 음악 품질 MOS는 헬름홀츠/힘 경계 정답이 아님. [S16] |
| 최신 오디오 언어모델 | UniAudio 2.0, 2026-02-04; TAC2, 2026-10-27 | 생성·captioning·시간정렬 문제라 현재 정량적 구조화 경계와 직접 맞지 않음. 유행 때문에 주모형에 넣지 않음. [S17–S18] |
| active boundary acquisition | B-ACT v2, 2026-06-12 | boundary-centric acquisition 자체가 새롭지 않다는 직접 경고. video segmentation과 다른 문제구조를 명시해야 함. [S19] |
| 비근시안 AFA | BRiG-AFA, 2026-08-03 | Bellman risk-to-go를 학습하는 대안. 현재 한 단계 정보이득이 제한되면 참고하되 보상/학습비용도 공정 비교. [S20] |
| noisy-label uncertainty | JMLR 2024 label-noise conformal, NeurIPS 2025 online noisy conformal | 분류 문턱 변화는 알려진 균일 label-noise가 아님. 정리의 가정을 충족하지 않으면 coverage guarantee를 가져올 수 없음. [S21–S22] |
| 자동 연구 검증 | DCP v2, 2026-09-28 | 자동 에이전트의 score만으로 discovery를 주장하지 않는 평가 철학 참고. 새로운 사전출판물이며 검증 표준으로 과장하지 않음. [S23] |

사용자 반응·다운로드·GitHub star는 채택과 관심의 지표일 수 있지만 연구 우수성이나 우리 문제 적합성의 근거가 아니다. 현재 추천은 유명 모델을 최대한 붙이는 것이 아니라, **확률 학습 → 구조화된 관측 선택 → 틀릴 수 없는 종료 판정**을 분리해 얻는 실제 이득을 측정하는 것이다.

## 5. 자동 연구 루프에 넣을 연구 관문

1. **문제 동결:** 공개 정보, 숨은 정보, target, 비용, 그룹 분할, 예산과 중단 기준을 JSON으로 고정하고 해시를 남긴다.
2. **정확성:** 모든 짧은 bit vector·partial mask의 인증, K-run posterior의 확률 합, query update를 전수 기준으로 비교한다. 어떤 모델도 checker를 수정하지 못한다.
3. **재현:** ascending/even/random + frozen learned priors + uniform-prior DP가 재현되지 않으면 새 제안을 채점하지 않는다.
4. **동예산:** 모델 학습·policy inference·oracle 추출 비용을 분리해 측정한다. CPU thread 수·seed·hardware를 기록한다.
5. **모형 제안:** 한 라운드에 한 가설과 구체적인 반박 조건만 바꾼다. 결과 파일은 evaluator만 쓴다.
6. **독립 감사:** 최고 dev score만 보지 않고 전체 가족, 실패·미결·최악 그룹·time overhead와 baseline 누락을 읽는다.
7. **새로움 감사:** EC²/ECED, SBFE, active LSE, learned acquisition, differentiable DP에서 그대로 나온 요소와 새 요소를 표로 갱신한다.
8. **확정:** 현재 노출 자료에서 나온 결과는 retrospective라고 남긴다. 새 외부 benchmark/새로 봉인한 미래 데이터 없이 전향적 일반화 성공이라 쓰지 않는다.
9. **종료:** 예산·방법군 소진, 일정 도달, 반복적 무개선, 재현 실패 시 명시적으로 멈춘다. 결론을 예쁘게 만들려고 평가 규약을 바꾸지 않는다.

## 6. 문헌 목록과 확인 깊이

아래 URL은 2026-10-30에 직접 검색·열람했다. `초록`은 원문 전체 검토 완료를 뜻하지 않는다.

- **S1.** Golovin, Krause, Ray. Near-Optimal Bayesian Active Learning with Noisy Observations. NIPS 2010. EC²와 equivalence classes. [공식 논문](https://papers.nips.cc/paper_files/paper/2010/hash/1e6e0a04d20f50967c64dac2d639a577-Abstract.html), [저자 PDF](https://www.cs.cmu.edu/~dgolovin/papers/nips10.pdf). 초록·도입 검토.
- **S2.** Chen, Hassani, Krause. Near-optimal Bayesian Active Learning with Correlated and Noisy Tests. AISTATS 2017 / EJS 확장본. [공식 proceedings](https://proceedings.mlr.press/v54/chen17b.html), [EJS PDF](https://yuxinchen.org/files/papers/chen17eced-ejs.pdf). 초록·§2–3 도입·조건 검토.
- **S3.** Golovin, Krause. Adaptive Submodularity: Theory and Applications in Active Learning and Stochastic Optimization. JAIR 42, 2011. [공식 페이지](https://ocs.aaai.org/Library/JAIR/Vol42/jair42-012.php). 초록. 본 연구의 entropy gain에 그 성질이 자동으로 성립하지 않는다.
- **S4.** Deshpande, Hellerstein, Kletenik. Approximation Algorithms for Stochastic Submodular Set Cover with Applications to Boolean Function Evaluation and Min-Knapsack. TALG 2016. [DOI](https://doi.org/10.1145/2876506). 초록.
- **S5.** Hellerstein et al. Adaptivity Gaps for the Stochastic Boolean Function Evaluation Problem. 2022. [arXiv](https://arxiv.org/abs/2208.03810). 초록.
- **S6.** Mensch, Blondel. Differentiable Dynamic Programming for Structured Prediction and Attention. ICML 2018. [공식 페이지](https://proceedings.mlr.press/v80/mensch18a.html). 초록·도입.
- **S7.** Gotovos et al. Active Learning for Level Set Estimation. IJCAI 2013. [학회 저자 목록](https://ijcai.org/authors/alkis-gotovos/), [저자 발표자료](https://las.inf.ethz.ch/files/gotovos13active-talk.pdf). 메타데이터·방법 개요.
- **S8.** Ngo et al. Active Level Set Estimation for Continuous Search Space with Theoretical Guarantee. ACML/PMLR 2024. [공식 페이지](https://proceedings.mlr.press/v222/ngo24a.html). 초록.
- **S9.** Ishibashi et al. An (epsilon,delta)-accurate level set estimation with a stopping criterion. 2025, v2 2026-06-09. [arXiv](https://arxiv.org/abs/2503.20272). 초록·날짜.
- **S10.** Balcan et al. Learning to Branch. ICML 2018. [공식 페이지](https://proceedings.mlr.press/v80/balcan18a.html). 초록.
- **S11.** Davis, Wu. Learning Lookahead Lemmas for Neural Network Verification. 2026-07-31. [arXiv](https://arxiv.org/abs/2607.29051). 초록. 여기서 learning은 lemma 유도의 의미도 있으므로 모두 신경망 학습이라는 뜻으로 옮기지 않는다.
- **S12.** TabPFN. [Nature v2 논문](https://www.nature.com/articles/s41586-024-08328-6), [공식 저장소](https://github.com/PriorLabs/TabPFN), [공식 블로그 날짜 목록](https://priorlabs.ai/blog). 저장소·공식 게시물 확인. 3.5 모델 보고서 URL은 도구 접근 오류여서 원문 수치 검증 미완료.
- **S13.** Goswami et al. MOMENT. ICML 2024. [arXiv](https://arxiv.org/abs/2402.03885). 초록·게재 상태.
- **S14.** Ansari et al. Chronos-2: From Univariate to Universal Forecasting. 2025-10-17. [arXiv](https://arxiv.org/abs/2510.15821). 초록.
- **S15.** Zhu et al. MuQ: Self-Supervised Music Representation Learning with Mel Residual Vector Quantization. 2025-01-02. [arXiv](https://arxiv.org/abs/2501.01108), [공식 코드](https://github.com/tencent-ailab/MuQ). 초록; 가중치는 이번에 다운로드하지 않음.
- **S16.** Zhu, Li. MuQ-Eval: An Open-Source Per-Sample Quality Metric for AI Music Generation Evaluation. 2026-03-24. [arXiv](https://arxiv.org/abs/2603.22677). 초록.
- **S17.** Yang et al. UniAudio 2.0. 2026-02-04. [arXiv](https://arxiv.org/abs/2602.04683). 초록.
- **S18.** Bhosale et al. TAC2: Timestamped Audio Captioning as composable functions. 2026-10-27. [arXiv](https://arxiv.org/abs/2610.36568). 초록·날짜. 최신이라는 이유만으로 채택하지 않음.
- **S19.** Helvaci, Cheung. Boundary-Centric Clip-Budgeted Active Learning for Temporal Action Segmentation. v2 2026-06-12. [arXiv](https://arxiv.org/abs/2604.15173). 초록·날짜.
- **S20.** Feng, Li, Li. BRiG-AFA: Bellman Risk-to-Go Learning for Non-Myopic Active Feature Acquisition. 2026-08-03. [arXiv](https://arxiv.org/abs/2608.02305). 초록. 정식 peer review 여부 미확인.
- **S21.** Einbinder et al. Label Noise Robustness of Conformal Prediction. JMLR 25, 2024. [공식 PDF](https://jmlr.org/papers/volume25/23-1549/23-1549.pdf). 초록·가정 개요.
- **S22.** Xi et al. Exploring the Noise Robustness of Online Conformal Prediction. NeurIPS 2025. [공식 페이지](https://proceedings.neurips.cc/paper_files/paper/2025/hash/423d0909791493b7c10916fd328c2913-Abstract-Conference.html). 초록. 알려진 균일 잡음률이라는 제한을 유지.
- **S23.** Ning et al. Scores Alone Do Not Prove Discovery: The Discovery Certification Protocol for Auditing AI Research Agents. v2 2026-09-28. [arXiv](https://arxiv.org/abs/2609.09219). 초록·날짜. 자동 연구의 성공을 평가하기 위한 최근 제안으로만 취급.

## 7. 다음 문헌 감사에서 반드시 반박할 질문

- 첫-run target의 exact information gain을 선형/다항 DP로 계산하는 것이 기존 sequential diagnosis/SBFE 문헌에 이미 있는가?
- 가설 분포를 완전한 bit vector 대신 automaton state로 압축한 EC² objective를 정확 계산할 수 있는가? 단순 entropy heuristic보다 실제·이론 모두 나은가?
- BCE 대신 first-run NLL로 학습하는 것은 기존 survival/CRF 목적과 정확히 어떻게 다른가? 단순 다른 손실 적용이면 novelty를 축소해야 한다.
- 공통 판정 문턱이 모든 bit에 미치는 상관을 무시하면 어떤 반례가 생기는가? 규칙별 독립 학습으로 개선을 설명할 수 있다면 복잡한 joint policy가 필요한가?
- 현재 학습 prior 개선은 힘·β의 쉬운 분포 차이만 이용하는가? 더 단순한 경험적 lookup과 calibration이 같은 이득을 내는가?
- 쿼리 비용이 실제 비싼 원시 파형 처리에서 발생하는가? 이미 JSON에 라벨이 있으면 파일 재생에서의 쿼리 수는 계산 시간 절감과 같지 않다.

이 질문에 답하기 전 허용되는 주장은 **“학습으로 관측 순서를 정하고, 고정된 분석 규칙의 경계를 정확히 인증하는 방법을 구현하여 평가한다”**까지다. 성능 개선·신규성·넓은 일반화는 후속 결과로 판단한다.
