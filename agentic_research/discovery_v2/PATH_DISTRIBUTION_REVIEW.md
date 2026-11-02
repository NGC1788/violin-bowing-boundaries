# 미래 창의 표본분산을 경로 분포에서 전파하는 방법: 제한 문헌 검토

2026-11-02. 문헌·수학적 명세만 검토했다. 원자료·정답·행별 예측·가중치를 열거나 학습·실험을 실행하지 않았다. 로컬 문맥은 `agentic_research/AGENTS.md`, `README.md`, `docs/RESEARCH_PLAN.md`, `discovery_v2/AGENTS.md`, `discovery_v2/PLAN.json`이다. 이미 종료한 캠페인이나 현재 v2의 실행 명세를 수정하지 않는다. 이 검토의 부정적 조건은 해당 가설의 반박 기준이며 사용자가 허용한 다른 탐색 전체를 닫는 이유가 아니다.

**판정: 직접 MDN과 구별되는, 시험할 만한 구조적 가설은 있다. 그러나 Gaussian 경로→중심화 이차형식→일반화 비중심 χ²라는 조합 자체는 새 방법이라고 할 수 없다.** 가장 가까워 보인 “High Order Statistics” 예측 논문은 원문에서 평균·추세 같은 **선형** 집계만 다룬다. 표본분산이라는 비선형 집계와 그 예측 확률을 학습하는 문제는 그 논문의 직접 해법 밖이다. 남는 연구 가치는 작은 표본에서 이 구조가 실제 통계 효율을 주는지, 두 채널의 공동 변동·미래 관측 마스크까지 정확히 처리할 수 있는지에 있다. 이 제한 검색으로 최초성이나 새로운 이론의 존재를 주장하지 않는다.

## 1. 가까운 직접 출처 6개

“원문 부분 검토”는 초록뿐 아니라 아래에 명시한 본문 구간을 읽었다는 뜻이다. 모든 증명·코드를 독립 검증하거나 구현을 실행했다는 뜻은 아니다. 2025 출판과 2026 개정·공식 소프트웨어까지 확인했다. 임의로 최신 모델 목록을 늘리지는 않았다.

| 출처와 확인 수준 | 직접적인 겹침 | 남는 차이·주의점 |
|---|---|---|
| Imhof, **Computing the distribution of quadratic forms in normal variables**, Biometrika 1961. [원문 PDF](https://doc.rero.ch/record/294852/files/48-3-4-419.pdf). 원문 부분 검토: §1, §3의 특성함수 역변환, §4 근사. | 비중심 Gaussian 이차형식의 분포와 수치 역변환은 고전적 문제다. 표본분산을 이차형식으로 바꿔 CDF를 계산하는 연산은 신규성이 없다. | 수치 적분·절단 오차를 구별한다. 학습된 경로 모형, 두 이차형식의 공동 CDF, 데이터 의존 관측 선택을 해결한 논문은 아니다. |
| Davies, **Algorithm AS 155: The distribution of a linear combination of χ² random variables**, Applied Statistics 1980. [출판사 기록](https://academic.oup.com/jrsssc/article/29/3/323/6953943), [저자의 설명·코드](https://www.statsresearch.co.nz/robert/QF.htm). 출판 기록과 저자 구현 설명 검토; 유료 논문 전체는 미검토. | 독립 비중심 χ²들의 가중합 및 독립 정규항의 확률을 특성함수 역변환으로 계산한다. 저자는 정확도·실패 flag와 낮은 자유도 지배 사례의 문제를 설명한다. | “Exact distribution”은 유한 계산의 무오류를 뜻하지 않는다. 원래 C/Fortran 프로그램을 그대로 감싼다고 neural parameter gradient가 제공되는 것은 아니다. |
| Das, **New methods to compute the generalized chi-square distribution**, Journal of Statistical Computation and Simulation 2025, arXiv:2404.05062. [2025 출판](https://www.tandfonline.com/doi/full/10.1080/00949655.2025.2501401), [2026 개정 원문](https://arxiv.org/html/2404.05062), [공식 코드](https://github.com/abhranildas/gx2-matlab). 원문 부분 검토: §1–4, §8–9; README의 CDF 미분 예시. | 일반화 χ²의 CDF/PDF/역CDF 계산법과 tail별 수치 차이를 다룬다. 현재 공식 코드에는 native 분포 parameter 및 Gaussian quadratic boundary에 대한 CDF gradient/Hessian이 있다. | 미분 가능한 이차형식 확률을 처음 만들었다는 주장은 성립하지 않는다. README 예시의 미분 대상을 정확히 구별해야 한다. 이 검토는 배치 PyTorch 학습, 반복 고유값, 퇴화 공분산에서 그 구현의 정확도·속도를 확인하지 않았다. |
| Rangapuram et al., **Deep State Space Models for Time Series Forecasting**, NeurIPS 2018. [공식 원문](https://proceedings.neurips.cc/paper/2018/file/5cf68969fb67aa6082363a6d4e6468e2-Paper.pdf). 원문 부분 검토: §3–4, 특히 §4.2–4.3. | 신경망이 선형 Gaussian state-space parameter를 정하고 미래 경로의 다변량 정규분포를 생성한다. 추세·계절 구조, 경로 표본, 결측 likelihood 처리도 이미 존재한다. | 미래 창의 표본분산 CDF를 주 목표로 학습하는 것은 이 논문의 실험 목표가 아니다. 관측이 없는 likelihood 항을 빼는 것과 결측/voicing 선택과정 자체를 예측하는 것은 다르다. |
| Salinas et al., **High-Dimensional Multivariate Forecasting with Low-Rank Gaussian Copula Processes**, NeurIPS 2019. [공식 원문](https://papers.neurips.cc/paper/8907-high-dimensional-multivariate-forecasting-with-low-rank-gaussian-copula-processes.pdf). 원문 부분 검토: §2–4, 부록 A. | 신경망으로 `Σ=D+VVᵀ`를 학습하는 low-rank PSD 구조는 직접 선행이다. 비정규 주변변수를 copula로 다루고, likelihood 계산 비용도 줄인다. | 여기의 covariance는 기본적으로 시점별 여러 시계열 축의 구조다. 미래 시간축 전체의 Gaussian law라고 그대로 읽으면 안 된다. 비선형 copula inverse 후의 원래 단위 표본분산은 Gaussian 이차형식이 아니다. |
| Deshpande & Sarawagi, **Coherent Probabilistic Aggregate Queries on Long-horizon Forecasts**, arXiv:2111.03394v2, 2022. v1(2021) 제목은 **Long Range Probabilistic Forecasting in Time-Series using High Order Statistics**. [확인한 v2 원문](https://arxiv.org/pdf/2111.03394). 원문 부분 검토: §2.1–2.3, 식 (1)–(3) 및 consensus Gaussian. | 경로와 집계 예측을 Gaussian consensus로 맞추고 새 집계의 분포를 구한다. “경로 분포에서 집계 질의를 답한다”는 큰 틀은 선행이다. | 집계는 명시적으로 고정 가중 **선형합**이며 예시는 평균과 선형추세다. **분포의 variance**와 **표본경로의 sample variance라는 확률변수**는 다르다. 이 원문은 후자의 일반화 χ² head를 제시하지 않는다. |

## 2. 계산할 수 있는 것은 무엇인가

다음은 실제 자료를 확인한 결과가 아니라 이 후보의 수학적 명세다. 부모 에이전트가 전달한 extractor 명세에 따르면 미래 1.25–1.75초 창은 **46프레임**, pitch target은 **voiced 프레임의 cents 표준편차**, power target은 **전체 46프레임의 `20log10(RMS)` dBFS 표준편차**이며 pitch guard는 voiced frame **23/46개 이상**이다. 이 검토자가 구현을 독립 재검증한 사실은 아니다. 96은 **prefix** 프레임 수이며 미래 창 길이와 다르다. `ddof` 등 남은 구현 세부는 정확하게 맞춰야 한다.

고정된 미래 창의 실수값 경로를 `Y|prefix ~ N(μ,Σ)`라 하고

`H=I−11ᵀ/n`, `A=H/(n−ddof)`, `V=YᵀAY`

로 두면 `V`는 그 단위의 표본분산이다. `ddof=0`이면 분모 `n`, `ddof=1`이면 `n−1`이다. 표준편차 문턱 `s≥0`에 대해서는 `P(std(Y)≤s)=P(V≤s²)`이므로 제곱근에 대한 별도 분포 근사가 필요 없다.

`Y=μ+LZ`, `Z~N(0,I)`, `LLᵀ=Σ`라 두고 `B=LᵀAL`, `b=LᵀAμ`, `c=μᵀAμ`라 하자. `B=U diag(λ) Uᵀ`, `d=Uᵀb`로 대각화하면

`V = κ + Σ_{λ_i>0} λ_i (Z_i+d_i/λ_i)²`,

`κ=c−Σ_{λ_i>0} d_i²/λ_i`.

즉 offset이 있을 수 있는 일반화 비중심 χ²다. `A`가 PSD이므로 `λ_i=0`인 방향의 선형항도 0이다. 완전 rank의 `Σ`에서는 보통 `κ=0`이지만, 순수 low-rank 경로에서는 deterministic mean 성분 때문에 `κ>0`일 수 있다. 영 고유값을 나누거나 offset을 버리면 잘못된 법칙이 된다. 공분산을 `D+VVᵀ`로 만들 때 `D>0`이면 비퇴화 경로를 유지할 수 있지만, 그 noise floor가 예측할 수 있는 작은 분산의 범위에도 영향을 준다.

조건부 모형에서 다음 moment도 바로 계산된다:

`E[V]=tr(AΣ)+μᵀAμ`,

`Var(V)=2 tr(AΣAΣ)+4 μᵀAΣAμ`.

이들은 미래 **경로의 변동**과 경로 **예측오차의 불확실성**이 다름을 보여 준다. 기울기·진동 mean이 있으면 작은 `Σ`에서도 미래 창의 sample variance는 클 수 있다. 반대로 경로의 모든 프레임에 같은 random offset이 더해지면 예측오차는 커져도 sample variance는 바뀌지 않는다.

시간 상관과 mean drift가 있는 경로에 단순 `χ²_(n−1)` 또는 표준 Wishart 법칙을 적용하면 안 된다. 일반화 이차형식은 그 상관을 보존하는 확장이다. 일반적인 Wishart 주장은 동일한 mean/covariance에서 독립 반복된 다변량 표본이라는 훨씬 강한 조건이 필요하다.

## 3. 중요한 적용 경계 세 가지

### 3.1 Gaussian을 어느 단위에서 가정하는가

전달받은 실제 target 단위는 **cents와 dBFS**다. 따라서 이 두 단위 자체에 Gaussian 경로를 두고 미래 mask를 고정한다면 두 표본분산 모두 이차형식이 맞다. **Raw Hz나 raw power로 지수 역변환할 필요가 없으며**, 그 때문에 이 후보가 배제되지는 않는다.

- Gaussian cents 경로의 voiced-frame std에는 위 계산이 적용된다. 향후 target을 Hz std로 바꾸면 지수변환 후의 경로는 Gaussian이 아니므로 그대로 적용되지 않는다.
- Gaussian dBFS 경로의 전체-frame std도 위 계산이 적용된다. 향후 **raw-power** std로 바꾸면 lognormal 경로의 중심화 이차함수가 되어 Gaussian χ² CDF를 그대로 사용할 수 없다.
- Target 전체를 `log1p(std)`로 변환한 MDN과 달리, path head는 경로의 원래 값·단위를 명세해야 한다. 단조적인 **최종 std 변환**은 문턱 역변환으로 처리할 수 있지만, 프레임별 비선형 변환은 분산과 교환되지 않는다.

비선형 단위변환이 필수이면 경로 표본을 실제 target 함수에 통과시키는 방법은 가능하다. 다만 그 경우 이차형식 CDF의 exactness·속도가 장점이라는 논거를 수정해야 한다.

### 3.2 두 variability의 주변분포만으로 공동 규칙을 답할 수 없다

Pitch 경로와 power 경로를 묶은 joint Gaussian을 두면 각 분산은 그 벡터에 대한 PSD 이차형식이다. 그러나 필요한 질의가 `P(V_pitch≤u, V_power≤v)`라면 이는 **두 의존 이차형식의 공동 CDF**다. 각각의 Imhof/Davies CDF를 구한 뒤 곱하는 것은 조건부 독립을 추가로 가정하는 것이다.

두 covariance block이 독립이면 곱이 맞지만, correlated Gaussian이나 shared low-rank factor를 쓰면 일반적으로 맞지 않는다. 일부 독립성 특수조건을 제외하면 비중심 mean까지 포함한 별도 검토가 필요하다. 따라서 여기서 가능한 정직한 선택은 다음과 같다.

- 독립 채널 모형임을 명시하고 그 제약을 직접 joint MDN 대비 가설로 시험한다.
- 두 이차형식의 공동 적분/특성함수 역변환을 사용하고 비용·오차·미분을 별도로 검증한다.
- 하나의 **공동 경로 표본 집합**을 모든 문턱에 공유하여 실제 규칙을 평가한다. 이는 일관된 cell 확률을 만들 수 있지만 Monte Carlo 오차와 학습용 estimator 문제가 생긴다.

일변량 이차형식 확률을 계산하는 기존 공식 코드가 두 채널의 공동 질의를 이미 해결한다고 쓰면 안 된다. 9개 주변 BCE를 평가한다고 출력 질의가 두 scalar target의 독립 주변 CDF만 필요하다는 뜻도 아니다.

### 3.3 관측 마스크와 invalid guard는 부가 항이 아니다

미래 voiced mask를 `m∈{0,1}^n`, `k=Σm_i`로 **고정해서** 주면, observed pitch의 중심화 행렬은

`A_m=[diag(m)−mmᵀ/k]/(k−ddof)`

이며 `k>ddof`에서 같은 이차형식 계산이 가능하다. 그러나 예측 시 미래 mask는 알려지지 않는다. 그 실제 mask를 입력하면 미래 정보 사용이다.

더욱이 mask가 pitch 값·power·추정기 신뢰도에 의해 결정되면, `Y|M=m`는 원래 Gaussian의 submatrix 주변분포와 다를 수 있다. 값에 의존하는 선택은 truncation/selection을 만든다. “결측 likelihood 항을 뺐다”는 것만으로 선택과정의 확률모형이 되지 않는다.

일반적으로는 `P(rule|prefix)=Σ_m P(M=m|prefix) P(rule|M=m,prefix)`를 실제 guard와 함께 계산해야 한다. 여기서는 pitch의 `k≥23`과 전체 46-frame power 경로를 함께 처리해야 하며 mask 패턴은 지수적으로 많다. invalid이면 전체 거짓이라는 규칙은 continuous variance density 밖의 분리된 사건이며, valid/invalid와 두 경로의 의존도 포함해야 한다. 실용적인 hurdle/voicing state 모형을 추가할 수 있지만 독립성·근사 수준을 밝혀야 한다. invalid 사례를 제외한 점수만 좋아졌다고 전체 목표를 개선했다고 할 수 없다.

## 4. Harmonic low-rank 경로의 약속과 비식별성

`Σ=D+Φ C Φᵀ`, `C⪰0`로 두고 `Φ`에 sine/cosine·drift basis를 넣으면 compact PSD 경로 모형을 만들 수 있다. 이 구조 자체는 Gaussian random basis/state-space/factor model의 알려진 구성이다. Prefix에서 배우는 것이 oscillation amplitude·frequency·drift라면 실제 temporal order가 미래 창의 변동분포에 영향을 주는 기전을 명시할 수 있다.

하지만 random phase의 sinusoid는 자동으로 Gaussian process가 아니다. Gaussian sine/cosine coefficient를 쓰는 law와 고정 amplitude·균일 phase law는 서로 다르다. 이 head를 실제 바이올린의 물리 생성모형이라고 부르기보다 제한된 descriptor path model로 해석해야 한다.

**두 scalar variability만 감독하면 path parameter는 강하게 비식별적이다.** 다음은 수식에서 직접 나오는 예다.

- `H1=0`이므로 mean에 상수 offset을 더해도 분산은 같다. 같은 방향의 random-level covariance도 보이지 않는다.
- `E[V]`는 mean의 변화량과 covariance trace가 합쳐진 값이다. deterministic drift와 stochastic 변동이 비슷한 scalar target을 만들 수 있다.
- 한 창의 이차형식 법칙은 고유값·비중심 성분의 조합만 보며, 많은 temporal eigenvector·factor rotation을 구별하지 못한다. `μ`와 `−μ`도 같은 단일 분산법칙을 줄 수 있다.
- 유한 문턱의 BCE만 감독하면 전체 scalar 분산분포조차 식별하지 못한다. 여러 mean/covariance가 같은 질의 확률을 줄 수 있다.

식별되지 않는 latent parameter도 좋은 예측기로 쓸 수는 있다. 다만 “학습한 covariance가 실제 vibrato dynamics를 회복했다”는 해석이나 관측하지 않은 창 길이의 정확한 확장에는 추가 근거가 필요하다.

미래 프레임 경로의 likelihood를 auxiliary supervision으로 쓰면 제약이 강해진다. 그러나 이는 직접 2-target MDN보다 **더 많은 미래 정답 정보와 다른 감독 단위**를 준다. Path-head 개선과 supervision 개선을 혼동하지 않도록 같은 future-path 보조감독을 받는 MDN/encoder 대조가 필요하다. 사람 내 여러 프레임이 생겼다고 독립 사람이 늘지는 않는다. 주 평가는 계속 참여자 macro 9-rule score여야 한다.

## 5. CDF 미분과 수치계산: 가능한 것과 아직 검증할 것

고전 CDF를 미분하거나 재매개변수화하여 학습하는 방향은 가능하다. 위 `Z` 표현에서는 characteristic function이

`φ(t)=exp(itc) det(I−2itB)^(−1/2) exp[−2t² bᵀ(I−2itB)^(−1)b]`

이므로 parameter 미분을 행렬식·선형계 풀이에 대해 표현할 수 있다. 적절한 조건 아래 적분과 미분을 교환할 수 있지만, 퇴화·경계·tail까지 자동으로 정당화되는 것은 아니다. 이 식을 구현했다는 사실만으로 새 방법은 아니다.

특히 다음을 구별해야 한다.

- `∂F/∂threshold`라는 PDF와 `∂F/∂μ,∂F/∂Σ`라는 학습 parameter gradient는 다른 대상이다.
- `B`의 고유분해를 통해 역전파하면 반복/근접 고유값에서 eigenvector derivative가 불안정할 수 있다. CDF 자체가 매끄러운 것과 특정 분해 구현의 gradient 안정성은 다르다.
- Das의 현재 공식 코드에는 boundary coefficient에 대한 analytic derivative 예시도 있으므로 **일반화 χ² CDF의 differentiability 자체가 빈 기술 공백은 아니다**. `Y=μ+LZ`를 표준정규 `Z`의 quadratic boundary로 다시 쓰고 연쇄미분하는 경로도 기존 기능과 직접 겹친다.
- 적응적 quadrature의 stop/분기, tiny eigenvalue 처리, 작은 cell 확률의 차분은 불연속 또는 큰 상대오차를 만들 수 있다. 자동미분이 실행된다고 수학적 gradient와 일치하는 것은 아니다.
- 단순 moment-matched Gamma/χ² 근사는 더 싸지만 tail·비중심성 정보를 버린다. 이를 정확 CDF와 구별해야 한다. 반대로 비싼 exact integration이 주 점수에서 그 비용에 맞는 이득을 주는지도 열린 질문이다.

**모델 학습 전** 구현 단계에서 필요한 검증은 Gaussian analytic 특수사례, 순수 deterministic drift, rank-deficient covariance+offset, 독립/상관 채널, 고정 mask의 일관성, tolerance를 바꿨을 때의 CDF·gradient 안정성이다. 기준값은 알려진 분포 또는 독립 구현으로 얻어야 한다. 본 검토에서는 이 검증을 실행하지 않았다.

경로를 reparameterize해 Monte Carlo로 최종 표본분산을 계산하는 것도 가능하다. 하지만 `1{V≤threshold}`의 보통 pathwise derivative는 거의 모든 표본에서 0이다. 매끄러운 indicator·score-function·조건부 적분 등 별도의 estimator를 쓰면 편향/분산/계산 비용을 밝혀야 한다. 단순히 “Gaussian이므로 reparameterization trick으로 BCE를 학습”한다고 끝낼 수 없다.

## 6. 의미 있는 좁은 기술 공백과 반박 조건

가설을 다음처럼 제한하면 탐색할 이유는 있다.

> 같은 prefix 정보에서, 제한된 mean/PSD covariance로 미래의 **실제 단위 경로**를 모델링하고 실제 nonlocal window-variance 함수로 확률을 전파하면, 두 variability를 직접 출력하는 MDN보다 작은 자료에서 더 나은 사람 단위 확률 예측을 하는가?

이는 “모든 새 연구를 종료”할 결론이 아니라 **별도 batch로 검토 가능한 새로운 표현 가설**이다. 현재 v2 결과에 맞춰 동결된 모델을 바꾸는 것은 아니다. 현 단계의 기대는 구조에 따른 sample efficiency이며, 선행을 결합한 것만으로 높은 방법론 신규성을 기대할 근거는 없다.

더 강한 방법 기여를 주장하려면 적어도 하나가 실제로 필요하다: (a) 두 dependent quadratic functional의 공동 확률과 gradient를 효율적·안정적으로 계산하는 기존법 대비 개선, (b) 미래 mask/guard 선택을 포함한 tractable pushforward, (c) task supervision만으로 필요한 functional law를 식별·제약하는 새로운 결과, (d) 여러 창/문턱으로 전이하는 구체적 효율·정확도 이득. 이 검토는 이 중 어느 것도 새로 해결하지 않았다.

필요한 최소 대조는 모델 목록 확대보다 아래 대비다.

1. **정보·학습예산을 맞춘 direct MDN와 path pushforward.** 같은 summary/prefix moment 접근, 같은 encoder, 같은 감독 범위를 고정한다.
2. **Path covariance 기여:** 동일 path mean에서 diagonal 대 temporal correlated/low-rank covariance를 비교한다. Correlation을 넣어 실제 variance law가 나아지는지가 기전이다.
3. **Oscillation 기여:** drift+noise 대 harmonic basis를 비교한다. Harmonic 장치가 필요하지 않으면 빼는 것이 맞다.
4. **Exactness 기여:** 같은 path law에서 정확한 quadratic CDF, moment approximation, 충분한 shared path sampling의 정확도·비용을 구별한다. CDF 엔진이 단순 표현 이득과 혼동되면 안 된다.
5. **감독 기여:** aggregate-only 대 future-path auxiliary 감독을 구별하고 direct MDN도 같은 보조감독을 받게 한다.

사전에 다음을 이 가설의 반박 또는 범위 축소로 인정해야 한다.

- Target 단위의 비선형 변환 때문에 claimed generalized-χ² pushforward가 실제 extractor와 다르다.
- Validity/voicing을 미래 실제 mask로 조건화하거나 invalid를 제외해야만 좋아진다.
- 두 주변 CDF의 독립 곱이 joint rule 오류를 지배하고, 공동 처리를 하면 계산 이점이 사라진다.
- Path likelihood는 좋아져도 실제 주 score·최악 사람 손실이 Markov/direct MDN보다 나아지지 않는다.
- Correlation·harmonic·정확 CDF 제거 대조가 동등하면 복잡한 장치의 실익은 미입증이다.
- 데이터 크기에서 mean/covariance가 collapse하거나 결과가 noise floor·rank·quadrature 설정에 주로 좌우된다.
- 익숙한 source folds의 결과만으로 새 사람 일반화를 확증하거나 latent harmonic parameter를 물리적 회복으로 해석해야만 주장이 성립한다.

**현재 결론:** 이 방향은 direct MDN과 다른 실질적인 nonlocal-statistic 표현을 제공하므로 문헌만으로 폐기할 이유는 없다. 전달된 cents/dBFS 단위는 Gaussian quadratic-form 접근과 맞는다. 남은 주 난점은 **미래 voiced mask/guard와 두 채널 공동의존의 pushforward**다. 이를 명시하면 작고 제한된 path law의 가능성을 별도 명세할 수 있다. 다만 학습된 Gaussian 경로, harmonic/low-rank PSD, 고전 quadratic CDF의 결합만으로 새 알고리즘이나 성능 개선을 주장해서는 안 된다.
