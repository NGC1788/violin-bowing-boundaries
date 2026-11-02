# Markov 마스크의 block-count 주변화 decoder: 직접 선행 검토

2026-11-02. **문헌·명세 검토만 수행했다.** 새 자료 접근, 모델/수치 실행, 준비·수학 캠페인의 재시작은 없다. 로컬에서는 상위 `AGENTS.md`와 공개 집계 `path_source_gate_v1/RESULTS.md`를 확인했다. 이전 문헌 검토에서 읽은 원문도 명시적으로 재사용했다. 이 문서 외 파일은 수정하지 않는다.

**판정:** 유한 마스크 사전의 미지 패턴 문제를 피하는 **실질적인 decoder 설계**다. 그러나 Markov-binomial/occupation count의 forward DP, Gaussian 표본분산의 비중심 χ², 잠재 mixture, 결측과 값의 공동 생성, 오차 경계에 따른 확률 abstraction/refinement는 각각 알려져 있다. 추가로 제안된 **adaptive beam+정확 guard mass+누락 valid mass**도 오래된 anytime probability bounding과 직접 겹친다. 단순 조합으로 높은 신규성을 주장할 수 없다. 좁게 남는 가능성은 **랜덤 관측 선택 아래 비선형 창 통계에 충분한 요약 상태를 찾아, 제한 예산에서 확률·local decoder gradient의 오차를 함께 추적하는 알고리즘**이다. 그 가능성은 아직 성능·계산비·직접 선행 대비 우위로 입증되지 않았다.

## 1. 직접 출처와 읽은 수준

기본 decoder 관련 7개와 후속 beam/gradient 명세에 특히 가까운 4개 출처로 제한했다. 저자/기관 원문과 학회 원문만 근거로 삼았다. “원문 부분 검토”는 초록 외 명시한 본문을 읽었다는 뜻이며 모든 부록·구현을 독립 검증했다는 뜻은 아니다. 이 제한 검색은 최초성의 증명이 아니다.

| 직접 출처 | 확인 수준 | 정확한 겹침과 차이 |
|---|---|---|
| Jens Ledet Jensen, **On a saddlepoint approximation to the Markov binomial distribution**, Aarhus/Thiele Research Report 02, 2011. [기관 원문](https://pure.au.dk/ws/files/44177127/imf_thiele_2011_02.pdf). | 원문 §1의 식 (1), 초기조건, §2 일부. | `P(S_t=k,M_t=j)`의 정확한 재귀를 직접 제시하고 Ladd(1975)의 선행을 명시한다. 현재 상태와 누적 success count를 유지해 모든 binary Markov 경로를 합산하는 것은 고전적이다. 여러 block의 count vector로 상태를 확장하는 것은 이 forward recursion의 자연스러운 확장이다. 이 보고서의 새 결과인 saddlepoint 근사를 우리 DP의 신규성 근거로 삼지 않는다. |
| J. P. Imhof, **Computing the distribution of quadratic forms in normal variables**, Biometrika 1961. [원문 PDF](https://doc.rero.ch/record/294852/files/48-3-4-419.pdf), [기관 서지](https://archive-ouverte.unige.ch/unige:101766). | 앞선 `PATH_DISTRIBUTION_REVIEW.md` 작성 때 원문 §1·3·4 부분 검토. 이번 재요청에서 PDF endpoint는 410을 반환했으므로 새로 전문을 읽었다고 하지 않는다. | 비중심 Gaussian 이차형식의 법칙·CDF는 알려져 있다. 이번 iid 등분산 가정에서는 일반화 χ²보다 단순한 **하나의 비중심 χ²**가 된다. 이 통계 항등식 자체는 새로운 head나 새로운 분포 이론이 아니다. |
| Ipsen, Mattei & Frellsen, **not-MIWAE: Deep Generative Modelling with Missing not at Random Data**, ICLR 2021. [공식 원문](https://openreview.net/pdf?id=tu29GQT0JFy). | 원문 §1–3, 식 (1)–(2), missingness factorization. | 데이터와 관측 마스크를 함께 생성·학습하는 것은 직접 선행이다. 논문은 `p(y)p(m|y)`로 값 의존 선택을 다룬다. 후보의 `p(h)p(m|h)p(y|h)`는 더 제한된 shared-latent 모형이다. 둘을 동일한 일반 MNAR 처리라고 부를 수 없다. |
| Ma & Zhang, **Identifiable Generative Models for Missing Not at Random Data Imputation**, NeurIPS 2021. [공식 원문](https://papers.neurips.cc/paper_files/paper/2021/file/e8a642ed6a9ad20fb159472950db3d65-Paper.pdf). | 초록·서론 수준 확인. 식별성 정리 증명은 미검토. | 명시적인 mask 모형을 붙여도 MNAR 생성모형이 자동으로 식별되지 않는다는 직접 경고다. 논문의 식별성 결과는 별도 구조적 가정이 있다. 이 후보에 그 정리를 적용하거나 이 후보가 underlying unvoiced pitch를 회복한다고 주장할 근거는 없다. |
| Kwiatkowska, Norman & Parker, **Game-based Abstraction for Markov Decision Processes**, QEST 2006. [저자 원문](https://www.cs.ox.ac.uk/david.parker/papers/qest06.pdf). | 원문 서론·§2–3의 probability bound와 abstraction 구분 부분 검토. | 작은 상태 공간에서 확률의 상·하한을 얻고 gap이 크면 refinement한다는 큰 원리는 선행이다. 여기의 mean approximation coupling 또는 count-state pruning과 같은 구체식은 아니며, 이 MDP 결과가 현재 통계모형의 유효성을 증명하지도 않는다. |
| Deshpande & Sarawagi, **Coherent Probabilistic Aggregate Queries on Long-horizon Forecasts**, arXiv:2111.03394v2, 2022. [원문](https://arxiv.org/pdf/2111.03394). | 원문 §2.1–2.3; 앞선 검토와 이번 원문 확인. | Gaussian 경로에서 집계 질의를 효율적으로 답하는 큰 틀은 겹친다. 하지만 실제 집계는 평균·추세 등 **선형 가중합**이다. 제목의 이전 표현 “High Order Statistics”를 sample variance의 분포를 이미 해결했다는 뜻으로 읽으면 안 된다. Random-mask nonlinear variance count decoder는 직접 제시하지 않는다. |
| Biswas, Zambon, Nespoli & Corani, **Nonlinear Probabilistic Forecast Reconciliation**, arXiv:2604.26668v3, 2026-09-17. [원문](https://arxiv.org/html/2604.26668). | 원문 §2, §3.1–3.3, §4의 probabilistic bottom-up 정의. 출판 상태를 별도로 검증하지 않았으므로 arXiv 원문으로 표기한다. | 비선형 map으로 예측분포를 pushforward하는 큰 주장은 최신 직접 선행에도 있다. Projection/conditioning과 UKF 근사를 다루며, continuous base-law 가정 및 mixed/discrete 확장의 어려움을 명시한다. 후보의 finite mask+연속 variance 혼합, 정확 count 주변화와 deterministic CDF brackets는 직접 해법 밖이다. |

후속 beam/gradient 주장에는 다음 네 개가 더 직접적이다.

| 직접 출처 | 확인 수준 | 정확한 겹침과 차이 |
|---|---|---|
| David Poole, **Logic Programming, Abduction and Probability**, FGCS 1992. [저자 원문](https://www.cs.ubc.ca/labs/lci/papers/docs1992/poole-fgcs92.pdf). | 원문 §4.1–4.2 및 서론. | 열거한 설명들의 probability sum을 하한으로, 미탐색 queue probability mass를 더한 것을 상한으로 쓰는 anytime inference를 명시한다. “Retained sum+unknown mass”의 매우 가까운 직접 선행이다. 후보의 exact guard DP로 **valid** residual만 남기는 특수화나 mask-logit gradient는 직접 다루지 않는다. |
| Liu, Regier, Tripuraneni, Jordan & McAuliffe, **Rao-Blackwellized Stochastic Gradients for Discrete Distributions**, ICML 2019. [공식 원문](https://proceedings.mlr.press/v97/liu19c/liu19c.pdf). | 원문 §1–2, 식 (5)–(8). | 확률이 큰 k개 latent category의 gradient를 정확히 합하고 remainder에서 표본을 뽑아 correction을 더한다. 계산 예산에 따른 head/tail 분리와 discrete gradient가 직접 겹친다. 이 논문은 remainder를 버리지 않아 unbiasedness를 유지한다. 후보의 deterministic retained-only gradient를 같은 의미의 unbiased Rao–Blackwell estimator라고 하면 틀린다. |
| Correia, Niculae, Aziz & Martins, **Efficient Marginalization of Discrete and Structured Latent Variables via Sparsity**, NeurIPS 2020. [공식 원문](https://papers.nips.cc/paper_files/paper/2020/file/887caadc3642e304ede659b734f79b00-Paper.pdf). | 원문 §1–2, sparse-support 전략 설명. | 큰 combinatorial latent space를 작은 support로 줄여 deterministic differentiable marginalization을 수행한다. 다만 sparsemax/SparseMAP은 **모형 자체**의 일부 확률을 0으로 만든다. 후보는 positive Markov law를 보존하고 누락 mass를 경계로 기록한다는 차이가 있다. Support를 줄인다는 말만으로 신규성을 주장할 수 없다. |
| Adnan Darwiche, **A Differential Approach to Inference in Bayesian Networks**, JACM 2003. [원문 사본](https://people.cs.pitt.edu/~milos/courses/cs3710/readings/differential-darwiche.pdf). | 원문 §2–4, 특히 arithmetic-circuit 평가·미분 설명 및 §4.3 rounding-error 논의. | 지수 개의 latent assignment 합을 compact sum-product circuit로 표현하고 그 확률·parameter partial을 효율적으로 계산하는 큰 틀은 직접 선행이다. 후보의 count DP 역시 이런 circuit다. Count-beam valid residual와 local logistic-score bound의 구체적 결합이 해당 논문의 새 결과는 아니다. |

## 2. 무엇이 정확한가: 필요한 생성 명세

아래는 후보의 논리 점검이며 새로운 수치 결과가 아니다. Prefix를 `x`, 미래 프레임 수를 `T=46`, 공유 regime을 `H∈{1,…,K}`, pitch block들을 `I_b`라 두자. 조건부로

`P(H=h|x)=π_h(x)`,

`P(M_1:T|x,h)=p_h(M_1|x) ∏_{t=2}^T P_h(M_t|M_{t−1},x)`

를 정하고 모든 허용 transition/initial 확률을 양수로 둔다. 이는 미관측 마스크에도 양의 확률을 주는 **모형 지원 범위**다. 미지 마스크의 확률이 적절하다는 calibration 보장은 아니다.

Pitch는 `Y_t=a_{b(t),h}(x)+σ_{p,h}(x) ε_t`이며, **regime·prefix가 주어졌을 때 `ε_t`들은 iid `N(0,1)`이고 마스크와 독립**이라고 해야 한다. Power의 전체 46-frame 경로에도 iid 등분산 Gaussian innovation을 두며, pitch/power innovation까지 조건부 독립이라면 다음의 product CDF가 성립한다. 공유 `H`가 채널과 마스크 사이의 주변 의존을 만들지만 이 조건부 독립 가정을 증명하는 것은 아니다.

`N_b=Σ_{t∈I_b}M_t`, `n=Σ_b N_b`라 두면, fixed count vector `N`에서 voiced pitch sample의 centered sum of squares는

`Q_p/σ_p² ~ χ²_{n−1}(λ_h(N))`,

`λ_h(N)=[Σ_b N_b a_b²−(Σ_b N_b a_b)²/n]/σ_p²`.

왜 정확한가: 같은 block의 mean은 같고 noise가 exchangeable iid이므로 어떤 프레임이 선택됐는지는 필요 없고 block별 개수만 필요하다. 따라서 count가 query-sufficient하다. 등분산·iid·mask 독립 중 하나가 깨지면 같은 count라도 선택 위치에 따라 법칙이 달라질 수 있다.

**분모를 명세해야 한다.** Candidate 수식의 sample std가 `ddof=1`이면 `(n−1) S_p²/σ_p²`가 위 χ²다. 실제 extractor가 `ddof=0`이면 `n S_p²/σ_p²`를 사용하며 자유도는 여전히 `n−1`이다. λ는 동일하다. 이 검토자는 실제 extractor를 열지 않았으므로 일치 여부를 확정하지 않는다. Guard는 `n≥23`이고 그 이하의 모든 mask 질량은 실제 all-false 규칙으로 보내야 한다.

Power를 전부 관측하고 조건부 독립이면 그 CDF `F_{w,h}(v)`는 count와 무관하다. Valid-only joint sub-CDF는

`P(valid,S_p≤u,S_w≤v|x)`

`=Σ_h π_h(x) F_{w,h}(v) Σ_{N:ΣN_b≥23} w_h(N|x) F_{p,h,N}(u)`.

전체 outcome law에는 invalid 질량의 all-false atom을 별도로 더한다. 모든 threshold query를 이 하나의 law에서 만들면 정합성은 구성상 성질이다. 정확한 생성모형 안의 주변화와 실제 자료의 확률 정확도는 구별해야 한다.

**Count-conditioned pattern-mixture 확장:** 위 독립 모형만 가능한 것은 아니다. `p(M|x,h)p(Y,W|N(M),x,h)`로 경로 parameter를 count vector에 의존시키면 mask와 값은 regime 안에서도 주변적으로 의존한다. 각 count에서 block mean·iid 등분산 pitch innovation을 유지하면 같은 χ² 형태가 성립한다. Power parameter도 `N`에 의존하면 위 식의 `F_{w,h}`를 `F_{w,h,N}`로 바꾸어 **count 합 안**에 넣는다. 이는 Markov mask law가 사후에 값을 보지 않고도 정의되는 유효한 pattern-mixture generative law다. Count가 같지만 mask 위치가 다른 경우에 같은 조건부 path law를 준다는 제한은 남는다. 일반 MNAR/selection identification을 해결했다는 주장이 아니며, count-conditioned parameter화와 count-independent mean approximation의 보증도 각각 다시 명세해야 한다.

## 3. DP의 직접 겹침과 계산 문턱

누적 count vector를 `c`, 현재 binary mask 상태를 `j`라 하고 `f_t(c,j)=P(N_{1:B}(1:t)=c,M_t=j|x,h)`로 둔다. 시간 `t+1`이 block `b`라면

`f_{t+1}(c+j e_b,j)=Σ_{i∈{0,1}} f_t(c,i) P_h(M_{t+1}=j|M_t=i,x)`.

마지막에 `j`를 합치면 `w_h(c|x)`다. 이는 고전 Markov-binomial recurrence의 count-vector 확장이다. Prefix의 마지막 mask를 알고 있다면 initial transition에 조건화할 수 있고, forecast gap이 있다면 그 gap 전이도 정확히 반영해야 한다. 미래 시작 마스크의 분포를 임의로 prefix 마지막 mask와 동일시하면 안 된다.

**Block 경계를 넘을 때 마지막 mask 상태를 지우면 틀린다.** Block count만 갖고 다음 block을 독립 시작하거나, block별 marginal count law를 곱하면 경계 Markov 의존을 잃는다. Block transfer kernel에 start/end state를 유지하는 방식은 같은 forward 계산의 다른 표현이다.

원시 `2^46` mask 열거를 피한다는 장점은 실질적이다. 하지만 dense count 격자의 크기는 `∏_b (|I_b|+1)`에 달한다. `B`가 작을 때의 tractability와, `B=T`에서 다시 binary-mask 수준으로 커지는 한계를 함께 써야 한다. “Mask 수와 무관하게 항상 polynomial”이라는 설명은 부정확하다. 고정 `B`에 대해 tractable한 것과 `B`까지 입력에 포함한 일반 문제를 구별한다.

이 DP의 gradient는 유한 합·곱의 differentiation으로 얻을 수 있으므로 differentiable DP라는 설명만으로 신규성이 생기지 않는다. 확률의 underflow, 거의 결정론적인 transition, χ² tail derivative, 작은 σ와 λ 계산의 cancellation은 구현 별도 문제다. Exact algebra는 floating-point exactness가 아니다.

## 4. Block 근사와 CDF brackets의 정확한 적용 범위

원래 learned mean `μ_t`를 block constant `ã_b`로 근사하고 같은 mask와 같은 Gaussian noise를 결합하면, sample std의 reverse triangle inequality로

`|S(μ+σε;M)−S(ã+σε;M)| ≤ ||H_M(μ_M−ã_M)||₂/√(n−1)`

`≤ √[Σ_{t:M_t=1}(μ_t−ã_{b(t)})²/(n−1)]`.

따라서 각 block의 **최대오차** `E_b≥max_{t∈I_b}|μ_t−ã_b|`를 써야 count별 보증

`ε(N)=√[Σ_b N_b E_b²/(n−1)]`

를 얻는다. Block의 RMS error만으로 이 bound를 대체하면 특정 mask가 큰 오차 프레임을 집중 선택할 수 있으므로 일반적으로 안전하지 않다. `ddof=0`이면 대응 분모를 `n`으로 바꾼다.

Valid count에서 power를 근사하지 않는다면

`F_block,N(u−ε(N),v) ≤ F_original,N(u,v) ≤ F_block,N(u+ε(N),v)`

이며 음수 std threshold의 CDF는 0으로 정의한다. 이를 **같은 mask law와 regime weights**로 합하면 joint sub-CDF brackets다. 두 채널 모두 근사하면 각 threshold를 해당 오차만큼 함께 이동한다.

이 coupling argument와 확률 상·하한을 gap에 따라 세분화하는 원리는 기존 abstraction/verified-probability 계열과 겹친다. 이 후보에서 유용한 부분은 count별 오차를 실용적으로 유지하는 구체화일 수 있다. 하지만 bound는 **원래 learned Gaussian model과 block 근사 model의 차이**를 제어한다. 실제 미지의 conditional mean, Gaussian 가정 오차, 데이터 shift까지 포함한 truth certificate가 아니다.

또한 CDF의 상·하한을 인접 문턱에서 빼면 cell probability의 최선 경계가 자동으로 나오지는 않는다. 예를 들어 한 cell의 하한은 `max(0,L(upper)−U(lower))`, 상한은 `min(1,U(upper)−L(lower))`처럼 일관되게 계산해야 한다. 모든 cell의 독립 구간 끝점을 동시에 달성할 수 있다고 해석해서도 안 된다.

## 5. Pruning과 all-false fallback은 서로 다른 주장이다

후속 후보의 핵심은 **guard-only 정확 DP와 비싼 block-count query beam을 분리**하는 것이다. `g_h=P_h(Σ_t M_t≥23|x)`는 `(total count,last mask)`의 작은 DP로 정확하게 구한다. Retained valid contribution의 질량을 `m_{R,h}`라 하면 omitted-valid mass는

`r_{G,h}=g_h−m_{R,h}`.

각 rule은 invalid에서 false이므로 retained query sum이 `s_h`일 때 전체 valid rule probability는 `[s_h,s_h+r_{G,h}]` 안에 있다. Regime을 합치면 `r_G=Σ_h π_h r_{G,h}`이며, invalid mass `1−Σ_hπ_hg_h`는 별도 all-false로 **정확히** 처리한다. 이 경계는 전체 omitted mass를 쓰는 것보다 tight할 수 있다. 그러나 retained+residual 자체는 Poole류 anytime bounding의 특수화이므로 새 원리라고 할 수 없다.

두 가지 beam 구현을 구별해야 한다. (a) 선택한 **완성 count state의 전체 확률**을 다른 constrained DP로 정확히 다시 계산하면 full state mass다. (b) 매 시점 beam에서 탈락한 경로를 버린 뒤 같은 count state로 도달한 **살아남은 경로들의 mass만** 갖고 있다면 full count mass보다 작다. (b)도 retained cohort가 서로소인 부분 경로집합으로 정확히 정의되면 valid residual bound가 가능하지만, 이를 `w_h(N)`의 정확 값이라고 부르면 안 된다. Double counting·재정규화·pruned 경로의 재등장 처리에서 잘못하면 bound가 깨진다.

예산을 늘렸을 때 독립적으로 다시 계산한 두 beam의 retained cohort가 nested라는 보장은 없다. 각 실행의 residual bound는 유효할 수 있지만, 단조롭게 좁아지는 anytime interval을 주장하려면 누적 retained set을 유지하거나 별도의 포함관계 증명이 필요하다. `r_G`가 정한 tolerance를 넘으면 예산 한도 안에서 refinement하거나 명시적으로 미달로 반환해야 한다. 허용오차에 도달하는 데 항상 작은 beam이면 충분하다는 보장은 없다.

### 5.1 Local query-gradient bound: 성립하는 정확한 범위

Query value를 `F_N∈[0,1]`라 하고 mask-logit decoder coordinate `θ`에 대해 **F_N과 regime weights를 고정**하자. Time-specific binary transition logit은 한 path에서 `|∂ log p(M)/∂θ|≤1`이다. 그러므로 fixed retained trace에 대해 omitted valid contribution의 derivative는

`|Σ_{M∈omitted,valid} p(M) F_{N(M)} ∂θ log p(M)|≤r_G`

로 제어된다. 같은 scalar logit을 여러 프레임이 공유하면 path score가 합쳐져, 최대 T번 쓰인 좌표에는 보수적으로 `T r_G`가 된다. 이 부등식은 bounded integrand와 bounded score의 직접 결합이며, 새로운 gradient 원리라는 주장에는 약하다.

다음 확장은 **자동으로 따라오지 않는다.**

- Gaussian CDF parameter·σ·mean·regime weight까지 동시에 바뀌면 추가 derivative 항이 있다. `F_N`이 mask logit에 직접 의존하는 parameterization이면 위 bound만으로 충분하지 않다.
- Shared network parameter로 가는 Jacobian이 임의로 크면 mask-logit partial bound가 전체 network gradient bound가 되지 않는다.
- Beam membership은 parameter에 따라 바뀐다. 같은 trace가 유지되는 지역에서의 partial과 top-k/beam 전환점의 미분·학습 trajectory 안정성을 구별한다.
- Probability-query derivative의 오차와 BCE/log-loss gradient의 오차는 다르다. Probability가 0/1 근처이면 loss의 chain-rule 계수가 커지므로 별도 floor/conditioning 분석이 필요하다.

Retained gradient만 쓰면 일반적으로 biased truncation이다. Liu et al.의 tail-sampling correction을 추가하면 다른 estimator가 되며 tail sampling 비용·분산을 검토해야 한다. 그 대조를 하지 않고 “Rao–Blackwellized이므로 unbiased하고 항상 낫다”고 표현할 수 없다.

### 5.2 일반 retained-mass 처리와 출력 정책

Retained count states의 **원래 확률을 그대로** 합한 질량을 `r`, 버린 질량을 `u=1−r`라 하자. 어떤 사건의 retained contribution이 `s`이면 전체 모형의 확률은 `[s,s+u]` 안에 있다. 이미 정확히 계산한 invalid 질량은 unknown mass에 섞을 필요가 없다. Regime별 pruning이면 π-weighted 누락 질량을 사용해야 한다.

- **확률 구간 출력:** retained contribution과 unknown contribution의 경계를 유지한다. 이는 원래 full decoder를 enclosure한다.
- **모든 unknown을 all-false에 배정:** 완결된 별도 approximate predictive law다. Retained law를 바꾸지 않으면 full law와의 모든 사건확률 오차 및 total variation은 `u` 이하이다. 그러나 pruned valid state가 실제 invalid였다는 뜻은 아니다.
- **Retained states만 재정규화:** 원래 mixture와 다른 law다. 재정규화 사실과 오차 분석을 명시해야 하며 `s`를 full event probability라고 부르면 안 된다.

작은 누락 질량이 작은 **log loss** 차이를 보장하지는 않는다. 희귀 정답에 0을 배정하면 log loss는 무한대일 수 있다. Positive probability floor 또는 확률비 제약이 없으면 CDF/TV 오차만으로 proper-log-loss 안정성을 주장할 수 없다.

매 학습 단계에서 top-k count states를 다시 고르면 retained set 선택이 parameter 의존적·불연속적일 수 있다. Forward mass의 bound와 **그 bound/근사 objective gradient**의 타당성은 별개다. Pruning tolerance·refinement policy를 label 성능을 보고 사후 변경하면 계산 gate와 성능 탐색도 혼동된다.

## 6. 치명적 공백과 아직 남길 수 있는 기여

가장 큰 위험은 계산식 자체보다 **count sufficiency를 성립시키는 강한 가정**이다.

1. **Mask–innovation/위치 의존:** baseline 구조에서 voicing이 같은 프레임의 pitch 오류·power·noise에 의존하면 count 조건부 Gaussian law가 바뀐다. Count-conditioned pattern-mixture는 일부 의존을 명시할 수 있지만, 같은 count의 mask 위치 차이와 값 의존 선택을 일반적으로 표현하지 않는다. Shared finite `h`도 일반적인 MNAR 해결이나 식별성 보장을 주지 않는다. 이를 허용하려고 자유로운 `P(M|Y,h)`를 도입하면 간단한 χ²/count DP가 대개 사라진다.
2. **iid·등분산:** 진짜 frame innovation의 시간 상관, block별 noise scale, 특정 frame의 추정오차는 count-only χ²를 깨뜨린다. 약간의 상관을 주어도 정확성을 주장하려면 새로운 충분상태나 검증 가능한 오차 bound가 필요하다.
3. **두 채널 조건부 독립:** 공유 regime만의 dependence와 residual cross-channel correlation은 다르다. 후자를 허용하면서 marginal CDF를 곱하면 잘못된 joint law다. 반대로 독립을 강제하면 큰 장점인 tractability와 표현력 사이의 tradeoff를 솔직히 밝혀야 한다.
4. **잠재 pitch의 해석:** unvoiced frame의 pitch라는 latent quantity가 물리적으로 정의되어 있는지와 별개로, observed-mask descriptor law를 모델링할 수는 있다. 다만 관측되지 않은 값을 생성했다는 이유로 실제 pitch를 복원했다고 해석할 수 없다.
5. **복잡도·표현력:** 작은 `B`가 필요하지만 작은 block error가 필요하면 `B`가 커진다. Count-state 수와 CDF 호출·gradient 비용이 그 tradeoff의 실제 문턱이다. 이를 측정하지 않고 “46개라 싸다”고 말할 수 없다.
6. **감독과 검증:** 미래 path/mask likelihood를 쓰면 직접 aggregate decoder보다 더 많은 supervision을 받는다. 같은 supervision의 대조가 필요하다. 사람 수는 늘지 않으며 과거 source folds의 반복 사용도 독립 확인시험이 아니다.

신규성 후보를 좁혀 말한다면 **“관측 선택에 충분한 count abstraction, exact guard mass, adaptive query beam, noncentral-χ²의 retained partial, mean/pruning 오차의 joint-CDF enclosure를 함께 제공하는 특정 nonlinear aggregate decoder”** 정도다. 구체적인 사용처와 계산-정확도 계약이 생긴 점은 유용하다. 그러나 Poole의 residual-mass bound, Liu의 head/tail gradient 분해, Darwiche의 differentiable sum-product와의 겹침 때문에 그것만으로 새 핵심 원리가 성립하지는 않는다. 실제 새 알고리즘·정리가 되려면 최소한 다음 중 하나를 선행 대비 입증해야 한다.

- 기존 Markov-additive forward 방식보다 나은 state compression 또는 계산/gradient 복잡도 결과.
- 단순 iid·independent-channel 특수모형 밖에서도 효율성을 보존하는 새로운 sufficient statistic 또는 rigorously bounded extension.
- Block refinement와 probability pruning을 결합했을 때 요청된 query tolerance를 유한 비용으로 만족하는 구체 알고리즘과 비자명한 비용/오차 결과.
- 같은 정보·감독·예산의 direct MDN, 단순 regime mixture 및 Monte Carlo path pushforward보다 유용한 정확도–비용 tradeoff.

현재 확보한 것은 그 문제를 명확하게 쓰는 수준이다. 문헌 검토만으로 동일한 조합을 직접 제시한 논문을 찾지 못했다는 사실은 최초성 증명이 아니다.

## 7. 이후 한 단위를 정당화하는 최소 문턱

이 문서는 새 실행을 허가하거나 수치 gate를 통과했다고 판정하지 않는다. 별도 명세에서 다음을 먼저 고정할 수 있다.

- 실제 extractor의 단위·분모·guard와 일치하는 수학적 law, 조건부 독립 가정, gap initialization.
- 정확 small-T 열거와의 대응, count/end-state 경계 처리, full-mass 보존, invalid/all-false 처리에 대한 증명 의무. 여기서는 실행하지 않았다.
- Full/count-pruned/block-approximated law의 오차를 각각 구별하는 계약. Model error와 computation error를 분리한다.
- 최대 `B,K`, state/CDF budget, 원하는 bracket width와 실패 시 처리. 기준을 결과 뒤에 올려 통과로 바꾸지 않는다.
- iid·mask-independence·channel-independence가 자료/기전에 맞지 않거나 유용한 brackets를 얻는 계산비가 예산을 넘으면 **이 decoder 가설을 기각/축소**한다. 다른 허가된 연구 전체의 불가능성으로 확대하지 않는다.

**최종 판단:** mask-catalogue-free 지원 범위와 정확 nonlinear-statistic 주변화라는 차이는 실질적이다. 반면 기본 DP·χ²·mixture·abstraction의 조합은 강한 직접 선행 위에 있다. 가장 정직한 다음 주장은 “명시한 제한 생성모형 안에서의 exact/controlled-approximate decoder feasibility”이며, 새로운 AI 방법과 실제 예측 개선은 아직 별도로 입증해야 한다.
