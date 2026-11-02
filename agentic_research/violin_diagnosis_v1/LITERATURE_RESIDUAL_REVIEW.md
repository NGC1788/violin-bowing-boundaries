# 기준선에 고정한 확률 residual과 참여자 단위 위험 제어: 제한 문헌 검토

작성일: 2026-11-02. 문헌 조사와 해석만 수행했다. 새 학습·예측·자료 접근·시험 정답 열람은 하지 않았다. 로컬에서는 이 디렉터리의 `AGENTS.md`, 기존 `violin_raw_v1/LITERATURE_SCOPE_20261102.md`, `violin_raw_v1/RESULTS.md`만 근거 문맥으로 읽었다. 아래 수식의 수치 예시는 자료에서 계산한 진단값이 아니다.

**판정: 현재 아이디어는 새 방법으로 인정할 근거가 부족하며, 기존 기법을 결합한 후속 연구 설계에 가깝다.** 확률 혼합, log-prior residual, 기준선 대비 minimax regret, 위험 검정 뒤 채택, 계층 자료의 conformal 보정은 각각 직접 선행이 있다. 남을 수 있는 기술 문제는 **적은 수의 독립 참여자, 겹치는 cross-fitting 학습 집합, 구조적 다변량 확률, 실제 평가하는 주변 log loss**를 함께 다루면서도 비자명한 이득을 허용하는 유한표본 인증이다. 이 검토는 그 해법을 발견하거나 신규성을 입증하지 않는다.

## 1. 검토 범위와 원문 확인 수준

가까운 8개 출처를 선정했다. 모두 저자 원문·학회 원문·arXiv 원문이다. 검색 결과에 나온 제3자 요약은 아래 주장 근거로 사용하지 않았다. 각 항목의 “원문 부분 검토”는 초록만 읽은 것이 아니라 명시한 목적함수·가정·정리 구간을 읽었다는 뜻이며, 모든 부록 증명을 독립 검증했다는 뜻은 아니다. 2024 CRC, 2025 LTT 출판 및 HCP 개정본, 2026 GHCP까지 확인했지만 전 세계 문헌의 부재를 증명하는 체계적 전수조사는 아니다.

| 직접 출처와 확인 수준 | 이미 알려진 겹침 | 이 문제에 자동으로 옮길 수 없는 부분 |
|---|---|---|
| Yao, Vehtari, Simpson & Gelman, **Using Stacking to Average Bayesian Predictive Distributions**, Bayesian Analysis 2018. [저자 원문](https://sites.stat.columbia.edu/gelman/research/published/stacking_paper_discussion_rejoinder.pdf). 원문 부분 검토: §3.1–3.2, 식 (3)–(4). | Proper scoring rule로 예측분포의 convex combination을 고르고, leave-one-out 예측으로 가중치를 추정한다. Markov와 residual 분포의 한 개 혼합 가중치를 고르는 설명은 이 계열이다. | 경험적 cross-validation 최적화가 유한표본 모집단 비악화 인증은 아니다. 이 자료에서 녹음 하나를 빼는 LOO는 새 사람 일반화의 평가 단위와 다르다. 참여자째 제외해야 한다. |
| Ghavamzadeh, Petrik & Chow, **Safe Policy Improvement by Minimizing Robust Baseline Regret**, NeurIPS 2016. [학회 원문](https://proceedings.neurips.cc/paper_files/paper/2016/file/9a3d458322d70046f63dfd8b0153ece4-Paper.pdf), [저자 장문 원문](https://www.cs.unh.edu/~mpetrik/pub/Petrik2016b.pdf). 원문 부분 검토: §2–4, 식 (1)–(4), 정리 5–7. | 동일한 불확실한 모형 아래에서 새 정책과 기준선의 차이를 최악화하고, 확실하지 않은 곳에서 기준선으로 복귀한다. “uncertainty set에서 기준선 regret를 제어”한다는 큰 아이디어의 매우 직접적인 선행이다. | MDP의 전이 오차 경계, 할인 보상과 정책을 다룬다. 여기의 조건부 10패턴 확률이나 log loss 정리를 제공하지 않는다. 정확한 목적과 근사 알고리즘의 보장을 구별해야 한다. |
| Agarwal & Zhang, **Minimax Regret Optimization for Robust Machine Learning under Distribution Shift**, COLT 2022. [공식 원문](https://proceedings.mlr.press/v178/agarwal22b/agarwal22b.pdf). 원문 부분 검토: §1–3, 목적 (3), 정리 2의 조건 및 squared-loss 특수 결과. | 절대 worst-case risk 대신 분포별 regret를 최소화한다. 잡음이 큰 집단이 DRO 목적을 지배할 수 있다는 지적은 여기의 참여자별 난도 차이에도 관련된다. | 비교 대상은 분포별 class-optimal predictor이며 고정 Markov 기준선과 다르다. 학습분포의 reweighting 집합, bounded/Lipschitz loss 등의 조건이 있다. 임의의 새 사람이나 unbounded log loss에 보장을 바로 적용할 수 없다. |
| Sagawa, Koh, Hashimoto & Liang, **Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization**, ICLR 2020. [공식 원문](https://openreview.net/pdf?id=ryxGuJrFvS). 원문 부분 검토: §2–4 및 토의. | Group DRO는 이미 알려져 있다. 과매개변수 모형에서 train loss가 거의 0이면 worst-group train loss도 거의 0이어서, DRO만으로 일반화 실패를 고치지 못한다는 진단이 특히 관련된다. | 사전에 정의된 관측 집단의 위험과 새 참여자 일반화는 다르다. 사람 ID를 각각 집단으로 넣는다고 미래 사람의 손실을 인증하지 않는다. 논문의 다른 자료에서 얻은 효과 크기를 바이올린 개선 예상치로 사용할 수 없다. |
| Angelopoulos, Bates, Candès, Jordan & Lei, **Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control**, arXiv 2021, Annals of Applied Statistics 2025. [원문](https://arxiv.org/html/2110.01052), [2025 출판 확인](https://www.gsb.stanford.edu/faculty-research/publications/learn-then-test-calibrating-predictive-algorithms-achieve-risk). 원문 부분 검토: §1.1, §2.1–2.4, 정리 1, HB p-value 조건. | 학습된 후보의 위험 가설을 calibration 자료에서 검정하고, FWER를 제어해 통과한 후보만 허용한다. “증거가 있을 때만 residual 채택”은 이 틀의 직접적인 응용이다. 비단조 위험도 다룬다. | 기본 calibration 표본은 i.i.d.이고 예측기는 calibration과 분리되어야 한다. HB 유한표본 검정에는 bounded loss가 필요하다. 여러 후보 p-value 사이의 의존 허용과 한 p-value를 만드는 표본의 의존 허용을 혼동하면 안 된다. |
| Angelopoulos, Bates, Fisch, Lei & Schuster, **Conformal Risk Control**, ICLR 2024. [원문](https://arxiv.org/html/2208.02814). 원문 부분 검토: §2.1 정리 1 및 비단조 위험 부록. | Exchangeable loss curve에서 단조·우연속·유계 손실의 미래 기대값을 제어한다. 위험 보정이라는 일반 목표는 선행이다. | residual 강도를 올릴 때 어떤 정답에서는 손실이 내려가고 다른 정답에서는 올라간다. 따라서 excess log loss는 보통 요구되는 점별 단조 loss curve가 아니다. 기본 정리는 기대위험 보장이며, calibration 조건부 고확률 비악화 주장과도 다르다. |
| Lee, Foygel Barber & Willett, **Distribution-free inference with hierarchical data**, arXiv 2023, 확인한 HTML은 v4, 2025-08-02. [원문](https://arxiv.org/html/2306.06342). 원문 부분 검토: §1.1, §2.2 정리 1, 반복측정 특수 모형. | 참여자/녹음 같은 계층을 무시한 완전 exchangeability 대신 계층 exchangeability를 사용한다. 새 집단의 관측값에 대한 HCP coverage와 특수 반복측정 조건의 더 강한 보장을 제시한다. | 이는 예측집합의 coverage이며 기준선 대비 평균 log loss 인증이 아니다. 사람 간 교환가능성, 사람 내 순서 교환가능성도 가정이다. 서로 다른 음·조건·순서가 있는 모든 녹음을 동일 조건의 독립 반복측정으로 볼 수 없다. |
| Mallick, Tchetgen Tchetgen, Dobriban & Lee, **Generalized Hierarchical Conformal Prediction**, arXiv:2608.15500v1, 2026-08-16 preprint. [원문](https://arxiv.org/html/2608.15500v1). 원문 부분 검토: §1.1 가정 A1–A3, §2 방법 및 정리 2.1, 부록 A.2. | 새 집단에서 이미 소수의 정답 관측이 있을 때 reference groups와 함께 보정한다. 표본 크기 선택·donation·전역/지역 예측을 이용한다. “소수의 새 사람 관측으로 안전하게 적응”이라는 넓은 주장은 이미 좁아졌다. | 현재 같은 녹음의 첫 1초 특징은 새 사람의 과거 **라벨 관측**과 다르다. 본 논문도 coverage를 다루며 regret 보장은 아니다. 집단 크기와 잠재 분포의 관계 등 명시 가정이 있다. 현재 프로토콜에 그대로 적용할 방법이 아니다. |

## 2. 먼저 실제 점수와 안전성 주장을 맞춰야 한다

기존 공개 결과에 따르면 훈련 녹음은 1,442개이지만 사람은 34명이고, 개발은 8명이다. 독립 사람으로의 일반화에서 1,442를 독립 calibration 표본 수로 쓰는 것은 정당하지 않다. 녹음은 조건부 예측 학습에 정보를 줄 수 있지만 사람 간 위험 불확실성은 사람 수를 반영해야 한다. `RESULTS.md`는 주 점수를 **사람별로 9규칙·녹음 BCE를 평균한 뒤 사람을 동일 비중으로 평균**한다고 명시한다.

다음 기호는 이 검토의 제안된 수학적 명세이며 기존 구현을 새로 확인한 결과가 아니다.

- 허용 패턴을 `z ∈ {1,…,10}`, 그 9규칙 벡터를 `a(z) ∈ {0,1}^9`라 하자.
- 기준선 joint 분포를 `b(x) ∈ Δ10`, 후보를 `q(x) ∈ Δ10`, 규칙 주변확률을 `p_j(q)=Σ_z a_j(z)q_z`라 하자.
- 기존 점수와 맞는 손실은 `S(q,z)=(1/9)Σ_j BCE(p_j(q),a_j(z))`다.
- 목표 regret는 새 사람 `G`에 대해 `E_G E_{recording|G}[S(q,Z)−S(b,Z)]`다. 사람 내 녹음 선택 규칙까지 고정해야 이 estimand가 정해진다.

**Joint categorical log loss `−log q_z`의 비악화는 위의 평균 주변 BCE 비악화를 자동으로 뜻하지 않는다.** 10패턴을 출력한다는 사실만으로 두 목적이 같아지지 않는다. Joint score는 joint 분포에 proper하고, 주변 BCE 합은 주변확률에 proper하다. 주변 벡터가 joint를 유일하게 결정하는지도 별도 확인 사항이다. 이 검토는 그 선형사상의 injectivity를 가정하지 않는다.

따라서 불확실 집합 `U(x) ⊂ Δ10`을 쓰려면 제약도 실제 점수로 써야 한다:

`sup_{r∈U(x)} Σ_z r_z [S(q(x),z)−S(b(x),z)] ≤ τ(x)`.

이는 **확실한 조건부 분포 집합이 주어졌을 때**의 robust baseline-regret 제약이다. 핵심 어려움은 최적화식을 쓰는 것이 아니라 `U(x)`가 실제 조건부분포를 포함한다는 유효하고 유용한 근거다. 60개 연속 특징과 34명으로 임의의 `x`마다 좁은 집합을 distribution-free로 추정할 수 있다고 주장할 수 없다. Markov 상태로 묶는다면 상태 내 동질성·평활성·집단 안정성과 상태별 사람 수를 명시해야 한다. 관측하지 않은 사람의 조건부 법칙이 임의로 바뀌는 shift에는 보장이 없다.

## 3. 간단한 결정론적 경계와 그 한계

아래는 문헌의 새 정리 인용이 아니라 정의에서 바로 나오는 검산이다. 신규성으로 주장할 내용이 아니다.

**Convex mixture.** `q_α=(1−α)b+αr`, `0≤α<1`이면 각 패턴에서 `q_α,z≥(1−α)b_z`다. 모든 규칙의 참/거짓 사건에도 같은 부등식이 성립하므로, 기준선이 양의 확률을 주는 정답에 대해

`S(q_α,z)−S(b,z) ≤ −log(1−α)`.

따라서 `α≤1−exp(−τ)`는 각 녹음의 초과손실을 `τ` 이하로 제한한다. `α=0`은 정확한 기준선이다. 이 사실은 혼합이 평균적으로 개선됨을 뜻하지 않으며, `τ>0`은 악화를 허용하는 값이다. 예를 들어 `α=0.1`의 상한은 약 0.10536으로, 작은 평균 개선을 입증하는 기준과는 상당히 다를 수 있다.

**Logit residual.** `q_z=b_z exp(g_z)/Σ_k b_k exp(g_k)`에서 `max g−min g≤c`를 실제로 강제하면 `q_z≥exp(−c)b_z`이므로 같은 주변 BCE 초과손실도 `c` 이하이다. Zero initialization만으로는 학습 후의 이 경계가 없다. Prior offset도 residual 크기를 제한하지 않는다. 신뢰영역 또는 상한 제약과 “초깃값이 기준선”이라는 사실을 구별해야 한다.

**완전 무가정의 비악화는 대체로 복귀를 뜻한다.** Joint log score에서 `U=Δ10`, `τ=0`이면 각 `z`에서 `q_z≥b_z`여야 하며 합이 1이므로 `q=b`다. 주변 BCE에서도 기준선 법칙 `b`를 `U`에 포함하고 그 기대점수 비악화를 강제하면 proper-score 성질상 기준선의 주변확률과 달라질 수 없다(정의된 사건이 비퇴화인 경우). 즉 “아무 실제 분포나 가능하지만 손실은 절대 나빠지지 않으면서 유용하게 수정”하는 요구를 해결했다고 포장할 수 없다. 비자명한 변경에는 양의 허용 regret, 유효하게 제한된 `U`, 또는 통계적 오류확률 중 무엇을 허용하는지 명시해야 한다.

## 4. 34명에서 통계 인증이 막히는 지점

LTT의 틀을 옮기는 가장 간단한 경로는 **먼저 고정한 기준선·후보·강도 목록**을 독립 calibration 사람들에게 평가하고, 사람별 paired loss difference로 유효한 검정을 하는 것이다. 이는 새로운 원리가 아니라 기존 LTT를 이 estimand에 맞춘 적용이다. 집단 전체를 하나의 관측 객체로 삼으면 사람 내 의존을 허용할 수 있지만 사람 간 i.i.d. 또는 해당 검정에 충분한 대체 가정이 남는다.

1. **Cross-fitting은 누수를 막지만 i.i.d. 검정 표본을 자동으로 만들지 않는다.** 사람 `g`의 예측에 그 사람의 정답이 들어가지 않아도, 다른 사람을 평가하는 모형의 학습에는 `g`가 들어간다. 겹치는 학습 집합 때문에 34개 out-of-fold 손실차는 서로 의존한다. 이를 LTT의 i.i.d. HB 검정에 그대로 넣는 것은 정당화되지 않는다. Cross-fitting 결과는 후보 설계·불확실성 탐색에는 유용하되, 인증에는 별도 증명·안정성 가정·독립 calibration split 등이 필요하다.
2. **Fold 모형과 최종 재적합 모형은 다르다.** 33명으로 학습한 알고리즘의 cross-fitted 점수가 34명 전체 재적합 predictor의 같은 위험을 인증하지 않는다. 기준선도 평가 사람을 제외해 추정하고, 전처리·상태 정의·목표 문턱까지 fold 내 학습 경계를 맞춰야 한다. 기존 34명 전체로 적합한 기준선을 재사용하며 같은 34명을 “독립 calibration”이라고 부를 수 없다.
3. **이미 본 dev/test는 새 인증 자료가 아니다.** 현재 가설은 실패 결과와 사후 진단에서 나온다. 개발 8명 또는 이미 노출된 시험을 사후 강도·집합·검정 선택에 쓰고 새 확인시험이라고 부르면 안 된다. 새로운 사람 또는 그와 동등하게 독립성이 보존되는 사전 명세 확인 절차가 필요하다.
4. **Unbounded log loss의 문제는 정확히 처리해야 한다.** 유한 `ε` floor를 **실제 예측 확률에** 강제하여 BCE를 유계화하거나, 두 predictor의 확률비에 양방향 경계를 주어 loss difference를 유계화할 수 있다. 점수만 사후 clip하면 보장 대상은 clipped score로 바뀐다. Joint 분포를 평활화하면 정합성을 보존할 수 있으나 개별 주변확률을 따로 clip하면 정합성이 깨질 수 있다. 단방향 mixture bound는 regret의 상한만 주므로 표준 양방향 유계 HB 적용을 혼자 정당화하지 않는다.
5. **표본 수 비용은 매우 클 수 있다.** 가상의 독립 calibration 사람 `m`, 고정 후보 하나, 실제 BCE 확률이 `[ε,1−ε]`이면 `B=−log ε` 및 사람별 차이 `D_g∈[−B,B]`다. 기본 Hoeffding 상한은 `mean(D)+B sqrt(2 log(1/δ)/m)`이다. `ε=.01, δ=.05, m=34`에서도 보정폭은 약 **1.93**이다. 이는 최상의 검정이 불가능하다는 하한이 아니라, 손쉬운 distribution-free 인증이 얼마나 보수적인지 보여 주는 예다. 실제로 학습/보정을 나누면 `m<34`이며, 여러 후보·여러 집단 동시보정은 비용을 더한다. 더 타이트한 확률비·분산·안정성 구조를 입증해야 유용해질 수 있다.

HCP/GHCP를 가져와 이 공백이 이미 해결됐다고 할 수 없다. 그 논문의 coverage theorem을 log-loss regret theorem으로 바꾸는 추가 논증이 필요하다. 또한 사람 **평균** 위험 인증, 관측 집단별 위험 인증, 모든 새 사람의 조건부 위험 인증은 서로 다른 목표다. 마지막 목표가 가장 강하고, 여기의 작은 자료에서는 특히 정당화하기 어렵다.

## 5. 남길 만한 연구 질문과 신규성 문턱

유의미하게 좁힌 질문은 다음과 같다:

> 10패턴의 구조적 simplex에서 실제 주변 proper score를 사용하고, 참여자 단위 학습·검증 의존을 정확히 반영하여, 강한 Markov 기준선보다 위험이 작다고 인증되는 영역에서만 비영(非零) residual을 허용할 수 있는가? 그렇지 않은 영역에서는 자동으로 기준선이 되는가?

현재 상태에서는 이를 **방법론 후보/연구 질문**이라고만 부를 수 있다. 다음 중 하나 이상의 실제 결과가 있어야 단순 조합 이상의 주장이 가능하다.

- 새 사람을 단위로 한 유효한 cross-fitted regret 상한을 증명하고, 34명 규모에서도 상한이 유용해지는 명시 조건을 제시한다. 기존 split LTT·stacking보다 어느 보장을 강화하거나 어느 가정을 줄였는지 적어야 한다.
- 조건부 집합 `U(x)`의 유효성과 크기 사이 관계를 증명하며, 집합을 학습하는 과정의 적응적 선택까지 포함한다. “10개 패턴이라 적다”만으로 충분하지 않다.
- 같은 단순 혼합으로 표현할 수 없는 유용한 수정과, 그 수정의 주변 score regret 제어를 보인다. 계산이 조금 쉬워진 것과 통계적 보장이 새로워진 것을 구별한다.

작은 자료에 적용하는 선행 조합도 정직한 응용 연구가 될 수 있지만, 바이올린에 적용했다는 사실·zero initialization·일관된 head·개별 사람이 fallback한다는 표현만으로 알고리즘 신규성이 생기지 않는다.

## 6. 실행 전 고정할 반박 조건과 필수 대조군

이 절은 실험 실행 허가나 추가 실행 계획이 아니라, 향후 별도 계약이 필요한 최소 설계 조건이다. 현재 종료한 fixed12를 다시 열지 않는다.

**가장 작은 대조군**은 기존 Markov, Markov+하나의 전역 shrinkage/convex-mixture 강도, 같은 residual에 대한 단순 크기 상한, 그리고 독립 사람 calibration에 기반한 기존 LTT식 채택이다. 새로운 uncertainty-set/gating 장치를 주장하려면 같은 후보군·정보·예산에서 이들을 넘어서는 이유가 있어야 한다. Neural residual이 필요하다는 증거가 없으면 작은 정규화 선형 residual도 비교해야 한다. 많은 backbone/seed를 나열하는 방식은 이 질문에 답하지 않는다.

사전에 아래를 실패로 인정해야 한다.

- **Residual 무효:** 정해 둔 독립 사람 검증에서 기준선 대비 실제 주 score 개선이 없거나 상한이 0을 넘는다. 전체 fallback이면 안전 장치의 동작은 확인할 수 있어도 개선 방법의 성공은 아니다.
- **인증 무효:** 행을 독립 표본으로 센다, 겹치는 cross-fit 손실에 i.i.d. p-value를 쓴다, frozen 전체-train 기준선이 calibration 사람을 이미 학습했다, 또는 이미 본 시험을 새 확인으로 취급한다.
- **목적 불일치:** joint NLL 또는 Brier만 개선되고 사전 주 점수인 사람 평균 주변 BCE는 개선되지 않는다.
- **무의미한 집합:** 유효성을 유지한 `U`에서 거의 항상 `q=b`이고, 유용한 변화를 얻으려면 정당화되지 않은 집합 축소가 필요하다. 이 경우 통계 정보가 부족하다는 결과를 기록한다.
- **추가 장치 불필요:** 같은 유효 위험수준·동일 정보에서 전역 혼합 또는 기존 LTT가 동등하거나 더 좋다. 복잡한 gate의 방법론적 실익은 반박된다.
- **가정 의존:** 사람 내 교환가능성, 상태 내 동질성, 표본 크기 무시가능성 또는 제한된 shift 가정의 작은 위반에서 보장이 붕괴한다. 해당 가정 아래의 조건부 결과로 한정해야 한다.

**현재 권고:** 새로운 모델을 임의로 더 돌릴 근거는 없다. 먼저 사후 진단이 residual 폭주·확신 오류·사람 간 shift라는 가설을 얼마나 지지하는지 확인하고, 그에 맞는 최소한의 결정론적 harm cap과 기존 통계 gate를 명세할 수 있다. 다만 그 조합을 “새로운 안전 잔차 방법”으로 부르려면 위의 의존·추정량·조건부 집합 문제에서 실제 새 결과가 필요하다. 34명과 이미 사용한 dev/test만으로 강한 새 사람 비악화와 성능 개선을 동시에 확증할 수 있다는 주장은 현재 근거가 없다.
