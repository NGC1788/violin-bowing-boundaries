# 조건부 보정의 성립성과 신규성: 추가 직접 검토

2026-11-02. 기존 `violin_diagnosis_v1` 검토를 보완한다. 새로운 알고리즘·정리를 주장하지 않는다. 아래 출처는 원문 또는 저자 공개 원문이며 전수 문헌 부재를 증명한 체계적 검색은 아니다.

## 1. 느슨한 Hoeffding 경계 하나로 중단을 정당화하면 안 된다

[Learn then Test 원문](https://arxiv.org/html/2110.01052) §2.1–2.3, Proposition1을 확인했다. 학습된 고정 후보와 iid calibration 관측을 전제로 하며, 단순 quadratic Hoeffding보다 강한 Hoeffding–Bentkus p-value를 제공한다. 여러 후보 p-value 사이의 의존을 허용하는 것과, 한 p-value의 관측치가 임의로 의존해도 되는 것은 다르다.

구현은 원문의 **ceil과 e 계수**를 유지한다. 사람 손실차 \(D_g\in[-c,c]\)에 대해 \(L_g=(D_g+c)/(2c)\)라 하고 평균 비악화의 null \(E[D_g]\ge0\), 즉 \(E[L_g]\ge1/2\)를 검정하는 방향이다. 여기의 사후 개발 결과에는 그 유효성을 주장하지 않는다.

따라서 지난 문서의 “단순 경계로522명”은 그 특정 충분조건의 비용이다. 필요한 최소 참여자 수나 불가능성 하한이 아니다. 더 강한 기존 방법도 검토한 뒤 현재 후보를 평가해야 한다.

## 2. 더 강한 bounded-mean 추론도 이미 존재한다

[Waudby-Smith & Ramdas, Estimating means of bounded random variables by betting](https://arxiv.org/html/2010.09686), JRSSB86(1),2024: product betting, confidence-sequence 구성, 조건부 기대값 증명 구간을 확인했다. 정해 둔 전략 또는 과거만 사용하는 전략으로 비음수 자산을 만들며, 좋은 검정력을 얻기 위한 방식 자체가 선행이다.

이 감사는 가장 단순한 유효 대조만 구현한다. \(U_g=-D_g/c\in[-1,1]\)와 미리 고정한 \(\lambda\in[0,1]\)에 대해

\[
E_\lambda=\prod_g(1+\lambda U_g),\qquad
E_{\mathrm{mix}}=\int_0^1 E_\lambda\,d\lambda.
\]

고정 predictor의 독립 calibration 또는 null 아래 매 단계 \(E[U_g\mid\mathcal F_{g-1}]\le0\)가 있으면 이 값의 기대는 1 이하이다. 계산은 양의 가중치를 갖는 Gauss–Legendre 적분으로 degree-n 다항식을 적분한다. 결과를 본 뒤 최선의 lambda를 골라 그 값을 그대로 e-value라고 부르지 않는다.

최신 [Martinez-Taboada & Ramdas, Gaussian-efficient testing by betting on the mean of bounded data, arXiv:2608.21694v2,2026-10-30](https://arxiv.org/html/2608.21694)도 §3.1과 Theorem5.1의 가정을 확인했다. 독립성 없이 유효할 수 있지만 **고정된 조건부 평균 가정**이 남는다. 겹치는 cross-fit 손실이라는 이유만으로 이 조건이 자동 성립하지 않는다. 이 최신 알고리즘을 재구현·벤치마크한 것은 아니다.

## 3. 작은 n의 양성 대조와 의존성 반례

8명 모두 \(U_g=1\)이라면 uniform-mixture e는 \((2^9-1)/9=56.777\ldots\)로20을 넘고, HB도 .05 미만일 수 있다. 독립적이고 미리 고정된 유효 검정이라면 작은 n에서도 매우 강하고 일관된 이득을 검출할 여지가 있다. “8명이니 무조건 안 된다”는 결론을 내리지 않는다.

반대로 대칭 \(Z\in\{-1,+1\}\)를 한 번 뽑고 모든 \(U_g=Z\)로 두면 각 관측의 주변 평균은 0이다. 그래도 같은 e의 평균은 \(256/9>1\),20을 넘을 확률은 .5이다. 매 단계의 조건부 평균 제약은 깨진다. 독립 대칭 Bernoulli에서는 모든 가능한 개수의 경우를 합산하면 e의 평균은1이다. 코드가 두 경우를 정확 합산해 검사한다.

이는 주변 평균과 적절한 조건부 평균이 다르다는 **기본 반례**다. 실제 보정 fold가 이 극단적인 의존 법칙을 따른다고 주장하지 않는다. 겹치는 손실에 iid 검정 공식을 대입하는 일반적 정당화가 없음을 보여 준다.

## 4. 독립성 없는 온라인 보장이라는 우회도 새 방법은 아니다

[Korotin, V’yugin & Burnaev, Mixability of Integral Losses, Pattern Recognition120,2021](https://arxiv.org/html/1912.07048) §2–3, Theorem3.1을 확인했다. 정규화된 여러 손실의 평균에서도 mixability/exp-concavity가 유지되는 결과가 이미 있다. “9개 주변 BCE와 사람 내 반복을 평균해도 된다”는 사실을 새로운 이론으로 주장할 수 없다.

실제로 clipping이 작동하지 않는 영역에서 사람 한 명의 평균 BCE \(\ell_g\)는

\[
e^{-\ell_g(q)}
=\left\{\prod_{r,j} P_q(Y_{rj})\right\}^{1/(9R_g)}
\]

이다. 오른쪽은 양의 affine 사건확률의 기하평균이라 concave이다. 같은 사람 안에서 공통 혼합 가중치를 쓰는 지수 가중 전문가 혼합은 각 전문가 대비 누적 regret \(\log(1/w_k)\)를 갖는다. 두 전문가를 반씩 시작하면 평균 상한은 \(\log2/G\)로,8명에서는 .08664,34명에서는 .02039다. 이는 보장된 개선이 아닌 악화의 상한이다.

평가 clipping을 그대로 두면 이 exp-concavity를 무조건 주장할 수 없다. 합성 검산에서는 모든 joint 분포에 고정 uniform smoothing1e-4를 적용해 모든 주변확률을 `(EPS,1-EPS)` 안으로 두었다. 원래 모델이나 점수는 바꾸지 않았다. 이 유효성 대조는 합성8라운드뿐이고 실제 개발 자료의 순서를 바꿔 새 성능을 보고하지 않는다.

[Feng 외, Bayesian logistic regression for online recalibration and revision of risk prediction models with performance guarantees](https://arxiv.org/abs/2110.06866)와 공개 HTML의 AppendixC도 확인했다. 고정 예측기 대비 온라인 누적 손실과 oracle 보정기 대비 regret를 구분하는 직접 선행이다. 새로운 결과가 공개된 뒤 순차 갱신하는 설정은 현재의 offline 새 사람 위험과 다른 질문이다.

## 5. 신규성 판정 기준

전역 혼합, 상태별 혼합, prior logit, span cap, 유계 손실의 HB/betting, 주변 손실의 온라인 aggregation은 직접 선행 또는 기본 성질로 설명된다. **새 이름·바이올린 적용·정합적인10패턴·사람 단위 평균**은 이 공백을 메우지 않는다.

남은 후보는 의존을 처리하는 새 정리나, 같은 정보·위험수준에서 기존 방법보다 비자명한 수정이 가능한 조건부 예산 추정법이다. 현재 그러한 결과가 없다. 이 감사에서 상태별 혼합이 좋아지더라도 기존 conditional stacking의 사후 결과이며 신규성이 아니다. 나빠지면 현재의 조건부 장치를 밀 경험적 근거도 약해진다. 어느 경우에도 모델을 더 돌리는 것으로 이론·자료 독립성의 공백을 해결했다고 할 수 없다.
