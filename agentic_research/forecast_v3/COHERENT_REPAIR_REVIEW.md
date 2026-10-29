# 확률 일관성 보정 검토 — 강한 보조 기준선

2026-10-30. 개발 C/D의 점수나 예측을 읽지 않고 문헌, 수식, 합성 예제만 검토했다. 결론은 **구현할 가치가 있는 알려진 추론 기준선**이다. 이 보정의 일반 정리나 KL projection을 새 방법론이라고 주장하면 안 된다. 원래 모델의 주 점수는 그대로 두고 보조 결과로 보고한다.

## 가장 가까운 선행

- **Predd et al. (2009), Probabilistic Coherence and Proper Scoring Rules.** 논리적으로 가능한 결과들의 convex hull이 일관된 확률 집합이라는 명제와, proper score에서 비일관된 예측을 모든 가능한 결과에서 개선하는 일관된 예측이 존재한다는 정리가 이미 있다. §4의 Proposition 3 및 §5는 Bregman projection의 방향과 비유계 점수의 내부점 조건까지 다룬다. 아래 log-loss 보장은 이 고전 결과의 특수화다. [저자 원문](https://arxiv.org/html/0710.3183), [최종 저널 서지](https://research-explorer.ista.ac.at/record/2388).
- **Predd et al. (2008), Aggregating Probabilistic Forecasts from Incoherent and Abstaining Experts.** 비일관된 확률의 계산적 보정·결합과 성능 보장을 이미 연구했다. 따라서 “확률 모순을 고쳐 예측을 개선한다” 자체도 새로운 연구 질문이 아니다. [저널 원문 페이지](https://pubsonline.informs.org/doi/10.1287/deca.1080.0119).
- **Anany Kotawala (2026), Arbitrage-Free Forecasts from Language Models via Coherence Projection**, Forecast@ICML26 workshop poster. 공식 검색 기록과 색인된 원문에서 coherent-marginal projection, Bernoulli-KL projection, Appendix Theorem A.7의 loss-matched simplex projection을 확인했다. 이는 최근 AI 예측 문맥에서도 가까운 선행이다. 학회 본회의 논문으로 표현하지 않는다. 일반 소개 페이지/PDF 직접 열기는 브라우저 확인 화면에 막혀 전체 원문을 읽었다고 주장하지 않는다. [공식 기록](https://openreview.net/forum?id=Tqos7VqQhH), [원문](https://openreview.net/pdf/5689b12fe09983e997b0f3047717405d94adf2ee.pdf).

문헌 검색은 표적 조사이며 전수 조사가 아니다. 위 선행만으로도 일반 coherentization·Pythagorean dominance를 우리 신규성으로 내세울 수 없다는 판단은 충분하다.

## 우리 규칙에 대한 수학 검산

감사된 9개 관측 규칙의 가능한 0/1 벡터를 \(V=\{v_1,\ldots,v_{10}\}\), 일관된 주변 확률 집합을 \(C=\operatorname{conv}(V)\)라 하자. 원 예측 \(q\in(0,1)^9\)에 대해 필요한 방향은 다음이다.

\[
p^*=\arg\min_{p\in C}D_\Phi(p,q),\qquad
D_\Phi(p,q)=\sum_{j=1}^{9}\left[p_j\log\frac{p_j}{q_j}+(1-p_j)\log\frac{1-p_j}{1-q_j}\right].
\]

\(\Phi(p)=\sum_j[p_j\log p_j+(1-p_j)\log(1-p_j)]\)의 엄격한 볼록성과 projection의 일차 조건을 이용하면, \(y\in C\)에 대해

\[
\langle\nabla\Phi(p^*)-\nabla\Phi(q),y-p^*\rangle\ge0,
\]

\[
D_\Phi(y,q)-D_\Phi(y,p^*)-D_\Phi(p^*,q)
=\langle\nabla\Phi(p^*)-\nabla\Phi(q),y-p^*\rangle\ge0.
\]

가능한 실제 출력 \(y\in V\)는 이진 벡터이므로 \(D_\Phi(y,q)\)가 9개 binary log loss의 합이다. 따라서 평균 손실 \(L\)에 대해서는

\[
L(y,p^*)\le L(y,q)-\frac{D_\Phi(p^*,q)}9\quad\text{for every }y\in V.
\]

q가 비일관적이고 exact projection을 구하면 오른쪽 감소량이 양수다. q가 이미 일관되면 바꾸지 않는 것이 맞다. 이 증명은 \(D(p\Vert q)\) 방향이다. 역방향 \(D(q\Vert p)\)이나 latent mass의 유클리드 projection에 동일 보장을 그대로 붙이면 안 된다.

여기서 결과 y가 반드시 감사된 V에 들어가야 한다. observer 변경, 다른 guard/fixed, 규칙별 따로 결측 처리 등으로 가능한 결과 집합이 바뀌면 보장도 다시 검토해야 한다. 물리적 참 운동 상태나 음질에 대한 보장이 아니다.

## 실제 평가기에서는 추가 검사가 필요하다

현재 평가기는 확률을 \(\epsilon=10^{-6}\)과 \(1-\epsilon\) 사이로 clipping한 뒤 평균 log loss를 계산한다. 위 증명은 원래 log score에 대한 것이다. 후보 확률 clipping, solver 오차, 수치상 제약 위반이 있으므로 정리만 인용하고 구현에 무조건적인 보장을 붙이지 않는다.

구현은 실제 미래 라벨을 보지 않고, 후보 p에 대해 가능한 10개 결과를 모두 검사한다.

\[
\Delta_{\max}(p,q)=\max_{v\in V}\big[L_\epsilon(v,p)-L_\epsilon(v,q)\big].
\]

float64로 계산한 \(\Delta_{\max}\le0\)인 경우만 받아들인다. 양의 허용오차를 넣지 않는다. `<= 1e-8`은 “최대 1e-8 악화 허용”이지 비악화가 아니다. 검사가 실패하면 원래 q를 그대로 반환한다.

이것은 **유한 출력 집합에 대한 부동소수점 검사**다. 실수의 정확한 interval arithmetic 증명이라고 부르지 않는다. 엄밀한 실수 오차 인증이 필요하면 outward-rounded log의 상한으로 \(\Delta_{\max}\)를 판정해야 한다. 현재 메타데이터에 이 수치적 범위를 명시한다.

fallback q가 비일관적일 수 있으므로 “모든 출력이 항상 일관됨”도 보장하지 않는다. `accepted_rows`, `fallback_original_rows`, `coherent_output_rows`, `incoherent_fallback_rows`, `max_certified_delta`를 함께 기록한다.

## 구현 및 통제된 비교

`coherent_repair.py`는 입력 확률만 받는다. 모델 재학습·개발 라벨·미래 특징·평가 그룹별 최적화가 없다.

1. 10개 패턴의 affine 좌표를 이용한 latent-mass simplex projection으로 초기화한다.
2. forward Bernoulli-KL 목적함수에 고정 최대 64회의 projected gradient와 제한된 backtracking을 적용한다. 정확한 최적해라고 주장하지 않는다.
3. 최종 후보가 10개 결과의 수치적 비악화 검사를 통과하면 사용한다. 실패하면 초기 후보를 같은 검사에 넣고, 그것도 실패하면 q를 반환한다. 이 순서는 평가 전에 고정한다.
4. HGB와 독립 9-head MLP의 **보조 추론 기준선**으로 적용한다. 원래 모델의 주 log loss를 덮어쓰지 않는다. `auxiliary.forward_kl_coherent_repair`에 점수와 검사 메타데이터를 따로 저장한다.
5. 독립 MLP에 이미 포함한 latent-mass 유클리드 보정도 별도로 유지한다. 그 보정은 marginal-space 최단 projection이 아니며 log-loss 비악화 보장도 없다.

이 기준선이 강하면 공동 neural head의 신규성 주장은 더 어려워진다. 학습 제약의 효과를 주장하려면 공동 head가 원래 독립 head뿐 아니라, 이런 사후 보정과 강한 HGB 계열까지 비교해야 한다. 반대로 보정의 이득 자체가 작으면 이를 커 보이는 성능 개선으로 포장하지 않는다.

## 검증 범위

추가 합성 테스트는 다음을 확인한다.

- 10개 패턴의 점수가 기존 evaluator의 elementary operation·clipping·평균과 **배열 단위로 같음**.
- 무작위 예측, 0/1 및 극단 확률에서 반환값의 어떤 가능한 목표 손실도 원래보다 커지지 않음.
- 악화 후보·NaN·범위 밖 후보가 원래 예측으로 fallback함.
- 아주 작은 양의 악화도 양의 tolerance로 통과시키지 않음.
- 이미 일관된 예측을 임의로 크게 바꾸지 않음.
- 반복 수 상한과 비정상 입력을 거부함.

고전 정리 방향의 별도 임시 검산에서도 무작위 합성 확률 120개에 대한 forward-KL SLSQP 후보가 10개 출력 전부에서 원 log loss를 낮췄다. 이는 수학적 증명을 대체하지 않으며, 실측 AI 성능 결과도 아니다. 이 모듈을 추가하면서 C/D를 채점하거나 source-only sanity를 재실행해 유리한 설정을 고르지 않았다.

## 신규성과 의미의 판정

현재 단계에서 확실한 정보기술은 미래 observer 출력을 예측하는 학습, 논리 구조를 보존하는 확률 표현, 고정 비용의 추론 보정, 실제 점수에 맞춘 유한 출력 검사다. 그러나 각각 일반적 원리에는 선행연구가 있다. **고전 원리를 정직하게 구현하고 강한 기준선으로 삼는 것이며, 이것만으로 높은 방법론 신규성이 확보되지는 않는다.**

향후 의미 있는 기여 후보는 검증된 관측 규칙의 시간 예측 문제에서 이 구조가 언제 도움이 되는지, 학습 중 제약과 사후 보정의 차이가 무엇인지, 추가 계산비에 비해 개선이 충분한지를 재현 가능한 결과로 밝히는 것이다. 아직 2% 이상 성능 개선, 새 이론, 실시간 경보, 바이올린 일반화는 입증하지 않았다.
