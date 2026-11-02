# 학습 방법 후보: 미래 관측 누락을 주변화하는 구조적 확률 예측

**설계 후보이며 아직 학습·성능 검증하지 않았다.** 수치 감사의 성공이 이 학습 방법의 성공을 뜻하지 않는다. 실행 전 별도 학습 계약·누수 시험·코드 검토·동결이 필요하다.

## 정보 흐름

\[
x_{\le1s}\xrightarrow{\text{small neural encoder}}
\{\pi_h,\;p_h(M_1),\;P_{h,b}(M_t\mid M_{t-1}),\;
\mu_{p,h,b},\mu_{d,h,b},\sigma_{p,h},\sigma_{d,h}\}
\xrightarrow{\text{count-marginalized decoder}}q_{1:9}.
\]

모든 추론 입력은 과거 prefix에서만 온다. 실제 미래 mask·guard·path는 추론 입력으로 사용하지 않는다.

Every inference input above comes from the past prefix. The actual future mask, guard and path are labels, never inference inputs. The proposed base model has two regimes and two future blocks. A mask's future initial distribution is predicted from the prefix rather than copied from the last prefix frame: a250ms forecast gap lies between them.

The encoder outputs the parameters of a conditional future law. The decoder marginalizes its unobserved regime and mask counts to produce probabilities of nine future events. This is structured probabilistic learning, rather than simply fitting a curve to a physical law. Actual cents/dBFS units can be standardized using inner-train-only constants and transformed back at the target thresholds.

Global future levels matter for auxiliary path likelihood but vanish from standard-deviation events. A common level and block contrasts can therefore be parameterized separately, with a length-weighted zero-sum constraint on contrasts. This addresses a known nonidentifiability in the parameterization; it is not itself a new identification theorem or recovery of physical dynamics.

## Learning objective

Let \(y_j\) be a future event label and \(Y_{\mathrm{obs}},M\) the future observed paths/mask during training. A proposed objective is

\[
\mathcal L=\frac19\sum_{j=1}^9\operatorname{BCE}(y_j,q_j(x))
+\alpha\frac{-\log\sum_h\pi_h(x)
  p_h(M\mid x)p_h(Y_{\mathrm{obs}}\mid M,x)}{T}.
\]

Pitch likelihood uses only voiced frames; unvoiced stored zeros are not pitch observations. Power and mask likelihoods retain every record, including invalid pitch targets. The generator must preserve a positive noise floor and explicitly bounded numerical parameter range. The actual optimizer, loss weight, bounds, number of epochs and clipping constants are not selected by this document; they must be fixed in the subsequent contract before any fitting.

Training observes labels to evaluate a likelihood. Inference does not condition on those labels. Source-fold quantiles, auxiliary standardization, scalers, priors and stopping decisions all belong to inner train. No development/test truth is opened or hashed. Historical source exposure remains an exploratory limitation.

## Minimal comparison that could justify an AI contribution

A possible bounded comparison uses four methods×three fixed source-person folds=12jobs:

1. Free10-pattern joint prediction head with the same generator's auxiliary supervision.
2. Direct two-statistic mixture-density head with identical auxiliary supervision and prefix information; future validity must be predicted, not supplied.
3. The proposed exact count-marginalized head, whose main loss differentiates through the same generator parameters.
4. The same generator and count law with an explicit conditional-variance moment approximation, isolating whether the noncentral-chi-square law matters.

Strong source Markov/HGB baselines remain necessary. The Markov baseline for rule transitions is distinct from the latent binary-mask Markov chain inside the proposal. All parameter counts, preprocessing, gradients, losses and CPU costs must be disclosed. The moment approximation is a control, not a claim of a new distribution. A Monte Carlo comparison under the same law is also necessary before claiming a computational advantage; the twelve fits alone do not supply it.

The exact choice of comparisons remains subject to the numerical audit and training-readiness review. This is not an already frozen or executed experiment. No widening of the existing budget is authorized: at most the remaining three12job batches,180seconds per job and1,800seconds per batch before2026-11-06 00:00KST.

## What would count as success

The proposed mechanism is that a small conditional path law supplies a useful inductive bias and represents future observation uncertainty, reducing sample needs compared with a free density head. It is a hypothesis. Numerical coherence, a correct DP, an appealing diagram or a lower auxiliary likelihood is insufficient. Improvement must appear in the fixed participant-macro future-event score against strong and equally supervised methods, without excluding invalid or unseen cases. An unfavorable result closes this candidate rather than changing the target after the fact.

The present claim is a concrete **learnable AI architecture proposal**. Methodological originality and real performance remain open empirical questions.
