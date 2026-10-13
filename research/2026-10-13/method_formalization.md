# Coherent observation-family set learning: formal notes

2026-10-13. This note records a candidate method and elementary deductions used to audit it. It is not a claim of novel theorems, a guarantee of physical truth, or evidence of successful experiments. The implementation and run manifests must be checked separately.

## 1. The uncertainty is in an observation operator

For trial `i`, let `x_i` contain bowing inputs available before observing the response, and let `r_i` contain response-derived waveform features used only to construct labels. A prespecified classifier setting `eta` defines

`y_eta,i = C_eta(r_i) in {0,1}`.

For example, `eta` can vary periodicity and relative-flyback thresholds. The same `eta` is applied to **all** trials; it is not chosen independently for each cell or speed. A finite allowed family `E` induces a set of complete target maps:

`F = {T_eta = {i : y_eta,i=1} : eta in E}`.

The valid scientific claim is robust agreement with this declared family. The statement `physical truth belongs to F` requires additional evidence and is not assumed here. Threshold variants are dependent hypotheses about one measurement operator, not independent expert votes.

Input-only predictors use `p_theta(x_i)`. None may take amplitude, periodicity, flyback statistics, response-derived trial labels, or other target features as predictive inputs. A target-domain calibration such as impedance fitted from flyback height must be declared and shared across baselines or removed.

An off-plateau trial is excluded consistently before constructing every map. A zero or negative force cannot be passed into a logarithm by silently replacing it with a small positive constant; keep an explicit common eligibility rule and report its effect.

## 2. Coherent family versus rectangular relaxation

Let `L = intersection_eta T_eta` and `U = union_eta T_eta`. Replacing `F` by

`B = {T : L subset T subset U}`

allows every ambiguous cell to choose its own operator. In general `F subset B`, with strict inclusion. We call this the rectangular relaxation: it loses dependence among labels produced by one global threshold choice.

For a prediction set `P`, nonnegative fixed cell weights `w`, and positive union mass, define

`J(P,T) = w(P intersect T) / w(P union T)`.

Elementary exact bounds over the relaxation are:

`min_{T in B} J(P,T) = w(P intersect L) / w(P union U)`;

`max_{T in B} J(P,T) = w(P intersect U) / w(P union L)`.

**Proof.** Inside `P`, making an uncertain cell positive increases the numerator without changing the union. Outside `P`, making it positive increases only the denominator. The minimizing choices are therefore negative inside and positive outside; the maximizing choices reverse them. Fixed nonnegative weights preserve the argument. Adopt and test an explicit empty-union convention.

Consequently,

`min_B J <= min_F J <= max_F J <= max_B J`.

This ordering is deterministic and says nothing about unseen examples or true-label coverage.

### The relaxation can discard almost all useful information

Let `U={a,b_1,...,b_n}` with equal weights, and let the coherent family consist of

`T_j = U minus {b_j}`, for `j=1,...,n`, with `n >= 2`.

Every admissible complete map contains `n` positive cells and differs from every other map in only two positions. Yet `L={a}`. For `P=U`, the coherent worst-case IoU is `n/(n+1)`, whereas the rectangular lower bound is `1/(n+1)`. The gap tends to one as `n` grows: the relaxation invents the target `{a}`, which no admissible operator produces.

The issue also occurs for nested threshold families. If `F={{a},{a,b,c}}` and `P={a,b}`, then the coherent worst score is `1/2`, but the rectangular worst score is `1/3`, attained at the impossible target `{a,c}`. Thus simple monotonicity of a threshold does not remove the gap.

This example motivates evaluating the true finite family; it is not evidence that the particular experimental family has a large relaxation gap. Measure that gap on the backed-up maps and on a fresh sample.

## 3. Why the naive envelope loss cannot learn the ambiguous boundary

Consider the often tempting soft lower-envelope objective

`J_env(p) = sum_i w_i p_i L_i / [sum_i w_i p_i + sum_i w_i (1-p_i) U_i]`.

Its denominator simplifies exactly to

`w(U) + sum_{i outside U} w_i p_i`.

Thus `d J_env / d p_i = 0` for every ambiguous cell `i in U minus L`. The numerator also ignores these cells. This is not a method for resolving their labels: it resembles masked/partial overlap training with a constant uncertainty mass in the denominator. Any predictions between `L` and `U` that perfectly classify the stable cells achieve the same maximal hard-envelope score `w(L)/w(U)`.

This loss is useful as a **baseline or conservative objective**, but its novelty should not be overstated. Required controls are masked cross-entropy, masked soft-IoU/partial Dice, family-mean cross-entropy, and nominal-label overlap loss. [Partial Dice training on known pixels is established](https://pmc.ncbi.nlm.nih.gov/articles/PMC10807363/).

## 4. Candidate coherent-family objective

For a complete binary map `y_eta`, use the ordinary differentiable overlap score

`J_soft(p,y_eta) = sum_i w_i p_i y_eta,i / sum_i w_i (p_i + y_eta,i - p_i y_eta,i)`.

An exact finite-family robust objective maximizes `min_eta J_soft(p,y_eta)`, with a complexity penalty on the predictor. A smooth implementation can use

`smin_tau(J_1,...,J_M) = -tau * log(sum_j exp(-J_j/tau))`.

For this **unnormalized** version,

`min_j J_j - tau log M <= smin_tau(J) <= min_j J_j`.

If dividing the exponential sum by `M`, the score instead lies between `min J` and `min J + tau log M`. Both variants have the same gradient because their difference is constant. Record which one is logged; a normalized smooth minimum must not be advertised as a lower bound.

At `tau=0.05` and `M=15`, the generic smoothing gap bound is about `0.1354`, large enough to matter. This is an optimization smoothing parameter, not a confidence level. Report exact minimum IoU at evaluation regardless of the training surrogate.

The gradient is a weighted sum of per-operator score gradients with weights proportional to `exp(-J_eta/tau)`. Unlike `J_env`, a trial with family disagreement can receive a nonzero gradient because complete maps remain coupled. Standard minimax/DRO reasoning already covers this construction; the potential contribution is an observation-operator-aware formulation and useful empirical behavior, not inventing minimax optimization.

### Keep coupling across conditions

For groups `g` (e.g. string type and speed), fixed group weights `a_g` summing to one, and complete-map scores `J_g,eta`, the globally coherent score is

`min_eta sum_g a_g J_g,eta`.

The alternative `sum_g a_g min_eta J_g,eta` permits a different operator for each group and is always no larger. Report the latter as a separate conservative diagnostic if useful; do not substitute it without explaining that its uncertainty set is larger. Likewise, pooled IoU and macro-averaged IoU are different objectives and must be named explicitly.

### What a soft score does not guarantee

`J_soft` is a surrogate, not expected hard IoU in general. Better smooth minimum during training does not prove improved binary IoU at threshold 0.5. Fix the decision threshold on development data or by protocol, report hard and soft scores separately, and do not use a test-specific threshold. Direct overlap optimization has substantial prior work, including [Lovász-Softmax, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Berman_The_LovaSz-Softmax_Loss_CVPR_2018_paper.pdf).

## 5. Optional exact adversary for all partial-label completions

For fixed soft predictions, exact minimization over `B` differs from the relaxed envelope expression above. Let

`A=sum_{i in L} w_i p_i`,

`B0=sum_i w_i p_i + sum_{i in L} w_i (1-p_i)`.

For ambiguous membership indicators `t_i in {0,1}`,

`J_soft(p,t) = [A+sum_{amb} w_i p_i t_i] / [B0+sum_{amb} w_i (1-p_i)t_i]`.

Sort ambiguous indices by `p_i/(1-p_i)` in ascending order (`p_i=1` has infinite ratio). Evaluate the ratio for every prefix including the empty prefix; a minimizing prefix gives an exact minimizer. To see why, for a proposed ratio `r`, including index `i` changes numerator-minus-`r`-denominator by `w_i[p_i-r(1-p_i)]`; all negative terms should be included. Ties can be chosen arbitrarily. The complexity is `O(m log m)` for `m` ambiguous cells.

This gives an inexpensive stronger rectangular-ambiguity baseline if needed. It does **not** make the rectangular family scientifically appropriate, and the soft adversarial game's optimum need not correspond to an optimal hard prediction. It is an elementary fractional-programming observation, not a claimed new result.

## 6. Physical covariance is a testable prior

Use dimensionless referenced coordinates, for example

`u_alpha = log(F/F_ref) - alpha log(v/v_ref)` and `b=log(beta/beta_ref)`.

`alpha=1` assumes that changing speed shifts the force set approximately linearly. It is a hypothesis supported in some classical limits, not an exact invariant of every bowed-string model. Velocity-dependent friction has an additional velocity scale, and thermal/contact histories can break this collapse. In particular, a law involving `v/v0` need not be invariant when `v` changes but `v0` is held fixed.

Necessary ablations compare `(log F,log beta,log v)`, fixed `alpha=1`, and any learned `alpha` selected only from training/development conditions. One training speed cannot identify `alpha`; its apparent cross-speed advantage is evidence for the imposed prior, not data-driven recovery of a scaling law. Do not call `log(F/v)` itself dimensionless without reference scales (or an independently known impedance `Z` so that `F/(Zv)` is dimensionless).

The predictor may be an RBF/kernel model or a small network. High model capacity is not required for methodological interest, and may conceal whether the physical prior is doing any work.

## 7. Nearest general-method prior and precise distinction

| Prior | Collision / distinction |
|---|---|
| [Arachie & Huang, Adversarial Label Learning, AAAI 2019](https://ojs.aaai.org/index.php/AAAI/article/download/4186/4064), [Stoch-GALL](https://arxiv.org/abs/1906.00512) | Already trains against labels constrained by weak supervision, including dependent errors. We do not know true error-rate bounds for our weak labelers. Restricting an adversary to a declared operator family is a different uncertainty specification, not a new minimax principle. |
| [Sagawa et al., Group DRO, ICLR 2020](https://arxiv.org/abs/1911.08731) | Already optimizes worst-group risk and emphasizes regularization. Our operator indices change the complete labels on the same inputs rather than identifying a subpopulation; this distinction alone does not prove novel methodology. |
| [Yu, Ding & Bach, AISTATS 2022](https://proceedings.mlr.press/v151/yu22c.html) | Learns from noisy partial labelers using a generative model. Threshold variants of one waveform rule are strongly dependent; identifiability statements from that framework cannot be imported without verifying its assumptions. |
| [Cauchois, Gupta, Ali & Duchi, JMLR 2024](https://www.jmlr.org/papers/volume25/23-0253/23-0253.pdf) | Establishes predictive inference notions under weak supervision. Our deterministic family bounds are not their probabilistic coverage guarantees and do not independently cover mechanical truth. |
| [Partial overlap segmentation](https://pmc.ncbi.nlm.nih.gov/articles/PMC10807363/) and [Lovász-Softmax](https://openaccess.thecvf.com/content_cvpr_2018/papers/Berman_The_LovaSz-Softmax_Loss_CVPR_2018_paper.pdf) | Ignoring unknown pixels and optimizing overlap are known. The rectangular envelope's zero-gradient property makes masked losses particularly important competitors. |

The promising, falsifiable contribution is therefore narrower: **for physical regime maps with an uncertain but globally shared observation rule, retaining that shared structure may avoid the information loss of pointwise uncertainty envelopes, and may transfer better when paired with a justified physical coordinate prior**. Establishing that requires the coherent-versus-rectangular gap, every loss/coordinate ablation, and a fresh untouched sample. If family minimax does not improve over ordinary overlap or cross-entropy with the same coordinates, report that result and use the simpler method.

## 8. Executable finite checks

`python3 research/2026-10-13/verify_set_bounds.py` exhaustively compares the closed-form hard IoU bounds and the sorted-prefix soft adversary with all allowed completions of 240 small weighted examples (980 completions, seed 20261013). The 2026-10-13 run passed with maximum numerical discrepancy 0.0. It also verifies a 100-cell example whose coherent minimum is 0.99 while the rectangular minimum is 0.01. These are finite algebra checks, not validation of the physical classifier or evidence of performance on experimental data.
