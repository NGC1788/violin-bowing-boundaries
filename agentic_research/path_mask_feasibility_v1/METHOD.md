# Count-state marginalization for masked future-window variability

**Proposed method, not an established novelty or a measured performance improvement.** This unit derives a possible algorithm and reviews its overlap. No data fitting, numerical runs or new data access are part of this document. The preceding finite-mask catalogue remains a failed operational screen.

## 1. Prediction target and joint model

The input \(x\) contains only the first second of a recording. The future contains \(T=46\) pitch-cents and dBFS frames. A pitch mask \(M\in\{0,1\}^T\) is unobserved at prediction time. The observed pitch sample standard deviation uses only mask-selected frames and is valid when \(N=\sum_tM_t\ge23\). Power uses all frames. Both use ddof1. The nine queries are three pitch thresholds crossed with three power thresholds, fitted on inner train only.

A small shared regime \(H\in\{1,\ldots,K\}\) induces dependence between channels and mask:

\[
p(H=h\mid x)=\pi_h(x),\qquad
p(M\mid h,x)=p(M_1\mid h,x)\prod_{t=2}^T p(M_t\mid M_{t-1},h,x).
\]

All Bernoulli probabilities lie strictly inside \((0,1)\), so the declared mask law supports every binary pattern, including previously unseen ones. This is a **generative law**, not a dictionary lookup. Numerical pruning will be an approximation to that law, not a claim that discarded masks are impossible.

Partition the future into \(B\) consecutive blocks \(I_b\), with lengths \(L_b\). Define \(N_b=\sum_{t\in I_b}M_t\). Conditional on \(h,x,\mathbf N\), pitch innovations are iid Gaussian with one shared scale:

\[
Y_{p,t}=a_{h,b}(x,\mathbf N)+\sigma_{p,h}(x,\mathbf N)\epsilon_{p,t},
\quad t\in I_b,\quad \epsilon_{p,t}\stackrel{iid}{\sim}N(0,1).
\]

The dBFS path has a declared mean vector \(d_{h,t}(x,\mathbf N)\) and iid Gaussian scale \(\sigma_{d,h}(x,\mathbf N)\). Precisely, **conditional on \(h,x,M\)**, all innovations are iid Gaussian within each channel and independent between channels, and the path-law parameters depend on \(M\) only through \(\mathbf N(M)\). Gaussian/iid behavior conditional only on the count vector would not suffice if mask arrangement selected innovations. Channels are generally **not** independent after marginalizing the shared regime/counts.

Count-dependent means/scales define a restricted pattern-mixture model \(p(M\mid x)p(Y\mid M,x)\). The simpler base version drops count dependence. Even the richer version does not represent arbitrary value-dependent voicing or general correlated innovations: patterns with the same block counts have the same conditional statistic law. This is an explicit modeling restriction, not a solution of general MNAR identification. Only the measured masked descriptors are targets; latent true pitch in unvoiced frames is not recovered.

## 2. Exact count sufficiency under this model

For \(n=\sum_bn_b\ge23\), the pitch conditional CDF is

\[
F_{p,h,\mathbf n}(u)=F_{\chi'^2_{n-1}(\lambda_{h,\mathbf n})}
  \left(\frac{(n-1)u^2}{\sigma_{p,h}^2}\right),
\qquad
\lambda_{h,\mathbf n}=\frac{
 \sum_bn_ba_{h,b}^2-(\sum_bn_ba_{h,b})^2/n}{\sigma_{p,h}^2}.
\]

All parameters in this formula may depend on \(x,\mathbf n\). The formula follows by centering a spherical Gaussian vector, not from a new chi-square identity. Importantly, **mask positions within each block disappear from this conditional statistic**. This is why count-state compression is exact for this restricted model even when the mask Markov chain is time-inhomogeneous and not exchangeable.

Power has the corresponding noncentral chi-square CDF with df45 and noncentrality \(\|C_Td_h\|^2/\sigma_{d,h}^2\). With count probabilities \(w_h(\mathbf n)=P(\mathbf N=\mathbf n\mid h,x)\), the joint query is

\[
q_x(u,v)=\sum_h\pi_h\sum_{\mathbf n:\sum n_b\ge23}
w_h(\mathbf n)F_{p,h,\mathbf n}(u)F_{d,h,\mathbf n}(v).
\]

The product is **inside the conditional mixture**. Invalid-mask mass remains in normalization and contributes to the all-false rule pattern. Full exact evaluation of the same positive mixture yields nonnegative cells on the common threshold grid. No posthoc coherence projection is a new ingredient.

## 3. Classical count DP, and why a small number of blocks matters

Let \(D_t(\mathbf n,s)\) be joint probability of block counts through time \(t\) and final mask bit \(s\). With \(b(t)\) the current block,

\[
D_t(\mathbf n,s)=\sum_{r=0}^1
D_{t-1}(\mathbf n-s e_{b(t)},r)P_t(s\mid r,h,x).
\]

Initialization uses \(p(M_1\mid h,x)\). The terminal sum over \(s\) gives \(w_h\). This is a standard Markov occupation-count recurrence with augmented state, **not a newly invented DP**.

Memory is \(O(K\prod_b(L_b+1))\); a simple time bound is \(O(KT\prod_b(L_b+1))\). This is polynomial in \(T\) only for fixed \(B\), and exponential in the number of blocks. There is no general polynomial-time solution for arbitrary means hidden in this expression.

For two blocks of23frames there are at most576count pairs, and300satisfy \(n\ge23\). With two regimes and three thresholds this needs1,800pitchCDF evaluations per example; count-dependent power also needs1,800. Without count dependence, power needs only six. These are operation counts, **not measured runtimes**. There are \(2^{46}\) mask paths; comparing only with their enumeration would be a weak experiment. Required alternatives include Monte Carlo/Rao–Blackwellized sampling and direct density heads.

## 4. Proposed practical extension: retained count states with valid-mass accounting

The candidate algorithm maintains at most \(L\) count/end-bit states at each step, keeping their **unnormalized** probabilities. It never renormalizes away the discarded mass. A trace records pruned prefix events. A surviving state contains only mass of paths whose prefixes survived every pruning step, not the full probability of that count state. Merging later paths does not restore already discarded paths. An independently computed, exact one-dimensional count DP gives

\[
g_x=P(N\ge23\mid x).
\]

Let \(a_x\) be the total retained terminal valid mass, after mixing regimes. Define the omitted **valid** mass

\[
r_G=g_x-a_x\ge0.
\]

The one-dimensional guard calculation costs \(O(KT^2)\). It does not see an actual future mask. It separates omitted computational mass from genuinely invalid-mask probability.

Let \(q_R(u,v)\) be the retained sum of conditional CDF products. Since each product lies in \([0,1]\),

\[
q_R(u,v)\le q_x(u,v)\le q_R(u,v)+r_G.
\]

This residual-mass inequality is elementary truncated marginalization, not a new theorem by itself. The proposed contribution would have to come from a useful **count-compressed, gradient-controlled, budget-adaptive implementation for this nonlinear masked statistic**, and demonstrable advantages over close existing methods.

On the registered finite nine-query grid, a coherent point approximation can assign all omitted valid mass to the all-false output pattern. This is an explicit **computational fallback**, not an inferred invalid-mask probability and not an exact finite-valued path law for arbitrary thresholds. Full-law query answers remain intervals. A high omitted mass is a failed approximation, not a confidence estimate.

Retrying a forward DP at doubled \(L\) uses the same prefix/model parameters, never labels. Total DP transition work of a doubling schedule is within a constant factor of its largest beam, with sorting/CDF work accounted separately. Different beam runs are not automatically nested and their error need not decrease monotonically. Recompute the actual residual and envelope and retain the best valid envelope rather than assuming improvement. If a maximum beam/latency is reached before the specified approximation criterion, return an accuracy failure rather than quietly accepting a looser result.

## 5. What can be said about gradients

Retained DP and ordinary CDF partial derivatives are differentiable on a fixed active trace. The mask-logit gradient estimate is the sum of retained event derivatives, not a claim that the top-L selection operation is everywhere smooth.

For an independent time-specific Bernoulli logit \(\theta_j\), each mask-path score satisfies \(|\partial_{\theta_j}\log p(M)|\le1\). Holding the conditional Gaussian-head parameters fixed,

\[
\left|\partial_{\theta_j}q_x-\widehat{\partial_{\theta_j}q_x}\right|\le r_G.
\]

A shared logit used at at most \(T\) steps gives the looser bound \(Tr_G\). Shared regime-softmax logits have a score bound of1. These are **decoder parameter partials**. A whole-network bound requires the actual output Jacobian and the Gaussian-head contribution; parameter sharing cannot be ignored.

For Gaussian mean partials, \(|\partial_\lambda F|\le1/2\) and

\[
\|\nabla_\mu F_{p,M}\|_2\le\frac{\|C_M\mu_M\|_2}{\sigma_p^2}.
\]

A compact parameter box with positive noise floor yields finite derivative envelopes; without such restrictions a small omitted probability does **not** bound all gradients. In particular, scale derivatives, tiny-noise regimes, threshold tails and active-set switches need separate checks. A CDF value bound alone is not a gradient certificate. No claim is made that refinement boundaries are continuous or that this derivative is the global gradient of a smooth beam-selected objective.

## 6. Arbitrary predicted means: an honest coarsening envelope

Suppose a declared Gaussian mean path \(\mu_t\) is approximated on each block by \(a_b\), and the noise realization and scale are shared in a coupling. Let \(e_b\ge\max_{t\in I_b}|\mu_t-a_b|\), using **predicted mean values only**. For any selected mask with count vector \(\mathbf n\),

\[
|S_p(Y)-S_p(\bar Y)|\le
\delta_p(\mathbf n)=\sqrt{\frac{\sum_bn_be_b^2}{n-1}}.
\]

This follows from the reverse triangle inequality and the operator norm1 of centering. It does not compare unrelated stochastic paths or differently specified covariance laws. A corresponding full-frame bound applies to power if its mean is also approximated. Constant mean offsets can make this bound conservative; centering the deterministic mean error before bounding can tighten it, but no unproved tightening is assumed here.

Consequently each retained state's joint CDF is bracketed by its conditional CDF evaluated at \((u-\delta_p,v-\delta_d)\) and \((u+\delta_p,v+\delta_d)\), with negative lower thresholds interpreted as impossible events. Summing weighted brackets and adding \(r_G\) to the upper bound gives an envelope for the uncompressed declared mean-path law. Coarsening and pruning errors are distinct. Refinement must reduce the actual reported envelope, not merely increase block count.

When refining blocks, the original declared path law must remain fixed. Count-dependent heads must keep their original conditioning counts: refined counts may be summed back to the original counts, but must not be fed as a new conditioning variable to the head. Otherwise the law changes rather than its approximation improving. The first proposed numerical coarsening test restricts heads to count-independent parameters to avoid this ambiguity.

These are real-arithmetic bounds under the specified law. SciPy CDF output and ordinary floating-point DP do not automatically provide validated intervals. If separate CDF and arithmetic errors are bounded by \(\eta\) and \(\xi\), they must be added explicitly. Passing fixtures would give empirical numerical evidence, not a universal floating-point certificate.

## 7. Relation to the training objective

For each query, an interval \([l,u]\) gives a label-free worst-case uncertainty in clipped binary log loss:

\[
\Delta_\ell\le\max\left\{\log\frac{u'}{l'},\log\frac{1-l'}{1-u'}\right\},
\quad l'=\operatorname{clip}_\epsilon(l),\;u'=\operatorname{clip}_\epsilon(u).
\]

Near0or1this may be large despite small absolute probability error. The bound concerns **numerical/model-compression error under the chosen law**, not calibration, generalization, or superiority to Markov. Nonzero gradient guarantees require additional derivative envelopes. No numerical tolerance or deployment-safe loss claim has been established in this written unit.

Train-only future masks and paths may supervise the generative law, but they are labels. At inference all \(M,\mathbf N,H\) are marginalized. Any direct-MDN comparison must receive identical auxiliary future supervision with the same backbone/information/compute. A generative-likelihood gain alone is not a win on participant-macro future rule loss.

## 8. What would falsify this proposal

- Precise prior already supplies an equivalent learned masked-statistic count/truncation decoder, leaving no meaningful new algorithm or use advantage.
- Exact count compression requires too few blocks to represent useful paths, while necessary refinement has exponential state growth.
- At realistic prefix-only parameter values, retained valid mass or CDF/gradient accuracy cannot meet a fixed tolerance inside the CPU budget.
- Head outputs exploit numerical CDF errors, pruning thresholds or fallback instead of improving the same mathematical law.
- Finite-regime conditional iid Gaussian assumptions dominate the error; count support alone does not repair the observer. Here50ms frames overlap at10ms hops. Shared raw samples can induce descriptor-error dependence, which a small regime mixture does not automatically remove. Raw descriptor autocorrelation alone is also not an estimate of conditional innovation covariance. Under genuinely correlated Gaussian innovations, equal block counts generally do not imply equal quadratic-form laws; the count-sufficiency proof no longer applies.
- Equal-supervision direct MDN or source Markov remains as good or better, including invalid rows and unfavorable participants.

None of these checks has been run in this written unit. The next decision must distinguish an interesting executable numerical hypothesis from an already novel and successful learning method.
