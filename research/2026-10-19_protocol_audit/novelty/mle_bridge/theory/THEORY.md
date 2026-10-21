# What a minimum geometric obstruction can and cannot say about MLE

2026-10-21. Prepared before inspecting the empirical bridge results. This is an interpretation and method document; it contains no new empirical outcome. The prior fixed-column certificates are reused as selected audit targets. All data were previously exposed.

## 1. The two questions differ

For one speed folder let x_j=log(beta_j), and let [l_j,u_j] be a first-run log-force bracket. The prior hard decision at fixed slope p asks whether there exists a such that every interval

    [l_j - p*x_j - epsilon, u_j - p*x_j + epsilon]

contains a. A minimum fixed obstruction is a smallest fixed set S of columns for which the intersection remains empty throughout a declared common waveform-threshold domain. Its exactness concerns that universal intersection predicate and that domain.

The existing interval-Gaussian likelihood instead uses

    q_j(a,b,sigma) = Phi((u_j-a-b*x_j)/sigma)
                    - Phi((l_j-a-b*x_j)/sigma),
    log L = sum_j log q_j,

with finite or one-sided brackets and sigma>0. In the original joint-speed fit, a+b*x is replaced by c+a_v*log(v)+b*log(beta). It is a different estimand from the single-speed fit. Both assign positive probability to discordant intervals and trade their likelihood costs against all other observations. A tolerance epsilon is not sigma, a confidence level, or a likelihood-ratio cutoff.

Therefore a minimum hard obstruction is neither a minimum likelihood-rejection subset nor a minimum deletion set. Removing one obstruction need not remove another. Adding or removing observations need not move a fitted slope or a likelihood ratio monotonically. This distinction is substantive, rather than merely a numerical implementation issue.

## 2. An exact counterexample: strong hard conflict with zero likelihood ratio

Choose centered log-position x=(-2,-1,1,2), any slope p0, z=(1,-1,-1,1), and equal-width intervals

    [l_j,u_j] = [p0*x_j+z_j-w, p0*x_j+z_j+w],  0<w<1.

The shifted fixed-p0 intervals are centered at +1 or -1. They are incompatible whenever epsilon<1-w; for w=.05 their required hard tolerance is .95. Two columns already witness this conflict.

Nevertheless, the free-slope interval-Gaussian MLE admits b=p0 as a global optimum. To see this, write d=b-p0. For fixed sigma, each negative log interval probability is convex in its predicted mean: translated Gaussian interval mass is log-concave (a standard consequence of preservation of log-concavity by integration; see Prékopa 1973 in LITERATURE.md). Thus the summed negative log likelihood is convex in (a,d). Mirrored positions have equal z and equal widths, so the objective is even in d. Convexity and symmetry imply NLL(a,0,sigma)<=NLL(a,d,sigma) for every a,d,sigma. Optimizing a and sigma does not change this inequality.

Consequently the optimized fixed-p0 and free-slope likelihoods are equal, and the likelihood-ratio statistic for b=p0 is exactly zero, even though the hard geometric obstruction is strong. No implication from hard incompatibility to MLE rejection is valid in general. The construction applies to any p0 by shifting the interval centers; x can be translated to a physically smaller beta range without changing the result after adjusting the intercept.

To match the original design, cross those four beta positions with three speeds and add c0+a0*log(v) to each interval center. The same symmetry applies independently at every speed, so setting b=p0 remains optimal for every c,a_v,sigma. The three-speed design has full rank three. `synthetic_bridge.py` verifies this 12-row construction for p0=-2 and -1.2, plus the four single-speed examples; its exact claim follows from symmetry and convexity rather than the optimizer output alone.

This elementary counterexample is an explanatory result, not a claim to have discovered a new general phenomenon of model misspecification or influence.

## 3. A valid but limited likelihood bound

For fixed p, transform the intervals to J_j=[l_j-p*x_j,u_j-p*x_j]. If J_i and J_j are disjoint, their masses under the SAME N(a,sigma^2) distribution sum to at most one. Therefore

    q_i*q_j <= 1/4,
    -sum_all log(q_j) >= log(4).

This bound holds uniformly in a and sigma. It also holds for the original intervals when their epsilon-expanded versions are disjoint. More generally k mutually disjoint intervals give product<=k^(-k), and NLL>=k*log(k), by the arithmetic-geometric mean inequality. With independent disjoint pairs of column indices their log(4) contributions can be added.

This is a bound on absolute fixed-law likelihood loss, not on the likelihood ratio against a fitted alternative. The alternative also incurs interval-width and residual losses; Section 2 shows the likelihood ratio can be zero. For individual pairs a tighter numerical bound can profile their two-interval product, but it still does not supply a valid significance claim or a minimal rejection certificate. Do not describe log(4) as a p-value.

The argument assumes the compared columns share the same residual location after the known law has been subtracted. It cannot mix independently free folder intercepts. Within one speed folder the nuisance speed term is a common shift, so the argument also applies inside the original joint-speed likelihood. Invalid first-run boundaries have no such probability until a censoring policy has been specified; an invalidity witness alone supplies no likelihood bound.

### A sharper fixed-scale bound and uniform-domain certificate

For an ordered disjoint pair J_i=[a_i,b_i] and J_j=[a_j,b_j] with gap g=a_j-b_i>0, enlarge them to opposing half-lines. Their masses satisfy

    q_i <= Phi((b_i-a)/sigma),
    q_j <= Phi((a-a_j)/sigma).

Log Phi is concave. Averaging its two arguments therefore gives

    -log(q_i)-log(q_j) >= -2*log Phi(-g/(2*sigma)).

The bound is sharp for opposing half-lines with their common mean at the gap midpoint. It is conservative for bounded intervals. As sigma tends to infinity it tends to log(4), recovering the scale-free bound above.

Suppose a numerical-only obstruction cover supplies a same-folder separating pair on every rectangle r in the declared waveform-rule domain, and let g_star be the minimum ORIGINAL, unexpanded gap over that finite cover. If all gaps are positive, then every rule in the domain and every externally allowed sigma<=s_max obey

    fixed-p full-data NLL >= -2*log Phi(-g_star/(2*s_max)).

The pair may change between rectangles; it suffices that one valid pair is available at every rule. If the recorded gap uses epsilon-expanded intercept intervals, add 2*epsilon to recover the original gap. Coverage relying on an invalid boundary in any rectangle does not establish this bound there. The inequality does not turn an optimizer's scale cap into a calibrated physical noise bound, nor does it supply a likelihood-ratio lower bound. It is a compact, uniformly valid minimum likelihood burden under an explicitly restricted scale.

## 4. A useful bridge: retain the whole analysis and audit selected deletions

The defensible empirical question is: do the previously selected geometric witness columns also substantially influence the originally fitted law? Freeze their IDs before computing the new deletion effects. Refit the original likelihood after deleting them, rather than fitting the tiny witness in isolation. Report the fitted beta exponent, speed exponent when applicable, sigma, fixed-p profiled NLL, free NLL, and numerical diagnostics.

For single-speed fits, use one intercept, one beta slope and sigma. Do not include a redundant log-speed regressor. For original three-speed fits use c+a_v*log(v)+b*log(beta), preserve all three speed folders, and state whether deletion removes only the witness's speed-specific column or the entire corresponding beta block across speeds. These are different perturbations. Main analyses should preserve the original one-sided intervals; originally finite-only fits can be a separately named sensitivity analysis, not a silent replacement of the population.

Compare each witness deletion against all size-matched subsets when feasible, otherwise a fixed seeded reference sample with explicit sampling error. The percentile is a finite-dataset rank, not a statistical p-value: the witness was selected from the same data and the subsets are not random treatment assignments. Report unconditioned controls and position-aware controls separately if their design was fixed before outcomes. Since extreme beta positions often both constrain interval feasibility and provide slope leverage, an unmatched rank cannot isolate special value beyond leverage.

Failing to cross a nominal likelihood-ratio cutoff does not establish that the underlying law is true. A Wilks chi-square reference, if shown, is an explicitly approximate model-based diagnostic: small samples, censoring, fitted scales on a bound, within-session dependence and selection can invalidate its nominal calibration. Do not compare a selected tiny-subset p-value to the old full-data confidence interval.

## 5. Exact decomposition of deletion effects

Let theta_1 and theta_0 denote the optimized full-data free-slope and fixed-p parameters; let ell_j(theta) be an individual log interval mass. Write

    T_full = 2 * sum_all [ell_j(theta_1)-ell_j(theta_0)].

For a deleted set S, define G_m as the increase in remaining-data log likelihood obtained by refitting model m instead of keeping theta_m. Each G_m>=0 at exact optima. Then the following identity is exact:

    T_after = T_full
              - 2 * sum_{j in S}[ell_j(theta_1)-ell_j(theta_0)]
              + 2 * (G_1-G_0).

The first change is the selected columns' direct full-fit likelihood contrast; the last change is the differential response of the remaining data to refitting. This identity explains why deletion is not monotone and why a column's residual, geometric role, or isolated likelihood contribution cannot replace a full refit. Individual likelihood contrasts can be negative. Their fractions are not probabilities or causal attribution. Store both terms if possible and verify reconstruction of T_after within numerical tolerance.

This is algebraic likelihood bookkeeping, not a novel influence theorem. Its purpose is to explain the domain-specific bridge without disguising a refit effect as direct evidence from the selected columns.

## 6. Why fitting only two or three witness columns is unsafe

Two distinct x values allow a free intercept and slope to pass through both interval interiors. Letting sigma tend to zero then sends each interval probability to one. The likelihood supremum is one but need not be attained at any positive sigma; a numerical optimizer often reports its imposed lower scale bound as a fit. Three columns may still admit a line through all intervals and have the same degeneracy. Residual degrees of freedom alone do not protect an interval model from this problem.

For a tiny fixed-p subset, disjoint intervals can yield a finite fitted scale, but selection for incompatibility makes its apparent significance circular. Neither case is evidence that those columns are erroneous. They identify a review priority, not permission to clean the data until a favored law is accepted.

## 7. A convex formulation provides a separate optimization check

The likelihood is not generally convex in (theta,log sigma), but it has an established convex parameterization. Let tau=1/sigma>0 and eta=tau*theta. Each standardized interval is

    [tau*l_j - X_j*eta, tau*u_j - X_j*eta].

The set of (z,eta,tau) satisfying these affine inequalities is convex. The product of its indicator and the standard normal density in z is log-concave; integrating out z preserves log-concavity. Thus each interval probability is log-concave in (eta,tau), and total negative log likelihood is convex. One-sided intervals are included by omitting the corresponding affine inequality. This is a known censored-regression construction, documented by Burridge (1981) and Ekvall & Bottai (2023), not new methodology here.

The fixed-slope constraint theta_beta=p becomes the linear equality eta_beta=p*tau. Equivalently subtract p*log(beta) from both endpoints and retain only the intercept/speed design. Bounds L<=log sigma<=U become the convex precision interval exp(-U)<=tau<=exp(-L). Both free and fixed-slope models can therefore be solved with a convex objective over a convex feasible set.

At an exact interior stationary point, convexity proves global optimality. More generally the correct first-order boundary conditions suffice. The smooth invertible change of variables also means an exact interior stationary point in the old parameterization maps to a global optimum; separate local minima are not the theoretical concern here. Existence, flat directions, scale-bound optima and floating-point convergence still require checks. A small reported numerical gradient on an unbounded eta domain is not by itself a rigorous finite objective-gap certificate, and convexity does not establish uniqueness without further strictness/identifiability assumptions. A separately implemented precision-parameter fit supplies a useful independent numerical cross-check without replacing frozen results.

## 8. Verification and admissible conclusions

Numerical integration or a second stable tail implementation should check interval probabilities and gradients. Compare multiple starts or an independent optimizer on baseline fits, all selected witness deletions, and extrema from the deletion controls. Include all failures and scale-bound fits in denominators. Enumerating all subsets certifies combinatorial coverage; it does not by itself certify every nonlinear optimization is globally exact.

Synthetic checks should include Section 2's hard-conflict/LR-zero example; a free-scale collapse example; both directions of deletion-induced LR change; and the decomposition identity in Section 5. The production method need not use an influence approximation. Independent direct refits matter because approximations can fail for grouped deletions even in OLS (see LITERATURE.md).

`synthetic_bridge.py` executes those checks without importing empirical or production code. It also checks four interval probabilities against numerical quadrature, 437 scale-free pair-bound inequalities, and the rank deficiency of same-folder two/three-row original designs. All passed in `synthetic_results.json`; the largest deletion-decomposition residual was approximately 8.9e-15. A rank-three, three-observation cross-speed example exactly interpolates all interval centers and demonstrates that full design rank alone does not prevent free-scale collapse.

The narrow contribution can be a verified connection or demonstrated separation between rule-uniform geometric witnesses and the existing likelihood analysis, with precise perturbation units, exhaustive small-deletion comparisons, and explicit preservation of censoring. If the geometric witnesses are not unusually influential, that negative result is important: it limits the method's interpretation and prevents conflating a worst-column tolerance test with evidence against a physical law. Neither outcome establishes classifier truth, a new minimum-force law, independence of sessions, or worldwide methodological priority.
