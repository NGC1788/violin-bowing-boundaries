# Existing MLE and the minimum obstruction: audit before bridge inference

2026-10-21. Read-only source and saved-result audit. No new empirical subset fitting or inference was performed for this note. Old code, data, frozen scores and pipelines were unchanged. `baseline_reference.json` extracts already saved fits and `source_sha256.json` identifies the inspected sources.

## Exact existing estimand

`novelty/lower_law/lower_law.py:fit` calls `final/compare.py:censored_fit` with design `[1, log(v), log(beta)]` and the lower-force intervals. It estimates **one diagram's** conditional latent log-boundary mean

    log F_min = c + a log(v) + b log(beta) + residual,
    residual ~ Normal(0, sigma^2).

The fit is joint over three speed folders (approximately .05, .10, .20 m/s), usually 40 beta columns per speed and hence 120 likelihood contributions. Speed and recording block are confounded. This is not a fit of 6,000 independent stroke responses, a universal law or a calibrated physical error model. There is no force-uncertainty epsilon in this likelihood. The new geometric epsilon must not be silently inserted into it.

Measured/window-force studies use force of the 0.2 s window starting at t=.2 s. Onset-force studies reuse those labels but replace the force by independently cached force-at-onset and **sort/re-extract** boundaries; they are different axes/estimands. The current obstruction fixture and certificates concern only window force. Onset-force results cannot inherit these certificates unchanged.

For a contribution with endpoints (lo,hi), in natural log-force units, the likelihood is `Phi((hi-mu)/sigma)-Phi((lo-mu)/sigma)`. First K=2 H-run begins at sorted force index s. If s=0 it is left-censored `(-inf,log max(F[0],.001)]`; if s>0 it is finite `(log max(F[s-1],.001),log max(F[s],.001)]`; if no run exists it is right-censored `(log max(F[last],.001),inf)`. The lower-boundary fit retains no-run columns. All force values, beta values, sorting, labels and censoring must be held fixed for an unchanged-baseline bridge.

## Computational implementations and existing numerical findings

- Legacy API: `novelty/lower_law/lower_law.py:fit(rows)` returns `(a,b)`; its dependency `final/compare.py:censored_fit(rows,design,'lower')` returns `[c,a,b,log_sigma]`. It subtracts ordinary normal CDFs, clips each probability to at least 1e-300, and uses unbounded Nelder-Mead from a fixed start. It discards optimizer success/gradient diagnostics. Do not use this as the only numerical verifier.
- Stable audit API: `verify_lower_law/statistics.py:fit(rows,init=None)` returns params, NLL, success, maximum analytic gradient, sigma, fallback, n. `arrays(rows)`, `objective(p,X,lo,hi)`, and `logprob(a,b)` are reusable. The stable probability uses log-normal-CDF, survival symmetry in the positive tail and `expm1` differences. The design is centered/scaled; initial optimization is L-BFGS-B with `log_sigma` in [-9,3.5], then unbounded Nelder-Mead on diagnostic failure. Thus **the fallback does not enforce the stated initial scale bounds**. Report actual scale/interiority and convergence rather than claiming a general bounded global optimizer.
- `arrays` asserts strict `lo<hi`; `fit` requires at least six rows and a rank-three design. Zero-width clamp artifacts are not valid normal interval observations. D_T2 was excluded from the current 13-diagram fixture for this reason. Do not silently widen intervals or treat them as point density measurements.
- Saved stable full-data baselines are `novelty/criterion_identifiability/response/results.json: sessions[*].baseline`. There are 13 diagrams, each n=120. Their beta coefficients differ from corresponding legacy values by at most **1.3067503e-5**. The baseline numerical agreement was already established.
- Four previously identified numerical failures were in *changed-label tilt variants*, not the unchanged baselines: D1 periodicity a=0; D1 ratio a=-1; C_T2 periodicity a=0; B_T2r1 periodicity a=0. `response/audit.json` preserves both results: legacy stable-gradient maxima 2.04–7.83 and stable NLL improvements .121–9.081. Reusing old changed-label results without stability checks is unsafe.

A standalone unchanged baseline can be reconstructed from `novelty/joint_rule/input_joint.json.gz`: every column carries session, speed, beta, sorted force and baseline labels. Use **all 1,560 columns**, not just the 1,132 marked eligible; group by session for the original three-speed model. Keep raw nonpositive forces and apply the original .001 clamp at extraction. No original feature caches are needed. Exact per-column identity is `id`; use IDs to remove or select a certificate, never a beta-nearest join.

## Inference that originally existed

`lower_law.py` fitted the point exponents and applied descriptive H_S boxes: a in [.9,1.5], b in [-2.4,-1.5]. Its four then-new diagrams used 500 percentile bootstrap replicates resampling beta-column indices paired across speeds (seed 61). The onset follow-up used 200 replicates; coefficients were summarized across diagrams with DerSimonian-Laird-type between-diagram variance and HKSJ intervals. `verify_lower_law/statistics.py` additionally tried blocks of 1,5,10 adjacent beta columns paired across speeds, and equal type×tension weighting. These are conditional post-hoc uncertainty assessments with remaining scan/session dependence.

**There is no single original per-speed p=-2 versus p=-1.2 MLE rejection test corresponding to the obstruction predicate.** A new fixed-slope likelihood ratio, its calibration, or a subset-deletion influence statistic must be declared as a new post-hoc bridge. It does not retroactively change frozen physical-model prediction scores or old bootstrap CIs.

## Exact mismatch with the obstruction certificates

The existing geometric target fixes a speed folder, a candidate beta slope p, and epsilon; it asks for an intercept common to **all originally finite-positive eligible columns**, under every common waveform rule in the declared domain. Censored/no-run/nonpositive columns are excluded at baseline (315 left-censored +113 no-run across the fixture). Any selected original eligible boundary becoming invalid defeats geometric compatibility. Gaussian MLE instead retains one-sided likelihood contributions, allows nonzero residual scatter, and maximizes a sum across all speeds.

A minimum obstruction contains two or three columns from the **same speed folder**. Its exact minimality means no smaller fixed subset obstructs the whole common-rule domain. It does not mean:

- it is a minimum subset causing an MLE slope rejection;
- deleting it restores the full-data likelihood conclusion;
- its members are uniquely influential, wrong labels, physically anomalous or independent;
- every conflict pair contributes a large share of total NLL;
- an MLE failure can be certified by its geometric epsilon alone.

A same-speed subset has rank at most two in `[1,log(v),log(beta)]`; it cannot identify the speed exponent. To fit it in isolation, profile its effective intercept `c+a log(v)` and beta slope, explicitly recognizing the changed design. With only 2–3 columns, free scatter can collapse or be poorly identified, so subset-only LR asymptotics are particularly unreliable. To preserve the old estimand, use the full 120-column session fit and a clearly marked deletion/case-weight diagnostic for exact certificate IDs.

## Most defensible bridge and interpretation

1. Reproduce full120 baselines unchanged against saved stable parameters and likelihoods.
2. Fit candidate p while profiling all remaining parameters and compare with the unconstrained full120 MLE, using the same observations/censoring. Any LR reference distribution is new and approximate, not inherited from the geometric certificate.
3. Evaluate exact certificate-case deletion or case weights while retaining other speed folders; compare with same-size **same-folder eligible-column** subsets to avoid claiming influence from selecting convenient extremes. Report coefficient shift, fitted scatter and loss/objective changes together. Deletion of a selected conflict is an explanatory perturbation, not new independent evidence.
4. If translating a geometric conflict into a likelihood bound, keep the residual scale assumption explicit. A pair's positive separation constrains simultaneous interval probability at fixed sigma, but free sigma can absorb some incompatibility; summing maxima independently does not reproduce the shared-intercept MLE. A valid computable bound can be useful even when it does not imply a significant LR.
5. Include counterexamples or observed failures where a geometric obstruction has little MLE impact. Such a result is scientifically meaningful: it separates worst-column feasibility from likelihood-weighted model adequacy instead of conflating them.

The likely contribution is a transparent bridge between a certificate's local geometric explanation and the original model's aggregate inferential behavior. Generic interval-censored Gaussian MLE, case-deletion influence, likelihood-ratio testing and minimum inconsistent sets are established concepts; novelty must lie in the verified waveform→K-run→boundary→likelihood connection and its concrete diagnostic result.
