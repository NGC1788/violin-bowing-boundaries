# Original MLE bridge: production artifact

Post-hoc descriptive experiment fixed in `../PLAN.md` before deletion scores. All old source files and frozen outcomes remain unchanged. The input fixture is a byte-identical copy of the existing joint-rule fixture, containing 13 diagrams, 39 speed folders, 1,560 columns and 78,000 original labels. `cases.json` freezes the 77 previously selected exact minimum obstruction sets; no set was selected for MLE influence. `reference_baselines.json` contains the prior stable full120 fits, and `prior_obstructions.json` preserves the complete old minimum-set family for the separately declared family extension.

## Reproduce without original archives or feature caches

Use Python with NumPy and SciPy in a new or empty directory outside this artifact:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python reproduce.py --out /tmp/violin-mle-bridge-production-reproduction
```

The wrapper copies code and compact inputs, reconstructs original censoring, fits all baselines and selected deletions, enumerates every control with at most two workers, recomputes exact finite-population ranks, compares outputs and verifies source hashes are unchanged. It never removes a preexisting directory. Re-running into a nonempty output directory is rejected. It takes approximately half a minute on the development laptop. The top-level artifact may supply additional independent oracle and theory checks.

Direct computation in a disposable copy consists of `python run.py --prepare`, `python run.py --workers 2`, `python summarize.py`. Existing control JSON files trigger resume; use a fresh disposable copy for full reproduction.

## Model and numerical checks

`model.py` exposes `extract_rows(columns, labels=None)`, `fit(rows, fixed_beta=None, init=None)`, `fit_arrays(X,lo,hi,p=None,init=None)`, `row_nll(params,rows)` and `logmass(a,b)`. Parameters are `[intercept, speed exponent, beta exponent, log sigma]`. The original force-order K=2 interval extraction and 1 mN clamp are preserved. Every full diagram includes all 120 columns, with finite, left-censored and no-run/right-censored observations. No geometric epsilon enters the likelihood.

Log-CDF/survival differences are computed stably with analytic derivatives. The design is standardized. Every optimization stage enforces log sigma in [-9,3.5]: initial L-BFGS-B, bounded Nelder-Mead followed by bounded L-BFGS-B, then bounded SLSQP for residual convergence failures. A fit is accepted only with optimizer success, finite NLL, maximum standardized gradient <=.002 and no sigma-bound hit. Fixed-slope NLL must be >= free NLL within 1e-7. Every full-data baseline matches its prior saved fit (maximum parameter difference 1.11e-7).

The initial benchmark encountered one precision-level L-BFGS abnormal termination (gradient 3.59e-7); the bounded SLSQP fallback was added before full enumeration. Final benchmark and all 60,234 control fits passed. Of these,59,924 used the first optimization stage, 214 the second, 96 the third; maximum accepted gradient .000314. These are numerical optimizers with diagnostics and independent verification, **not interval-arithmetic certificates of objective error or exact analytic MLE solutions**. The deletion population is exhaustively enumerated; that combinatorial completeness must not be conflated with mathematically exact floating-point optimization.

## Stored outputs

- `rows.json`: original intervals, ID, diagram/session, speed folder/group, original beta-column index, eligibility, log speed and log beta. Sorted by exact ID. Use IDs, never a nearest-beta merge.
- `baselines.json`: free and both fixed-slope fits for 13 diagrams; all parameters, NLL, success, gradient, sigma/bound flag, iteration/attempt counts, censor counts, ordered IDs and reference differences.
- `selected.json`: 77 cases with target slope, immutable selected IDs, original geometry feasibility, original/after-deletion fits, beta/speed effects, movement toward target, T before/after, mechanism decomposition, same-beta-index across-three-speed block deletion, and all/span-matched ranks. T=2(NLL_fixed−NLL_free) is descriptive; no chi-square cutoff, calibrated p-value or formal rejection is claimed.
- `controls/00..40.json`: 41 unique folder×deletion-cardinality jobs, eligible row indices, target slopes, control count, fit count, duration and every unresolved failure (none).
- `controls/00..40.npz`: `combinations` indexes **that job's session-filtered rows.json in its original order**. `log_beta_span` accompanies every subset. For `free` and each required slope (`-2.0`/`-1.2`), keys are `_params`, `_nll`, `_grad_max`, `_success`, `_scale_hit`, `_iterations`, `_attempts`. There are 22,110 distinct subsets and 60,234 fits. A free fit shared by the two slope cases is computed once.
- `summary.json`: all 77, classifier-specific 52, geometry-infeasible 25, and per-slope summaries. Units are overlapping diagram×folder×law comparisons, not 77 independent experiments.

Ranks use the complete same-folder eligible-column deletion population, including the selected set itself. The reported midrank is `(strictly below + .5*tied)/N`; absolute differences <=1e-9 count as ties. The matched population retains log-beta spans in [.8,1.2] times the selected span. Ranks are descriptive finite-population positions, **not randomization p-values**. Exact certificate membership and exact enumerated index equality are independently auditable despite the numerical MLE tolerance.

## Result and mechanism

For the 52 classifier-specific cases, selected minimal obstructions changed the original fitted beta exponent by a median absolute .00613; 17 moved toward the candidate slope and 35 away. The median absolute-influence rank was .542, falling to .347 after span matching. Only 2/52 were at or above the 95th percentile of absolute beta influence. Thus small exact geometric explanations are generally not sets dominating the original likelihood. The family-extension analysis addresses dependence on the frozen lexicographic representative separately.

The identity checked for each selected deletion is

    T_full - T_remaining
      = 2*(removed NLL_fixed_at_full - removed NLL_free_at_full)
        + 2*(fixed refitting gain - free refitting gain).

Maximum numerical identity error was 2.69e-11. In B_T2r2/v1 at target −1.2, deleting the fixed geometric certificate moves beta −1.816→−1.955 and raises T 54.55→98.79. Its removed contrast contributes −25.24 and differential refitting −18.99. This is a useful counterexample to interpreting geometric obstruction membership as evidence against the fixed-slope likelihood. It does not prove any physical law correct or establish independent classifier accuracy.
