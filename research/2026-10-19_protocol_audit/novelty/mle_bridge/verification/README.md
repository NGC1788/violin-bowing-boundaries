# Independent verification of the MLE bridge

This folder is owned by the independent verifier. `oracle.py` does not import
the production fitter or earlier project likelihood code. It computes Gaussian
interval probabilities with separate positive, negative, and zero-crossing
branches, and rejects zero-width or reversed intervals rather than clipping
their probability to a small number. `synthetic.py` contains no empirical data
reads.

Run from this folder using the project Python environment:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python synthetic.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python independent_refits.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python compare_results.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python control_audit.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python family_audit.py
```

Dependencies: NumPy, SciPy, mpmath. The high-precision reference uses 100-decimal
`mpmath.erfc` / `erf`, independently of SciPy's normal log-CDF implementation.
Analytic score formulas are checked against a five-point objective stencil.
An intercept/scale fit is compared with a dense 8,181-point grid; this is a
numerical check, not a general global-optimality certificate. The test also
checks invalid intervals, rank deficiency, all-one-sided censoring, no
informative intervals, and scale-bound flags.

## Interpretation safeguards

- A fixed-epsilon geometric contradiction does **not** imply zero Gaussian
  likelihood. With positive dispersion, every nonempty interval has positive
  mass. `synthetic.json` includes an explicit two-interval counterexample.
- A two- or three-column obstruction selected within one speed does not have
  full rank for `[intercept, log speed, log beta]`. It must not be fitted alone
  and presented as a significance test of that original three-parameter law.
- Optimizer success does not establish identification. One-sided censoring
  can let an intercept escape; interval-perfect fits can make scale collapse;
  rank deficiency and scale-bound hits require separate reports.
- The legacy stable implementation's Nelder–Mead fallback has no scale bounds,
  even though its initial L-BFGS-B fit uses log-scale bounds `[-9, 3.5]`.
  A new bounded audit must not silently overwrite old frozen numerical scores.
- Selected-set deletion can be reported as a deterministic sensitivity
  diagnostic. It does not provide an unselected p-value, population uncertainty,
  or a causal explanation of a physical law.

These safeguards do not reject mixed censoring or finite-interval MLE in
general. They state where a successful optimization is insufficient evidence.

## Empirical validation completed

The fixed empirical plan was committed as `aed035e`. All verifier inputs are
inside this artifact folder; no raw data or project feature cache is required.

- `independent_refits.py` reconstructs all 1,560 intervals using a literal first
  consecutive-two detection. It fits 13 full baselines with both fixed slopes,
  and both models after all 77 selected deletions: 193 fits.
- The independent fitter uses the canonical variables `eta=theta/sigma` and
  `tau=1/sigma`, bounded positive tau, two starts derived from the intervals,
  and numerical optimization derivatives. It never uses production parameter
  values as starts. This differs from the production fitter's coefficient/log-
  scale parameterization and analytic score.
- `compare_results.py` checks the production rows exactly, adds 154 independent
  three-speed-block fits and 78 independent coherent-rule fits, and checks all
  77 likelihood-change decompositions. All 425 fit comparisons agree within
  7.12e-7 in parameters and 3.36e-11 in negative log likelihood. The coherent
  replay also checks label counts, changed intervals, censoring counts, and
  original eligible boundary losses.
- `control_audit.py` verifies exhaustive enumeration of 22,110 subsets in 41
  jobs; independently recalculates all 60,234 objectives; independently refits
  the first, middle and last subset for every model in every job (354 fits);
  and replays 462 descriptive rank summaries. Maximum objective disagreement
  is 2.77e-11; sampled refits agree within 1.30e-6 in parameters and 4.61e-11
  in negative log likelihood. These are computational checks, not statistical
  sample sizes.
- `family_audit.py` joins all 7,742 previously certified minimum obstructions
  to the stored control fits by exact IDs, verifies 23,226 member ranks, all 77
  chosen maximizers, and six aggregate groups. It accepts the previously
  verified geometric lists; it does not claim a new proof of their geometry.

The enumeration and ID membership checks are discrete and exact. The ranks
are computed exactly from the stored numerical MLE results, with the declared
1e-9 tie tolerance. MLE solutions remain numerical approximations; neither
small stationarity residuals nor independent agreement provide a rigorous
finite objective-gap certificate. Canonical convexity explains why exact
stationarity would be sufficient, as documented with prior art in `../theory/`.
