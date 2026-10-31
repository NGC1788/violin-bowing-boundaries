# Projection audit v2 — final bounded numerical check

This is a standard active-set Newton implementation for classical forward Bernoulli-KL projection. It is an auxiliary numerical audit, not a novel forecasting method. No observed targets are loaded or scored.

The numerical protocol is unchanged: same 64 ID-hash selected records and the same four previously frozen prediction arrays; full-simplex Frank–Wolfe gap at most `1e-8`; feasibility error at most `1e-10`; no positive tolerance for the worst of the ten admissible targets' clipped log-loss differences. The fixed selection prefix remains `projection_audit_v1:` to preserve the same records. There is one actual audit with a 600-second deadline. Stop after it regardless of the outcome. No third projection audit or expanded model search is authorized by this protocol.

The solver uses the actual interior marginals, analytic gradient and Hessian, an affine-eliminated Newton step on the positive-weight support, and a vertex step to admit inactive coordinates. Armijo backtracking maintains feasibility; a numerical-equality step is allowed only if the full-simplex gap improves. Every final quantity is recomputed from the actual candidate. A successful optimization status alone is insufficient.

Already coherent inputs with nonnegative recovered mass and reconstruction error at most `1e-12` are returned bitwise unchanged. An identity input outside `[1e-6,1-1e-6]` receives no fabricated FW bound; it is recorded separately as an unchanged coherent input. For interior identity cases, the gap is recomputed at the actual represented point `w@bank`, and its reconstruction residual from the unchanged q is recorded; the gap is never hardcoded to zero. This matters at exact 0/1 vertices. All other candidates require both the original gap and score checks before replacing the input. The returned fallback may be incoherent, but its score is unchanged.

Synthetic validation, from repository root:

```bash
.venv-mac/bin/python -m pytest agentic_research/projection_audit_v2 -q
```

After independent preflight approval only:

```bash
.venv-mac/bin/python agentic_research/projection_audit_v2/run_audit.py freeze
.venv-mac/bin/python agentic_research/projection_audit_v2/run_audit.py run
```

`freeze_manifest.json` locks all v2 Python files, protocol, the imported v1 solver, the existing repair implementation and its dependencies, and the four prediction archives against the prior artifact snapshot. No v1/v3 files are changed. The runner checks all frozen hashes before and after the audit. Exclusive creation of `run_receipt.json` prevents a second actual audit, even after failure. A process lock prevents concurrent runs. Failures preserve the completed rows, current record ID and exception. All code and data are under the same local user, so hashes detect changes rather than providing a security boundary.

Interpret only optimization adequacy on the chosen predictions. This is not an observed-label performance benchmark or a fair batch-throughput comparison. Smaller KL objective alone neither proves better forecasting nor establishes global convergence. The authorized single audit is complete: 239/256 passed the gap check, 11 were outside-clip coherent identities without a gap certificate, and six optimizations failed. Returned forecasts were non-worsening on all ten possible targets, but only 237/256 were coherent. See RESULTS.md. This branch is closed; do not run a third projection audit.
