# Fresh-workspace reproduction audit

Completed: 2026-10-13 08:58 UTC. Result: **PASS for the audited numerical pipeline**.
The core benchmark and chained two-speed comparisons reproduced without any
pre-existing reports, checkpoints, raw downloads, private handoff files or
original-workspace working directory. No production code changes were needed.

## Isolation and limits

A new temporary workspace was created under ignored
`work/reproduction_audit_20261013_r2u3gdc2/`. Only `scripts/`, `tests/`,
`research/` and requirements files were copied. Copying excluded `__pycache__`,
`.pyc`, raw/report/private directories and `HANDOFF*`. No `data/` directory or
virtual environment was copied, and `reports/` did not exist when the run began.

Every command used the original checkout's existing `.venv-mac/bin/python`,
but with its working directory set to the temporary workspace and `PYTHONPATH`
set to that workspace's `scripts/`. Numerical-library thread counts were set
to one. This tests filesystem/input portability and deterministic reconstruction
in the recorded Mac environment. It does **not** test installing dependencies
on another machine, a different architecture or different package versions.

The workspace is a file snapshot taken during final development, not a newly
installed Git clone. The public compressed input tables were copied as part of
`research/inputs/`. Historical public summary JSON files were present for
comparison, but no historical fitted model or private report was available.
The comparison commands received newly generated predecessor run paths through
their actual CLI options.

## Executed commands

The table lists the arguments passed to the interpreter. Each process returned
exit status zero. Timings include process startup; they are not performance
claims about the methods.

| Step | Command arguments | Seconds |
|---|---|---:|
| Public source inputs | `scripts/benchmark_data.py --include-ambiguous --out reports/portable_inputs.json` | .47 |
| Full A-source benchmark | `scripts/benchmark_sets.py --source A --out reports/basic_A` | 10.42 |
| Exact checkpoint reconstruction | `scripts/restore_frozen_models.py --verify-rebuild` | 1.79 |
| Two-speed source scaling | `scripts/multispeed_scaling.py --output reports/newms` | 11.27 |
| Affine flow | `scripts/affine_transport.py --output reports/newaf --cello-baselines reports/newms` | 22.01 |
| Established asymmetric comparator | `scripts/twospeed_known_prior.py --output reports/newprior --flow-results reports/newaf/results.json` | 1.03 |
| Matched width ablation | `scripts/affine_width_ablation.py --output reports/newwidth --full-run reports/newaf` | 7.51 |
| Snapshot test suite | `-m pytest -q` | 31.47 |

Total sequential process time was 85.95 seconds. Setup/copying and the final
comparison audit are outside that total. The chained CLI options worked with
arbitrary new run-directory names; the pipeline did not fall back to old reports.

The public source-input loader recovered 24,000 measured rows, selected 23,301
eligible rows, retained 211 ambiguous eligible rows, and selected 1,750 A/.1
training rows. Source checksums were verified by the loader. The resulting
input manifest fingerprint was

`c2a25e04defd60c76113b5da7cda2c39f025793b822d7c7cb9eb280e1ff6bfef`.

## Scientific-result comparisons

Comparisons ignored wall-clock times, output paths and creation timestamps.
All listed metrics and fitted flow/exponent parameters agreed with the tracked
historical summaries to absolute tolerance 1e-12. Discrete values agreed
exactly. The audit checked **390 numeric values** across these groups:

| Comparison with tracked result | Numeric values | Outcome |
|---|---:|---|
| Eight A-source selected-model family-min macro IoUs | 8 | PASS |
| Multispeed aggregate scores and selected exponents | 48 | PASS |
| Affine aggregate scores, selected warps and all 18 task scores | 194 | PASS |
| Known-prior means, all six task comparisons and paired outcome | 45 | PASS |
| Width-ablation aggregates, all 18 tasks and multistart audit | 95 | PASS |

Tracked references were `benchmark_cross_source_summary.json`,
`multispeed_scaling_summary.json`, `affine_transport_summary.json`,
`twospeed_known_prior_summary.json`, and `affine_width_ablation_summary.json`.
No performance values were retuned or rewritten to obtain agreement.

Representative reproduced results include cello affine versus matched k=0
family-min macro IoU .6947942448043762 versus .6935022384440498, and published
D-string affine versus known asymmetric IoU .8612645058677045 versus
.8445121481511775. These reproduce the existing mixed evidence; successful
reproduction does not convert it into a general superiority result.

## Frozen models and tests

All **14** missing model files were reconstructed from the public source inputs
and frozen settings. Every rebuilt file's SHA-256 matched the original frozen
checkpoint bytes exactly. A separate audit read each newly created file and
rechecked its hash against the portable manifest. The portable manifest's
internal fingerprint was

`6191519763c552ef57f22062933714b0ad71403c159e479171d210c659544913`.

The reconstruction command verified the frozen implementation hashes and
training membership before fitting. It did not fit or select models using target
labels, perform a new hyperparameter search, or modify the frozen manifest.

The copied snapshot's suite reported **259 passed, 2 skipped, 102 subtests
passed** in 30.73 seconds of pytest-reported time. The two unavailable optional
conditions were the real 7-Zip executable on `PATH` and the deliberately absent
private backed-up source artifacts. Portable-source tests ran; NumPy, PyArrow,
Torch, Numba and plotting dependencies were available. The skip conditions are
explicit in `test_prepare_first.py` and `test_benchmark_data.py`.

## Preserved evidence and work not covered

Ignored local evidence is retained in the temporary workspace:

- `audit_setup.json`: isolation setup and interpreter provenance.
- `audit_execution.json`: argument lists, return codes, timings and log paths.
- `audit_comparisons.json`: all comparison outcomes and exact checkpoint checks.
- `audit_logs/`: complete stdout/stderr for each process.
- Newly generated `reports/`: complete fresh-run results, parameters and logs.

This audit did not download/decode A2, run its supplemental evaluator, reinstall
the environment, re-extract the copyrighted source PDF, or repeat the separate
high-resolution physics grids. Those are distinct reproducibility scopes.
The source A benchmark, public inputs, fourteen-checkpoint restoration and
the requested multispeed → affine → known-prior / width-ablation CLI chain
had no hidden ignored-file dependency in this fresh-workspace test.
