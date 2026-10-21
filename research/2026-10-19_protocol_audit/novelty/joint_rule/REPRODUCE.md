# Reproduce the joint-rule artifact without raw recordings

The runner reads this folder, makes a compact copy in a **new or empty folder
outside the artifact**, and executes all scientific steps only in that copy.
It never invokes the exporter, downloads files, or overwrites the original
inputs, profiles, or results. The copied fixture includes all 78,000 windows;
raw waveforms are not needed for this particular method reproduction.

Use a Python environment with NumPy. Matplotlib is needed only for `--figures`.
No GPU, network, package installation, or pickle loading is required.

```bash
python reproduce.py --output-dir /tmp/joint-rule-reproduction
```

To regenerate the publication figures and compare their scientific selections:

```bash
python reproduce.py --output-dir /tmp/joint-rule-reproduction-with-figures --figures
```

Omitting `--output-dir` creates a unique temporary folder. The directory cannot
be the source artifact, one of its descendants, an existing nonempty directory,
an ordinary file, or a symlink. Existing contents are never deleted. Supply
`--python /path/to/python` if a different interpreter is needed. The default
timeout is 600 seconds **per subprocess** and is configurable through
`--timeout-per-step`.

## What is recomputed

1. Independent exhaustive synthetic joint-rule tests and input validation.
2. The separate rectangle/first-run analytical oracle tests.
3. Fixture identity, baseline labels, eligibility, and old one-axis slice audit.
4. Every local joint threshold profile and the complete global search.
5. Independent replay of every local profile and its obstruction certificates.
6. Independent global optimum and minimum-tolerance verification.
7. Every selected witness against previously certified arbitrary-edit lower
   bounds, using the same eligible columns and direction-compatible changes.
8. Continuous tolerance cutoffs, including their literal floating-point
   endpoints, and the explanatory obstruction search with independent checks.
9. Optionally, all `fig_*.png` figures and `figure_data.json` or
   `figure_selection.json`, when present in the source artifact.

Every source verifier must be on the runner's explicit inventory. A newly added
verifier causes the runner to stop until its execution and result comparison
are explicitly accounted for.

The source profile directory must contain paired numeric result archives and
metadata JSONs. The runner compares **all** of these files, `global_atoms.npz`,
`results.json`, analytical-oracle output, and every verification output. All
NumPy arrays are opened with `allow_pickle=False`; object arrays are forbidden.
Array keys, shapes, and dtypes must match. Integer and Boolean data match
exactly; finite float results use absolute and relative tolerances of `1e-12`;
matching NaNs in numerical profiles are allowed. JSON structures and all
scientific decisions are compared recursively.

Only top-level `seconds`, `elapsed_seconds`, `duration_seconds`, and
`source_sha256` fields are excluded from scientific comparison, because runtime
and copied-path provenance can differ. Their differing values are listed in
the report. No nested scientific field, threshold selection, count, hypothesis
decision, witness, or budget result is skipped.

The runner does not assert PNG pixel identity: plot rendering can differ by
font/backend version. It checks that each expected image was freshly generated,
records its hash, and compares its scientific data selection. The original
figures require separate visual inspection.

## Output and integrity

The output contains:

- `work/`: copied code and inputs, followed by freshly generated results.
- `reference/`: immutable-at-start snapshots of the source numeric results.
- `logs/`: complete output from each subprocess.
- `reproduction_report.json`: step status, runtime versions, all comparisons,
  snapshot/output/log hashes, and original-input integrity checks.

The report verifies that all snapshotted original files are unchanged after
execution and that both copied compressed fixtures and the arbitrary-edit
reference remain byte-identical.
A scientific mismatch, changed source, missing result, failed subprocess, or
timeout makes the final status `failed` and returns a nonzero process exit.

The original-data exporter is deliberately not copied or called. Baseline and
slice consistency are checked against `prior_input_features.json.gz`, included
beside the joint fixture, so the audit does not need access to another project
directory. The compact arbitrary-edit reference is also copied; its earlier
certification is not re-derived here. This reproduces the computations on the frozen feature fixture;
it does not independently validate the original waveforms or establish an
independent physical replication.

The artifact's `reproduction/` directory preserves the final run report, logs,
and output-path safety checks. Consult the report for the actual compared-file
and array-element counts rather than interpreting computation counts as a
scientific sample size.

## Recorded run

The completed copy-only run on 2026-10-21 passed all 12 stages in 162.21 seconds:
92 compared files (52 JSON, 40 NPZ), 1,063 arrays containing 16,248,845 elements,
and 197,575 numeric JSON values. It regenerated all three figures. Original
files and all three copied fixtures were unchanged. Full evidence is in
`reproduction/reproduction_report.json` and its 12 subprocess logs.

During active development, two earlier attempts stopped at the verifier
inventory guard when additional verification modules appeared. They executed
no scientific subprocess; `reproduction/early_inventory_stops.json` retains
those setup stops. The completed run includes every final verifier.
