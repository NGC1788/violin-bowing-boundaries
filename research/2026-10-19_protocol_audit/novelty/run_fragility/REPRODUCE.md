# Reproduce the compact fragility artifact

This package contains the already derived force-ordered labels in
`input_columns.json.gz`. Reproduction does not download or re-extract raw
archives, rerun the waveform classifier, or modify the original results.
It validates the operational calculation conditional on this frozen input.

## Environment and command

Python 3.11.15, NumPy 2.4.6 and Matplotlib 3.11.2 were used. NumPy is the
only nonstandard dependency for all numerical calculations and independent
verification. Matplotlib and its dependencies are needed only for figures.

```bash
python3 -m venv /tmp/run-fragility-env
/tmp/run-fragility-env/bin/python -m pip install -r requirements-reproduce.txt
/tmp/run-fragility-env/bin/python reproduce.py --figures
```

By default, `reproduce.py` creates a new temporary directory and prints its
location. To choose a directory, supply `--output-dir /path/to/new-folder`.
The destination must be new or empty, cannot be a symlink, and cannot lie
inside this source artifact. A nonempty directory is rejected; it is never
cleaned or overwritten. An alternate interpreter can be supplied with
`--python /path/to/python`.

## What is run and compared

The driver takes a small immutable snapshot of scripts, the compact input,
plans, and stored reference JSONs. It runs the following in a separate
`work/` directory containing **no pre-existing output files**:

1. Every independent verification script: exhaustive first-run costs,
   input validation, exhaustive global interval choices, global input
   validation, exhaustive end-to-end edits, and compact-input auditing.
2. The fixed synthetic examples, all empirical calculations, and the
   support/sensitivity diagnosis.
3. Independent application of saved empirical edit witnesses.
4. With `--figures`, the two figures and their deterministic selection.

All ten result JSONs are compared with the stored references; with figures,
`figure_selection.json` is also compared. Dictionaries must have the same
scientific keys, lists must have the same length and order, and decisions,
strings, booleans and integer counts must agree exactly. Floating-point
values use absolute and relative tolerances of 1e-12. This includes the
individual edit witnesses and eligibility records, not just final averages.

Runtime timings and explicitly listed top-level provenance fields are
excluded from scientific comparison and listed in the report. For example,
the empirical-result file hash changes when its duration changes. The
copied compact-input hash is still checked exactly, and all original
source/reference files are checked for changes at the end. PNG bytes are
not compared because font/rasterization environments can differ.

`input_audit.py --external-dependencies skip` audits all compact input
values and identities while skipping hashes of the original research
environment. Those external files are not needed to reproduce the
operational calculation. This does not claim to independently validate
the earlier raw-signal extraction or classifier labels.

The driver rejects unexpected new verification scripts until its explicit
execution order is updated. It never calls `export_inputs.py`.

## Outputs and failure behavior

- `reproduction_report.json`: environment versions, script exit codes,
  exact source hashes, comparison counts, ignored metadata and mismatches.
- `logs/`: complete stdout/stderr from every step.
- `reference/`: copied original result JSONs.
- `work/`: regenerated calculations and optional figures.

A failed subprocess, missing output, scientific mismatch or changed source
produces a nonzero exit code. The output directory remains available for
inspection. Each subprocess has a default 300-second timeout, adjustable
with `--timeout-per-step`.

## Local reproduction record

On 2026-10-21, a fresh virtual environment was created containing only
NumPy, Matplotlib and Matplotlib's required dependencies. The installed
distributions were copied from the recorded environment without network
access. SciPy and scikit-learn were absent.

The complete `--figures` run passed in **33.67 seconds**:

- All 11 execution steps returned zero.
- All 11 scientific result files matched: **95,803 numerical values** and
  **19,167 other values**, with all dictionary/list structures checked.
- All original source/reference files and the copied fixture were unchanged.
- Both figure PNGs were regenerated; figure-selection metadata matched.
- Differences were confined to runtimes, explicitly skipped external
  provenance, and the empirical-result file hash affected by its runtime.

The full local report and step logs are preserved at:

```text
~/violin-rne/reproduction_runs/run_fragility_20261021/reproduction_report.json
~/violin-rne/reproduction_runs/run_fragility_20261021/logs/
```

The report records all 11 installed dependency versions and copied source
hashes. This is evidence of reproduction from the compact fixture in a
minimal environment on the same computer; it is not an independently
collected dataset or an independent audit of the waveform classifier.

A small, portable receipt is included at
[`reproduction/reproduction_report.json`](reproduction/reproduction_report.json),
with all 11 subprocess logs in [`reproduction/logs/`](reproduction/logs/).
The larger reference/work copies are not duplicated in this receipt.
