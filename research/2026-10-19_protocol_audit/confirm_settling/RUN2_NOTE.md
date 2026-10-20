# Run 1 crashed; run 2 = bug fix only (2026-10-20 ~10:25)

Run 1 (`score_run1_crashed.log`, claim kept as `SCORE_CLAIM_run1_crashed.json`) parsed the labels and then raised
KeyError('WARM') before ANY level or difference was computed or printed: the folder-to-session mapping tested `"cold" in
folder`, which is true for both folders because the archive directory is named "…_coldwarm". The only change for run 2 is
the mapping (leaf directory ends with `_cold` / `_warm`); protocol, thresholds, references, bootstrap and decision rules
are unchanged (`score_settling_run1.py` = run-1 code). New hash receipt FREEZE_RECEIPT_run2.json. Run 2 is the only run
that produced numbers.
