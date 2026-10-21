# Independent verification — coherent threshold decisions

2026-10-21. This audit was developed separately from `threshold_profile.py` and
`empirical.py`. `oracle.py` imports neither: it reapplies each rule to every
window and finds the first K consecutive positives with literal Python loops.
No raw waveform, archived download, original label file or frozen score changed.

## Completed checks

| Check | Scope | Result |
|---|---:|---|
| Production sweep vs literal synthetic oracle | 828 fixtures; 2,748 atoms; 8,362 endpoint/interior probes; 34,620 slope-profile comparisons; 43,236 minima checks | All match |
| Input rejection and regression checks | 30 malformed inputs; 12 tied-score, singleton/infinite-domain, zero-eligible and fixed-NaN cases | Pass |
| Independent record-extrema formula | 6,552 status/rule/K fixtures; 58,968 first-run comparisons | All match |
| Actual `attainability.py` closed-form API | 14,560 fixtures; 94,024 endpoint/interior comparisons, including empty inputs, K>n and literal +infinity | All match |
| Feature fixture vs prior input | 78,000 unique windows; 1,560 columns; 39 folders; force, beta, IDs and baseline labels | Exact match |
| Nominal feature rules | All 78,000 windows for each of four features | Zero baseline mismatches |
| Complete empirical profiles | 182,057 global patterns; 191,987 literal local tables; 157,115,191 saved scalar values | All match |
| Empirical summary optima | 48 global, 288 budget and 1,872 folder optima | All match |
| Direct final-threshold replay | 48 full-data witnesses (3,744,000 window evaluations), plus 51 feasible folder witnesses | All match |

These are computational comparison counts, **not independent experiments,
participants, data sets or confirmations of physical truth**. Floating-point
log endpoints match within `rtol=atol=1e-13`; integer counts, Boolean decisions,
threshold events and interval closures match exactly.

The complete real-data oracle independently reconstructs the global threshold
arrangement from all guarded scores. It recomputes every column at its own
local thresholds (at most 51), then maps global thresholds into those local
tables. It does not reuse the production dirty-column update algorithm.

## What the empirical audit establishes

Under the fixed K=2, epsilon=.2 operational predicate, the narrow ratio rule
can restore the beta^-2 shape for one folder (`B_T2r1/v2`). Its smallest eligible
edit cost is 44, compared with the unconstrained arbitrary-edit lower bound 5.
Applying that same scalar rule everywhere changes 88 labels in that folder and
3,200 across all data, while nine other folders lose at least one finite
eligible boundary. The common periodicity rule restores no folders. Amplitude
and height stress tests each restore one beta^-1.2 folder but invalidate 23 and
22 other folders, respectively. Broad periodicity/ratio ranges do not add
restorable folders at this epsilon.

A post hoc diagnostic follows from the fully checked profiles: if a common
rule must preserve finite boundaries in **all** originally eligible columns,
none of the six examined ranges restores either shape in any folder at
epsilon=.2. This is an explicitly additional condition, not the original
primary criterion.

## Material qualifications

1. Every result concerns an already exposed data set and the declared
   first-run/all-column compatibility predicate. It does not certify the old
   MLE confidence interval, classifier correctness or a universal force law.
2. Exact impossibility is only within the specified one-feature threshold
   family/domain and fixed originally eligible columns. Combined features,
   other classifiers or a censoring-aware decision are different questions.
3. The all-column predicate is deliberately demanding: both slopes begin
   incompatible in all 39 folders at epsilon=.2. Failure to restore therefore
   cannot establish which physical law is correct.
4. No finite-boundary disappearance is counted as compatibility. Ineligible
   original columns are still classified and charged in the global cost.
5. The chosen threshold has been selected using these data. A single common
   rule is more constrained than arbitrary label edits, but is not an externally
   validated or calibrated classifier improvement.
6. The repeated saved profile entries are highly dependent. Their large count
   is evidence of implementation coverage, not scientific sample size.

## Files and reruns

Run the Python scripts in this folder. `exhaustive.py`, `validation_cases.py`,
`fixture_audit.py`, `record_envelope.py`, `attainability_api.py` and
`real_profiles.py` write correspondingly named JSON/log evidence. The fixture
audit uses `../prior_input_columns.json.gz` when present, otherwise the prior
study's fixture. `real_profiles.py` requires the six saved profile NPZ files,
`results.json`, `input_features.json.gz` and `arbitrary_reference.json`.

All checked inputs and implementations are SHA-256 pinned in the resulting
JSON files. No material discrepancy remained at completion.
