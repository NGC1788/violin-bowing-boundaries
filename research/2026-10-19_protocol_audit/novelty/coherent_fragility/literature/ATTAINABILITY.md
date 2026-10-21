# Record-envelope attainability implementation

2026-10-21. This structural extension was specified before reviewing this phase's empirical results. It supplements [the derivation](RECORD_ENVELOPE.md), which treats finite thresholds. The implementation also handles literal +∞ to match the production profile's domain convention.

`attainability.py` has no dependency on the production sweep and no empirical I/O. The API is:

```
first_run_threshold_intervals(values, guard, fixed, k, relation, domain)
no_run_threshold_interval(values, guard, fixed, k, relation, domain)
contains(interval, threshold)
```

The classifier is `fixed OR (guard AND comparison(values, threshold))`, with disjoint fixed/guard boolean arrays. Relation `ge` uses `>=`; relation `lt` uses `<`. Active scores must be finite. Nonactive positions may have NaN and remain fixed. Domain `[lo,hi]` is closed, nonnegative and may include literal +∞. An infinite observed score is rejected. A −∞ threshold is outside this API's scope.

For `ge`, always-H positions receive effective score +∞ and always-zero positions −∞. The K-window activation is its minimum effective score, q_s. First start s occurs exactly on `(max_{r<s} q_r, q_s]`. For `lt`, always-H uses −∞ and always-zero +∞; q_s is the maximum effective score and the corresponding state is `(q_s, min_{r<s} q_r]`. Both are intersected with the caller's closed domain. Empty sets are omitted, all endpoint closures are retained and fixed positions do not change at t=+∞. These sentinels are correct over the declared nonnegative extended-real domain.

The no-run interval is `(max_s q_s,+∞]` for `ge` and `[0,min_s q_s]` for `lt`, intersected with the domain. If there are fewer than K observations every threshold has no run. Start zero is returned explicitly, allowing the consumer to distinguish left censoring from no run. Whether a positive start also has finite positive, nonzero-width force endpoints remains a separate check using the force grid.

A positive start not in the returned list is unreachable under this one-parameter rule. Even a returned state can conflict with other columns through disjoint threshold intervals. Thus the finite-grid relaxation and the coherent-rule question have distinct obstruction mechanisms.

The API uses a simple O(nK) sliding-window calculation and prefix extrema, independently of the event-update production solver. It is intended as an exact structural audit/oracle, not a novel asymptotic improvement or a new general record-statistics algorithm. A separate verifier must compare it with direct label evaluation before empirical diagnostics are trusted.
