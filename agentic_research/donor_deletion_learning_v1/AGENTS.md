# Donor-deletion learning preparation

- AI learning/forecasting is primary. This unit implements/tests/reviews, then binds only approved original-train inputs; actual fitting belongs to the next separately frozen execution unit.
- Before independent source approval do not read or hash actual NPZ, row/person arrays or weights. Use synthetic arrays only. Never open/hash old dev/test truth; no raw reprocessing/downloads/standalone numerical audits.
- Preserve all frozen older files/results. Four objectives share prefix encoder/generator/common aux/init/minibatches/memory/prior. Same-person memory excluded. Held truth never reaches training/prediction payloads.
- All12 full-memory and donor-counterfactual outputs must be parent-pinned before held event scoring. Worst held donor is a secondary post-seal evaluation only.
- Independent code/contract review is delegated to an existing reviewer; preparation cannot bypass their unresolved blockers.
- Exact support deficits, invalid rows, failures, compute/storage costs and historical-reference limitations must be retained. No extra fitting or batch rescue.
- Public code/docs/synthetic/aggregates/hashes only. Keep all actual memory/data/predictions/weights/traces/logs private in local/.
