# Reviewed source-only execution contract

- AI forecasting is primary. Preserve frozen learning code and old completed campaigns unchanged. No extra standalone numerical audit.
- Implement/tests/review before reading or hashing actual NPZ. After code approval, freeze allowed original-train inputs and row identities in this same preparation unit; no actual fitting until the subsequent run unit.
- Only original train34people/1442rows. Never open/hash old dev/test truth. Inputs come from the sealed path-source preparation and sealed source HGB predictions.
- All feature/label transforms and learned priors fit innertrain only. Held future labels never enter job payloads or stopping. Seal all12predictions before held-label scoring.
- Fourmethods×threefixedpersonfolds; no new seeds or settings. Same initial shared parameters, optimizer, person weights and minibatch sequence. Invalid15rows stay.
- HGB forecasts are reused, not refit; disclose historical60dim/clipped8 representation versus neural73derivedfeatures. Three analytic Markov count fits are fixed reference work within total worker budget;12scheduled jobs are exactly12neural fits, not18hidden model runs.
- Atomically register batch/job starts under canonical budget lock; failed/time-out attempts remain consumed. Parent watchdog terminates child jobs within allocated wall time; actual execution includes launch, loading, training and persistence. First failed job closes the whole batch without promotion. No rescue/rerun/refreeze.
- All row outputs, weights, traces and tokens stay local/private. Public artifacts are code, aggregates and hashes. No main merge, raw deletion, payments or external messages.
