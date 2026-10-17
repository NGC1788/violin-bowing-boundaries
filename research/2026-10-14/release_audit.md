# Independent release audit — 14 October 2026

Initial review scope: the newly added conditional transport experiment,
shared-observer comparison, and source-boundary identification audit. That
review preceded the completed width-law certificate. No A2
waveforms, results or evaluation files were opened, and no frozen file was
edited. This is a software and reporting review, not independent physical
validation or confirmation of methodological novelty. The integration addendum
below records later checks separately, including completed A2 work and
publication metadata cleanup.

## Result

No new P1/P2 numerical defect was found in the three reviewed analyses. The
negative conditional-transport result is reported consistently and is not
promoted into a generally improved method. The observer comparison distinguishes
shared-rule dominance from differences of independently minimized scores. The
source geometry distinguishes exact bands, minimum-disagreement alternatives,
censoring and unidentified empty alternatives.

The four targeted test files completed with **42 tests passed in 22.81 seconds**:

```sh
bash scripts/run_mac.sh python -m pytest \
  tests/test_conditional_transport.py \
  tests/test_conditional_transport_report.py \
  tests/test_observer_dominance_audit.py \
  tests/test_source_identifiability_audit.py -q
```

The report test includes replay in a new filesystem tree containing only public
scripts, derived inputs, published-cell data and the conditional experiment's
gzip bundle. It reuses the installed interpreter and packages; it is not a
fresh dependency-install test. The existing independent conditional audit
recomputes all 18 source selections and all numerical results. Its successful
comparison count must not be interpreted as experimental sample size.

## Artifact checks

- The public observer table contains 5,596 unique task/scope/model-pair records
  and is byte-identical to the archived CSV. All six count summaries were
  independently recomputed from that table, including grid reversals,
  continuous reversals, grid-missed reversals and the three effect margins.
- All five code hashes recorded in the observer protocol match the released
  source files.
- The public source-identification summary matches the archived audited
  summary. Its recorded script hash and all six complete-run file hashes match.
  The stated 7,244 column/rule cases are consistent with the artifact.
- Before publication-path cleanup, compressed artifact sizes were 769,646 bytes for the complete
  conditional experiment, 505,615 bytes for the complete observer pair table,
  and 31,689 bytes for the source-identification summary bundle.
- `HANDOFF_CODEX.md` remains untracked and unstaged. It is not a release input.

The earlier independent geometry/observer review additionally checked every
stored observer extremum witness: 22,384 IoUs across all 5,596 comparison rows,
with maximum discrepancy 2.22e-16. This release review verified the public
table's identity and summary arithmetic rather than counting that previous
review as a second independent rerun of the same witnesses.

## Documentation issues found for integration

1. `conditional_transport_assessment.md` links to the nonexistent
   `schoonderwaldt2008/audit.md`. The existing dataset overview is
   `schoonderwaldt2008/README.md`; the second visual inspection is
   `schoonderwaldt2008/SECOND_VISUAL_AUDIT.md`.
2. The observer reproduction instructions refer to `docs/mac_research.md` for
   the matched-width run, but that page initially omitted the
   `affine_width_ablation.py` command. Its benchmark examples also use
   `reports/reproduction_A` through `_D`, whereas the observer script expects
   `benchmark_A` through `_D` beneath one `--benchmark-base` directory.

The complete compatible development command chain is:

```sh
bash scripts/run_mac.sh benchmark --out reports/reproduction/benchmark_A --source A
bash scripts/run_mac.sh benchmark --out reports/reproduction/benchmark_B --source B
bash scripts/run_mac.sh benchmark --out reports/reproduction/benchmark_C --source C
bash scripts/run_mac.sh benchmark --out reports/reproduction/benchmark_D --source D
bash scripts/run_mac.sh python scripts/multispeed_scaling.py \
  --output reports/multispeed_scaling/reproduction
bash scripts/run_mac.sh python scripts/affine_transport.py \
  --cello-baselines reports/multispeed_scaling/reproduction \
  --output reports/affine_transport/reproduction
bash scripts/run_mac.sh python scripts/affine_width_ablation.py \
  --full-run reports/affine_transport/reproduction \
  --output reports/affine_width_ablation/reproduction
bash scripts/run_mac.sh python scripts/observer_dominance_audit.py \
  --benchmark-base reports/reproduction \
  --full-run reports/affine_transport/reproduction \
  --k0-run reports/affine_width_ablation/reproduction \
  --out reports/observer_dominance/reproduction
bash scripts/run_mac.sh python scripts/source_identifiability_audit.py \
  --flow-run reports/affine_transport/reproduction \
  --output reports/source_identifiability/reproduction
```

Choose new empty output paths if these already exist. The full fitting chain
above was inspected for path compatibility during this review, not rerun here;
the targeted public-bundle replay is the execution evidence for this release
review. Neither documentation issue changes any stored score.

## Integration and public reproduction — 15 October

The broken dataset link and incompatible reproduction command paths have been
corrected. The root README, Mac guide, research overview, manuscript and
paper-readiness assessment now report the completed negative A2 primary results.
Historical freezes and all fifteen bound implementation files are byte-identical
to commit `fce8d32`; neither original primary endpoint was changed.

The independent [A2 audit](lockbox_independent_audit.md) reverified the actual
whole archive and reconstructed all twenty-six predictions and metrics. A
separate [public replay](lockbox_A2/README.md) validates the complete payload
inventory, cross-file provenance, model/source identities, local recorded
chronology and contrast arithmetic. Tampered but rehashed payloads are covered
by rejection tests, not merely ordinary checksum tests.

The [width-law assessment](width_law_assessment.md) records all 186 nominal-control
cases and independent constructive-witness checks. A second scientific review
identified an overstrong causal interpretation of its small extra disagreement
count. The revised text makes clear that a small optimum in a relaxed source
problem does not upper-bound full-model error or held-out IoU degradation.
This is a limitation of inference, not a changed numerical result.

Publication cleanup replaced the host prefix in one published-source path in
the conditional experiment bundle and the source-identification summary.
Transformation notes and original byte hashes are retained. All forty other
conditional files are byte-identical to the original run; the remaining input
manifest values match exactly. Source-audit scientific values match the original
run, and all six archived file hashes were rechecked. Current gzip sizes are
769,843 / 505,615 / 36,162 bytes for conditional / observer-pair / source-summary
artifacts. The width bundle has its own path-transformation note and original
input-manifest hash. None of these publication edits changed predictions.

A new temporary filesystem tree contained only public candidate-release files,
with no copied `reports`, raw data or checkpoints. It reconstructed all fourteen
original checkpoints with exact frozen SHA256 values, replayed all twenty-six A2
models with 2,886 numerical values and zero discrepancy, and reproduced the
conditional and width independent-audit JSONs byte-for-byte. The installed
Python environment was reused. The width figure title was subsequently clarified
and its report replay rerun; numerical audit output was unchanged. Full captured
outputs are in [public_reproduction.json](public_reproduction.json).

The canonical suite at this stage passed **349 tests and 102 subtests** in
50.32 seconds with no skips. After the bounded-width sensitivity addition, the
October 17 final run passed **361 tests and 102 subtests** in 51.56 seconds.
The twelve added tests include uncompressed optimization and constructive
reclassification on sixty random small force-band problems. The bounded result
also replayed in a fresh public-only filesystem tree with every scientific value
identical, excluding its timestamp; see [the captured evidence](public_bound_reproduction.json)
and [the final validation log](../../docs/validation.md).
Software checks do not establish physical truth, general predictive superiority
or new-method novelty. `HANDOFF_CODEX.md` is still excluded from this release.
