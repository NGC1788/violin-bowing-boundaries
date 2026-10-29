# 2026-10-30 local research and data audit

Scope: read-only inspection of the existing work, inputs, process state and compute environment. This file is the only output changed by this audit. No training, archive extraction, download, pipeline restart or GitHub action was performed.

## 1. Current state

- `~/violin-bowing-boundaries` is clean at `027fd6e` on the existing research branch. Latest scientific work is the October 21 classification-choice/geometry/MLE study. October 24 added final Korean R&E documents, the LaTeX report and Claude PPT handoff, not new experiments.
- `~/violin-rne/COLLAB_LOG.md` is the best current status source. Its latest note is the October 24 completion; older claims in its historical section are superseded.
- `pgrep` found no active `replicate.py`, `recover_B1.py`, `archive_companion.py`, `parallel_download.py` or `aria2c` at inspection. This is a point-in-time process observation, not permission to modify those pipelines. A stale `replicate.instance.lock` file remains; a file's existence is not proof that a process is running.
- `timeres/replicate_state.json` has **17 archives, all `done`**, including the late cold/warm archive. Its last processing timestamps are October 20.
- Existing frozen confirmations are spent. **No untouched confirmation dataset is established by the current records.** Unread individual waveform files from an already-used specimen do not make that specimen a new independent lockbox.
- No `AGENTS.md` was found in either repository or their checked `/`, `/Users`, `/Users/jeongjihoon` ancestors.

The existing contribution is already computational: exact minimum label edits, joint-rule enumeration, minimum fixed obstruction sets and the connection to censored MLE. It is not an AI model-training contribution. Simply adding an LLM research controller would automate the work without making the scientific method itself an AI contribution.

## 2. Machine that is usable now

| Item | Observed value |
|---|---|
| CPU / memory | Apple M2 / 16 GiB |
| Free local disk | approximately 115 GiB at inspection |
| Python | `~/violin-bowing-boundaries/.venv-mac/bin/python` |
| Numerical packages | NumPy 2.4.6, SciPy 1.17.1 |
| ML packages | scikit-learn 1.9.1, PyTorch 2.14.1 |
| Accelerator | `torch.backends.mps.is_available() == True`; CUDA unavailable |
| Data / tests | pyarrow 26.0.0, joblib 1.6.0, pytest 9.1.1 |
| Not installed in this environment | pandas, xgboost, lightgbm |

No installation is required for NumPy/PyArrow ingestion, logistic/HistGradientBoosting baselines, or a small PyTorch network. Limit concurrent fits on this laptop. Loading all time-window CSV tables concurrently is wasteful; ingest one archive at a time, select fixed prefix/target windows, then concatenate compact float32 arrays.

## 3. Small, fully portable inputs

All paths in this section are relative to `~/violin-rne/` and also have counterparts under the Git snapshot `~/violin-bowing-boundaries/research/2026-10-19_protocol_audit/`.

### Joint-rule fixture

`novelty/joint_rule/input_joint.json.gz` (2,290,913 bytes):

- SHA-256 `85ab696054670f80bc9690873291c39bb119b56670b4513da59bae994a2b0d70`.
- 1,560 column dictionaries; each has 50 force-ordered observations. Total 78,000 observations, 13 diagrams, 39 speed folders.
- Fields: `id`, `group`, `session`, `folder`, `column`, `trial`, `speed`, `beta`, `force`, `baseline`, `eligible`, `guard`, `fixed`, `periodicity`, `ratio`.
- `session` values: A1, A2, B1, C1, D1, A3, A4, D2, A_T2, B_T2s1, C_T2, B_T2r1, B_T2r2. Each has 120 columns. D_T2 is excluded by the original clamp/zero-width protocol issue.
- `group = session|folder` is one speed-specific recording block. `id` adds beta-column identity. The arrays are sorted by measured force, not acquisition time. `trial` must not be treated as a harmless predictive feature.
- No waveform samples. Guard/fixed/baseline contain observer logic, not independent physical annotations.

### Four-feature fixture

`novelty/joint_rule/prior_input_features.json.gz` (3,742,311 bytes):

- SHA-256 `e5650e0a51402382d2545702885c536c44ae0efd66a698dbb0cc8103e0d686d4`.
- Same observations, forces and force order.
- Four features: `periodicity`, `second_flyback_ratio`, `amplitude`, `flyback_height`. Each has `values`, `guard`, `fixed` lists. Missing nonfinite features are serialized as `null`.
- Suitable for reproducing/classifying rule response and developing structured sensitivity methods. **Predicting `baseline` from these same observer features is rule imitation, not recognition against independent physical truth.** A deterministic rule is then a compulsory near-perfect baseline.

### Minimal geometry fixture

`novelty/run_fragility/input_columns.json.gz` (927,669 bytes):

- SHA-256 `df926dbacbec45a01a50aa71e6f4172b34d68d9346a30f291655cbfa9de03167`.
- Force, beta, speed, labels and trial IDs; no waveform features.
- Best kept as the unchanged reference oracle for downstream scientific-decision evaluation.

## 4. Richer data immediately available for a predictive ML task

### Time-resolved features: the best first pilot input

Directory: `~/violin-rne/timeres/features/`.

There are **17 compressed CSVs, 532.64 MiB combined**. Schema:

```text
folder, trial, beta, window_s, t, plateau_s,
force, speed, amplitude_std, periodicity, f0_hz,
flybacks_per_period, second_flyback_ratio, flyback_height, error
```

`t` is relative to the supplied constant-speed plateau landmark. `force` and `speed` are averages within the associated window. Each stroke has many overlapping rows, so row counts are not independent experiment counts.

This audit read the relevant columns of **all 17 tables**. Every table had no recorded errors and complete row counts for all six fixed windows below:

- input `[0,.20]`, `[0,.10]`, `[.05,.15]`, `[.10,.20]`;
- future target `[.25,.45]` and sensitivity target `[.30,.50]`.

Counts per requested window:

| Archive category | Strokes per requested window |
|---|---:|
| 12 ordinary archives (A1, B1, C1, D1, A2, A3, A4, D2, A_T2, B_T2s1, C_T2, D_T2) | 6,000 each = 72,000 |
| A1 cold/warm | 4,000 |
| B_T2 sample2 (July 3; four speeds) | 6,400 |
| Part 6 setup repetition (two repetitions) | 12,000 |
| HighBowForceRange_1 | 9,000 |
| HighBowForceRange_2 | 9,000 |
| Total | **112,400** |

The original 13-diagram set is the 11 ordinary archives excluding D_T2 plus the two Part 6 repetitions: **78,000**. The extra tables are available, but they are also exposed and have incompatible designs (different speed counts and force counts). They require explicit manifests rather than silently forcing a 3-speed/40×50 layout.

Existing code to reuse as a specification, not to overwrite:

- `astra_20261018/forecast/benchmark.py`: creates nonoverlapping future-window prediction inputs and source-fitted baselines.
- `astra_20261018/forecast/PROTOCOL.md`: exact temporal scope and leakage cautions.
- `timeres/audit_20261017/window_audit.py`: `load_features`, duplicate/missing-key checks, beta-grid recovery and fixed-label logic.
- `astra_20261018/forecast/results/`: existing exposed A1↔A2 benchmark; must not call a re-run new independent confirmation.

**Important observer-version distinction:** old future-forecast labels use periodicity/f0/amplitude/flyback count via `window_audit.vector_labels`; later joint-rule fixtures use the feature/guard rule exported through `novelty/tilt_principle` and `novelty/label_sensitivity`. They are not automatically interchangeable. Pin the observer implementation and its constants in the new harness, verify a reconstruction fixture, and report which one is predicted. Do not silently relabel to favor the new model.

### Compact multivariate sequences

`astra_20261018/archive_companion/*/blocks_*.npz` provides 13 archive files, generally about 14 MiB each. Twelve ordinary archives have `data.shape == (6000, 10, 160)`; July 3 B_T2 sample2 has `(6400, 10, 160)`. Separate `claude_extra/blocks/blocks_Part6.npz` also exists, plus A1/A2 copies.

Fields:

```text
speed, position, force_dc, force_ac, force_ac_lo,
bridge_dc, nut_dc, bridge_p2p, bridge_edge, coh
```

`block_s == .05`, arrays float32, at most 160 blocks (8 seconds), missing tail padded by NaNs. `folder`, `trial`, `beta` are per-stroke metadata. All 13 companion files checked had finite beta metadata.

These sequences are **aligned to original acquisition start, not to the plateau landmark**. Position is integrated speed. Force/bridge/nut channels describe mechanically measured signals, not microphone audio. Training a Transformer or small temporal network is feasible on these arrays, but several leakage fixes are essential:

- use only blocks ending before the prediction origin;
- do not supply the full padding mask, total duration, full-trace position maximum or future speed maximum as prefix features;
- estimate landmarks using a fixed externally supplied protocol or validate a causal online detector;
- normalize folder names before joining: some NPZs store `v1` while CSVs store full paths; A1 stores dated basename folders; Part 6 has repetitions with repeated `v1` names and needs the full path;
- join by archive + unambiguous folder + trial and assert one-to-one completeness.

### Preserved waveforms / archives

- `~/violin-bowing-boundaries/data/cache/2024-04-19_TypeA_sample2/`: 6,000 `window_*.npy`, each example shape `(19999,4)`, float32; total cache about 2 GiB. These are legacy acquisition windows, not necessarily the fixed prefix/future windows.
- `claude_extra/wf_20261018/skeptic_force/raw/raw_v{1,2,3}.npz`: A2 subsets include `att_b` (bridge attack samples), `att_f`, harmonic summaries and onset indices. These are processed caches with their own timing/selection semantics.
- `data/raw/zenodo/17749111/2024-04-10_TypeB_sample1.7z`: retained B1 archive, approximately 14,168 MiB.
- `astra_20261018/archive_companion/pins/2024-03-25_TypeA_sample1_coldwarm.7z`: retained cold/warm archive, approximately 8,362 MiB.

Do not assume all 78,000 raw waveforms remain locally available. Foundation audio-model experiments would need deliberate extraction/domain analysis; these are bridge/nut force signals and a 50 kHz acquisition, not ordinary audio recordings. A microphone-audio foundation model is not automatically a suitable representation.

## 5. Ground truth, splitting and selection limitations

1. There are no independent physical H labels or listening-quality labels in the reviewed fixtures. Observer fidelity, future observer prediction and real physical correctness are different endpoints.
2. Every current diagram and frozen confirmation has been used. A new fixed split can protect future *model fitting* from leakage but cannot erase historical researcher exposure. Label it retrospective development/evaluation.
3. Never split individual overlapping windows randomly. Group by stroke at minimum; use whole diagram or known physical-specimen families for the outer split. A1/A2 are different specimens, whereas A1 cold/warm and repeated setup blocks can share the same physical string. Preserve this nesting and avoid claiming all archive labels independent.
4. Speed is confounded with recording block. Leave-speed-out is a distribution shift stress test, not a causal speed effect.
5. Feature preprocessing, hyperparameter selection, calibration, early stopping, thresholding and class weights must use source training/validation groups only. Evaluation labels must never enter candidate mutation prompts.
6. Future `force`, future waveform features, `plateau_s`, onset metadata computed from full traces, trial/order IDs, and target session IDs are prohibited prefix inputs. Controlled nominal speed/beta availability must be stated.
7. All-window imputation can leak future support. Use masks known at the origin only; report missing features instead of selecting clean target cases.
8. Statistical units are scarce recording/specimen groups. Report paired group-level differences and each held-out group, not tiny observation-level confidence intervals over tens of thousands of correlated windows.
9. Endlessly optimizing on a fixed validation set overfits it. Record the number of attempted hypotheses, use nested development folds, and reserve a separately locked later external evaluation before strong generalization claims.

## 6. Concrete pilot interface recommended to root

A low-risk computational pilot can be built immediately from fixed-window CSV rows, without downloads:

```python
# one record per stroke; all X rows end at supplied origin .20 s
record = {
  'stroke_id': (archive, full_folder, trial),
  'diagram_group': diagram_name,
  'specimen_family': explicitly_audited_mapping,
  'x_prefix': float32[4, 16],  # existing transforms + missing flags
  'x_controls': float32[3],  # force_asinh, log_beta, log_speed from past only
  'y_future_proxy': bool,     # pinned rule on [.25,.45] only
  'prefix_end_s': .20,
  'target_start_s': .25,
  'target_end_s': .45,
  'observer_version': pinned_hash,
}
```

Start with the original 13-diagram / 78,000-stroke manifest or an explicitly smaller development pilot, not all designs merged blindly. Baselines: source-calibrated persistence/Markov, controls-only HistGradientBoosting, snapshot HGB, history HGB, regularized logistic regression. A proposed learned sequence/uncertainty/structured model must beat the strong snapshot and history baselines with equal information and budget, not only a weak physics curve.

Primary metric: future-proxy log loss, Brier as secondary, plus entry/exit subgroups, worst-group loss, calibration, and runtime/memory. Existing exact boundary machinery can measure downstream **proxy decision stability**, not independently verify truth. A candidate that predicts all-H/all-non-H must fail coverage/discrimination guards. Log loss and abstention coverage need to be scored jointly so abstention cannot hide errors.

The scientifically meaningful upgrade is a learning objective or architecture for prediction/decision robustness under observer uncertainty and grouped domain shift, validated by ablations and additional data. It is **not** established merely by this audit, a new neural baseline, or the agentic loop. Start the loop with explicit falsifiable gates and preserve the October report as the completed prior study.
