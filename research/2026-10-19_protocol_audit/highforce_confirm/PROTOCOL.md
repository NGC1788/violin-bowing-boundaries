# High-force upper-boundary confirmation — dedicated parser/scorer protocol

Claude, 2026-10-19 (written after the control-only audit `controls_audit.json`, BEFORE any classifier column of
HighBowForceRange_1/2 was parsed). Governing frozen predictions: `novelty/thermal_upper/FROZEN_v2.json`
(created 10:31:51; sha256 bf9a14c9…) and its addendum `FROZEN_v3.json` (10:39:15; `hypotheses.high_force`).
Both archives started processing after 14:00 (replicate.log), so v2/v3 govern. Nothing in v2/v3 is changed here.
This protocol only adds the parsing and bookkeeping decisions that v2/v3 left open; every such addition is
marked **[added]**.

## 1. What the controls showed (no labels read)

- HighBowForceRange_1 = C_T2 sample2 (2024-06-22) + A_T2 sample2 (2024-06-24);
  HighBowForceRange_2 = D_T2 sample2 (2024-06-27) + B_T2 sample2 (2024-06-29). Receipts `validated`, 9,000 strokes
  each, output hashes match.
- Every folder: 1,500 strokes = **30 β columns × 50 strokes**, β 0.0199–0.0701 (log-spaced; max within-column
  spread ≤ 0.003 log, min between-column gap ≥ 0.040 log), measured speed 0.05 / 0.10 / 0.20 m/s (v1/v2/v3,
  verified from measured speed), plateau force ≈ 3.8–12.6 N, log-spaced (median step 0.023 ≈ ln 3 / 49).
- All four window times (0, 0.1, 0.2, 0.3 s) exist for all 1,500 strokes of every folder; no error windows.
- Trials are β-major (trials 1–50 = smallest β).

## 2. Targets [added: split by string]

Four separate targets, one per string session, each with its own published Z (POMA Table 1, as in FROZEN_v2
`z_published`): HF_C (C T2, Z 1.1154), HF_A (A T2, 1.1884), HF_D (D T2, 1.2021), HF_B (B T2, 1.1899).
No B T2 default for A/C/D. No merging with any sample1 (low-force) diagram; sample1 diagrams of the same string types
were used to fit the frozen models, so these targets are **unseen sessions and an unseen force range of seen string
types**, not unseen strings. HF_B: the friction law of the thermal models was fitted on a B T2 session (07-03) —
same string type, different session.

## 3. Parser [added; mirrors final/data.py + final/compare.py except for the column grouping]

- Windows with `window_s == 0.2` and t ∈ {0, 0.1, 0.2, 0.3}; label = `V.rm.classify(amplitude_std, periodicity,
  f0_hz, flybacks_per_period, 98.08, V.APPLIED) == "helmholtz"` (same wrapper as all earlier diagrams).
- Column = (full folder path, rank of β within folder // 50), with an assertion that each folder has exactly 1,500
  strokes, 30 columns of 50, and every between-column gap exceeds every within-column spread. Speed = folder median
  measured speed. lb = log median β of the column.
- Boundaries exactly as `compare.intervals` (K = 2 adjacent strokes in measured-force order at the same window
  time, log max(F, 1e-3)): lower from the first run, upper from the last run; H at the top force → upper
  right-censored (F_max ≥ top); no K-run in the column → **no-band** (upper NaN).
- Grid step for the secondary hit statistic: STEP_HF = ln(12/4)/49 = 0.02242 [added; the frozen scorer's ln 40/49
  belongs to the low-force grid].

## 4. Primary analyses (t = 0.2 s, K = 2), each run once

P1 — `hypotheses_high_force` (FROZEN_v2): per target, free censored fit log F_max = c + e_v log v + e_β log β over
the upper-boundary columns (no-band excluded = frozen `rows_for` behaviour). Column bootstrap, NB = 500, seed 2026,
as `score_target.free_fit`. Zone: H_thermal e_β ≥ −0.95; H_plain −1.05 < e_β < −0.95; H_steep e_β ≤ −1.05.
"supported" if the point estimate's zone is the only zone touched by the 95 % CI, else "not decidable".

P2 — `scoring_rule` (FROZEN_v2): per target and per frozen model in `fits` that exists in `thermal_law2.MODELS`:
level error = `offset_fit(mu, lo, hi)`, within the model's frozen `pi90.level`; held-out censored NLL per column
with the frozen σ; median |error| on finite columns (N reported); hit (± STEP_HF); design exponents over the
target's own columns vs the free-fit exponents. Thermal (T1k_break, T1mu, T1k_peak) vs S1_strict: smaller |level
error| wins only if the difference ≥ 0.03 log, else "not distinguishable". A code-identity check asserts that
`thermal_law2.predict` reproduces the frozen `high_force[group][model].Fmax_N` grid values for that group.

P3 — `FROZEN_v3.hypotheses.high_force`: speed curvature e(v2→v3) − e(v1→v2) with the ceiling lifted, pooled over
the four targets with target offsets and a common β slope (the `analysis2b` b1 design). Decision: sign of the point
estimate and whether the 95 % CI excludes 0. CI [added]: cluster bootstrap over the 4 targets (NB 200, seed 5,
as analysis2b) — primary; stratified column bootstrap within targets (NB 200) — reported alongside. Per-target
curvatures reported descriptively. Concave (< 0, CI excludes 0) removes the low-force ceiling explanation;
convex favours ceiling artefact + thermal contrast (v3 text).

## 5. Pre-declared sensitivity and descriptive outputs (reported whatever they show)

S1 [added] — no-band as left-censored F_max < lowest measured force of the column. This is an **assumption**
(observation_model/README: no-band cannot tell "above F_max" from "below F_min"); it is plausible here only if
the no-band columns sit at large β / low v. P1 and P3 are repeated under S1. If the P1 zone or the P3 sign/CI
decision changes, the result is reported as "censoring-assumption-dependent" and the primary decision is not
upgraded.

S2 — t = 0.1 and 0.3 s: P1 point estimates only, descriptive.

D1 — per target and speed: counts of finite / right-censored / no-band upper columns and of columns with ≥ 2
separate H runs (multi-band; not hidden).

D2 — lower boundary: counts only (finite / left-censored at 4 N / other). No lower-boundary hypothesis exists for
these targets; nothing is scored.

## 6. Integrity

`score_highforce.py` refuses to run unless: both receipts are `validated` with matching output hashes; FROZEN_v2/v3
sha256 match their `.sha256` files; this PROTOCOL.md and the scorer match the hashes in `FREEZE_RECEIPT.json`.
It creates `SCORE_CLAIM.json` with O_EXCL before reading any classifier column, and writes `score_result.json` with
O_EXCL (no overwrite). A failure after the claim is recorded as a failure; no second scoring run.
