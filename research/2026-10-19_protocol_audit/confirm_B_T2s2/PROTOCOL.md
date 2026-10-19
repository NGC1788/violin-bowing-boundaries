# B_T2 sample 2 (2024-07-03) — dedicated once-only scoring protocol

Claude, 2026-10-20 ~09:00 KST, written after the control-only audit (`controls_audit.json`) and BEFORE any
classifier column of this archive is parsed by this code. Background: the archive was processed 2026-10-19 23:03 via
the bounded stream path; both generic frozen scorers crashed on its design (predict_from_controls: "expected 3 speed
folders"; score_target: 2,000-stroke assert inside final/data.py) and wrote no output. score_target had parsed the
classifier columns in memory before its assert; nothing was printed or saved.

## Design (controls only)
4 speed folders `2024-07-03_Schelleng/v1..v4` with measured speeds 0.05, 0.10, 0.20 and **0.15** m/s; each folder =
1,600 strokes = **40 β columns (0.02–0.20, log-spaced) × 40 forces** (onset force ≈ 0.25–7.5 N, log step ≈ 0.075);
all four window times present for every stroke; no error windows.

## Exposure / contamination (disclosed)
FROZEN_v2's receipt records that a friction-reconstruction session of the same date and string type (030724, Helicore
T2) had been seen and that the thermal friction law was fitted on it; "4 same-string upper brackets and 5 lower bounds
visible in friction_offline.log [D]". The upper-boundary result here is therefore **partially contaminated** and is
reported with that label.

## 1. FROZEN_v2 upper laws (applicable) — primary
Governing file FROZEN_v2.json (10:31, sha256 bf9a14c9…, also in GitHub d6b4081); target group "B T2", Z = 1.1899.
- Parser [added]: windows `window_s == 0.2`, t ∈ {0, 0.1, 0.2, 0.3}; label = `V.rm.classify(..., 98.08, V.APPLIED) ==
  "helmholtz"` (same wrapper). Columns = (folder, rank of β // 40) with asserts: 1,600 strokes per folder, 40 columns of
  40, between-column gaps > within-column spreads. Speed = folder median measured speed. lb = log median β of column.
- Boundaries exactly as `final/compare.intervals` (K = 2, measured window force, log max(F, 1e-3)); t = 0.2 s.
- Primary = all four speeds (the scoring rule says "the target's upper columns"; the laws are continuous in v).
  Statistics as `novelty/thermal_upper/score_target.py`: free censored fit (NB 500, seed 2026), per model level error =
  offset_fit vs PI90, NLL per column with frozen σ, finite-column median error, hit (± STEP = 0.075 [added: this grid's
  log step; score_target's log 40/49 belongs to the 50-step grid]), design exponents; thermal vs S1_strict at 0.03.
- Secondary [added]: the same on the three standard speeds only (0.05/0.10/0.20). Reported, not decisive.

## 2. PACE* lower estimator (method_protocol) — NOT APPLICABLE, not scored
Its frozen definition assumes the 50-step nominal grid F_nom = 0.1·40^((rank−1)/49), "nominal ranks 1–20", three
folders of 2,000 and the 40×50 column geometry. This archive has a 40-step grid starting near 0.3 N and four speeds.
Any adaptation would be a new estimator chosen after seeing the design, so no PACE* claim is made for B_T2 sample 2.

## 3. FROZEN_v3 pooled hypotheses (confirm_v3) — loader addendum
confirm_v3 needs B_T2 sample 2 upper rows. They are produced here (t = 0.2, K = 2) for the three standard speeds only
(the v3 curvature is defined on 0.05/0.10/0.20; 0.15 m/s excluded [added]) and saved as `bt2s2_v3_rows.npz` (O_EXCL),
together with the score. confirm_v3 gets a dated addendum that only changes how B_T2 sample 2 rows are loaded.

## Integrity
`score_bt2s2.py --freeze` (label-free) hashes this file, the code, FROZEN_v2, the features/receipt; `--score` verifies
them, creates `SCORE_CLAIM.json` (O_EXCL) before parsing labels, writes `score_result.json` and `bt2s2_v3_rows.npz`
(O_EXCL). One run.
