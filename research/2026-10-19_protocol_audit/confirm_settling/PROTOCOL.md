# Frozen prediction: does string settling (play-in) shift the playability-boundary levels? (A1 cold/warm)

Claude, 2026-10-20 ~10:10 KST, written BEFORE the coldwarm archive (2024-03-25_TypeA_sample1_coldwarm.7z, Zenodo part 3)
is processed: no feature, label or control of it exists yet (the pipeline is still waiting for disk). Only the archive
listing was read: two folders, `2024-03-25_cold/v2` and `2024-03-26_warm/v2`, 2,000 strokes each (0.1 m/s, 40 β × 50 F).
Per the dataset paper (POMA 58, 035013, §settling) these are the same type-A T1 string recorded immediately after
mounting and after a playing session of several thousand strokes; the paper reports "minimal" playability-area
differences. The A1 diagram analysed so far is the same string type/sample recorded on 2024-03-27 (folders `_r_v1..v3`).

## Why
Post hoc variance decomposition (`novelty/session_variance/`): beyond Z scaling, between-sample/session SD ω is 0.069
(upper) and 0.158 (lower), not smaller than the between-type SD, while same-day repeats agree to 0.002 (upper). If
settling of one physical string moves its levels by ~ω, the "session" component is (partly) a lifetime effect, not
manufacturing variance.

## Quantities (fixed)
t = 0.2 s, K = 2 force-adjacent strokes, measured window force, columns = 40 β groups of 50 strokes (asserted), compare.
intervals logic; zero-width intervals widened by ±0.01 N (as in session_variance). Levels by thermal_law.offset_fit:
- upper: relative to FROZEN_v2 S1_strict (θ fixed, Z of A T1 = 1.0592);
- lower: relative to the A1+A2 free-exponent lower law without Z (as in session_variance.py; deterministic refit).
Sessions: COLD, WARM (unseen) and A1-main restricted to its 0.10 m/s folder (seen).
Differences: Δ_settle = WARM − COLD; Δ_day = A1main(0.10) − WARM. Uncertainty: β-column cluster bootstrap with the
same β columns resampled in both sessions of a difference (B 2000, seed 53).

## Decision rule (per boundary, per difference)
Thresholds h = ω/2 from the session-variance analysis: upper 0.035, lower 0.079.
- "matters": 95 % CI excludes 0 and |point| ≥ h;
- "negligible": the whole 95 % CI lies inside (−h, +h);
- otherwise "not decided".
Primary: Δ_settle for both boundaries. Secondary: Δ_day. Descriptive: scan direction and low-force load decrease
(label-free) of COLD/WARM; upper/lower interval counts.

## Integrity
`score_settling.py --freeze` hashes this protocol, the code and the inputs it will need except the coldwarm features
(which do not exist yet); `--score` checks the hashes, verifies the coldwarm stream receipt, writes an O_EXCL claim
before parsing labels, then `score_result.json` (O_EXCL). One run.
