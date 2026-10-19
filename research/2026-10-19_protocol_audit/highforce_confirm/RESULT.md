# High-force frozen confirmation — result (scored once, 2026-10-19 21:39–21:41 KST)

Protocol `PROTOCOL.md`; freeze `FREEZE_RECEIPT.json` (21:39:37, label-free; frozen-grid identity 1e-16);
claim `SCORE_CLAIM.json` (21:39:42); output `score_result.json` (read-only, O_EXCL). Governing predictions:
FROZEN_v2 (10:31) and FROZEN_v3 addendum (10:39). Targets are **unseen sessions (sample2, 2024-06-22/24/27/29) and an
unseen force range (≈4–12 N) of string types whose sample1 diagrams trained the models** — not unseen strings.
Window t = 0.2 s, K = 2, measured force. Natural-log units.

## Design (controls only, before labels)
Each target: 3 speeds (0.05/0.10/0.20 m/s) × 30 β columns (0.02–0.07, log-spaced) × 50 forces (≈3.8–12.6 N,
log-spaced) = 4,500 strokes. Upper-boundary columns per target (no-band excluded): C 77, A 77, D 80, B 81 of 90.

## P2 — frozen level predictions: **failed in all four strings**

Level error = observed − predicted (offset fit); frozen tolerance = each model's PI90 (≈0.15).

| Target (Z) | S1_strict (cZv/β) | T1k_break (thermal) | E3_free_Z (empirical) | best inside PI90 |
|---|---:|---:|---:|---|
| HF_C (1.1154) | +0.364 | +0.447 | +0.337 | none |
| HF_A (1.1884) | +0.224 | +0.316 | +0.204 | S2m +0.056, T2km_break +0.140 |
| HF_D (1.2021) | +0.165 | +0.262 | +0.165 | S2m +0.007, T2km_break +0.097 |
| HF_B (1.1899) | +0.357 | +0.432 | +0.311 | none |

Every frozen model **under-predicts** the 4–12 N upper boundary of the new sessions by 0.17–0.36 log (18–43 %);
S1_strict is outside PI90 for 4/4 targets. Thermal vs plain Schelleng (0.03 rule): **S1_strict better in 4/4**
on level error; thermal NLL worse by +0.14 to +0.66 per column. The two models inside PI90 for A and D (S2m,
T2km_break: free β exponent m fitted ≈1.15–1.19) are not a pre-specified decision and are 2 of 10 models.

## P1 — β exponent zone (hypotheses_high_force)

| Target | e_β [95 % CI] | e_v [95 % CI] | frozen decision | S1 (no-band → F_max < F_min of column) |
|---|---|---|---|---|
| HF_C | −0.75 [−1.27, −0.44] | 0.79 [0.59, 1.11] | not decidable | −1.05 [−1.50, −0.73], not decidable |
| HF_A | −0.82 [−1.30, −0.47] | 0.91 [0.70, 1.23] | not decidable | −1.03 [−1.44, −0.76], not decidable |
| HF_D | −0.48 [−0.79, −0.24] | 0.79 [0.67, 1.01] | **H_thermal supported** | −0.61 [−0.86, −0.37], supported |
| HF_B | −0.39 [−0.68, −0.16] | 0.57 [0.45, 0.77] | **H_thermal supported** | −0.40 [−0.74, +0.04], supported |

Caveat that changes the reading: the H_thermal zone (e_β ≥ −0.95) is one-sided. The thermal models themselves predict
e_β −0.86 to −0.94 on this design; the D and B slopes (−0.48, −0.39) are far shallower than **any** frozen model,
and the thermal models lost to S1 on level and NLL. "H_thermal supported" therefore means only "shallower than
Schelleng's β^−1", not support for the thermal friction law.

## P3 — v3 speed curvature with the ceiling lifted: **not decided**

Pooled (4 targets, target offsets): e12 0.82, e23 0.63, curvature −0.19; cluster-bootstrap CI [−0.46, +0.17],
stratified [−0.38, +0.04] → concave point, CI includes 0. Per target heterogeneous: A +0.13, C +0.21, B −0.53,
D −0.44. S1: −0.46, cluster CI [−0.92, +0.06] (includes 0), stratified [−0.76, −0.24] (excludes 0) →
censoring-assumption-dependent; primary decision not upgraded.

## Descriptive (pre-declared D1/D2)

- No-band columns: 13/13/10/9 (C/A/D/B), almost all at v = 0.05 and β ≥ 0.042 (consistent with F_max < 4 N there;
  HF_B also 2 columns at v = 0.20, β ≈ 0.02).
- Right-censored upper (H at 12 N): mostly v = 0.20 (22–29 of 30 columns) and v = 0.10 (10–20).
- Multi-band columns (≥ 2 separated K = 2 H bands): 48/90 (C), 54/90 (A), 48/90 (D), 38/90 (B). **Post hoc
  (`posthoc_multiband.py`): this is not specific to high force** — the 11 low-force diagrams show 37–73 % of
  banded columns with ≥ 2 bands (HF 47–70 %). Extra bands are short (median 2–4 strokes) and separated from the main
  band (median 20–24 strokes at high force) by a median gap of **one** non-H stroke, i.e. mostly single-stroke
  dropouts, not separate H regions. The frozen last-run rule was the same in training and test.
- Lower boundary: mostly left-censored (H already at ≈4 N); no lower hypothesis was scored.
- t = 0, 0.1, 0.3 s (descriptive): e_β −0.34 to −0.80, e_v 0.50–0.90; same picture.

## Reading

1. The frozen level transfer that worked for A_T2/D_T2 (same force range, sample1 sessions) **did not** transfer to
   the sample2 sessions in the 4–12 N range. Causes are not separated: session (string sample, date), force range,
   and the boundary-extraction rule (multi-band) are confounded. No post hoc rescue is used to overturn this.
2. Post hoc rule sensitivity (not used to rescue anything): replacing the frozen last-band upper rule with the
   longest-band rule moves the S1_strict level errors to +0.02/−0.14/−0.08/+0.19 (C/A/D/B) and flattens e_β to
   −0.28…+0.05; the first-band rule gives positive e_β. In this force range the upper boundary estimate is fragile
   with respect to how single-stroke dropouts are handled, so the frozen failure mixes session/force-range
   differences with rule fragility. Neither is separable here; the frozen verdict stands.
3. Nothing here supports the thermal friction law over plain Schelleng.
