# Verification of the minimum-force law result (novelty/lower_law) — verdict

Computations by Astra (2026-10-20 10:4x, `statistics.py`, `axis/audit_axis.py`, plan `PLAN.md`); Astra stopped before
writing a verdict, so this summary was written by Claude from `audit_statistics.json`, `axis/axis_results.json` and the
logs. Everything is post hoc.

| Check | Result |
|---|---|
| (folder, trial) → onset-force mapping | complete and unique in every session; no missing t = 0.2 s strokes; β and speed identical between caches and ramp metadata |
| β-block bootstrap (block 1 / 5 / 10 adjacent β columns) | onset-axis RE β −1.22 [−1.36, −1.09] / −1.23 [−1.36, −1.09] / −1.26 [−1.39, −1.12]; τ 0.15–0.20 |
| Onset minus window (paired over sessions) | β +0.21 [+0.11, +0.32] (block 1), +0.22, +0.22; speed −0.08 [−0.13, −0.03] → the axis effect is real across sessions |
| Equal weight per type × tension group | window β −1.58 [−1.88, −1.29]; onset β −1.29 [−1.44, −1.14] |
| Nominal-command axis (proxy grid 0.1·40^(rank/49)) | per-session β −0.79 … −2.23 (as scattered as the window axis); the onset result is not the command grid |
| Censoring variants on the onset axis (complete case / omit no-band / common 0.25 N / 0.35 N) | per-session β −0.86 … −1.61; none near −2 |
| D_T2 (onset axis, added) | β −1.22 |

**Verdict: survives (as a post hoc association), magnitude widened.** On the force at bow onset, the minimum-force
boundary of 11–12 sessions scales close to v^0.95–1.0 · β^(−1.2 to −1.3), with between-session SD ≈ 0.15–0.20; censoring
choices move single sessions between β −0.9 and −1.6, but no variant approaches Schelleng's −2. On the window force after
travel the exponent steepens by ≈0.2 and the scatter roughly doubles. Not done by this check: the within-scan
β/time-order confound test and the prior-art search (item 7 of the plan). Speed remains confounded with recording block;
the onset axis was chosen after seeing the data.

## Prior art (Astra, literature/README.md) — changes the reading
v/β was already proposed from simulations together with a window-dependence of the fitted β exponent (Beigi & Conneely
2026 v3, Eq. 8–9); the load decrease and the use of the window-averaged force are in van Walstijn et al. 2026 §3.2;
establishment vs maintenance thresholds were measured by Galluzzo (≈4×); Schoonderwaldt et al. 2008 measured lower slopes
not different from −2. Remaining contribution: a quantified sensitivity of the minimum-force exponent and its
between-session scatter to the time at which the force coordinate is read, on the same real sessions — not a new law.
