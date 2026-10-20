# Which minimum-force law? Per-session exponents on sessions never used for this statistic (post hoc; plan first)

Claude, 2026-10-20 ~11:55. Labels of A3, A4, D2 and B_T2 sample 2 were parsed earlier for other frozen statistics
(levels, boundaries), but nobody has computed their lower-boundary speed/β exponents. The expectations below come only
from the 10 development diagrams (novelty/lower_protocol/analysis2.log §3) and are written before computing.

## Development pattern (already known)
Per-diagram free fit log F_min = a + b_v log v + b_β log β (interval-censored, t = 0.2 s, K = 2, measured window force):
eight non-A_T1 diagrams b_v +1.09…+1.42, b_β −1.61…−2.18 (Schelleng v/β² direction); A1/A2 b_v +0.67/+0.94,
b_β −0.97/−1.09. Speed is confounded with recording block (each speed is its own scan).

## Expectations (fixed)
- H_S (Schelleng-like, as the eight): b_v ∈ [0.9, 1.5] and b_β ∈ [−2.4, −1.5].
- For the type-A T1 sessions A3, A4: "A-type specific" if b_β > −1.3 (like A1/A2); "A1/A2 atypical" if b_β < −1.5;
  otherwise "in between".
- Sessions: A3, A4, D2 (3 speeds, 40 × 50) and B_T2 sample 2 (4 speeds 0.05/0.10/0.15/0.20, 40 × 40, forces from
  ≈0.25 N, so more left-censoring; also fitted on its three standard speeds).
- Uncertainty: β-column-index bootstrap paired across speeds, B = 500, seed 61.
- Sensitivity (pre-declared): readings below 0.2 N treated as left-censored (detection limit), same fit.
- The 10 development diagrams are re-fitted with the same code for a like-for-like table.
- Every result is reported; no other axes, K or windows are tried here.

## RESULTS (post hoc; results.json, floor_check.json, common_limit.json, onset_axis.json)
1. Planned check: the development pattern did NOT replicate. Measured-axis β exponents: A3 −1.22 [−1.32, −1.11],
   A4 −1.42 [−1.59, −1.27], D2 −1.11 [−1.20, −1.01], B_T2 s2 −1.31 [−1.42, −1.20] (std speeds); none in the H_S box. A3 reads
   "A-type specific", A4 "in between". Same-type contrasts: D1 −1.64 vs D2 −1.11; B_T2 dev −1.82…−2.18 vs s2 −1.31.
   The same-day repeat pair itself differs (B_T2r1 −2.18 vs r2 −1.82): a single diagram's β exponent carries ≈ ±0.3.
2. Follow-ups decided after (1): (a) β exponent vs session force floor ρ −0.36 (p 0.21); vs number of left-censored
   columns ρ −0.81 (p 0.0004); (b) a common detection limit (0.35 N for every session) shrinks the spread only from
   SD 0.37 to 0.28 → not a floor artefact; (c) ONSET-force axis (force at bow onset from label-free ramp metadata, 11
   sessions; B1 and Part 6 lack ramp data): random-effects mean β −1.22 [−1.36, −1.09] (HKSJ), between-session SD 0.18;
   speed exponent 0.95 [0.86, 1.05], SD 0.13 — versus the measured axis (14 sessions) β −1.54 [−1.75, −1.33], SD 0.37;
   speed 1.09 [0.97, 1.21], SD 0.23.
Reading (post hoc, axis chosen after seeing the data): on the onset-force axis the sessions agree on F_min ∝ v^0.95 β^−1.2,
far from Schelleng's v/β² and close to the simulated v/β form; the travel-affected window-force axis steepens the β
exponent and doubles its between-session scatter. Speed remains confounded with recording block.

## Robustness of the onset-axis result (onset_sensitivity.json; post hoc; 10 sessions with cached data, B_T2 s2 only in LOO)
RE β exponent: t = 0.1 s −1.16 [−1.32, −0.99] (SD 0.22); t = 0.3 s −1.25 [−1.39, −1.12] (SD 0.16); K = 3 −1.14 [−1.29, −0.99] (SD 0.22; interval reaches −1);
leave-one-session-out mean −1.19…−1.25. The onset-axis β ≈ −1.1 to −1.3 is stable across these choices; none approaches −2.

## B1 added after its recovery (2026-10-20, onset_with_B1.json; post hoc)
B1 onset axis: v +0.99 [+0.84, +1.15], β −1.32 [−1.40, −1.25] (window axis v +1.09, β −1.66). Random effects over 12
sessions: β −1.23 [−1.35, −1.11], SD 0.18; v +0.96 [+0.87, +1.04], SD 0.12 — unchanged.
