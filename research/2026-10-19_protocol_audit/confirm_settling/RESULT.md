# Settling test (A1 cold → warm → main) — result

Frozen protocol `PROTOCOL.md` (+ `ADDENDUM_20261020_1015.md`: the cold folder's legacy features already existed, unopened).
Run 1 crashed before computing anything (folder-name bug); run 2 = bug fix only (`RUN2_NOTE.md`). Evidence level:
frozen while processing was under way, no label parsed before the freeze; ONE string, ONE speed (0.10 m/s).

## Frozen decisions (thresholds ω/2: lower 0.079, upper 0.035)
| Difference | Lower boundary level | Upper boundary level |
|---|---|---|
| warm − cold (primary) | **+0.256 [+0.003, +0.542] → matters** | −0.041 [−0.099, +0.014] → not decided |
| A1 main − warm (secondary) | +0.109 [−0.001, +0.273] → not decided | −0.046 [−0.139, +0.026] → not decided |

Levels (lower): cold −0.323, warm −0.067, A1 main (0.10 m/s) +0.042. Finite lower columns 23 / 38 / 40 of 40.

## Post hoc checks (after the score)
- Protocol (label-free, `posthoc_protocol_check.py`): cold and warm share the scan direction (ascending) and the
  low-force attack (0.104 m/s²); A1 main is the reverse scan (0.563 m/s²). The load decrease is LARGER in warm
  (dec5 −0.065 vs −0.025 N), which by the development association would predict a LOWER warm level — the opposite of
  what was observed. The lowest settings delivered much more force in cold (median onset 0.48 N vs 0.17 N warm, 0.15 N
  main): the elevation-to-force mapping changed as the new string settled.
- Censoring (`posthoc_floor_matched.py`): because cold never reached the low forces, many of its lower boundaries are
  left-censored. Matching the force floor (dropping 651 warm strokes below each column's lowest cold force) gives
  warm − cold = **+0.317 [+0.106, +0.631]**; main − warm +0.053 [−0.057, +0.194]. The shift is not a censoring artefact.

## Reading
For this one string, play-in raised the minimum-force boundary level by ≈0.26–0.32 ln (≈30–37 %), comparable to the
between-type spread, while the maximum-force level did not move detectably; most of the change happened between
mounting and the end of the play-in, not between day 2 and day 3. The dataset paper reported "minimal" playability-area
differences for the same comparison. One string and one speed: this is a single prospective observation, not a law,
and settling is confounded with everything else that changed between 03-25 and 03-26 (rosin state, room conditions,
force calibration).

## Adversarial verification (workflow wf_7f5846d0-4a8, independent re-run; post hoc) — supersedes the "Reading" above
- Reproduced exactly. **Erratum:** the bootstrap paired β columns by round(log β, 2), which matched only 29 of 40 columns
  (cold/warm β differ by 0.0005–0.0066 in log). Index pairing: warm − cold lower +0.256 [+0.071, +0.466] (frozen decision
  unchanged: "matters"); main − warm lower +0.109 [+0.002, +0.229] (would read "matters"); upper unchanged (not decided).
  The frozen outputs are kept as computed.
- Direction supported: 19 vs 6 β columns with non-overlapping intervals; Helmholtz fraction at 0.30–0.60 N 0.50 (cold) →
  0.29 (warm). Classifier thresholds cannot explain it (f0 reference and min_amplitude cannot flip a Helmholtz label at
  all; periodicity thresholds only flip windows above 1 N).
- Size NOT established: 17 of 40 cold lower boundaries are left-censored, so the cold level is a censored-normal
  extrapolation (cold high-β half −1.13). Complete case +0.10 [−0.04, +0.26]; onset-force axis +0.12 [+0.02, +0.22];
  t = 0.3 s +0.09 [−0.16, +0.36]; column-wise sign tests not significant on any physical axis; adding −0.05 N to warm forces
  removes it (warm has 31 negative-force windows, which argues for the opposite correction). The floor-matched check did
  not remove the cold censoring, so "not a censoring artefact" was overstated.
- Corrected reading: for this one string the minimum-force boundary probably moved up during play-in, by roughly
  +0.1 to +0.3 ln; the shift is confounded with the changed elevation-to-force mapping, with β/time order inside each
  session and with the force zero offset. The upper boundary showed no change in any variant.
- Prior art (prior-art agent): Lampis, Mayer & Chatziioannou, JASA-EL 4, 113201 (2024) saw a string's transient playable
  region change after remounting one month later; Schoonderwaldt et al. 2008 §3.6 found the minimum force ≈20 % lower in
  a diagram repeated five months later. A controlled new-vs-played-in comparison of steady-state F_min/F_max with a
  bowing machine was not found.
