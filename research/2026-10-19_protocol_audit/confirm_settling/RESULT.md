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
