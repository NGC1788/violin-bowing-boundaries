# Sustained-motion (stationarity) labels on the measured data — result (run once; Claude, 2026-10-21 ≈13:30 KST)

`stationarity.py` → `results_stationarity.json`, `run_stationarity.log`. Plan: `PLAN_stationarity.md` (c6f0be5).

- Slip fraction: **not done** (no flyback duration in the features; raw waveforms not on disk except one archive that
  cannot be extracted within the disk reserve).
- No stroke lacked the t_late window (dropped 0 everywhere).
- Mean per-session β exponent (window axis, 14 sessions | onset axis, 11), paired shift from base:
  - **P1** persistence (Helmholtz also at t_late): −1.519, +0.013 [−0.006, +0.031] | −1.224, +0.012 [−0.001, +0.026].
  - **P2** persistence in every window to t_late: −1.502, +0.030 [+0.007, +0.052] | −1.212, +0.025 [+0.010, +0.039].
  - **A1** amplitude at t_late ≥ 0.9 × at 0.2 s: −1.533, −0.001 [−0.005, +0.003] | −1.237, −0.001 [−0.005, +0.003].
  - All "small" (|shift| < 0.1); none moves toward Schelleng (all means > −1.8); no β-confound flag (base-Helmholtz
    strokes removed: ≤ 1 % in the smallest-β tercile, 0–8 % in the largest).
- Updated range over all unflagged variants (periodicity, flyback, relative amplitude, stationarity): S = 0.34 (window,
  "label-dependent", set by the flyback range) and 0.17 (onset, "moderate") — unchanged by the new variants.

## Reading
At t = 0.2 s after the plateau starts, almost every measured Helmholtz stroke stays Helmholtz and keeps its amplitude
for the rest of the window span (0.6 s at 0.05 and 0.1 m/s; 0.1 s at 0.2 m/s, the plateau limit). The measured boundary
therefore is not contaminated by decaying or transient Helmholtz-like motion, and a sustained-motion requirement leaves
the law unchanged (≤ 0.03). This is the opposite of the simulations, where requiring sustained motion vs a fixed
amplitude ratio moved the exponent from −2.7 to −4.9; the fragility of the simulated minimum force is a property of the
idealised model, not of the measured boundary. Caveat: the later window sees a lower window force (load decrease) —
this would, if anything, make persistence harder, and it still removed almost nothing.
