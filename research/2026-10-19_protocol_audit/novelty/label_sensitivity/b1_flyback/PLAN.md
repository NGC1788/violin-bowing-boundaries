# B1: corner-width (flyback-duration) labels from the raw waveforms (pre-registered)

Claude, 2026-10-21 (before any code; the commit time is the record). Post hoc w.r.t. the boundaries.

## What can be measured
The bow-point slip fraction used in the simulations is not observable in the bridge force. What is observable is how
long the Helmholtz corner takes to pass the bridge: the duration d_fb of the tallest flyback leg in the folded mean
cycle (`regime_map.flyback_structure` → `flyback_ms`), i.e. the corner width. A rounded corner is the data-side
counterpart of the simulations' broadened-slip motion (v8). It is a proxy, not the slip duration.

## Data and processing (fixed)
Only B1 (2024-04-10_TypeB_sample1) has its raw archive on disk (14.9 GB .7z). It is streamed member by member with the
existing 7zz (`x -so`, inventory from `l -slt`, as `astra_20261019/stream_features.py`); nothing is written to disk
except a small per-stroke table. For each stroke, the plateau and the t = 0.2 s, 0.2 s window are found exactly as
`timeres/replicate.windows_of`; periodicity, f0, flybacks_per_period, second_flyback_ratio and flyback_ms are computed
there. **Gate**: periodicity, flybacks_per_period and second_flyback_ratio must equal the stored B1 features for
≥ 99.9 % of strokes, else nothing is interpreted. The archive is not modified or deleted.

## Variants (fixed; on top of the base classifier label at t = 0.2 s)
- Absolute corner: d_fb ≤ k · β · T0 (T0 = 1/f0 of the window), k ∈ {1, 2, 4}.
- Relative corner: d_fb ≤ q × the median d_fb of base-Helmholtz strokes in the same (folder, β column), q ∈ {1.5, 2}.

## Readings (fixed)
- B1 minimum-force β exponent (window axis and onset axis) per variant; shift from base with a paired β-column
  bootstrap (200) 95 % interval. |shift| < 0.1 small, 0.1–0.3 moderate, ≥ 0.3 large; "moves to Schelleng" if ≤ −1.8.
- β-confound flag as before (removal of base-Helmholtz strokes differs by > 20 points between the smallest- and
  largest-β tercile); flagged variants are not read as label sensitivity.
- Descriptive: log d_fb regressed on log β, log v and log F among base-Helmholtz strokes (does the corner width scale
  with the bow–bridge round trip βT0?).
- One session only; no generalisation.
