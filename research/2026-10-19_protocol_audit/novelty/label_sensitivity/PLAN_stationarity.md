# Sustained-motion (stationarity) labels on the measured data (pre-registered)

Claude, 2026-10-21 (before any code for this test; the commit time is the record). Post hoc w.r.t. the boundaries.

## Why
The independent audit noted that the measured sweep varied thresholds inside one classifier but never tested the two
label *kinds* that mattered in the simulations: a slip-fraction check and a stationarity (sustained-motion) check.
- **Slip fraction: not testable now.** The feature files do not store flyback duration, the raw waveforms are not on
  disk except the B1 archive (14.9 GB .7z), and extracting even one folder would breach the 60 GB + 2 GB disk reserve
  (about 3 GB of slack). Recorded as not done.
- **Stationarity: testable** from the stored features, which cover 0.2 s windows every 0.05 s across the plateau.
  Plateau lengths (label-free): 3.4 s at 0.05 m/s, 1.04 s at 0.1 m/s, 0.52 s at 0.2 m/s.

## Design (fixed)
Same sessions, rebuild, columns, K = 2 boundary and per-session fit as `label_sweep.py`; the force coordinate is still the
t = 0.2 s window force (window axis) or the onset force (onset axis). Base label = classifier Helmholtz at t = 0.2 s.
Later time t_late = 0.8 s at 0.05 and 0.1 m/s, 0.3 s at 0.2 m/s (last full window).
- **P1 persistence**: Helmholtz at t = 0.2 s and at t_late.
- **P2 persistence-all**: Helmholtz in every window from 0.2 s to t_late (0.05 s steps).
- **A1 amplitude stationarity**: Helmholtz at 0.2 s and amplitude_std(t_late) ≥ 0.9 × amplitude_std(0.2 s).
Strokes without a t_late window are dropped from that variant only (reported).
Caveat fixed in advance: the measured force falls with bow travel, so a later window sees a lower force; persistence
can fail because the force dropped, not because the motion was not self-sustaining. The onset-force axis does not
remove this (the motion is still judged later); it is reported alongside.

## Readings (fixed)
- Per variant and axis: mean of per-session β exponents, paired shift from base with a 95 % t-interval.
  |shift| < 0.1 small, 0.1–0.3 moderate, ≥ 0.3 large. "Moves to Schelleng" if the mean ≤ −1.8.
- β-confound flag: share of base-Helmholtz strokes removed, by β tercile; flagged if the largest-β tercile loses
  > 20 percentage points more than the smallest (or vice versa).
- The updated range over all unflagged variants (with the earlier periodicity / flyback / relative-amplitude ones),
  read with the same bands as PLAN.md (S < 0.1 robust; 0.1–0.3 moderate; ≥ 0.3 label-dependent).
