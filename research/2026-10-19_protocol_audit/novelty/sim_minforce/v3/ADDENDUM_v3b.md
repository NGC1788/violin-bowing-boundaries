# v3 result and v3b addendum (written after v3 ran, before v3b runs)

## v3 result (run once, `run_v3.log`, `results_v3.json`)
- **C0b validation passed**: F_min/F_S = 2.08 at β = 0.1, b_β = −1.90 (gate [1, 4], −2 ± 0.3).
- **C1b failed the pre-registered FIR quality gate**: max log error for n ≤ 30 = 0.19 (gate 0.1), DC gain 0.83
  (C1b-flat 0.74). A per-period gain of 0.83 at low frequency is ~100 × the true fundamental loss, so C1b "never
  Helmholtz" is a filter artifact. Cause: K = 64 is far too short for the target response (it changes on a scale of
  ~30 harmonics, i.e. an impulse response of ~±130 samples at D = 4000). Not interpreted.

## v3b (fixed now)
- Same as v3 but K = 300 and only β ∈ {0.08, 0.1, 0.12, 0.2} (bridge loop 320–800 samples > K). Smaller β cannot hold
  this filter at D = 4000 and are not run.
- FIR gate: max |log error| ≤ 0.1 for n ≤ 30 and |DC gain − 1| ≤ 0.01; else not interpreted.
- C0b values reused from v3 (same code, unaffected by K).
- Readings as PLAN_v3 at β = 0.1 (r, Q, speed exponent from v = 0.05/0.1/0.2); b_β only over 0.08–0.2 ("large-β
  exponent", 4 points), compared with C0b over the same 4 β. Convergence: D = 8000, K = 600 at β = 0.12.

## v3b result (run once, `run_v3b.log`, `results_v3b.json`)
- **FIR gate failed again** (C1b: max log error n ≤ 30 = 0.196, DC gain 0.989; C1b-flat: 0.074 but DC gain 0.990).
  A DC gain of 0.989 is a per-period fundamental loss of 1.1 % against the measured 0.18 % (6×), so the C1b minimum
  forces (2.0 N at β 0.1, r ≈ 120, b_β −3.4) are dominated by filter error. Not interpreted.
- Likely cause: the digitized ζ_n are bumpy from mode to mode (e.g. mode 3 = 6.7e-4, mode 4 = 3.6e-4), so the target
  response needs a long impulse response, which does not fit into the bridge-side loop even at β = 0.08.
