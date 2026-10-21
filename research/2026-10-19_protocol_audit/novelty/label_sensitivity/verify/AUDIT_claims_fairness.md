# Claims and fairness audit of `label_sensitivity` (verifier, 2026-10-21)

Scope: RESULT.md, draft_v2 §3.8 sentence ("robust to how the Helmholtz label is thresholded"), 최종보고서 §10.8 paragraph
("판정 기준을 바꿔도 결론이 유지되나"). Only files in this `verify/` folder were written. Scripts: `audit_check.py`
(→ `audit_check.json`), `extend_flyback.py` (→ `extend_flyback.json`).

## 1. Number check (all quoted numbers vs results.json / posthoc_amplitude.json)
All numbers match after rounding:
- base −1.532 | −1.236; gate max |diff| 0.000 on both axes (but **13** window sessions were gated: D_T2 has no stored
  window exponent and was silently skipped, so it enters the 14-session mean ungated).
- periodicity shifts ≤ +0.011 | +0.021 ("≤ 0.02" in the paper rounds 0.021).
- flyback 0.3/0.4/0.6/0.7: +0.169/+0.066/−0.095/−0.171 | +0.112/+0.049/−0.044/−0.058. Correct.
- fixed floor means −0.136/+0.705 (window), −0.068/+0.630 (onset); S 2.408 / 1.925. Correct.
- β terciles: smallest 0.2–6.5 %, largest 41.7–58.4 % ("0–6 %", "42–58 %"). Correct.
- relative amplitude: +0.014/+0.164 | +0.007/+0.102. Correct.
- β-neutral range 0.340 (−1.703 to −1.362) | 0.170 (−1.295 to −1.124). Correct.
- **Wrong label**: "preferentially at low force *within a column* (corr −0.34 to −0.61)". The code correlates removal
  with log F across all Helmholtz windows of the session (β and force are confounded across columns). After removing
  (speed, β)-column means the correlation is **−0.21 to −0.52**.
- "Helmholtz amplitude scales with v/β": approximately. Fitted over base-H windows (controlling log F):
  amplitude ∝ v^0.85–0.97 β^−0.70 to −0.81.

## 2. Honesty of the post hoc β-confound argument
- The classifier comment ("a fixed floor alone would erase weak but regular oscillation at large beta") was committed
  2026-09-17 (8e3f6c0), a month before the plan (32271d1, 12:25). The plan's "Why" paraphrases `classify`, so the
  author knew it, yet put amp1x/amp2x in the sweep **without flagging** them and without a rule to exclude any
  variant from S. The mechanism data (42–58 % vs 0–6 %) do support the confound, and RESULT.md labels it post hoc,
  so RESULT.md is honest. The paper sentence is not: it leads with "robust ... with one instructive exception",
  which reverses the pre-registered verdict ("label-dependent" on both axes).
- Even the post hoc β-neutral subset is **not** "robust" by the plan's own vocabulary (robust = S < 0.1): window 0.34
  → "label-dependent", onset 0.17 → "moderate". "Robust" is used for something the plan defined differently.
- "±0.17, the same size as the force-reference-time effect" compares a half-span with a full shift. The flyback full
  span is 0.34 (window), larger than 0.21 [0.11, 0.32]; on the onset axis 0.17, smaller.
- The flyback range 0.3–0.7 sets the ±0.17. Outside it (verifier, `extend_flyback.py`): f 0.8 → mean −1.30
  (shift +0.23, 18 % labels changed), f 0.9 → −0.59 (+0.94, 32 %), f ≥ 1 degenerate. Those rules admit near-equal
  double flybacks (multiple-slip-like windows), so they are not real Helmholtz labels. But they show the trend is not
  monotone and that the ±0.17 depends on where the plan put the range. They move away from −2, not toward it.
- The narrow claim survives: no β-neutral variant puts the mean near −2 (fly0.7 mean −1.70, 95 % CI [−1.89, −1.51]).
  But fly0.6/0.7 move 10–11 of 14 sessions toward −2, and single sessions reach −2.40 (A_T2) and −2.26 (B_T2s1).

## 3. Fairness of the simulation analogue and comparison
- The simulation's check (`vsim3.label3`) compares the median |Δv| over *slip* samples with 0.5 v/β, an absolute
  ideal-Helmholtz scale, together with a slip-fraction check 0.5β ≤ sf ≤ 2β. The measured "relative" criterion uses
  the std of the sensor channel over the whole window and normalises it by the column's own median. So it cannot see
  a deficit that covers the whole column (for example broadened slip in every small-β window, which is the v8
  mechanism). It is a loose analogue, not "the analogue".
- Closer analogue (verifier): amplitude ≥ r·k·v/β, with k = the session median of amp·β/v over base-H windows.
  - r 0.5: shift +0.000 [−0.012, +0.012] window, −0.002 onset. It removes 0–7 % of windows in every β tercile, so it
    is β-neutral.
  - r 0.7: +0.132 [+0.077, +0.188] window, +0.076 onset.
  - This **corroborates** the parent: an amplitude check scaled to v/β does not steepen the measured exponent.
- "−2.7 vs −4.9" is the v8 smooth-friction result only. Two caveats:
  - The paper's own §3.9 gives −2.7 to −6.5 across friction laws.
  - The L3 value −2.69 includes a grid-censored point (−2.82 without it).
  - Quoting the narrower range makes the simulation look less label-dependent, so it is conservative for the claim.
  - The simulation varied label *kinds* (period-only, amplitude + slip-fraction, stationarity). The measured sweep
    varied thresholds inside one classifier, and no slip-duty or stationarity label was tested. "Far more robust" is
    fair only for the amplitude part: the counterpart of the simulation's −4.9 label moved the measured mean by
    ≤ +0.16.

## Verdict: weakened
The numbers are correct and the narrow conclusion holds ("gentler than β^−2 under every β-neutral threshold
tested"). "Robust" is an overstatement by the plan's own bands. The within-column correlation is mislabelled. The
comparison with the force-reference-time effect is unfair. The comparison with the simulation needs a scope
qualifier. Suggested exact rewordings are in the structured output.
