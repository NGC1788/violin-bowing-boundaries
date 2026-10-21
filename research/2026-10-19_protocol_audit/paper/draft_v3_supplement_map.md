# Supplement map for draft v3

Companion to `paper/draft_v3.md` (2026-10-21). Draft v3 keeps one central contribution: how regime-classification choices constrain bow-force laws estimated from robotic Schelleng diagrams (v2.10 §§3.8, 3.10–3.15). Everything else in working draft v2.10 is indexed here, with one or two sentences each and pointers to the v2.10 section and the result folders (paths relative to `/Users/jeongjihoon/violin-rne`). Numbers below are copied from v2.10; nothing was recomputed. `draft_v2.md` (v2.10) remains the complete record and was not edited.

## A. Results excluded from the main text by design

1. **Protocol audit** (post hoc; v2.10 §3.1; v2.10 Fig. 1). In A2 the measured force of nominal-force ranks 1–20 fell by 0.044–0.174 N after 2–14 cm of travel, the attack acceleration follows recording order (the reverse-scanned A1 gave its low-force strokes 15×, 5.4× and 1.35× harsher attacks than A2), and each speed is a separate session. Forces below about 0.2 N are uncertain by tens of millinewtons. Pointers: `paper/repro/attack_scan_all.log`, `novelty/attack_data/`, `novelty/lower_protocol/`, `paper/fig1_protocol_en.png` (script `paper/scripts/fig_paper_en.py`).

2. **Frozen maximum-force tests** (v2.10 §3.2 with Tables 2–3 and Figs. 2, 4, 5; §3.6 "Why the high-force transfer failed" and "Maximum-force results"). Across nine frozen once-only tests of the strict cZv/β law, 4/9 passed and the level-error SD was 0.20, and all ten frozen upper-boundary models under-predicted the 4–12 N boundary of all four high-force sessions; session, force range and extraction rule are confounded. Pointers: `highforce_confirm/`, `confirm_v3/`, `confirm_B_T2s2/`, `novelty/thermal_upper/` (FROZEN_v1–v3 with hashes), `posthoc_hf/`, `robust_upper/`, `novelty/range_effect/`, `novelty/upper_final/`, `novelty/upper_law/`; figures `paper/fig2_upper_en.png`, `paper/fig_highforce.png`, `paper/fig5_frozen_upper_summary.png`.

3. **String damping and the absolute minimum force** (post hoc, zero fitted parameters; a replication of Schoonderwaldt et al. 2008; v2.10 §3.9, first half). With zero fitted parameters, Schelleng's minimum force filled from published Z, the friction jump and digitized modal damping lay below the measured boundary by Q = 8.7–20 (onset axis) and 4.6–14 (window axis), replicating the order-of-magnitude shortfall of Schoonderwaldt et al.; damping did not order the minimum force across string types. Pointers: `novelty/string_damping/` (plan commit `b2886c4` before digitizing).

4. **Simulation programme** (v2.10 §3.9, second half). A corner-exact digital waveguide reproduced Schelleng for frequency-independent loss (β exponent −2.0 to −2.05), but with measured frequency-dependent damping every quantity depended on the friction law and the Helmholtz definition (β exponent −2.7 to −6.5 among sustained-motion labels); it does not identify the cause of the shortfall (post hoc; not independently verified). Pointers: `novelty/sim_minforce/` (eight versions; all failures kept).

5. **Settling** (v2.10 §3.7, second paragraph). For one string at one speed (A1, 0.10 m/s, cold → warm → main diagram), a frozen comparison indicated a probable upward shift of the minimum-force boundary of roughly 0.1–0.3 during play-in, confounded with the elevation-to-force mapping, and no detectable change of the maximum force. Pointers: `confirm_settling/` (PROTOCOL.md, RESULT.md, fig_settling.png), `refute_settling/`.

## B. Other v2.10 material not used in the v3 main text

6. **Frozen minimum-force tests and the dec5_pl association** (v2.10 §3.3; §3.6 "Minimum-force association"; v2.10 Fig. 3). The post hoc association of minimum-force levels with a same-session load-decrease descriptor (r = 0.88, n = 10) is fragile; the D_T2 primary endpoint was untestable and the A4/D2 PACE* tests are not read as a method advantage. Pointers: `robust_lower/`, `paper/fig3_lower_en.png`.

7. **Prediction of an unseen string** (v2.10 §3.4, Table 1). On the development diagrams the lookup and the free-exponent law tied on the lower boundary (0.269 vs 0.268), and no variant beat the plain lookup in leave-one-diagram-out (0.258). Pointer: `final/ANALYSIS_PLAN_FINAL.md` (the published-Z comparison is a post hoc change).

8. **Change-point boundary extraction** (v2.10 §3.5). A Bayesian single-band change-point extractor showed no consistent test–retest improvement over the K = 2 rule on B_T2r1/r2, so this direction was stopped. Pointer: v2.10 §3.5 (uses v2.10 ref. [11]).

9. **Session variance decomposition** (post hoc; partly refuted by an independent check; v2.10 §3.7, first paragraph). Within some types, sessions differ as much as impedance-normalised types; with five within-type degrees of freedom the design cannot rank the two components. Pointers: `novelty/session_variance/`, `refute_session_variance/`.

10. **Further minimum-force-law details** (v2.10 §3.8 beyond what v3 §§3.1–3.2 keep). These are the label-free load transform (1 + c = 1.12–1.63; a partial protocol explanation that failed its pre-declared criterion), local exponents of Schoonderwaldt's modified law, B1 corner-width labels (the βT0-tied variant steepened B1 from −1.66 to −2.53 but is β-confounded), sustained-motion label details, and onset-axis robustness to t and K. Pointers: `novelty/load_transform/`, `novelty/lower_law/` (onset_sensitivity.json, onset_with_B1.json, scan_order_check.json), `verify_lower_law/`, `novelty/label_sensitivity/` (b1_flyback/, RESULT_stationarity.md, posthoc_amplitude.json, verify/).

11. **Full tilt and criterion-replay record** (v2.10 §§3.10–3.11). This covers the complete 14-diagram tilt curves (height, periodicity, ratio, B1 corner width), the sufficient Gaussian model and its u_r formula, the 13,720 replay fits (219 numerical fallbacks), the matched-mask and stable-target audits, and the support diagnostics. Pointers: `novelty/tilt_principle/`, `novelty/criterion_identifiability/` (theory/REPORT.md, theory/counterexamples.py, response/, locality/, review/).

12. **Friction** (v2.10 §4, "Friction"). In a public friction reconstruction the friction contrast did not collapse on F v, and thermal-friction upper laws lost to plain Schelleng in all four high-force sessions; no thermal-friction claim is made. Pointers: `novelty/friction_contrast/`, v2.10 ref. [14].

13. **No-band censoring direction** (v2.10 §2). The binary data cannot identify whether a no-run column lies below or above the measured range. v3 §2.3 states this assumption; the derivation is in `astra_20261019/observation_model/README.md`.

## C. Details of the central contribution moved to the supplement

- **§4 (exact edits).** Synthetic design and negative results (`novelty/run_fragility/SYNTHETIC.md`), proofs (`THEORY.md`, `GLOBAL_THEORY.md`), all K, ε and direction-restricted results (`results.json`, `diagnostics.json`), input audit and verification (`verification/REVIEW.md`), witness figure (`fig_witness.png`).
- **§§5.1–5.4 (scalar shared rules).** Attainability proof and addendum (`novelty/coherent_fragility/literature/ATTAINABILITY.md`, `ADDENDUM_ATTAINABILITY.md`, `THEORY.md`), all threshold profiles (`profiles/*.npz`), budget and per-folder oracle results (`results.json`, `diagnostics.json`), threshold-profile figure (`fig_threshold_profile.png`).
- **§§5.5–5.8 (joint rule).** Per-case obstruction table and summary (`novelty/joint_rule/theory/obstruction_table.md`, `obstruction_summary.md`), construction (`theory/THEORY.md`, `solver_notes.md`), independent verification (`verification/REVIEW.md`), joint-region figure (`fig_joint_regions.png`). Its marked B_T2r1 point minimizes tolerance and changes 3,255 labels, a different objective from the 3,200-change budget optimum.
- **§6 (MLE bridge).** β-index block deletions across speeds (`novelty/mle_bridge/production/selected.json`, documented in `production/README.md`), the full certificate family (`theory/family.json`, `theory/FAMILY_REVIEW.md`), theory and literature audit (`theory/THEORY.md`, `theory/LITERATURE.md`), independent verification (`verification/README.md`), plans (`PLAN.md`, commit `aed035e`; `FAMILY_EXTENSION.md`, commit `45014c5`).

## D. Supplementary figure candidates

- S1. `novelty/lower_law/fig_lower_law.png`: per-session minimum-force β exponents on the window and onset axes (post hoc). Panel (b) carries an annotation of a refit of Schoonderwaldt 2008 that v2.10 does not describe, so check it before use.
- S2. `novelty/criterion_identifiability/response/response_curves.png`: replay response curves. The bands are Monte Carlo error, not population intervals.
- S3. `novelty/criterion_identifiability/locality/fig_endpoint_locality.png`: lower/upper endpoint changes per feature (13 sessions; descriptive fractions).
- S4. `novelty/run_fragility/fig_witness.png` (v2.10 Fig. 7 caption).
- S5. `novelty/coherent_fragility/fig_threshold_profile.png` (v2.10 Fig. 9 caption).
- S6. `novelty/joint_rule/fig_joint_regions.png` (see the 3,255-label note above).

## E. References

v3 cites 25 of the 33 v2.10 references, renumbered by first appearance. Mapping v2.10 → v3: 1→1, 2→2, 3→3, 4→4, 5→5, 6→6, 7→7, 8→8, 9→9, 10→10, 21→11, 19→12, 20→13, 23→14, 22→15, 24→16, 27→17, 28→18, 32→19, 33→20, 25→21, 29→22, 30→23, 31→24, 26→25.

The dropped v2.10 references belong to the excluded material: [11] Fearnhead (change point, item 8); [12] Schumacher and [13] Mores 2016 (maximum-force Z scaling, item 2); [14] friction-reconstruction data (item 12); [15] Johnson et al. and [16] Leek et al. (batch effects, items 2 and 9); [17] Lampis, Mayer and Chatziioannou 2024, Acta Acust. (session effects, friction jump in item 3); [18] Lampis, Mayer and Chatziioannou 2024, JASA Express Lett. (remount effect, item 5).
