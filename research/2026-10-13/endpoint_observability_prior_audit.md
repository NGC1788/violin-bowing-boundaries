# Endpoint-force reconstruction: prior-art and identifiability audit

Checked 2026-10-13. This is a critical feasibility assessment, not evidence that the proposed method is new. No new measurements, raw-data downloads, or author contact were performed for this audit.

## Verdict

Endpoint-force inversion, redundant reconstruction from both endpoints, and tuning delays/gains by their agreement are established. Finite real-bow width has also been addressed by the producers of the public cello dataset. A new paper cannot credibly center its novelty on any of those operations. A narrower investigation of **which contact-state claims remain identifiable under explicit model uncertainty** could still be useful, but requires calibrated uncertainty sets and comparisons to the actual inverse-method literature.

## Closest primary sources

1. **Woodhouse, Schumacher & Garoff (2000), JASA 108(1), 357–368**, [DOI](https://doi.org/10.1121/1.429529), [author-hosted full paper](https://euphonics.org/wp-content/uploads/2022/03/Schumacher_Garoff1.pdf). Sections I–III reconstruct force and transverse velocity at the bow using both endpoint reactions. They provide dispersive time-domain FIR and frequency-domain formulations, require transfer/reflection information, and use plucks to calibrate delays, channel gains, impedance and stiffness. Section III.A fine-tunes delays and relative gain by minimizing the difference between two reconstructed velocities. Synthetic signals validate reconstruction. Section IV discusses torsion: string-center velocity differs from surface contact velocity. These are direct precedents for the proposed inverse identity and dual-end consistency check.

2. **Debut, Delaune & Antunes (2010), International Journal of Mechanical Sciences 52, 1419–1436**, [publisher article](https://www.sciencedirect.com/science/article/abs/pii/S0020740310000536), DOI 10.1016/j.ijmecsci.2010.03.010. This work identifies contact force and motion from support reactions through a modal representation. It regularizes the inverse operator and optimizes imperfect modal frequencies. The publisher's accessible text describes simulation, noisy-data tests, and physical experiments. Therefore “inverse reconstruction robust to noise and uncertain dynamics” is also insufficient as an unqualified novelty claim. This audit read the accessible publisher text, not a complete downloaded manuscript.

3. **Lampis, Matusiak, Mayer & Chatziioannou (2025), “Friction force reconstruction in bowed cello strings,” JASA 157(4 Supplement), A203**, DOI 10.1121/10.0037894. The [authors' official publication list](https://www.mdw.ac.at/iwk/wasilakis/publications/) identifies this as a **conference abstract**, not a full journal article. Publisher full text was inaccessible here. Separately, the [lab's current dissertation summary](https://iwk.mdw.ac.at/?PageId=209), section “Was passiert am Kontaktpunkt?”, explicitly reports extension of the earlier method to finite-width real bows, with reconstructed friction–velocity loops in different regimes. That primary lab account is enough to prohibit claiming finite-width cello inversion as untouched territory; detailed algorithm comparisons require the underlying dissertation/full manuscript.

4. **Bavu, Smith & Wolfe (2005), “Torsional waves in a bowed string,”** [author manuscript](https://arxiv.org/html/physics/0505059). The contact-surface velocity is the transverse velocity plus radius times angular velocity. Simultaneous measurements show the torsional contribution is not generally negligible. Consequently an endpoint reconstruction of transverse motion alone cannot be equated with slip speed under an actual bow.

5. **Mayer & Lampis (2024), “A Versatile Monochord Setup: An Industrial Robotic Arm as Bowing and Plucking Device,”** [institutional technical report](https://pub.mdw.ac.at/pubmdw/publication/80ae2c3f-2d93-41f3-bfe3-0114d35c7001?view=full), DOI 10.21939/iwk-tech-report-1-2024. The apparatus deliberately reduces body-motion complications and uses strain-gauge sensing; approximate rigid supports are much more defensible here than on a complete violin. This does not independently establish a numerical bound on support compliance or sensor response.

6. **Lampis et al. (2023), “Examination of the static and dynamic bridge force components of a bowed string,” POMA 51, 035002**, DOI 10.1121/2.0001755. Its [author-uploaded supporting dataset](https://zenodo.org/records/10619541) was released in 2024 and contains MATLAB code plus static/dynamic bridge measurements. Distinguish the 2023 article from its 2024 data release and from the related JASA meeting abstract. The dataset description verifies purpose and provenance; the complete paper was not obtained from its primary publisher in this audit.

## Consequences for the proposed experiment

The following are our mathematical/experimental-design deductions, not new findings claimed by the cited authors:

- For the ideal rigid nondispersive point-contact model, the delayed difference of one endpoint reaction yields a **transverse point velocity**, with its sign determined by force orientation. Verify signs from the forward simulator and calibration conventions; do not choose signs to maximize apparent sticking.
- Agreement of two endpoint estimates cannot identify their common absolute scale. If both force channels and impedance share a compensating gain, the agreement residual is unchanged while the inferred velocity relative to the measured bow speed changes. Independent tension, length, frequency and calibrated force metadata can constrain that scale. Assuming a stick plateau to estimate impedance is circular when sticking is the quantity being audited.
- Time-delay error is amplified where force changes rapidly. Dispersion, sampling filters, relative channel lag and gain must be represented before disagreement is interpreted as physical failure. Fixed offsets cancel in a delayed difference; drifting offsets do not. DC friction requires an additional offset treatment even when a velocity reconstruction is usable.
- With finite bow width, the two endpoints provide exterior wave information, not arbitrary spatial resolution of distributed hair–string contact. A single binary “all contact sticks” state cannot be certified from a center-point approximation.
- True relative velocity also involves torsion, bow-hair motion and potentially partial contact/lift-off. These quantities require either measurements or explicitly validated dynamical restrictions. Merely bounding transverse reconstruction error is insufficient.
- Two reconstructions agreeing is a necessary model-consistency check, not an independent ground-truth measurement: their errors can share a systematic component. A low residual must not be converted directly into a calibrated probability of correct sticking labels.

## A defensible next step, conditional on available metadata

1. Reproduce the existing ideal identity and dispersion-aware reconstruction as named baselines; use independently parameterized simulated transverse/torsional/finite-width systems to quantify model mismatch.
2. Declare specimen-level nuisance sets from physical metadata and calibration evidence, before evaluating regime outcomes. Avoid per-trial fitting that forces velocity to equal bow velocity.
3. Return a set of compatible contact behaviors and abstain whenever both sticking and slipping remain feasible. Report the proportion of recordings for which the set is informative, as well as counterexamples and bound failures.
4. Present any remaining uncertainty as conditional sensitivity unless nuisance-set coverage has been independently validated. No current source found here establishes calibrated torsional, bow-hair and sensor bounds for every archived trial.

This route is scientifically interesting but presently higher-risk than a transparent audit of operational waveform labels. The literature search is not a proof of absence of other relevant prior work.
