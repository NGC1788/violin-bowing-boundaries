# Data provenance and scope

The measured data are derived from A. Lampis's public **A collection of bowed responses of cello strings**, IWK / University of Music and Performing Arts Vienna, Zenodo parts 1–4 and 6, CC BY 4.0. Record identities and original cache hashes are preserved in `input_joint.json.gz` and `input_audit.json`; attribution and version details are also in `../coherent_fragility/DATA.md`.

This artifact contains transformed numerical features and force-ordered observations, not audio archives or paper PDFs. It retains 13 diagrams / 39 speed folders / 1,560 beta columns / 78,000 windows. The fixed law-comparison population is the same 1,132 originally finite-positive K=2 columns (56,600 windows) as the earlier exact-edit audit. All measurements are robotic cello-string data. No violin recording is claimed.

`export_inputs.py` reads existing cached feature tables, reproduces the original waveform classifier, and exports its remaining physical gate with periodicity and second-flyback ratio. Missing feature values are locked to original labels. No originally positive observation needed this lock. The baseline pair (.8,.5) reproduces all 78,000 labels. Old single-feature guards are insufficient for the joint rule: 105 observations become admissible only when both thresholds are relaxed.

`prior_input_features.json.gz` and `prior_arbitrary_reference.json` are byte-identical copies from the scalar audit, used for independent identity/slice checks and lower-bound comparison. Original force measurements, labels and earlier results are unchanged. Hypothetical reclassifications are counterfactual analysis outputs and are not corrected labels.

All analyses are retrospective on already exposed data. The local plan commit `406d44f` predates the new joint costs but is not an independent confirmation or public preregistration. The fixed-subset obstruction analysis was added after seeing the joint results and is separately labelled post hoc. Cell counts are computational arrangements, not independent experiments, distinct label-vector counts, or probabilities over classifier choices.

`reproduce.py` uses only these compact artifacts; no raw data export, network, GPU or installation is performed. Re-running the exporter requires the original local feature-cache environment, whereas reproducing the scientific outputs does not.
