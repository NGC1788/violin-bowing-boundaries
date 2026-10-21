# Reproduce the complete MLE bridge audit

No audio archives, network, GPU, server or original feature caches are required. This directory contains the compact immutable inputs and all code. The measured data are the public robotic cello-string recordings attributed in DATA.md. Do not describe this as new violin measurements.

From this directory, use the project's existing Python environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  ~/violin-bowing-boundaries/.venv-mac/bin/python reproduce.py
```

Or use another Python environment with the packages in `requirements.txt` already installed. The runner never installs anything. It creates its own temporary copy, reruns all production fits, independent likelihood checks, family joins, and scientific figures, and removes only that temporary copy on exit. Original inputs/results remain unchanged. Run logs and the final report go to `reproduction/`.

The ten stages are production refits, coherent-rule replay, theoretical counterexamples, synthetic numerical oracle, independent full/selected refits, independent comparison including blocks/rules, all-control checks, all-certificate-family join, independent family audit, and figures. At most two production workers are used, with BLAS threads limited to one.

Validated in the recorded local environment: 48.19 s, ten stages PASS; 908 production arrays / 673,916 elements and four JSON outputs identical; another eight JSON outputs / 228,770 numerical values agree at the runner's declared tolerance. The exact JSON equality in the production runner is deliberately strict for this same-environment check; a different numerical library may require investigating harmless last-bit differences rather than assuming identical files.

`production/README.md` documents individual production steps and fit diagnostics. `verification/README.md` documents the independent implementation. `theory/family.json` stores all 7,742 minimum certificates, including unsuccessful and adverse effects. Family selection and deletion ranks are descriptive, not calibrated tests. Exhaustive enumeration does not certify a floating-point optimizer's exact real-arithmetic objective gap.

Plans: PLAN.md was committed locally as `aed035e` before deletion scoring. FAMILY_EXTENSION.md was committed as `45014c5` after observing the designated coherent replays and one selected case, before family scoring. Both concern already exposed data, not independent confirmation or public preregistration.
