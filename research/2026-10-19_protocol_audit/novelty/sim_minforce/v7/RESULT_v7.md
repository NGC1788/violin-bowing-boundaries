# Simulation v7 — result (run once; Claude, 2026-10-21 ≈12:17 KST)
- Code gates passed: the chain code with one contact point reproduces v6 exactly (P0 = H0m, P1 = H1m at every β).
- **W1 (10 mm, 8 contacts, measured damping): no Helmholtz run at any β → reading "undefined".** The same for W0
  (frequency-independent loss), 5 mm / 5 contacts, Coulomb friction and the ramp start.
- Post hoc diagnostic (`diag_v7.py`, β 0.1, frequency-independent loss): **two contacts only 2 samples (0.7 mm) apart
  already destroy Helmholtz motion**, while one contact gives a clean band. A 0.7 mm bow is physically a point bow, so
  this is a model artifact: rigid sticking contacts with lossless string between them trap high-frequency waves that
  trigger spurious slips (no damping or bow-hair compliance inside the contact patch). v7 says nothing about the
  physical effect of bow width. A usable finite-width model would need bow-hair compliance and loss in the patch
  (Pitteroff & Woodhouse type); not attempted.
