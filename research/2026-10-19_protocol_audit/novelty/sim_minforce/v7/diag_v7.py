"""POST HOC diagnostic (after v7 showed no Helmholtz for any finite-width bow, even with frequency-independent loss):
how does the failure depend on the number of contact points M and the width W? Labels at beta 0.1 for the W0 string
(frequency-independent loss, smooth friction, ideal start). Claude, 2026-10-21."""
import math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import vsim7 as V7
a, p = V7.V4.powerlaw(); g = np.array([math.exp(-2 * math.pi * math.exp(a))])
for W, M in ((0, 1), (1, 2), (2, 2), (5, 2), (10, 2), (10, 3), (10, 4), (2, 8), (10, 8)):
    fm, pat = V7.fmin(0.1, W, M, g, 0.0)
    print(f"W {W:>2} mm M {M}: F_min {fm}  {pat}", flush=True)
