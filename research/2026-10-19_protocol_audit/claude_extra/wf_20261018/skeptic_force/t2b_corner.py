"""T2b: corner sharpness as a contact detector (load-cell independent). A bowed string in contact keeps a sharp
Helmholtz corner (the bow re-sharpens it every period); a free string loses its high partials ~10x faster than its
fundamental (Matusiak 2024 damping r1 = 0.005 m^2/s => mode-10 decay ~10/s vs mode-1 ~1/s). So after real contact
loss, log(edge) should fall much faster than log(p2p). Event-aligned at the block where the force reading reaches its
floor (same definition as t2_contact.py), constant-speed blocks only; also the same quantities after the bow stops.
Usage: python -I t2b_corner.py
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import Spec, med
from t2_contact import floor_time, amp
import numpy as np

for A in ("A2", "A1"):
    S = Spec(A)
    print(f"\n=== {A}")
    e0 = np.nanmedian([np.nanmedian(S.bridge_edge[i, :S.onset[i]]) for i in range(S.n)])
    print(f"  pre-onset edge (noise) median {e0:.1f} N/s")
    for s in ("v1", "v2", "v3"):
        ev = [(i, floor_time(S, i)) for i in np.flatnonzero(S.valid & (S.folder == s) & (S.rank <= 30))]
        ev = [(i, t) for i, t in ev if t > 0]
        lags = np.arange(-8, 13, 2)
        Mp = np.full((len(ev), lags.size), np.nan); Me = Mp.copy()
        stopP, stopE, enoise = [], [], []
        for k, (i, tf) in enumerate(ev):
            lp = np.log(amp(S, i))
            en = np.nanmedian(S.bridge_edge[i, :S.onset[i]])
            enoise.append(en)
            le = np.log(np.sqrt(np.clip(S.bridge_edge[i] ** 2 - en ** 2, 1e-3, None)))
            for q, L in enumerate(lags):
                b = tf + L
                if 0 <= b < S.nb and S.plateau[i, b]:
                    Mp[k, q] = lp[b] - lp[tf]; Me[k, q] = le[b] - le[tf]
            lm = S.last_move[i]
            nv = S.nvalid[i]
            if nv - lm >= 4:
                stopP.append(lp[nv - 1] - lp[lm]); stopE.append(le[nv - 1] - le[lm])
        print(f"  {s}: n {len(ev)}  lag(blocks): " + " ".join(f"{L:+d}" for L in lags))
        print("     dlog p2p : " + " ".join(f"{med(Mp[:, q]):+.2f}" for q in range(lags.size)))
        print("     dlog edge: " + " ".join(f"{med(Me[:, q]):+.2f}" for q in range(lags.size)))
        print(f"     from last moving block to end of record (bow stopped): dlog p2p {med(stopP):+.2f}  dlog edge {med(stopE):+.2f}")
