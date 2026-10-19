"""T2c: does the force channel still 'feel' the string when its DC reading is at the floor? coh = correlation of
the force-channel fluctuation with the bridge fluctuation inside each 50 ms block (blocks npz); force_ac = rms of
the force channel above ~200 Hz. Event-aligned at the floor block (t2_contact.floor_time). Usage: python -I t2c_coh.py"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import Spec, med
from t2_contact import floor_time
import numpy as np

for A in ("A2", "A1"):
    S = Spec(A)
    print(f"\n=== {A}")
    for s in ("v1", "v2", "v3"):
        ev = [(i, floor_time(S, i)) for i in np.flatnonzero(S.valid & (S.folder == s) & (S.rank <= 30))]
        ev = [(i, t) for i, t in ev if t > 0]
        lags = np.arange(-8, 13, 2)
        C = np.full((len(ev), lags.size), np.nan); Fa = C.copy(); Fl = C.copy()
        pre_c = []
        for k, (i, tf) in enumerate(ev):
            pre_c.append(np.nanmedian(np.abs(S.coh[i, :S.onset[i]])))
            for q, L in enumerate(lags):
                b = tf + L
                if 0 <= b < S.nb and S.plateau[i, b]:
                    C[k, q] = abs(S.coh[i, b]); Fa[k, q] = S.force_ac[i, b]; Fl[k, q] = S.force_ac_lo[i, b]
        print(f"  {s}: n {len(ev)}  pre-onset |coh| {med(pre_c):.3f}; lags " + " ".join(f"{L:+d}" for L in lags))
        print("     |coh|      : " + " ".join(f"{med(C[:, q]):.3f}" for q in range(lags.size)))
        print("     force_ac mN: " + " ".join(f"{1000*med(Fa[:, q]):.2f}" for q in range(lags.size)))
        print("     force_ac_lo: " + " ".join(f"{1000*med(Fl[:, q]):.2f}" for q in range(lags.size)))
    # reference: |coh| and force_ac vs reading level on plateau blocks, all ranks
    print("  plateau blocks, all strokes: median |coh| and force_ac (mN) by force reading bin")
    for s in ("v1", "v2", "v3"):
        m = S.valid[:, None] & (S.folder == s)[:, None] & S.plateau
        f = S.force_dc[m]; c = np.abs(S.coh[m]); fa = S.force_ac[m]
        out = []
        for lo, hi in ((-1, 0.0), (0.0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1), (1, 5)):
            mm = (f >= lo) & (f < hi)
            if mm.sum() > 30:
                out.append(f"[{lo},{hi}) {med(c[mm]):.3f}/{1000*med(fa[mm]):.2f}")
        print(f"   {s}: " + "  ".join(out))
