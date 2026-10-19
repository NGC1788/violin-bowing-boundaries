"""T5b: is the bridge channel sensitive to the bow's normal force when nothing moves?  Pre-onset (bow placed at
F_on = 0.1 ... 4.4 N, not moving) bridge and nut DC regressed on F_on.  If the bridge channel had vertical cross-talk
or the hair pushed the string sideways (tilted bow), the pre-onset bridge DC would scale with F_on (unless the channel
were re-zeroed at every recording start).  Also: does the stroke-to-stroke (detrended) bridge DC change follow the
detrended force change (common cause beyond the design)?  And the nut DC change at low force.
Usage: python -I t5b_bridge_static.py"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import Spec, med
from t3b_carry import hf_resid
import numpy as np

for A in ("A2", "A1"):
    S = Spec(A)
    print(f"\n=== {A}")
    for s in ("v1", "v2", "v3"):
        m = np.flatnonzero(S.valid & (S.folder == s))
        bpre = np.array([np.nanmedian(S.bridge_dc[i, :S.onset[i]]) for i in m])
        npre = np.array([np.nanmedian(S.nut_dc[i, :S.onset[i]]) for i in m])
        f = S.f_on[m]
        pb = np.polyfit(f, bpre, 1); pn = np.polyfit(f, npre, 1)
        # within-column (detrended by beta column mean) slope
        bb = bpre.copy(); ff = f.copy()
        for c in range(40):
            q = S.col[m] == c
            bb[q] -= bb[q].mean(); ff[q] -= ff[q].mean()
        sw = np.sum(bb * ff) / np.sum(ff * ff)
        print(f"  {s}: pre-onset bridge DC vs F_on slope {1000*pb[0]:+.2f} mN/N (within-column {1000*sw:+.2f}), corr {np.corrcoef(f,bpre)[0,1]:+.2f}; "
              f"nut slope {1000*pn[0]:+.2f} mN/N corr {np.corrcoef(f,npre)[0,1]:+.2f}")
        # low-force strokes: changes 2 -> 15 cm
        q = np.flatnonzero(S.valid & (S.folder == s))
        q = q[np.argsort(S.trial[q])]
        dB = np.full(q.size, np.nan); dF = np.full(q.size, np.nan); dN = np.full(q.size, np.nan)
        for k, i in enumerate(q):
            j0, j1 = S.at_travel(i, 0.02), S.at_travel(i, 0.15)
            if j0 >= 0 and j1 >= 0:
                dB[k] = S.bridge_dc[i, j1] - S.bridge_dc[i, j0]
                dF[k] = S.force_dc[i, j1] - S.force_dc[i, j0]
                dN[k] = S.nut_dc[i, j1] - S.nut_dc[i, j0]
        rB = hf_resid(dB, S.rank[q], S.col[q]); rF = hf_resid(dF, S.rank[q], S.col[q])
        low = S.rank[q] <= 20
        ok = low & np.isfinite(rB) & np.isfinite(rF)
        print(f"     ranks 1-20: nut DC change 2->15 cm median {1000*med(dN[low]):+.1f} mN (bridge {1000*med(dB[low]):+.0f}, force {1000*med(dF[low]):+.0f}); "
              f"detrended per-stroke corr(bridge change, force change) {np.corrcoef(rB[ok], rF[ok])[0,1]:+.2f} (n {ok.sum()})")
        hi = S.rank[q] >= 41
        ok2 = hi & np.isfinite(rB) & np.isfinite(rF)
        print(f"     ranks 41-50: bridge change {1000*med(dB[hi]):+.0f} mN, force change {1000*med(dF[hi]):+.0f} mN; detrended corr {np.corrcoef(rB[ok2], rF[ok2])[0,1]:+.2f}")
