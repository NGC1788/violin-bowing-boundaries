"""T2 contact test, independent of the load cell. When the force reading reaches its floor, has the hair really left
the string? If yes, the string is no longer bowed: its bridge-force amplitude must start a free decay at that moment,
and the later deceleration/stop of the bow must leave no mark on the string. If the true force were still ~0.1-0.2 N
(reading wrong), the string stays driven: constant amplitude at constant bow speed, amplitude following the bow
speed during the deceleration.
Amplitude = sqrt(p2p^2 - p2p_noise^2) of the bridge channel (5-95 % range per 50 ms block).
Usage: python -I t2_contact.py
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import Spec, med
import numpy as np

FLOOR_THR = 0.01   # reading level regarded as 'at floor' (sessions' floors are -0.02..+0.01 N)


def amp(S, i):
    n0 = np.nanmedian(S.bridge_p2p[i, :S.onset[i]])
    p = S.bridge_p2p[i]
    return np.sqrt(np.clip(p * p - n0 * n0, 1e-6, None))


def floor_time(S, i):
    """first plateau block from which the 3-block running median reading stays < FLOOR_THR until the bow decelerates."""
    pl = np.flatnonzero(S.plateau[i])
    if pl.size < 10:
        return -1
    f = S.force_dc[i]
    sm = np.array([np.nanmedian(f[max(b - 1, 0):b + 2]) for b in range(S.nb)])
    end = pl[-1]
    for b in pl[2:]:
        if np.all(sm[b:end + 1] < FLOOR_THR) and end - b >= 4:
            return int(b)
    return -1


if __name__ != "__main__":
    pass
else:
  for A in ("A2", "A1"):
    S = Spec(A)
    print(f"\n=== {A}")
    for s in ("v1", "v2", "v3"):
        cand = np.flatnonzero(S.valid & (S.folder == s) & (S.rank <= 30))
        ev, ctrl = [], []
        for i in cand:
            tf = floor_time(S, i)
            a = amp(S, i)
            pl = np.flatnonzero(S.plateau[i])
            if tf > 0:
                ev.append((i, tf))
        if not ev:
            print(f"  {s}: no floor strokes")
            continue
        # event-aligned log-amplitude, constant-speed blocks only
        lags = np.arange(-10, 13)
        M = np.full((len(ev), lags.size), np.nan)
        for k, (i, tf) in enumerate(ev):
            a = np.log(amp(S, i))
            for q, L in enumerate(lags):
                b = tf + L
                if 0 <= b < S.nb and S.plateau[i, b]:
                    M[k, q] = a[b] - a[tf]
        # decay rates: pre-floor plateau, floor plateau, deceleration, post-stop (per s)
        def rate(a, b0, b1):
            idx = np.arange(b0, b1)
            idx = idx[(idx >= 0) & (idx < S.nb)]
            y = a[idx]
            ok = np.isfinite(y)
            if ok.sum() < 3:
                return np.nan
            return np.polyfit(idx[ok] * S.dt, y[ok], 1)[0]
        pre, flo, dec, post, spd_ratio, amp_ratio, driven_ratio = [], [], [], [], [], [], []
        for i, tf in ev:
            a = np.log(amp(S, i))
            pl = np.flatnonzero(S.plateau[i])
            pre.append(rate(a, max(pl[0] + 3, tf - 8), tf))
            flo.append(rate(a, tf, pl[-1] + 1))
            lm = S.last_move[i]
            dec.append(rate(a, pl[-1] + 1, lm + 1))
            post.append(rate(a, lm + 1, S.nvalid[i]))
            # deceleration: amplitude ratio vs bow-speed ratio at the block where speed falls to 30 %
            sp = S.speed[i]
            jj = np.flatnonzero((np.arange(S.nb) > pl[-1]) & (sp < 0.3 * np.nanmax(sp)))
            if jj.size:
                j = jj[0]
                spd_ratio.append(sp[j] / np.nanmax(sp))
                amp_ratio.append(np.exp(a[j] - a[pl[-1]]))
                # free-decay prediction from the floor-plateau decay rate
                driven_ratio.append(np.exp(flo[-1] * (j - pl[-1]) * S.dt))
        print(f"  {s}: floor strokes {len(ev)} (rank<=30); beta median {med(S.beta[[e[0] for e in ev]]):.3f}")
        print("    event-aligned median log-amplitude change vs lag (blocks of 50 ms; 0 = reading reaches floor):")
        print("     " + " ".join(f"{L:+d}:{med(M[:, q]):+.2f}" for q, L in enumerate(lags)))
        print(f"    log-amplitude rate (1/s), medians: pre-floor plateau {med(pre):+.2f} | floor plateau {med(flo):+.2f} | "
              f"deceleration {med(dec):+.2f} | after stop {med(post):+.2f}")
        print(f"    at 30 % bow speed in the deceleration: amplitude ratio {med(amp_ratio):.2f} (driven-Helmholtz expectation "
              f"~ speed ratio {med(spd_ratio):.2f}; free-decay continuation {med(driven_ratio):.2f})")
