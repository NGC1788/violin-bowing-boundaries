"""Per-stroke contact-state timeline for one config/speed (diagnostic; Claude 2026-10-18).
Usage: python -B diag_strokes.py CONFIG SPEED [--attack iwk|ramp] [--every 2]
Prints, per force: ramp length, then per 0.1 s window from plateau start (and 2 windows before it):
'H' Helmholtz, 'm' multiple slips (>1.3 onsets/period), 's' stuck/no slip, '.' other; then onsets/period and
slip fraction of the 0-0.2 s window."""
import sys
import numpy as np
import experiment as X
import hairsim as H

name, v = sys.argv[1], float(sys.argv[2])
attack = sys.argv[sys.argv.index("--attack") + 1] if "--attack" in sys.argv else "iwk"
every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 2
beta = 0.1
fp, kw = X.config(name)
pv, pf, starts, total = X.profiles(v, beta, attack, False)
out, diag = H.simulate(X.STRING, fp, [beta] * len(X.FORCES), pv, pf, X.PR, X.RATE, 0.0, int(total * X.RATE), 2, 0, **kw)
fl = diag["slip_flags"]
win = int(0.1 * X.RATE)
def sym(s):
    s = s.astype(np.int8)
    st = int(np.sum((s[1:] == 1) & (s[:-1] == 0)))
    per = st / (len(s) / X.RATE * X.F0)
    if X.contact_label(s, beta):
        return "H"
    if st == 0:
        return "s" if s.mean() < 0.5 else "S"
    return "m" if per > 1.3 else "."
for k in range(0, len(X.FORCES), every):
    a = int(starts[k] * X.RATE)
    pre = "".join(sym(fl[a + i * win:a + (i + 1) * win, k]) if a + i * win >= 0 else " " for i in (-3, -2, -1))
    post = "".join(sym(fl[a + i * win:a + (i + 1) * win, k]) for i in range(15))
    s = fl[a:a + 2 * win, k].astype(np.int8)
    st = int(np.sum((s[1:] == 1) & (s[:-1] == 0)))
    xw = out[a // 2:(a + 2 * win) // 2, k].astype(float)
    wl = X.wave_label(xw)
    print(f"F {X.FORCES[k]:6.3f} ramp {starts[k]:.3f}s  pre[{pre}] |{post}|  0-0.2s: onsets/per {st / (0.2 * X.F0):.2f} slipfrac {s.mean():.3f}"
          f"  wave {'H' if wl else '-'} std {xw.std():.3f} N")
