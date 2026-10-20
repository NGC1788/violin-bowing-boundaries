"""Independent verifier code (not the builder's).  Two independent solvers:
  WG  : digital waveguide (McIntyre-Woodhouse), integer delays, D = fs/f0 samples per round trip,
        frequency-independent loss (zeta_n = zeta1/n exactly), no band limit -> corner stays 1 sample sharp.
  MOD : own modal solver, coefficients from scipy expm of the augmented state matrix (not the builder's closed form).
Friction: Coulomb mus 1.2 / mud 0.6 (C0, C1), Friedlander/McIntyre hysteresis (default) or stick-if-possible.
Classifier: own implementation of the PLAN rule (nominal-period bins, exactly one stick->slip per bin)."""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(HERE, "nbcache"))
sys.dont_write_bytecode = True
import numpy as np
from numba import njit, prange
from scipy.linalg import expm

T, MU, F0 = 145.31, 7.721e-3, 97.8
Z = math.sqrt(T * MU)
C = math.sqrt(T / MU)
L = C / (2 * F0)
M = MU * L / 2
MUS, MUD = 1.2, 0.6
ACC = {0.1: 0.104}
_d = json.load(open(os.path.join(HERE, "..", "..", "string_damping", "damping_digitized.json")))
_pts = sorted(_d["panels"]["A T1"]["points"], key=lambda p: p["mode"])
DN = np.array([p["mode"] for p in _pts], float)
DZ = np.array([p["zeta"] for p in _pts], float)
ZETA1 = float(DZ[0])
_P, _A = np.polyfit(np.log(DN), np.log(DZ), 1)


def zeta_c1(n):
    n = np.asarray(n, float)
    return np.where(n <= DN.max(), np.exp(np.interp(n, DN, np.log(DZ))), np.exp(_A + _P * np.log(n)))


def F_S(beta, v=0.1):
    return Z * (math.pi * ZETA1) * v / (2 * beta ** 2 * (MUS - MUD))


R_STEP = 10 ** (3 / 23)
EXT = 0.005 * R_STEP ** np.arange(-8, 24)


# ---------------- friction (own) ----------------
@njit(cache=True)
def fric(w, Y, F, slipping, s, hyst):
    if slipping and hyst:
        u = s * w - Y * F * MUD
        if u > 0.0:
            return -s * MUD * F, s * u, True, s
    fst = -w / Y
    if abs(fst) <= MUS * F:
        return fst, 0.0, False, s
    s2 = 1.0 if w > 0.0 else -1.0
    u = abs(w) - Y * F * MUD
    return -s2 * MUD * F, s2 * u, True, s2


# ---------------- waveguide ----------------
@njit(cache=True)
def wg_one(F, db2, dn2, gb, gn, fs, vb, acc, nsteps, i0, ob0, on0, odv, of, hyst):
    ob = ob0.copy(); on = on0.copy(); Y = 1.0 / (2.0 * Z)
    slipping = False; s = 1.0
    for k in range(nsteps):
        vbow = acc * (k + 1) / fs
        if vbow > vb:
            vbow = vb
        ib = k % db2; inn = k % dn2
        vib = -gb * ob[ib]; vin = -gn * on[inn]
        w = vib + vin - vbow
        f, dv, slipping, s = fric(w, Y, F, slipping, s, hyst)
        ob[ib] = vin + Y * f
        on[inn] = vib + Y * f
        r = k - i0
        if r >= 0 and r < odv.size:
            odv[r] = dv; of[r] = f


@njit(parallel=True, cache=True)
def wg_batch(Fs, db2, dn2, gb, gn, fs, vb, acc, nsteps, i0, ob0s, on0s, odv, of, hyst):
    for i in prange(Fs.size):
        wg_one(Fs[i], db2, dn2, gb, gn, fs, vb, acc, nsteps, i0, ob0s[i], on0s[i], odv[i], of[i], hyst)


def wg_run(beta, Fs, D=1000, loss="dist", start="ramp", hyst=True, vb=0.1, win=(0.2, 0.4)):
    fs = F0 * D
    db2 = int(round(beta * D)); dn2 = D - db2
    assert abs(db2 - beta * D) < 1e-9
    sig = ZETA1 * 2 * math.pi * F0
    if loss == "dist":
        gb, gn = math.exp(-sig * db2 / fs), math.exp(-sig * dn2 / fs)
    elif loss == "bridge":  # all loss at the bridge (Schelleng's resistive termination)
        gb, gn = math.exp(-sig * D / fs), 1.0
    else:
        gb, gn = 1.0, 1.0
    Fs = np.asarray(Fs, float)
    ob0 = np.zeros((Fs.size, db2)); on0 = np.zeros((Fs.size, dn2))
    if start == "ideal":   # exact Helmholtz travelling waves, mid-stick at t = 0, friction mud F
        acc = 1e12; tpl = 0.0
        t0 = -D // 2
        for i, F in enumerate(Fs):
            f0 = MUD * F
            a = lambda j: vb / (beta * D) * ((j - t0) % D) - vb / (2 * beta)
            jb = np.arange(-db2, 0); jn = np.arange(-dn2, 0)
            ob0[i, jb % db2] = a(jb) + f0 / (4 * Z)
            on0[i, jn % dn2] = -a(jn - db2) + f0 / (4 * Z)
    else:
        acc = ACC[vb]; tpl = vb / acc
    i0 = int(round((tpl + win[0]) * fs)); i1 = int(round((tpl + win[1]) * fs))
    odv = np.zeros((Fs.size, i1 - i0)); of = np.zeros_like(odv)
    wg_batch(Fs, db2, dn2, gb, gn, fs, vb, acc, i1, i0, ob0, on0, odv, of, hyst)
    return odv, of, fs


# ---------------- own modal ----------------
def modal_coeffs(cfg, beta, fs, N):
    n = np.arange(1, N + 1, dtype=float)
    w = 2 * np.pi * F0 * n
    z = ZETA1 / n if cfg == "C0" else zeta_c1(n)
    dt = 1 / fs
    A = np.zeros((N, 2, 2)); Bv = np.zeros((N, 2))
    for j in range(N):
        Mx = np.array([[0, 1, 0], [-w[j] ** 2, -2 * z[j] * w[j], 1 / M], [0, 0, 0]]) * dt
        E = expm(Mx)
        A[j] = E[:2, :2]; Bv[j] = E[:2, 2]
    phi = np.sin(np.pi * n * beta)
    Y = float(np.sum(phi ** 2 * Bv[:, 1]))
    return dict(n=n, w=w, z=z, A=A, B=Bv, phi=phi, Y=Y, fs=fs)


@njit(cache=True)
def mod_one(F, phi, a11, a12, a21, a22, b1, b2, Y, dt, vb, acc, nsteps, i0, odv, of, hyst):
    N = phi.size; q = np.zeros(N); p = np.zeros(N)
    slipping = False; s = 1.0
    for k in range(nsteps):
        vbow = acc * (k + 1) * dt
        if vbow > vb:
            vbow = vb
        vf = 0.0
        for j in range(N):
            vf += phi[j] * (a21[j] * q[j] + a22[j] * p[j])
        f, dv, slipping, s = fric(vf - vbow, Y, F, slipping, s, hyst)
        for j in range(N):
            g = phi[j] * f
            qn = a11[j] * q[j] + a12[j] * p[j] + b1[j] * g
            p[j] = a21[j] * q[j] + a22[j] * p[j] + b2[j] * g
            q[j] = qn
        r = k - i0
        if r >= 0 and r < odv.size:
            odv[r] = dv; of[r] = f


@njit(parallel=True, cache=True)
def mod_batch(Fs, phi, a11, a12, a21, a22, b1, b2, Y, dt, vb, acc, nsteps, i0, odv, of, hyst):
    for i in prange(Fs.size):
        mod_one(Fs[i], phi, a11, a12, a21, a22, b1, b2, Y, dt, vb, acc, nsteps, i0, odv[i], of[i], hyst)


def mod_run(cfg, beta, Fs, fs=96000.0, N=300, hyst=True, vb=0.1):
    md = modal_coeffs(cfg, beta, fs, N)
    acc = ACC[vb]; tpl = vb / acc
    i0 = int(round((tpl + 0.2) * fs)) - 1; i1 = int(round((tpl + 0.4) * fs)) - 1
    Fs = np.asarray(Fs, float)
    odv = np.zeros((Fs.size, i1 - i0)); of = np.zeros_like(odv)
    A, B = md["A"], md["B"]
    mod_batch(Fs, md["phi"], A[:, 0, 0].copy(), A[:, 0, 1].copy(), A[:, 1, 0].copy(), A[:, 1, 1].copy(),
              B[:, 0].copy(), B[:, 1].copy(), md["Y"], 1 / fs, vb, acc, i1, i0, odv, of, hyst)
    return odv, of, md


# ---------------- own classifier (PLAN rule) ----------------
def label(dv, fs, vb=0.1, eps=0.01, min_slip=1):
    slip = np.abs(dv) > eps * vb
    if min_slip > 1:   # flicker-tolerant variant: drop slip runs shorter than min_slip samples
        e = np.flatnonzero(np.diff(np.r_[0, slip.astype(int), 0]))
        for a, b in zip(e[0::2], e[1::2]):
            if b - a < min_slip:
                slip[a:b] = False
    tr = np.flatnonzero(~slip[:-1] & slip[1:]) + 1
    T0 = fs / F0
    info = dict(n_trans=int(tr.size), slip_frac=float(slip.mean()))
    if tr.size < 3:
        return False, info
    P = (tr[-1] - tr[0]) / (tr.size - 1) / T0
    start = tr[0] - T0 / 2
    nb = int((dv.size - start) // T0)
    cnt = np.histogram(tr, start + T0 * np.arange(nb + 1))[0]
    frac = float(np.mean(cnt == 1))
    info.update(frac=frac, relP=float(P))
    return bool(frac >= 0.9 and abs(P - 1) <= 0.05), info


def fminK(Fs, labs, K=2):
    for i in range(len(Fs) - K + 1):
        if all(labs[i:i + K]):
            return float(Fs[i])
    return None


def slope(bs, Fm):
    x = [math.log(b) for b, f in zip(bs, Fm) if f is not None]
    y = [math.log(f) for f in Fm if f is not None]
    return float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 else None
