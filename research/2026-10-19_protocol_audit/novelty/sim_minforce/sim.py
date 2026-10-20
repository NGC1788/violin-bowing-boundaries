"""Modal bowed-string simulator for the pre-registered minimum-force ablation (PLAN.md).

Single-polarisation transverse modal string, rigid (pinned) ends, point bow at x = beta*L,
constant normal force F, bow speed ramped linearly 0 -> v at the measured median acceleration,
then held.  Mode n: shape sin(n pi x / L), modal mass mu L / 2, damping ratio zeta_n.

Time stepping (per mode, exact): the friction force f is held constant over each step
(zero-order hold) and every damped oscillator is advanced with its exact discrete transition
matrix (no numerical dispersion, unconditionally stable).  The bow-point velocity at the END of
the step is linear in that step's force:  v = v_free + Y_eff * f,  with
Y_eff = sum_n phi_n^2 * b2_n + dY, where dY is the instantaneous (ZOH) admittance of the modes
that are not carried (truncation correction, see build_modes).  The friction law is solved
implicitly against this line every step (stick if |f_stick| <= mu_s F, else slip on the
friction curve; Friedlander/McIntyre hysteresis rule: an ongoing slip continues while a
same-direction slip solution exists, the larger root on the falling hyperbolic branch).
"""
import json
import math
import os

import numpy as np
from numba import njit, prange

HERE = os.path.dirname(os.path.abspath(__file__))
SD = os.path.join(HERE, "..", "string_damping")

# ---- string A T1 (PLAN.md) ----
T = 145.31            # N
MU = 7.721e-3         # kg/m
EI = 3.032e-4         # N m^2
F0 = 97.8             # Hz
Z = math.sqrt(T * MU)                 # 1.0592 kg/s
CW = math.sqrt(T / MU)
L = CW / (2 * F0)
B_INH = math.pi ** 2 * EI / (T * L ** 2)   # stiff string f_n = n f0 sqrt(1 + B n^2)
MODE_MASS = MU * L / 2

# ---- friction (PLAN.md) ----
DMU = 0.6
MUS = 1.2
MUD = 0.6
MUINF = 0.35
V0 = 0.1

ACC = {0.05: 0.088, 0.1: 0.104, 0.2: 0.417}
BETAS = [0.02, 0.03, 0.05, 0.08, 0.12, 0.2]
BETA_EXTRA = 0.1      # extra run for Q_sim(0.1) and the absolute comparison (not in the slope fit)
R_STEP = 10 ** (3 / 23)                       # plan grid: 24 log-spaced 0.005..5 N
EXT_FORCES = 0.005 * R_STEP ** np.arange(-8, 24)   # same ladder continued 8 steps down (k>=0 = plan grid)
N_EXT_BELOW = 8
FS_BASE = 96000.0
FMAX_FRAC = 0.3066    # carry modes with f_n <= 0.3066 fs (29.4 kHz at 96 kHz, N = 301): Si(N w1 dt) = pi/2,
                      # so the modal ZOH bow admittance alone equals 1/(2Z) (no memoryless residual term)
PLATEAU = 0.5
WIN = (0.2, 0.4)      # label window, s after plateau start


def plan_forces():
    return EXT_FORCES[N_EXT_BELOW:]


# ---- damping ----
def load_damping():
    d = json.load(open(os.path.join(SD, "damping_digitized.json")))
    pts = d["panels"]["A T1"]["points"]
    n = np.array([p["mode"] for p in pts], float)
    z = np.array([p["zeta"] for p in pts], float)
    o = np.argsort(n)
    n, z = n[o], z[o]
    p, a = np.polyfit(np.log(n), np.log(z), 1)
    return n, z, float(a), float(p)


DN, DZ, DA, DP = load_damping()
assert DN[0] == 1
ZETA1 = float(DZ[0])


def zeta_measured(nn):
    nn = np.asarray(nn, float)
    inside = np.exp(np.interp(nn, DN, np.log(DZ)))          # log zeta linear in n between digitized modes
    beyond = np.exp(DA + DP * np.log(nn))                   # power law fitted to all digitized modes
    return np.where(nn <= DN.max(), inside, beyond)


_RES = json.load(open(os.path.join(SD, "results.json")))
S10 = float(_RES["damping"]["A T1"]["S10"])
ZR_EFF = 6 / math.pi * S10
ZR_C0 = math.pi * ZETA1
F_MEAS_ONSET = float(_RES["onset"]["T1"]["A T1"]["F_meas"])
F_MEAS_WINDOW = float(_RES["window"]["T1"]["A T1"]["F_meas"])


def F_S(beta, v, cfg):
    zr = ZR_C0 if cfg == "C0" else ZR_EFF
    return Z * zr * v / (2 * beta ** 2 * DMU)


# ---- modes ----
def mode_freqs(cfg, nn):
    if cfg == "C3":
        return nn * F0 * np.sqrt(1 + B_INH * nn ** 2)
    return nn * F0


TAPER = None   # numerical band-edge regularisation (see RESULT.md); dict or None


def build_modes(cfg, beta, fs=FS_BASE, fmax_frac=FMAX_FRAC, nmodes=None, trunc_corr=False, taper="default"):
    if isinstance(taper, str):
        taper = TAPER
    dt = 1.0 / fs
    nn_all = np.arange(1, 200001, dtype=float)
    f_all = mode_freqs(cfg, nn_all)
    N = int(np.sum(f_all <= fmax_frac * fs)) if nmodes is None else int(nmodes)
    nn = nn_all[:N]
    fn = f_all[:N]
    w = 2 * np.pi * fn
    zeta = ZETA1 / nn if cfg == "C0" else zeta_measured(nn)
    W = np.ones(N)
    if taper is not None and taper["kind"] == "damping":      # extra damping rising ~ s^2 above f1
        sx = np.clip((fn - taper["f1_frac"] * fs) / (fn[-1] - taper["f1_frac"] * fs), 0, 1)
        zeta = zeta + taper["zeta_max"] * sx ** 2
    if taper is not None and taper["kind"] == "coupling":     # raised-cosine roll-off of bow coupling f1 -> f2
        f1, f2 = taper["f1_frac"] * fs, taper["f2_frac"] * fs
        sx = np.clip((fn - f1) / (f2 - f1), 0, 1)
        W = 0.5 * (1 + np.cos(np.pi * sx))
    m = MODE_MASS
    sig = zeta * w
    wd = w * np.sqrt(1 - zeta ** 2)
    E = np.exp(-sig * dt)
    c = np.cos(wd * dt)
    s = np.sin(wd * dt)
    a11 = E * (c + sig * s / wd)
    a12 = E * s / wd
    a21 = -E * w ** 2 * s / wd
    a22 = E * (c - sig * s / wd)
    b1 = (1 - a11) / (w ** 2 * m)
    b2 = a12 / m
    phi = np.sin(np.pi * nn * beta) * np.sqrt(W)
    Ymod = float(np.sum(phi ** 2 * b2))
    # exact ZOH end-of-step admittance of the continuous string = sum over ALL modes.
    # flexible string: analytically sum_n sin^2(n pi beta) sin(n th)/n = pi/4 (th < 2 pi beta) -> 1/(2Z)
    # stiff string: carried modes + numerically summed undamped tail (absolutely convergent).
    if cfg == "C3":
        nt = nn_all[N:]
        wt = 2 * np.pi * f_all[N:]
        Ytail = float(np.sum(np.sin(np.pi * nt * beta) ** 2 * np.sin(wt * dt) / wt) / m)
        Ytarget = Ymod + Ytail
    else:
        Ytarget = 1.0 / (2 * Z)
    # A memoryless residual dY = Ytarget - Ymod was tried first and REJECTED: it acts as a series dashpot
    # (passes DC, dissipates f^2 dY during stick, lets the string creep under a stuck bow at high F).
    dY = (Ytarget - Ymod) if trunc_corr else 0.0
    return dict(cfg=cfg, beta=beta, fs=fs, dt=dt, N=N, W=W, taper=taper, n=nn, f=fn, zeta=zeta, phi=phi,
                a11=a11, a12=a12, a21=a21, a22=a22, b1=b1, b2=b2,
                Ymod=Ymod, Ytarget=Ytarget, dY=dY)


# ---- kernel ----
@njit(cache=True)
def _slip_u(ws, YF, fric, mud, mus, muinf, v0):
    """|slip speed| u>0 solving u + Y F mu(u) = ws (ws = signed free rel. velocity along slip dir)."""
    if ws <= 0.0:
        return -1.0
    if fric == 0:
        u = ws - YF * mud
        return u if u > 0.0 else -1.0
    b = v0 - ws + YF * muinf
    c = v0 * (YF * mus - ws)
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return -1.0
    u = 0.5 * (-b + math.sqrt(disc))   # larger root (sliding branch)
    return u if u > 0.0 else -1.0


@njit(cache=True)
def _mu(u, fric, mud, mus, muinf, v0):
    if fric == 0:
        return mud
    return muinf + (mus - muinf) * v0 / (v0 + u)


@njit(cache=True)
def sim_one(F, phi, a11, a12, a21, a22, b1, b2, dY, dt, vb, acc, nsteps, i0, out,
            fric, mus, mud, muinf, v0):
    N = phi.size
    q = np.zeros(N)
    p = np.zeros(N)
    Y = dY
    for j in range(N):
        Y += phi[j] * phi[j] * b2[j]
    YF = Y * F
    slipping = False
    sdir = 1.0
    nrec = out.size
    for k in range(nsteps):
        t = (k + 1) * dt
        vbow = acc * t
        if vbow > vb:
            vbow = vb
        vfree = 0.0
        for j in range(N):
            vfree += phi[j] * (a21[j] * q[j] + a22[j] * p[j])
        w = vfree - vbow
        f = 0.0
        dv = 0.0
        done = False
        if slipping:
            u = _slip_u(sdir * w, YF, fric, mud, mus, muinf, v0)
            if u > 0.0:
                f = -sdir * F * _mu(u, fric, mud, mus, muinf, v0)
                dv = sdir * u
                done = True
        if not done:
            fst = -w / Y
            if abs(fst) <= mus * F:
                f = fst
                dv = 0.0
                slipping = False
            else:
                sdir = 1.0 if w > 0.0 else -1.0
                u = _slip_u(abs(w), YF, fric, mud, mus, muinf, v0)
                if u < 0.0:
                    u = 0.0
                f = -sdir * F * _mu(u, fric, mud, mus, muinf, v0)
                dv = sdir * u
                slipping = True
        for j in range(N):
            g = phi[j] * f
            qn = a11[j] * q[j] + a12[j] * p[j] + b1[j] * g
            p[j] = a21[j] * q[j] + a22[j] * p[j] + b2[j] * g
            q[j] = qn
        r = k - i0
        if r >= 0 and r < nrec:
            out[r] = dv


@njit(parallel=True, cache=True)
def sim_batch(Fs, phi, a11, a12, a21, a22, b1, b2, dY, dt, vb, acc, nsteps, i0, out,
              fric, mus, mud, muinf, v0):
    for i in prange(Fs.size):
        sim_one(Fs[i], phi, a11, a12, a21, a22, b1, b2, dY, dt, vb, acc, nsteps, i0, out[i],
                fric, mus, mud, muinf, v0)


@njit(cache=True)
def free_decay(q0, phi, a11, a12, a21, a22, out):
    N = phi.size
    q = q0.copy()
    p = np.zeros(N)
    for k in range(out.size):
        y = 0.0
        for j in range(N):
            y += phi[j] * q[j]
        out[k] = y
        for j in range(N):
            qn = a11[j] * q[j] + a12[j] * p[j]
            p[j] = a21[j] * q[j] + a22[j] * p[j]
            q[j] = qn


def simulate(cfg, v, beta, forces, fs=FS_BASE, nmodes=None, fmax_frac=FMAX_FRAC):
    md = build_modes(cfg, beta, fs=fs, fmax_frac=fmax_frac, nmodes=nmodes)
    acc = ACC[v]
    t_pl = v / acc
    i0 = int(round((t_pl + WIN[0]) * fs)) - 1          # record index r <-> time (i0+1+r)/fs
    i1 = int(round((t_pl + WIN[1]) * fs)) - 1
    nsteps = i1          # the label only depends on t <= plateau+0.4 s (causal); last 0.1 s of plateau not needed
    out = np.zeros((len(forces), i1 - i0))
    fric = 1 if cfg in ("C2", "C3") else 0
    sim_batch(np.asarray(forces, float), md["phi"], md["a11"], md["a12"], md["a21"], md["a22"],
              md["b1"], md["b2"], md["dY"], md["dt"], float(v), float(acc), nsteps, i0, out,
              fric, MUS, MUD, MUINF, V0)
    return out, md


# ---- classifier ----
def classify(dv, fs, vb, eps_frac=0.01, f0=F0):
    """Helmholtz label from the bow-point relative velocity in the 0.2-0.4 s window.
    slip <=> |dv| > eps_frac * vb.  Stick->slip transition times t_i; a nominal period is 'one
    stick->slip' when the interval to the next transition lies in [0.5, 1.5) T0 (0 extra, 0 missing).
    frac = #good intervals / max(#intervals, floor(window f0) - 1).  Helmholtz if frac >= 0.9 and
    mean period (t_last - t_first)/(M-1) within +-5 % of 1/f0."""
    thr = eps_frac * vb
    slip = np.abs(dv) > thr
    idx = np.nonzero((~slip[:-1]) & slip[1:])[0] + 1
    T0 = 1.0 / f0
    n_exp = int(math.floor((WIN[1] - WIN[0]) * f0))
    if idx.size < 2:
        return False, dict(n_trans=int(idx.size), frac=0.0, mean_period_rel=None,
                           slip_frac=float(slip.mean()))
    tt = idx / fs
    iv = np.diff(tt)
    good = int(np.sum((iv >= 0.5 * T0) & (iv < 1.5 * T0)))
    frac = good / max(iv.size, n_exp - 1)
    rel = float((tt[-1] - tt[0]) / iv.size * f0)
    ok = bool(frac >= 0.9 and abs(rel - 1) <= 0.05)
    return ok, dict(n_trans=int(idx.size), frac=float(frac), mean_period_rel=rel,
                    slip_frac=float(slip.mean()))


def fmin_K(forces, labels, K=2):
    for i in range(len(forces) - K + 1):
        if all(labels[i:i + K]):
            return float(forces[i]), i
    return None, None
