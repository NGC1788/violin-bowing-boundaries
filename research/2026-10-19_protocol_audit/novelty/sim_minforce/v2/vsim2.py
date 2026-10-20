"""Simulation v2 (PLAN_v2.md): maintenance test in a corner-exact digital waveguide with all loss lumped in the nut-side
loop, either frequency-independent (C0n) or the measured frequency-dependent modal damping as a linear-phase FIR (C1n).
Built on the verifier's waveguide (../verify/vsim.py: friction rule, ideal start, classifier). Claude, 2026-10-21."""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(HERE, "nbcache"))
sys.path.insert(0, os.path.join(HERE, "..", "verify"))
import numpy as np
from numba import njit, prange
import vsim as VS  # verifier's module: constants, zeta_c1, F_S, fric, label, fminK, slope, EXT

Z, F0, MUD, MUS = VS.Z, VS.F0, VS.MUD, VS.MUS
F_S_C1 = 0.0199  # PLAN_v2: F_S(C1) at beta 0.1, v 0.1 (string_damping results.json)


def zeta_flat(n):
    n = np.asarray(n, float)
    return np.where(n <= VS.DN.max(), np.exp(np.interp(n, VS.DN, np.log(VS.DZ))), VS.DZ[-1])


def design_fir(D, K, zfun):
    """Linear-phase FIR (length 2K+1, delay K) whose magnitude at f = n f0 is exp(-2 pi n zeta_n) (per period)."""
    M = 1 << 16
    f = np.arange(M // 2 + 1) / M                      # cycles/sample
    n = f * D                                          # harmonic number (f0 = fs/D)
    A = np.exp(-2 * np.pi * n * zfun(np.maximum(n, 1e-9)))
    A[0] = 1.0
    h = np.fft.irfft(A, M)                             # zero-phase, symmetric around 0
    h = np.roll(h, K)[:2 * K + 1] * np.hanning(2 * K + 3)[1:-1]
    # realised magnitude at harmonics 1..100
    nn = np.arange(1, 101); w = 2 * np.pi * nn / D
    H = np.abs(np.exp(-1j * np.outer(w, np.arange(2 * K + 1) - K)) @ h)
    tgt = np.exp(-2 * np.pi * nn * zfun(nn))
    err = np.abs(np.log(H / tgt))
    return h, dict(max_log_err_n_le_100=float(err.max()), max_log_err_n_le_30=float(err[:30].max()), dc_gain=float(h.sum()))


@njit(cache=True)
def run_one(F, db2, dn2, h, K, vb, nsteps, i0, ob0, xh0, odv, hyst):
    ob = ob0.copy(); Y = 1.0 / (2.0 * Z)
    Lx = xh0.size; x = xh0.copy()          # circular history of waves entering the nut loop
    ldel = dn2 - K
    slipping = False; s = 1.0
    for k in range(nsteps):
        ib = k % db2
        vib = -ob[ib]
        # returning nut wave: FIR over x delayed by ldel; x index of time t is t mod Lx (t may be negative)
        acc = 0.0
        for j in range(2 * K + 1):
            t = k - ldel - j
            acc += h[j] * x[t % Lx]
        vin = -acc
        w = vib + vin - vb
        f, dv, slipping, s = VS.fric(w, Y, F, slipping, s, hyst)
        ob[ib] = vin + Y * f
        x[k % Lx] = vib + Y * f
        r = k - i0
        if r >= 0 and r < odv.size:
            odv[r] = dv


@njit(parallel=True, cache=True)
def run_batch(Fs, db2, dn2, h, K, vb, nsteps, i0, ob0s, xh0s, odv, hyst):
    for i in prange(Fs.size):
        run_one(Fs[i], db2, dn2, h, K, vb, nsteps, i0, ob0s[i], xh0s[i], odv[i], hyst)


def sim(beta, Fs, h, K, D=1000, vb=0.1, T=0.5, win=(0.3, 0.5), hyst=False):
    fs = F0 * D; db2 = int(round(beta * D)); dn2 = D - db2
    assert abs(db2 - beta * D) < 1e-9 and dn2 > K
    Fs = np.asarray(Fs, float)
    Lx = dn2 + 2 * K + 2
    ob0 = np.zeros((Fs.size, db2)); xh0 = np.zeros((Fs.size, Lx))
    t0 = -D // 2
    a = lambda j: vb / (beta * D) * ((j - t0) % D) - vb / (2 * beta)
    for i, F in enumerate(Fs):
        f0 = MUD * F
        jb = np.arange(-db2, 0); ob0[i, jb % db2] = a(jb) + f0 / (4 * Z)
        jn = np.arange(-Lx, 0); xh0[i, jn % Lx] = -a(jn - db2) + f0 / (4 * Z)
    i0 = int(round(win[0] * fs)); i1 = int(round(win[1] * fs))
    odv = np.zeros((Fs.size, i1 - i0))
    run_batch(Fs, db2, dn2, np.asarray(h, float), K, vb, i1, i0, ob0, xh0, odv, hyst)
    labs = [VS.label(odv[i], fs, vb) for i in range(Fs.size)]
    slipfrac = [float(np.mean(np.abs(odv[i]) > 0.01 * vb)) for i in range(Fs.size)]
    return labs, slipfrac


def fmin(beta, h, K, D=1000, vb=0.1):
    Fs = VS.EXT
    labs, sf = sim(beta, Fs, h, K, D=D, vb=vb)
    Hl = [bool(l[0]) if isinstance(l, tuple) else bool(l) for l in labs]
    fm = VS.fminK(Fs, Hl, K=2)
    sfH = [s for s, l in zip(sf, Hl) if l]
    return fm, Hl, (float(np.median(sfH)) if sfH else None)


def main():
    out = {"F_S_C1_beta0.1": F_S_C1}
    betas = [0.02, 0.03, 0.05, 0.08, 0.12, 0.2]
    cfgs = {}
    for D in (1000, 2000):
        g = math.exp(-2 * math.pi * VS.ZETA1)
        cfgs[("C0n", D)] = (np.array([g]), 0, None)
        K = 64 * D // 1000
        h, q = design_fir(D, K, VS.zeta_c1); cfgs[("C1n", D)] = (h, K, q)
        h2, q2 = design_fir(D, K, zeta_flat); cfgs[("C1n-flat", D)] = (h2, K, q2)
    res = {}
    for name in ("C0n", "C1n", "C1n-flat"):
        h, K, q = cfgs[(name, 1000)]
        rows = {}
        for b in betas + [0.1]:
            fm, Hl, sf = fmin(b, h, K)
            rows[str(b)] = dict(fmin=fm, n_H=int(sum(Hl)), slipfrac_H_median=sf)
            print(f"{name} beta {b}: F_min {fm} (H labels {sum(Hl)}/32, slip fraction of H runs {sf})", flush=True)
        fm6 = [rows[str(b)]["fmin"] for b in betas]
        ok = [i for i, v in enumerate(fm6) if v]
        bb = float(np.polyfit(np.log([betas[i] for i in ok]), np.log([fm6[i] for i in ok]), 1)[0]) if len(ok) >= 3 else None
        res[name] = dict(rows=rows, b_beta=bb, fir=q)
    # convergence (C1n, D = 2000)
    h, K, q = cfgs[("C1n", 2000)]
    res["conv_C1n_D2000"] = {str(b): fmin(b, h, K, D=2000)[0] for b in (0.03, 0.12)}
    f0n, f1n = res["C0n"]["rows"]["0.1"]["fmin"], res["C1n"]["rows"]["0.1"]["fmin"]
    FSC0 = VS.F_S(0.1)
    res["readings"] = dict(C0n_over_FS=f0n / FSC0 if f0n else None, r=(f1n / f0n) if (f0n and f1n) else None,
                           Q_C1n=(f1n / F_S_C1) if f1n else None, measured_Q=dict(onset=20.2, window=13.7),
                           measured_b_beta=dict(onset=-1.23, window=-1.54))
    json.dump(res, open(os.path.join(HERE, "results_v2.json"), "w"), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("readings", "conv_C1n_D2000")}, indent=1, default=float))
    print({n: (res[n]["b_beta"], res[n]["fir"]) for n in ("C0n", "C1n", "C1n-flat")})


if __name__ == "__main__":
    main()
