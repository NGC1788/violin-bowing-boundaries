"""Simulation v3 (PLAN_v3.md): maintenance test, all loss in the bridge-side loop (C0b: frequency independent;
C1b: measured modal damping as a linear-phase FIR), amplitude-checked Helmholtz label. Claude, 2026-10-21."""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(HERE, "nbcache"))
sys.path.insert(0, os.path.join(HERE, "..", "verify")); sys.path.insert(0, os.path.join(HERE, "..", "v2"))
import numpy as np
from numba import njit, prange
import vsim as VS     # verifier: constants, friction rule, classifier, fminK, zeta_c1, F_S, EXT
import vsim2 as V2    # design_fir, zeta_flat

Z, F0, MUD = VS.Z, VS.F0, VS.MUD
F_S_C1 = 0.0199


@njit(cache=True)
def run_one(F, db2, dn2, h, K, vb, nsteps, i0, xb0, on0, odv, hyst):
    Y = 1.0 / (2.0 * Z)
    xb = xb0.copy(); Lx = xb.size; on = on0.copy()
    ldel = db2 - K
    slipping = False; s = 1.0
    for k in range(nsteps):
        acc = 0.0
        for j in range(2 * K + 1):
            acc += h[j] * xb[(k - ldel - j) % Lx]
        vib = -acc
        inn = k % dn2
        vin = -on[inn]
        w = vib + vin - vb
        f, dv, slipping, s = VS.fric(w, Y, F, slipping, s, hyst)
        xb[k % Lx] = vin + Y * f
        on[inn] = vib + Y * f
        r = k - i0
        if r >= 0 and r < odv.size:
            odv[r] = dv


@njit(parallel=True, cache=True)
def run_batch(Fs, db2, dn2, h, K, vb, nsteps, i0, xb0s, on0s, odv, hyst):
    for i in prange(Fs.size):
        run_one(Fs[i], db2, dn2, h, K, vb, nsteps, i0, xb0s[i], on0s[i], odv[i], hyst)


def label3(dv, fs, beta, vb):
    ok, info = VS.label(dv, fs, vb)
    slip = np.abs(dv) > 0.01 * vb
    sf = float(slip.mean())
    amp = float(np.median(np.abs(dv[slip]))) if slip.any() else 0.0
    good = bool(ok and 0.5 * beta <= sf <= 2 * beta and amp >= 0.5 * vb / beta)
    return good, dict(classifier=bool(ok), slip_frac=sf, slip_amp_over_v_beta=amp * beta / vb)


def fmin(beta, h, K, D=4000, vb=0.1, T=0.5, win=(0.3, 0.5)):
    fs = F0 * D; db2 = int(round(beta * D)); dn2 = D - db2
    assert abs(db2 - beta * D) < 1e-9 and db2 > K
    Fs = VS.EXT; n = Fs.size
    Lx = db2 + 2 * K + 2
    xb0 = np.zeros((n, Lx)); on0 = np.zeros((n, dn2))
    t0 = -D // 2
    a = lambda j: vb / (beta * D) * ((j - t0) % D) - vb / (2 * beta)
    for i, F in enumerate(Fs):
        f0 = MUD * F
        jb = np.arange(-Lx, 0); xb0[i, jb % Lx] = a(jb) + f0 / (4 * Z)
        jn = np.arange(-dn2, 0); on0[i, jn % dn2] = -a(jn - db2) + f0 / (4 * Z)
    i0 = int(round(win[0] * fs)); i1 = int(round(win[1] * fs))
    odv = np.zeros((n, i1 - i0))
    run_batch(Fs, db2, dn2, np.asarray(h, float), K, vb, i1, i0, xb0, on0, odv, False)
    labs = [label3(odv[i], fs, beta, vb) for i in range(n)]
    H = [l[0] for l in labs]
    fm = VS.fminK(Fs, H, K=2)
    idx = [i for i in range(n) if H[i]]
    diag = dict(n_H=len(idx), n_classifier_only=sum(1 for l in labs if l[1]["classifier"] and not l[0]),
                median_slip_frac_H=float(np.median([labs[i][1]["slip_frac"] for i in idx])) if idx else None)
    return fm, diag


def main():
    betas = [0.02, 0.03, 0.05, 0.08, 0.12, 0.2]
    g = math.exp(-2 * math.pi * VS.ZETA1)
    cf = {}
    for D, K in ((4000, 64), (8000, 128)):
        cf[("C0b", D)] = (np.array([g]), 0, None)
        h, q = V2.design_fir(D, K, VS.zeta_c1); cf[("C1b", D)] = (h, K, q)
        h2, q2 = V2.design_fir(D, K, V2.zeta_flat); cf[("C1b-flat", D)] = (h2, K, q2)
    res = {}
    for name in ("C0b", "C1b", "C1b-flat"):
        h, K, q = cf[(name, 4000)]
        rows = {}
        for b in betas + [0.1]:
            fm, dg = fmin(b, h, K)
            rows[str(b)] = dict(fmin=fm, **dg)
            print(f"{name} β {b}: F_min {fm} {dg}", flush=True)
        ok = [b for b in betas if rows[str(b)]["fmin"]]
        bb = float(np.polyfit(np.log(ok), np.log([rows[str(b)]["fmin"] for b in ok]), 1)[0]) if len(ok) >= 3 else None
        res[name] = dict(rows=rows, b_beta=bb, n_beta_used=len(ok), fir=q)
    h, K, q = cf[("C1b", 4000)]
    sp = {str(v): fmin(0.1, h, K, vb=v)[0] for v in (0.05, 0.1, 0.2)}
    okv = [v for v in (0.05, 0.1, 0.2) if sp[str(v)]]
    res["speed_C1b_beta0.1"] = dict(fmin=sp, b_v=float(np.polyfit(np.log(okv), np.log([sp[str(v)] for v in okv]), 1)[0]) if len(okv) >= 2 else None)
    h, K, q = cf[("C1b", 8000)]
    res["conv_C1b_D8000"] = {str(b): fmin(b, h, K, D=8000)[0] for b in (0.03, 0.12)}
    f0b, f1b, FSC0 = res["C0b"]["rows"]["0.1"]["fmin"], res["C1b"]["rows"]["0.1"]["fmin"], VS.F_S(0.1)
    val = bool(f0b and 1 <= f0b / FSC0 <= 4 and res["C0b"]["b_beta"] is not None and abs(res["C0b"]["b_beta"] + 2) <= 0.3)
    r = f1b / f0b if (f0b and f1b) else None
    res["readings"] = dict(validation_pass=val, C0b_over_FS=(f0b / FSC0) if f0b else None, C0b_b_beta=res["C0b"]["b_beta"],
                           r=r, r_reading=None if (not val or r is None) else ("substantial (r>=3)" if r >= 3 else ("not (r<=1.5)" if r <= 1.5 else "partial")),
                           Q_C1b=(f1b / F_S_C1) if f1b else None, b_beta_C1b=res["C1b"]["b_beta"], b_beta_C1b_flat=res["C1b-flat"]["b_beta"],
                           measured=dict(Q_onset=20.2, Q_window=13.7, b_onset=-1.23, b_window=-1.54))
    json.dump(res, open(os.path.join(HERE, "results_v3.json"), "w"), indent=1, default=float)
    print(json.dumps(res["readings"], indent=1, default=float)); print("speed", res["speed_C1b_beta0.1"], "conv", res["conv_C1b_D8000"])
    print("FIR", {n: res[n]["fir"] for n in ("C1b", "C1b-flat")})


if __name__ == "__main__":
    main()
