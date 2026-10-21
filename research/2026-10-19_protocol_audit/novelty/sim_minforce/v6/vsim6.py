"""Simulation v6 (PLAN_v6.md): hyperbolic (velocity-dependent) friction curve, otherwise the v4/v5 waveguide;
maintenance (ideal start) and the measured start-from-rest ramp. Claude, 2026-10-21."""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(HERE, "nbcache"))
for d in ("verify", "v2", "v3", "v4"):
    sys.path.insert(0, os.path.join(HERE, "..", d))
import numpy as np
from numba import njit, prange
import vsim as VS
import vsim3 as V3
import vsim4 as V4

Z, F0 = VS.Z, VS.F0
MUS, MUI, V0 = 1.2, 0.35, 0.1


@njit(cache=True)
def mu_of(u):
    return MUI + (MUS - MUI) * V0 / (V0 + u)


@njit(cache=True)
def fric_h(w, Y, F):
    """stick-if-possible; slip on the hyperbolic curve. w = free relative velocity; returns (f, dv)."""
    fst = -w / Y
    if abs(fst) <= MUS * F:
        return fst, 0.0
    s = 1.0 if w > 0.0 else -1.0
    A = abs(w) - Y * F * MUI; B = Y * F * (MUS - MUI)
    b1 = V0 - A
    u = 0.5 * (-b1 + math.sqrt(b1 * b1 + 4.0 * V0 * (A - B)))
    if u < 0.0:
        u = 0.0
    return -s * F * mu_of(u), s * u


@njit(cache=True)
def run_one(F, db2, dn2, h, L, ldel, vb, acc_b, fs, nsteps, i0, xb0, on0, odv):
    Y = 1.0 / (2.0 * Z)
    xb = xb0.copy(); Lx = xb.size; on = on0.copy()
    for k in range(nsteps):
        vbow = vb
        if acc_b > 0.0:
            vbow = acc_b * (k + 1) / fs
            if vbow > vb:
                vbow = vb
        acc = 0.0
        for j in range(L):
            acc += h[j] * xb[(k - ldel - j) % Lx]
        vib = -acc
        inn = k % dn2
        vin = -on[inn]
        w = vib + vin - vbow
        f, dv = fric_h(w, Y, F)
        xb[k % Lx] = vin + Y * f
        on[inn] = vib + Y * f
        r = k - i0
        if r >= 0 and r < odv.size:
            odv[r] = dv


@njit(parallel=True, cache=True)
def run_batch(Fs, db2, dn2, h, L, ldel, vb, acc_b, fs, nsteps, i0, xb0s, on0s, odv):
    for i in prange(Fs.size):
        run_one(Fs[i], db2, dn2, h, L, ldel, vb, acc_b, fs, nsteps, i0, xb0s[i], on0s[i], odv[i])


def fmin(beta, h, tau, start, D=4000, vb=0.1, acc_b=0.104):
    fs = F0 * D; db2 = int(round(beta * D)); dn2 = D - db2
    assert abs(db2 - beta * D) < 1e-9
    L = len(h); ldel = db2 - int(round(tau)); assert ldel >= 1
    Fs = VS.EXT; n = Fs.size; Lx = ldel + L + 2
    xb0 = np.zeros((n, Lx)); on0 = np.zeros((n, dn2))
    if start == "ideal":
        t0 = -D // 2
        a = lambda j: vb / (beta * D) * ((j - t0) % D) - vb / (2 * beta)
        mu_slip = MUI + (MUS - MUI) * V0 / (V0 + vb / beta)
        for i, F in enumerate(Fs):
            f0 = mu_slip * F
            jb = np.arange(-Lx, 0); xb0[i, jb % Lx] = a(jb) + f0 / (4 * Z)
            jn = np.arange(-dn2, 0); on0[i, jn % dn2] = -a(jn - db2) + f0 / (4 * Z)
        ab, w0, w1 = 0.0, 0.3, 0.5
    else:
        ab = acc_b; tpl = vb / acc_b; w0, w1 = tpl + 0.2, tpl + 0.4
    i0 = int(round(w0 * fs)); i1 = int(round(w1 * fs))
    odv = np.zeros((n, i1 - i0))
    run_batch(Fs, db2, dn2, np.asarray(h, float), L, ldel, vb, ab, fs, i1, i0, xb0, on0, odv)
    labs = [V3.label3(odv[i], fs, beta, vb) for i in range(n)]
    H = [l[0] for l in labs]
    return VS.fminK(Fs, H, K=2), "".join("H" if l[0] else ("c" if l[1]["classifier"] else ".") for l in labs)


def main():
    betas = [0.02, 0.03, 0.05, 0.08, 0.12, 0.2]
    a, p = V4.powerlaw()
    h, q = V4.design_minphase(4000, 1024, a, p); assert q["gate"]
    g = math.exp(-2 * math.pi * math.exp(a))
    res = {}
    for name, hh, tau, start in (("H0m", np.array([g]), 0.0, "ideal"), ("H1m", h, q["phase_delay_f0"], "ideal"),
                                 ("H1r", h, q["phase_delay_f0"], "ramp"), ("H0r", np.array([g]), 0.0, "ramp")):
        rows = {}
        for b in betas + [0.1]:
            fm, pat = fmin(b, hh, tau, start)
            rows[str(b)] = dict(fmin=fm, labels=pat)
            print(f"{name} β {b}: F_min {fm}  {pat}", flush=True)
        res[name] = dict(rows=rows, b_beta=V4.slope(betas, [rows[str(b)]["fmin"] for b in betas]),
                         n_beta_defined=sum(1 for b in betas if rows[str(b)]["fmin"]))
    dmu = MUS - (MUI + (MUS - MUI) * V0 / (V0 + 0.1 / 0.1))
    FS = Z * math.pi * math.exp(a) * 0.1 / (2 * 0.1 ** 2 * dmu)
    f0m = res["H0m"]["rows"]["0.1"]["fmin"]; f1m = res["H1m"]["rows"]["0.1"]["fmin"]; f1r = res["H1r"]["rows"]["0.1"]["fmin"]
    bH0 = res["H0m"]["b_beta"]
    gate = bool(f0m and 0.5 <= f0m / FS <= 4 and bH0 is not None and abs(bH0 + 2) <= 0.3)
    est = res["H1r"]["n_beta_defined"] >= 3; br = res["H1r"]["b_beta"]
    r = (f1m / f0m) if (f1m and f0m) else None
    res["readings"] = dict(
        gate=gate, H0m_over_FS=(f0m / FS) if f0m else None, H0m_b_beta=bH0, dmu_eff=dmu,
        establishment=None if not gate else ("smooth friction does not restore establishment" if not est else
                     ("restores the measured flattening (b >= -2.0)" if br >= -2.0 else ("does not (b <= -3.0)" if br <= -3.0 else "partial"))),
        H1r_b_beta=br, H1r_F_min_beta0p1=f1r, H1r_level=None if f1r is None else ("matches (0.2-0.55 N)" if 0.2 <= f1r <= 0.55 else f"ratio to 0.33 N = {f1r / 0.33:.2f}"),
        r_maintenance=r, r_reading=None if (not gate or r is None) else ("substantial (r>=3)" if r >= 3 else ("not (r<=1.5)" if r <= 1.5 else "partial")),
        H1m_b_beta=res["H1m"]["b_beta"], measured=dict(b_onset=-1.23, b_window=-1.54, F_onset=0.40, F_window=0.27), v4_maintenance_coulomb=dict(b_beta=-4.12, F=0.335))
    json.dump(res, open(os.path.join(HERE, "results_v6.json"), "w"), indent=1, default=float)
    print(json.dumps(res["readings"], indent=1, default=float))


if __name__ == "__main__":
    main()
