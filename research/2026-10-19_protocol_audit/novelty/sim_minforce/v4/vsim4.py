"""Simulation v4 (PLAN_v4.md): smooth power-law damping target, minimum-phase FIR loss filter on the bridge side,
maintenance test with the validated v3 waveguide and amplitude-checked label. Claude, 2026-10-21."""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(HERE, "nbcache"))
sys.path.insert(0, os.path.join(HERE, "..", "verify")); sys.path.insert(0, os.path.join(HERE, "..", "v3"))
sys.path.insert(0, os.path.join(HERE, "..", "v2"))
import numpy as np
from numba import njit, prange
import vsim as VS
import vsim3 as V3

Z, F0, MUD, MUS = VS.Z, VS.F0, VS.MUD, VS.MUS
F_S_C1 = 0.0199
NFFT = 1 << 18


def powerlaw(maxmode=None):
    m = VS.DN <= (maxmode or 1e9)
    p, a = np.polyfit(np.log(VS.DN[m]), np.log(VS.DZ[m]), 1)
    return float(a), float(p)


def loss_target(n, a, p):
    n = np.asarray(n, float)
    return np.minimum(2 * np.pi * n * np.exp(a) * np.power(np.maximum(n, 1e-12), p), 20.0)


def design_minphase(D, L, a, p):
    k = np.arange(NFFT); f = np.minimum(k, NFFT - k) / NFFT          # symmetric |f| in cycles/sample
    logmag = -loss_target(f * D, a, p)
    c = np.fft.ifft(logmag).real
    w = np.zeros(NFFT); w[0] = 1; w[1:NFFT // 2] = 2; w[NFFT // 2] = 1
    h = np.fft.ifft(np.exp(np.fft.fft(c * w))).real[:L]
    nn = np.arange(1, 101); om = 2 * np.pi * nn / D
    H = np.exp(-1j * np.outer(om, np.arange(L))) @ h
    lr = -np.log(np.abs(H)); lt = loss_target(nn, a, p)
    rel = np.abs(lr / lt - 1)
    gate = bool(rel[:30].max() <= 0.10 and rel[30:].max() <= 0.30)
    tau = float(-np.angle(H[0]) / om[0])
    return h, dict(L=L, gate=gate, max_rel_err_n1_30=float(rel[:30].max()), max_rel_err_n31_100=float(rel[30:].max()),
                   loss_n1_real=float(lr[0]), loss_n1_target=float(lt[0]), phase_delay_f0=tau)


def pick_filter(D, a, p, Ls):
    tried = []
    for L in Ls:
        h, q = design_minphase(D, L, a, p); tried.append(q)
        if q["gate"]:
            return h, q, tried
    return None, None, tried


@njit(cache=True)
def run_one(F, db2, dn2, h, L, ldel, vb, nsteps, i0, xb0, on0, odv):
    Y = 1.0 / (2.0 * Z)
    xb = xb0.copy(); Lx = xb.size; on = on0.copy()
    slipping = False; s = 1.0
    for k in range(nsteps):
        acc = 0.0
        for j in range(L):
            acc += h[j] * xb[(k - ldel - j) % Lx]
        vib = -acc
        inn = k % dn2
        vin = -on[inn]
        w = vib + vin - vb
        f, dv, slipping, s = VS.fric(w, Y, F, slipping, s, False)
        xb[k % Lx] = vin + Y * f
        on[inn] = vib + Y * f
        r = k - i0
        if r >= 0 and r < odv.size:
            odv[r] = dv


@njit(parallel=True, cache=True)
def run_batch(Fs, db2, dn2, h, L, ldel, vb, nsteps, i0, xb0s, on0s, odv):
    for i in prange(Fs.size):
        run_one(Fs[i], db2, dn2, h, L, ldel, vb, nsteps, i0, xb0s[i], on0s[i], odv[i])


def fmin(beta, h, tau, D=4000, vb=0.1, win=(0.3, 0.5)):
    fs = F0 * D; db2 = int(round(beta * D)); dn2 = D - db2
    assert abs(db2 - beta * D) < 1e-9
    L = len(h); ldel = db2 - int(round(tau)); assert ldel >= 1
    Fs = VS.EXT; n = Fs.size; Lx = ldel + L + 2
    xb0 = np.zeros((n, Lx)); on0 = np.zeros((n, dn2)); t0 = -D // 2
    a = lambda j: vb / (beta * D) * ((j - t0) % D) - vb / (2 * beta)
    for i, F in enumerate(Fs):
        f0 = MUD * F
        jb = np.arange(-Lx, 0); xb0[i, jb % Lx] = a(jb) + f0 / (4 * Z)
        jn = np.arange(-dn2, 0); on0[i, jn % dn2] = -a(jn - db2) + f0 / (4 * Z)
    i0 = int(round(win[0] * fs)); i1 = int(round(win[1] * fs))
    odv = np.zeros((n, i1 - i0))
    run_batch(Fs, db2, dn2, np.asarray(h, float), L, ldel, vb, i1, i0, xb0, on0, odv)
    labs = [V3.label3(odv[i], fs, beta, vb) for i in range(n)]
    H = [l[0] for l in labs]
    return VS.fminK(Fs, H, K=2), dict(n_H=int(sum(H)), n_classifier_only=sum(1 for l in labs if l[1]["classifier"] and not l[0]))


def slope(xs, ys):
    ok = [(x, y) for x, y in zip(xs, ys) if y]
    return float(np.polyfit(np.log([o[0] for o in ok]), np.log([o[1] for o in ok]), 1)[0]) if len(ok) >= 3 else None


def main():
    betas = [0.02, 0.03, 0.05, 0.08, 0.12, 0.2]
    a, p = powerlaw(); a_lo, p_lo = powerlaw(11)
    res = dict(powerlaw=dict(a=a, p=p, zeta1_fit=math.exp(a), zeta1_measured=float(VS.ZETA1)), powerlaw_low=dict(a=a_lo, p=p_lo))
    # C0f baseline
    g = math.exp(-2 * math.pi * math.exp(a)); FS0 = Z * math.pi * math.exp(a) * 0.1 / (2 * 0.1 ** 2 * (MUS - MUD))
    rows = {str(b): fmin(b, np.array([g]), 0.0) for b in betas + [0.1]}
    res["C0f"] = dict(fmin={k: v[0] for k, v in rows.items()}, diag={k: v[1] for k, v in rows.items()},
                      b_beta=slope(betas, [rows[str(b)][0] for b in betas]), F_S=FS0)
    f0f = rows["0.1"][0]
    val = bool(f0f and 1 <= f0f / FS0 <= 4 and res["C0f"]["b_beta"] is not None and abs(res["C0f"]["b_beta"] + 2) <= 0.3)
    res["C0f"].update(validation_pass=val, ratio_to_FS=(f0f / FS0) if f0f else None)
    print("C0f", res["C0f"]["fmin"], "b", res["C0f"]["b_beta"], "valid", val, flush=True)
    for name, (aa, pp) in (("C1s", (a, p)), ("C1s-low", (a_lo, p_lo))):
        h, q, tried = pick_filter(4000, aa, pp, (64, 256, 1024))
        res[name] = dict(filter_tried=tried, filter=q)
        print(name, "filter", tried, flush=True)
        if h is None:
            continue
        rows = {str(b): fmin(b, h, q["phase_delay_f0"]) for b in betas + [0.1]}
        res[name].update(fmin={k: v[0] for k, v in rows.items()}, diag={k: v[1] for k, v in rows.items()},
                         b_beta=slope(betas, [rows[str(b)][0] for b in betas]))
        print(name, res[name]["fmin"], "b", res[name]["b_beta"], flush=True)
        if name == "C1s":
            sp = {str(v): fmin(0.1, h, q["phase_delay_f0"], vb=v)[0] for v in (0.05, 0.1, 0.2)}
            okv = [v for v in (0.05, 0.1, 0.2) if sp[str(v)]]
            res[name]["speed"] = dict(fmin=sp, b_v=float(np.polyfit(np.log(okv), np.log([sp[str(v)] for v in okv]), 1)[0]) if len(okv) >= 2 else None)
            h8, q8 = design_minphase(8000, 2 * q["L"], aa, pp)
            res[name]["conv_D8000"] = dict(filter=q8, fmin={str(b): fmin(b, h8, q8["phase_delay_f0"], D=8000)[0] for b in (0.03, 0.12)})
            print("speed", res[name]["speed"], "conv", res[name]["conv_D8000"], flush=True)
    f1 = res.get("C1s", {}).get("fmin", {}).get("0.1")
    v3c0b = json.load(open(os.path.join(HERE, "..", "v3", "results_v3.json")))["C0b"]["rows"]["0.1"]["fmin"]
    r = (f1 / f0f) if (f1 and f0f) else None
    res["readings"] = dict(validation_C0f=val, filter_gate=res["C1s"]["filter"] is not None, r=r,
                           r_reading=None if (not val or r is None) else ("substantial (r>=3)" if r >= 3 else ("not (r<=1.5)" if r <= 1.5 else "partial")),
                           r_vs_v3_C0b=(f1 / v3c0b) if f1 else None, Q=(f1 / F_S_C1) if f1 else None,
                           b_beta_C1s=res["C1s"].get("b_beta"), b_v_C1s=res["C1s"].get("speed", {}).get("b_v"),
                           measured=dict(Q_onset=20.2, Q_window=13.7, b_onset=-1.23, b_window=-1.54, F_onset=0.40, F_window=0.27))
    json.dump(res, open(os.path.join(HERE, "results_v4.json"), "w"), indent=1, default=float)
    print(json.dumps(res["readings"], indent=1, default=float))


if __name__ == "__main__":
    main()
