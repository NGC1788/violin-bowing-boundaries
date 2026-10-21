"""Simulation v8 (PLAN_v8.md): label robustness of the beta dependence. Same v6 (smooth friction) and v4 (Coulomb)
strings, ideal start, 1.5 s run, three labels on the same runs. Claude, 2026-10-21."""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(HERE, "nbcache"))
for d in ("verify", "v2", "v3", "v4", "v6"):
    sys.path.insert(0, os.path.join(HERE, "..", d))
import numpy as np
import vsim as VS
import vsim3 as V3
import vsim4 as V4
import vsim6 as V6

Z, F0 = VS.Z, VS.F0
D = 4000


def runs(beta, h, tau, law, vb=0.1, t_rec=(1.0, 1.5)):
    fs = F0 * D; db2 = int(round(beta * D)); dn2 = D - db2
    L = len(h); ldel = db2 - int(round(tau)); assert ldel >= 1
    Fs = VS.EXT; n = Fs.size; Lx = ldel + L + 2
    xb0 = np.zeros((n, Lx)); on0 = np.zeros((n, dn2)); t0 = -D // 2
    A = lambda j: vb / (beta * D) * ((j - t0) % D) - vb / (2 * beta)
    mu0 = VS.MUD if law == "coulomb" else V6.MUI + (V6.MUS - V6.MUI) * V6.V0 / (V6.V0 + vb / beta)
    for i, F in enumerate(Fs):
        f0 = mu0 * F
        jb = np.arange(-Lx, 0); xb0[i, jb % Lx] = A(jb) + f0 / (4 * Z)
        jn = np.arange(-dn2, 0); on0[i, jn % dn2] = -A(jn - db2) + f0 / (4 * Z)
    i0 = int(round(t_rec[0] * fs)); i1 = int(round(t_rec[1] * fs))
    odv = np.zeros((n, i1 - i0)); hh = np.asarray(h, float)
    if law == "coulomb":
        V4.run_batch(Fs, db2, dn2, hh, L, ldel, vb, i1, i0, xb0, on0, odv)
    else:
        V6.run_batch(Fs, db2, dn2, hh, L, ldel, vb, 0.0, fs, i1, i0, xb0, on0, odv)
    return odv, fs


def labels(odv, fs, beta, vb=0.1):
    n0 = lambda t: int(round((t - 1.0) * fs))
    out = []
    for dv in odv:
        w = dv[n0(1.3):n0(1.5)]
        L1, _ = VS.label(w, fs, vb)
        L2 = V3.label3(w, fs, beta, vb)[0]
        r1 = float(np.sqrt(np.mean(dv[n0(1.0):n0(1.1)] ** 2))); r2 = float(np.sqrt(np.mean(dv[n0(1.4):n0(1.5)] ** 2)))
        st = r2 / r1 if r1 > 0 else 0.0
        out.append((bool(L1), bool(L2), bool(L1 and st >= 0.95), st))
    return out


def main():
    betas = [0.02, 0.03, 0.05, 0.08, 0.12, 0.2]
    a, p = V4.powerlaw()
    h, q = V4.design_minphase(D, 1024, a, p); assert q["gate"]
    g = np.array([math.exp(-2 * math.pi * math.exp(a))])
    res = {}
    for name, hh, tau, law in (("H0m", g, 0.0, "smooth"), ("H1m", h, q["phase_delay_f0"], "smooth"), ("C1s", h, q["phase_delay_f0"], "coulomb")):
        rows = {}
        for b in betas + [0.1]:
            odv, fs = runs(b, hh, tau, law)
            labs = labels(odv, fs, b)
            r = {}
            for k, lab in (("L1", 0), ("L2", 1), ("L3", 2)):
                Hs = [x[lab] for x in labs]
                r[k] = dict(fmin=VS.fminK(VS.EXT, Hs, K=2), pattern="".join("H" if x else "." for x in Hs))
            r["stationarity"] = [round(x[3], 3) for x in labs]
            rows[str(b)] = r
            print(f"{name} β {b}: " + " | ".join(f"{k} {r[k]['fmin']} {r[k]['pattern']}" for k in ("L1", "L2", "L3")), flush=True)
        res[name] = dict(rows=rows, b_beta={k: V4.slope(betas, [rows[str(b)][k]["fmin"] for b in betas]) for k in ("L1", "L2", "L3")})
        print(name, "b_beta", res[name]["b_beta"], flush=True)
    c = res["H0m"]["b_beta"]
    control = all(c[k] is not None and abs(c[k] + 2) <= 0.3 for k in ("L2", "L3"))
    b2, b3 = res["H1m"]["b_beta"]["L2"], res["H1m"]["b_beta"]["L3"]
    if not control:
        reading = None
    elif (b2 is not None and b2 > -2.5) or (b3 is not None and b3 > -2.5):
        reading = "label-dependent"
    elif b2 is not None and b3 is not None and b2 <= -3.0 and b3 <= -3.0:
        reading = "survives"
    else:
        reading = "weakened"
    res["readings"] = dict(control_pass=control, H0m_b=c, H1m_b=res["H1m"]["b_beta"], C1s_b=res["C1s"]["b_beta"], reading=reading)
    json.dump(res, open(os.path.join(HERE, "results_v8.json"), "w"), indent=1, default=float)
    print(json.dumps(res["readings"], indent=1, default=float))


if __name__ == "__main__":
    main()
