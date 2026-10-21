"""Simulation v7 (PLAN_v7.md): finite bow width as M friction junctions in a waveguide chain. Claude, 2026-10-21."""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(HERE, "nbcache"))
for d in ("verify", "v2", "v3", "v4", "v6"):
    sys.path.insert(0, os.path.join(HERE, "..", d))
import numpy as np
from numba import njit, prange
import vsim as VS
import vsim3 as V3
import vsim4 as V4
import vsim6 as V6

Z, F0, L_STR = VS.Z, VS.F0, VS.L
D = 4000; N = D // 2          # one-way samples along the string


@njit(cache=True)
def run_one(F, pos, h, ldel, n_nut2, vb, acc_b, fs, nsteps, i0, jc, coul, xb0, xn0, rr0, ll0, segoff, seglen, odv):
    M = pos.size; Y = 1.0 / (2.0 * Z); Fj = F / M
    xb = xb0.copy(); xn = xn0.copy(); rr = rr0.copy(); ll = ll0.copy()
    Lxb = xb.size; Lh = h.size
    a = np.zeros(M); b = np.zeros(M); fo = np.zeros(M)
    for k in range(nsteps):
        vbow = vb
        if acc_b > 0.0:
            vbow = acc_b * (k + 1) / fs
            if vbow > vb:
                vbow = vb
        # reads
        acc = 0.0
        for i in range(Lh):
            acc += h[i] * xb[(k - ldel - i) % Lxb]
        a[0] = -acc
        for j in range(1, M):
            s = j - 1
            a[j] = rr[segoff[s] + k % seglen[s]]
        b[M - 1] = -xn[k % n_nut2]
        for j in range(M - 1):
            b[j] = ll[segoff[j] + k % seglen[j]]
        # friction
        dvc = 0.0
        for j in range(M):
            w = a[j] + b[j] - vbow
            if coul:
                f, dv, sl, sg = VS.fric(w, Y, Fj, False, 1.0, False)
            else:
                f, dv = V6.fric_h(w, Y, Fj)
            fo[j] = f
            if j == jc:
                dvc = dv
        # writes
        xb[k % Lxb] = b[0] + Y * fo[0]
        for j in range(1, M):
            s = j - 1
            ll[segoff[s] + k % seglen[s]] = b[j] + Y * fo[j]
        xn[k % n_nut2] = a[M - 1] + Y * fo[M - 1]
        for j in range(M - 1):
            rr[segoff[j] + k % seglen[j]] = a[j] + Y * fo[j]
        r = k - i0
        if r >= 0 and r < odv.size:
            odv[r] = dvc


@njit(parallel=True, cache=True)
def run_batch(Fs, pos, h, ldel, n_nut2, vb, acc_b, fs, nsteps, i0, jc, coul, xb0s, xn0s, rr0s, ll0s, segoff, seglen, odv):
    for i in prange(Fs.size):
        run_one(Fs[i], pos, h, ldel, n_nut2, vb, acc_b, fs, nsteps, i0, jc, coul, xb0s[i], xn0s[i], rr0s[i], ll0s[i], segoff, seglen, odv[i])


def junctions(beta, W_mm, M):
    c = beta * N
    if M == 1:
        return np.array([int(round(c))]), c
    Ws = W_mm / 1000 / L_STR * N
    return np.round(np.linspace(c - Ws / 2, c + Ws / 2, M)).astype(np.int64), c


def fmin(beta, W_mm, M, h, tau, start="ideal", coul=False, vb=0.1, acc_b=0.104):
    fs = F0 * D
    pos, c = junctions(beta, W_mm, M)
    assert np.all(np.diff(pos) >= 1)
    ldel = 2 * int(pos[0]) - int(round(tau)); assert ldel >= 1
    n_nut2 = 2 * (N - int(pos[-1]))
    seglen = np.diff(pos).astype(np.int64) if M > 1 else np.zeros(0, np.int64)
    segoff = (np.r_[0, np.cumsum(seglen)[:-1]] if M > 1 else np.zeros(0)).astype(np.int64)
    Lh = len(h); Lxb = ldel + Lh + 2
    Fs = VS.EXT; n = Fs.size
    xb0 = np.zeros((n, Lxb)); xn0 = np.zeros((n, n_nut2)); tot = int(seglen.sum()) if M > 1 else 1
    rr0 = np.zeros((n, max(tot, 1))); ll0 = np.zeros((n, max(tot, 1)))
    jc = int(np.argmin(np.abs(pos - c)))
    if start == "ideal":
        db2 = 2 * c; t0 = -D // 2
        A = lambda t: vb / (beta * D) * ((t - t0) % D) - vb / (2 * beta)
        mu_slip = 1.2 if coul else V6.MUI + (V6.MUS - V6.MUI) * V6.V0 / (V6.V0 + vb / beta)
        mu0 = 0.6 if coul else mu_slip
        for i, F in enumerate(Fs):
            f0 = mu0 * F / (4 * Z)
            gL = lambda t: A(t) + f0
            gR = lambda t: -A(t - db2) + f0
            t = np.arange(-Lxb, 0); xb0[i, t % Lxb] = gL(t - (c - pos[0]))
            t = np.arange(-n_nut2, 0); xn0[i, t % n_nut2] = gR(t - (pos[-1] - c))
            for s in range(M - 1):
                d = seglen[s]; t = np.arange(-d, 0); ps, pe = pos[s], pos[s + 1]
                rr0[i, segoff[s] + t % d] = gR(t - (ps - c)) if ps >= c else -gL(t - c - ps)
                ll0[i, segoff[s] + t % d] = gL(t - (c - pe)) if pe <= c else -gR(t - (N - c) - (N - pe))
        ab, w0, w1 = 0.0, 0.3, 0.5
    else:
        ab = acc_b; tpl = vb / acc_b; w0, w1 = tpl + 0.2, tpl + 0.4
    i0 = int(round(w0 * fs)); i1 = int(round(w1 * fs))
    odv = np.zeros((n, i1 - i0))
    run_batch(Fs, pos.astype(np.int64), np.asarray(h, float), ldel, n_nut2, vb, ab, fs, i1, i0, jc, coul, xb0, xn0, rr0, ll0, segoff, seglen, odv)
    labs = [V3.label3(odv[i], fs, beta, vb) for i in range(n)]
    H = [l[0] for l in labs]
    return VS.fminK(Fs, H, K=2), "".join("H" if l[0] else ("c" if l[1]["classifier"] else ".") for l in labs)


def main():
    betas = [0.03, 0.05, 0.08, 0.12, 0.2]
    a, p = V4.powerlaw()
    h, q = V4.design_minphase(D, 1024, a, p); assert q["gate"]
    g = np.array([math.exp(-2 * math.pi * math.exp(a))])
    v6 = json.load(open(os.path.join(HERE, "..", "v6", "results_v6.json")))
    res = {}
    cfgs = [("P0", 0, 1, g, 0.0, "ideal", False), ("P1", 0, 1, h, q["phase_delay_f0"], "ideal", False),
            ("W0", 10, 8, g, 0.0, "ideal", False), ("W1", 10, 8, h, q["phase_delay_f0"], "ideal", False),
            ("W1-5mm", 5, 5, h, q["phase_delay_f0"], "ideal", False), ("W1-coul", 10, 8, h, q["phase_delay_f0"], "ideal", True),
            ("W1-ramp", 10, 8, h, q["phase_delay_f0"], "ramp", False)]
    for name, W, M, hh, tau, start, coul in cfgs:
        rows = {}
        for bt in betas + [0.1]:
            fm, pat = fmin(bt, W, M, hh, tau, start, coul)
            rows[str(bt)] = dict(fmin=fm, labels=pat)
            print(f"{name} β {bt}: F_min {fm}  {pat}", flush=True)
        res[name] = dict(rows=rows, b_beta=V4.slope(betas, [rows[str(bt)]["fmin"] for bt in betas]),
                         n_beta_defined=sum(1 for bt in betas if rows[str(bt)]["fmin"]))
    step = 10 ** (3 / 23)

    def same(x, y):
        return (x is None and y is None) or (x is not None and y is not None and 1 / step - 1e-9 <= x / y <= step + 1e-9)
    gP0 = all(same(res["P0"]["rows"][k]["fmin"], v6["H0m"]["rows"][k]["fmin"]) for k in res["P0"]["rows"])
    gP1 = all(same(res["P1"]["rows"][k]["fmin"], v6["H1m"]["rows"][k]["fmin"]) for k in res["P1"]["rows"] if v6["H1m"]["rows"][k]["fmin"])
    bW1 = res["W1"]["b_beta"]
    res["readings"] = dict(
        code_gate_P0=gP0, code_gate_P1=gP1,
        b_beta_W1=bW1, W1_reading=None if not (gP0 and gP1) else ("undefined" if res["W1"]["n_beta_defined"] < 3 else
                   ("bow width restores a measured-like flattening (b >= -2.0)" if bW1 >= -2.0 else ("bow width does not (b <= -4.0)" if bW1 <= -4.0 else "partial"))),
        b_beta_W0=res["W0"]["b_beta"], b_beta_point_v6_H1m=v6["H1m"]["b_beta"], b_beta_point_v6_H0m=v6["H0m"]["b_beta"],
        W1_F_min_beta0p1=res["W1"]["rows"]["0.1"]["fmin"], b_beta_W1_5mm=res["W1-5mm"]["b_beta"], b_beta_W1_coul=res["W1-coul"]["b_beta"],
        W1_ramp_establishment=res["W1-ramp"]["n_beta_defined"] >= 3, measured=dict(b_onset=-1.23, b_window=-1.54))
    json.dump(res, open(os.path.join(HERE, "results_v7.json"), "w"), indent=1, default=float)
    print(json.dumps(res["readings"], indent=1, default=float))


if __name__ == "__main__":
    main()
