"""Threshold-tilt principle test (PLAN.md). Claude, 2026-10-21."""
import csv, gzip, json, math, sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

H = Path.home(); HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(H / "violin-rne/novelty/label_sensitivity"))
import label_sweep as LS  # noqa: E402

A_LOW = [-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0]
A_UP = [0.0, 0.25, 0.5, 0.75, 1.0]
BASE = LS.VARIANTS["base"]


def read2(stem, rep):
    rows, beta, fh = [], {}, []
    with gzip.open(LS.D.FEAT / f"{stem}.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            if r["error"] or r["window_s"] != "0.2":
                continue
            if rep and f"/{rep}/" not in r["folder"] + "/":
                continue
            k = (r["folder"], int(r["trial"])); beta[k] = float(r["beta"])
            if round(float(r["t"]), 2) == 0.2:
                rows.append((k, float(r["force"]), float(r["speed"]), LS.fnum(r["amplitude_std"]), LS.fnum(r["periodicity"]),
                             LS.fnum(r["f0_hz"]), LS.fnum(r["flybacks_per_period"]), LS.fnum(r["second_flyback_ratio"])))
                fh.append(LS.fnum(r["flyback_height"]))
    return (rows, beta), np.array(fh)


def slope_b(X, beta, folder, H0):
    m = H0 & np.isfinite(X) & (X > 0)
    x = np.log(beta[m]); y = np.log(X[m]); f = folder[m]
    for g in np.unique(f):
        s = f == g; x[s] -= x[s].mean(); y[s] -= y[s].mean()
    return float((x * y).sum() / (x * x).sum())


def tilted(X, beta, folder, H0, a, frac, upper=False):
    Hv = H0.copy()
    score = np.where(np.isfinite(X) & (X > 0), np.log(np.where(X > 0, X, 1)) - a * np.log(beta), np.nan)
    for g in np.unique(folder):
        s = (folder == g) & H0 & np.isfinite(score)
        if not s.any():
            continue
        if upper:
            q = np.quantile(score[s], 1 - frac); Hv[s & (score > q)] = False
        else:
            q = np.quantile(score[s], frac); Hv[s & (score < q)] = False
    return Hv


def fits(d, sess):
    out = {"window": LS.fit(d)[1]}
    if sess in LS.ONSET:
        out["onset"] = LS.fit(LS.to_onset(d, sess))[1]
    return out


def zero_cross(a_list, m):
    for i in range(len(a_list) - 1):
        if (m[i] <= 0 <= m[i + 1]) or (m[i] >= 0 >= m[i + 1]):
            if m[i + 1] == m[i]:
                return a_list[i]
            return float(a_list[i] + (0 - m[i]) * (a_list[i + 1] - a_list[i]) / (m[i + 1] - m[i]))
    return None


def tests(a_list, per_sess_delta, b_vals, sign):
    m = np.nanmean(np.array([per_sess_delta[s] for s in per_sess_delta]), axis=0)
    rho = spearmanr(a_list, m)[0]
    rho_s = [spearmanr(a_list, per_sess_delta[s])[0] for s in per_sess_delta]
    n_ok = int(np.sum(np.array(rho_s) * sign >= 0.7))
    zc = zero_cross(a_list, m); bmean = float(np.mean(b_vals))
    rho3 = spearmanr([abs(a - bmean) for a in a_list], np.abs(m))[0]
    T1 = bool(rho * sign >= 0.9 and n_ok >= int(np.ceil(11 / 14 * len(rho_s))))
    T2 = bool(zc is not None and abs(zc - bmean) <= 0.25)
    T3 = bool(rho3 >= 0.8)
    reading = "principle confirmed" if (T1 and T2) else ("direction only" if T1 else "not confirmed")
    return dict(mean_delta=m.tolist(), rho=float(rho), per_session_rho=[float(x) for x in rho_s], n_sessions_rho_ok=n_ok,
                zero_cross=zc, b_mean=bmean, rho_size=float(rho3), T1=T1, T2=T2, T3=T3, reading=reading)


def main():
    base_res = json.loads((H / "violin-rne/novelty/label_sensitivity/results.json").read_text())["sessions"]
    res = {"sessions": {}}
    for sess, st in LS.STEM.items():
        stem, rep = st if isinstance(st, tuple) else (st, None)
        raw, fh = read2(stem, rep)
        d0 = LS.build(raw, BASE); H0 = d0["H"]; beta = d0["beta"]; folder = d0["folder"]
        amp = np.array([r[3] for r in raw[0]])
        base = fits(d0, sess)
        e = {"base": base, "b": {}, "delta": {}}
        for qname, X in (("amplitude", amp), ("flyback_height", fh)):
            e["b"][qname] = slope_b(X, beta, folder, H0)
            for frac in (0.15, 0.30):
                for a in A_LOW:
                    f = fits(dict(d0, H=tilted(X, beta, folder, H0, a, frac)), sess)
                    e["delta"][f"{qname}|{frac}|{a}"] = {ax: f[ax] - base[ax] for ax in f}
        res["sessions"][sess] = e
        print(f"{sess:7s} base {base['window']:+.3f} (stored {base_res[sess]['base']['window_b_beta']:+.3f}) b_amp {e['b']['amplitude']:+.2f} b_fh {e['b']['flyback_height']:+.2f} | amp Δ(a) "
              + " ".join(f"{e['delta'][f'amplitude|0.15|{a}']['window']:+.2f}" for a in A_LOW), flush=True)
    # B1 corner width (upper bound)
    rec = {(r["folder"], int(r["trial"])): float(r["flyback_ms"]) if r["flyback_ms"] not in ("", "nan", "None") else math.nan
           for r in csv.DictReader(open(H / "violin-rne/novelty/label_sensitivity/b1_flyback/b1_flyback.csv"))}
    raw, _ = read2(LS.STEM["B1"], None); d0 = LS.build(raw, BASE); H0 = d0["H"]
    dfb = np.array([rec.get(r[0], math.nan) for r in raw[0]])
    bB1 = slope_b(dfb, d0["beta"], d0["folder"], H0); baseB1 = fits(d0, "B1")
    b1 = {"b": bB1, "base": baseB1, "delta": {}}
    for frac in (0.15, 0.30):
        for a in A_UP:
            f = fits(dict(d0, H=tilted(dfb, d0["beta"], d0["folder"], H0, a, frac, upper=True)), "B1")
            b1["delta"][f"{frac}|{a}"] = {ax: f[ax] - baseB1[ax] for ax in f}
    res["B1_corner"] = b1
    print(f"B1 corner b {bB1:+.2f} Δ(a) " + " ".join(f"{b1['delta'][f'0.15|{a}']['window']:+.2f}" for a in A_UP), flush=True)
    # tests
    summ = {}
    for qname in ("amplitude", "flyback_height"):
        for frac in (0.15, 0.30):
            for ax, ss in (("window", list(LS.STEM)), ("onset", LS.ONSET)):
                per = {s: [res["sessions"][s]["delta"][f"{qname}|{frac}|{a}"][ax] for a in A_LOW] for s in ss}
                bv = [res["sessions"][s]["b"][qname] for s in ss]
                summ[f"{qname}|{frac}|{ax}"] = tests(A_LOW, per, bv, +1)
    for frac in (0.15, 0.30):
        for ax in ("window", "onset"):
            m = [res["B1_corner"]["delta"][f"{frac}|{a}"][ax] for a in A_UP]
            rho = spearmanr(A_UP, m)[0]; zc = zero_cross(A_UP, m)
            summ[f"B1_corner|{frac}|{ax}"] = dict(mean_delta=m, rho=float(rho), zero_cross=zc, b=bB1,
                                                   direction_ok=bool(rho <= -0.9), neutral_ok=bool(zc is not None and abs(zc - bB1) <= 0.25))
    res["tests"] = summ
    (HERE / "results.json").write_text(json.dumps(res, indent=1, default=float))
    for k, v in summ.items():
        if k.startswith("B1"):
            print(f"{k}: Δ {[round(x, 3) for x in v['mean_delta']]} ρ {v['rho']:+.2f} zero {v['zero_cross']} b {v['b']:+.2f} dir {v['direction_ok']} neutral {v['neutral_ok']}")
        else:
            print(f"{k}: mean Δ {[round(x, 3) for x in v['mean_delta']]} ρ {v['rho']:+.2f} ({v['n_sessions_rho_ok']} sessions ok) zero {v['zero_cross']} b {v['b_mean']:+.2f} ρ_size {v['rho_size']:+.2f} → {v['reading']}")


if __name__ == "__main__":
    main()
