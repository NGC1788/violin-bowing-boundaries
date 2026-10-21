"""Refined tilt principle (PLAN_edge.md): edge-window scaling + confirmatory periodicity sweep. Claude, 2026-10-21."""
import csv, json, math, sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import tilt as T  # noqa: E402
LS = T.LS; C = LS.C
A_PER = [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6]


def edge_mask(d):
    m = np.zeros(len(d["H"]), bool)
    for r in C.intervals(d, 0.2):
        idx = r["idx"]; h = d["H"][idx]
        for s in range(len(h) - 1):
            if h[s] and h[s + 1]:
                m[idx[s]] = True; m[idx[s + 1]] = True
                break
    return m


def main():
    res = {"sessions": {}}
    for sess, st in LS.STEM.items():
        stem, rep = st if isinstance(st, tuple) else (st, None)
        raw, fh = T.read2(stem, rep)
        d0 = LS.build(raw, T.BASE); H0 = d0["H"]; beta = d0["beta"]; folder = d0["folder"]
        X = {"amplitude": np.array([r[3] for r in raw[0]]), "flyback_height": fh, "sfr": np.array([r[7] for r in raw[0]]),
             "periodicity": np.array([r[4] for r in raw[0]])}
        E = edge_mask(d0)
        e = {"n_edge": int(E.sum()), "b_all": {k: T.slope_b(v, beta, folder, H0) for k, v in X.items()},
             "b_edge": {k: T.slope_b(v, beta, folder, E) for k, v in X.items()}, "base": T.fits(d0, sess), "delta": {}}
        for frac in (0.15, 0.30):
            for a in A_PER:
                f = T.fits(dict(d0, H=T.tilted(X["periodicity"], beta, folder, H0, a, frac)), sess)
                e["delta"][f"{frac}|{a}"] = {ax: f[ax] - e["base"][ax] for ax in f}
        res["sessions"][sess] = e
        print(f"{sess:7s} edge n {e['n_edge']} | b_all/b_edge amp {e['b_all']['amplitude']:+.2f}/{e['b_edge']['amplitude']:+.2f} fh {e['b_all']['flyback_height']:+.2f}/{e['b_edge']['flyback_height']:+.2f} "
              f"sfr {e['b_all']['sfr']:+.2f}/{e['b_edge']['sfr']:+.2f} per {e['b_all']['periodicity']:+.3f}/{e['b_edge']['periodicity']:+.3f} | per Δ(a) "
              + " ".join(f"{e['delta'][f'0.15|{a}']['window']:+.2f}" for a in A_PER), flush=True)
    # B1 corner width edge scaling
    rec = {(r["folder"], int(r["trial"])): float(r["flyback_ms"]) if r["flyback_ms"] not in ("", "nan", "None") else math.nan
           for r in csv.DictReader(open(Path.home() / "violin-rne/novelty/label_sensitivity/b1_flyback/b1_flyback.csv"))}
    raw, _ = T.read2(LS.STEM["B1"], None); d0 = LS.build(raw, T.BASE)
    dfb = np.array([rec.get(r[0], math.nan) for r in raw[0]])
    res["B1_corner"] = dict(b_all=T.slope_b(dfb, d0["beta"], d0["folder"], d0["H"]), b_edge=T.slope_b(dfb, d0["beta"], d0["folder"], edge_mask(d0)))
    ss = list(LS.STEM)
    mean = lambda k, w: float(np.mean([res["sessions"][s][w][k] for s in ss]))
    known = {"amplitude": -1.0862, "flyback_height": -1.0647, "sfr": None, "B1_corner": 0.7629}
    cons = {k: dict(zero_known=known[k], b_all=mean(k, "b_all"), b_edge=mean(k, "b_edge")) for k in ("amplitude", "flyback_height", "sfr")}
    cons["B1_corner"] = dict(zero_known=known["B1_corner"], **res["B1_corner"])
    res["consistency"] = cons
    summ = {}
    for frac in (0.15, 0.30):
        for ax, sl in (("window", ss), ("onset", LS.ONSET)):
            per = {s: [res["sessions"][s]["delta"][f"{frac}|{a}"][ax] for a in A_PER] for s in sl}
            m = np.mean(np.array(list(per.values())), axis=0); rho = spearmanr(A_PER, m)[0]
            n_ok = int(np.sum(np.array([spearmanr(A_PER, v)[0] for v in per.values()]) >= 0.7)); need = int(np.ceil(11 / 14 * len(sl)))
            zc = T.zero_cross(A_PER, m)
            be = float(np.mean([res["sessions"][s]["b_edge"]["periodicity"] for s in sl])); ba = float(np.mean([res["sessions"][s]["b_all"]["periodicity"] for s in sl]))
            t1 = bool(rho >= 0.9 and n_ok >= need); t2 = bool(zc is not None and abs(zc - be) <= 0.25 and abs(zc - be) < abs(zc - ba))
            uninf = bool(np.max(np.abs(m)) < 0.05)
            summ[f"{frac}|{ax}"] = dict(mean_delta=m.tolist(), rho=float(rho), n_ok=n_ok, need=need, zero_cross=zc, b_edge=be, b_all=ba,
                                        reading="uninformative" if uninf else ("refined rule confirmed" if (t1 and t2) else ("direction only" if t1 else "not confirmed")))
    res["summary"] = summ
    (HERE / "results_edge.json").write_text(json.dumps(res, indent=1, default=float))
    print("consistency (known zero crossings vs b_all / b_edge):")
    for k, v in cons.items():
        print(f"   {k}: zero {v['zero_known']} | b_all {v['b_all']:+.2f} | b_edge {v['b_edge']:+.2f}")
    for k, v in summ.items():
        print(f"periodicity {k}: mean Δ {[round(x, 3) for x in v['mean_delta']]} ρ {v['rho']:+.2f} ({v['n_ok']}/{v['need']}) zero {v['zero_cross']} b_edge {v['b_edge']:+.3f} b_all {v['b_all']:+.3f} → {v['reading']}")


if __name__ == "__main__":
    main()
