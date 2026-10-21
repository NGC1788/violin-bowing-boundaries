"""Generalisation tests of the threshold-tilt principle (PLAN_generalize.md). Claude, 2026-10-21."""
import json, sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import tilt as T  # noqa: E402
LS = T.LS; C = LS.C

A_SFR = [-1.0, -0.5, 0.0, 0.5, 1.0]
A_AMP = T.A_LOW


def fit_upper(d):
    rows = [r for r in C.intervals(d, 0.2) if np.isfinite(r["upper"][0]) or np.isfinite(r["upper"][1])]
    p = C.censored_fit(rows, lambda r: (np.array([1.0, r["lv"], r["lb"]]), 0.0), "upper")
    return float(p[2])


def fits_upper(d, sess):
    out = {"window": fit_upper(d)}
    if sess in LS.ONSET:
        out["onset"] = fit_upper(LS.to_onset(d, sess))
    return out


def evaluate(a_list, per, b_vals, sign):
    m = np.mean(np.array(list(per.values())), axis=0)
    rho = spearmanr(a_list, m)[0]; rs = [spearmanr(a_list, v)[0] for v in per.values()]
    n_ok = int(np.sum(np.array(rs) * sign >= 0.7)); need = int(np.ceil(11 / 14 * len(rs)))
    zc = T.zero_cross(a_list, m); bm = float(np.mean(b_vals))
    t1 = bool(rho * sign >= 0.9 and n_ok >= need); t2 = bool(zc is not None and abs(zc - bm) <= 0.25)
    return dict(mean_delta=m.tolist(), rho=float(rho), n_ok=n_ok, need=need, zero_cross=zc, b_mean=bm, T1=t1, T2=t2,
                reading="generalises" if (t1 and t2) else ("direction only" if t1 else "does not generalise"))


def main():
    res = {"sessions": {}}
    for sess, st in LS.STEM.items():
        stem, rep = st if isinstance(st, tuple) else (st, None)
        raw, _ = T.read2(stem, rep)
        d0 = LS.build(raw, T.BASE); H0 = d0["H"]; beta = d0["beta"]; folder = d0["folder"]
        sfr = np.array([r[7] for r in raw[0]]); amp = np.array([r[3] for r in raw[0]])
        e = {"b_sfr": T.slope_b(sfr, beta, folder, H0), "b_amp": T.slope_b(amp, beta, folder, H0),
             "base_min": T.fits(d0, sess), "base_max": fits_upper(d0, sess), "g1": {}, "g2": {}}
        for frac in (0.15, 0.30):
            for a in A_SFR:
                f = T.fits(dict(d0, H=T.tilted(sfr, beta, folder, H0, a, frac, upper=True)), sess)
                e["g1"][f"{frac}|{a}"] = {ax: f[ax] - e["base_min"][ax] for ax in f}
            for a in A_AMP:
                f = fits_upper(dict(d0, H=T.tilted(amp, beta, folder, H0, a, frac)), sess)
                e["g2"][f"{frac}|{a}"] = {ax: f[ax] - e["base_max"][ax] for ax in f}
        res["sessions"][sess] = e
        print(f"{sess:7s} b_sfr {e['b_sfr']:+.2f} | G1 Δmin(a) " + " ".join(f"{e['g1'][f'0.15|{a}']['window']:+.2f}" for a in A_SFR)
              + f" | base Fmax β {e['base_max']['window']:+.2f} G2 Δmax(a) " + " ".join(f"{e['g2'][f'0.15|{a}']['window']:+.2f}" for a in A_AMP), flush=True)
    summ = {}
    for frac in (0.15, 0.30):
        for ax, ss in (("window", list(LS.STEM)), ("onset", LS.ONSET)):
            summ[f"G1|{frac}|{ax}"] = evaluate(A_SFR, {s: [res["sessions"][s]["g1"][f"{frac}|{a}"][ax] for a in A_SFR] for s in ss},
                                               [res["sessions"][s]["b_sfr"] for s in ss], -1)
            summ[f"G2|{frac}|{ax}"] = evaluate(A_AMP, {s: [res["sessions"][s]["g2"][f"{frac}|{a}"][ax] for a in A_AMP] for s in ss},
                                               [res["sessions"][s]["b_amp"] for s in ss], -1)
    b_sfr_all = [res["sessions"][s]["b_sfr"] for s in LS.STEM]
    summ["flyback_fraction_consistency"] = dict(b_sfr_mean=float(np.mean(b_sfr_all)), n_positive=int(np.sum(np.array(b_sfr_all) > 0)),
                                                observed="f 0.3 → +0.169, f 0.7 → −0.171 (window)",
                                                explained=bool(np.mean(b_sfr_all) > 0))
    res["summary"] = summ
    (HERE / "results_generalize.json").write_text(json.dumps(res, indent=1, default=float))
    for k, v in summ.items():
        if k.startswith("flyback"):
            print(k, v)
        else:
            print(f"{k}: mean Δ {[round(x, 3) for x in v['mean_delta']]} ρ {v['rho']:+.2f} ({v['n_ok']}/{v['need']} needed) zero {v['zero_cross']} b {v['b_mean']:+.2f} → {v['reading']}")


if __name__ == "__main__":
    main()
