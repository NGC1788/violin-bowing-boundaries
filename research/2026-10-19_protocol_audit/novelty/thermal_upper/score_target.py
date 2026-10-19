"""Score ONE processed archive against a frozen thermal_upper prediction (Claude, 2026-10-19).
Run only AFTER the archive is 'done' in timeres/replicate_state.json, once, and keep the output.
Usage: python score_target.py <archive stem> <FROZEN json> [group]   e.g.
       python score_target.py 2024-04-24_TypeA_sample4 FROZEN_v2.json
       python score_target.py 2024-06-17_TypeD_T2_sample1 FROZEN_v2.json "D T2"   (dry run on an already-seen diagram)
group (string type for the published Z) defaults to the frozen target's group, else 'B T2' (high-force archives).
Writes score_<stem>__<FROZEN stem>.json. Implements the scoring_rule text stored in the frozen file."""
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import rows as R
import thermal_law as TL
import thermal_law2 as TL2

HERE = Path(__file__).parent
STEP = np.log(40.0) / 49
NB = 500


def free_fit(lv, lb, lo, hi, x0=None):
    X = np.column_stack([np.ones_like(lv), lv, lb])
    f = lambda p: TL.censored_nll(X @ p[:3], np.exp(p[3]), lo, hi)
    x0 = [0.0, 0.7, -1.0, np.log(0.3)] if x0 is None else x0
    return minimize(f, x0, method="Nelder-Mead", options={"maxiter": 40000, "xatol": 1e-8, "fatol": 1e-10}).x


def main():
    stem, fz = sys.argv[1], sys.argv[2]
    F = json.loads((HERE / fz).read_text())
    fz_hash = hashlib.sha256((HERE / fz).read_bytes()).hexdigest()
    tgt = F.get("targets", {}).get(stem)
    grp = sys.argv[3] if len(sys.argv) > 3 else (tgt["group"] if tgt else "B T2")
    z = R.PROP[grp][1]
    tag = "TGT_" + stem
    R.D.STEMS[tag] = stem
    R.ZPUB[tag] = z
    rs = R.rows_for(tag)
    lv, lb, lz, lo, hi = TL.arrays(rs, R.ZPUB)
    out = dict(stem=stem, frozen=fz, frozen_sha256=fz_hash, frozen_created=F["created_kst"], group=grp, Z=z,
               scored_kst=dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(), n_columns=len(rs),
               n_finite=int(np.sum(np.isfinite(lo) & np.isfinite(hi))), n_right=int(np.sum(np.isfinite(lo) & ~np.isfinite(hi))),
               n_left=int(np.sum(~np.isfinite(lo) & np.isfinite(hi))), models={})
    p = free_fit(lv, lb, lo, hi)
    rng = np.random.default_rng(2026)
    bs = np.array([free_fit(lv[ii], lb[ii], lo[ii], hi[ii], p) for ii in (rng.integers(0, len(lv), len(lv)) for _ in range(NB))])
    out["observed"] = dict(e_v=float(p[1]), e_v_ci=np.percentile(bs[:, 1], [2.5, 97.5]).tolist(), e_beta=float(p[2]),
                           e_beta_ci=np.percentile(bs[:, 2], [2.5, 97.5]).tolist(),
                           level_A=TL.offset_fit(R.base_A(lv, lb), lo, hi)[0])
    for m, f in F["fits"].items():
        if m not in TL2.MODELS:
            continue
        mu = TL2.predict(m, f["theta"], lv, lb, lz)
        fin = np.isfinite(lo) & np.isfinite(hi)
        hit = np.mean([(q >= a - STEP if np.isfinite(a) else True) and (q <= b + STEP if np.isfinite(b) else True) for q, a, b in zip(mu, lo, hi)])
        ex = TL2.design_exponents(m, f["theta"], lv, lb, lz)
        pi = f.get("pi90", {})
        le = TL.offset_fit(mu, lo, hi)[0]
        out["models"][m] = dict(level_error=le, level_within_pi90=(abs(le) <= pi["level"]) if pi else None,
                                nll_per_col=float(TL.censored_nll(mu, f["sigma"], lo, hi) / len(rs)),
                                med_abs_finite=float(np.median(np.abs(mu[fin] - 0.5 * (lo[fin] + hi[fin])))) if fin.any() else None,
                                hit=float(hit), e_v_pred=ex["e_v"], e_beta_pred=ex["e_beta"],
                                e_v_err=ex["e_v"] - float(p[1]), e_beta_err=ex["e_beta"] - float(p[2]))
    ms = out["models"]
    verdict = {}
    for th in ("T1k_break", "T1mu", "T1k_peak"):
        if th in ms and "S1_strict" in ms:
            d = abs(ms[th]["level_error"]) - abs(ms["S1_strict"]["level_error"])
            verdict[th + "_vs_S1_level"] = "not distinguishable" if abs(d) < 0.03 else (th if d < 0 else "S1_strict")
            verdict[th + "_vs_S1_nll"] = ms[th]["nll_per_col"] - ms["S1_strict"]["nll_per_col"]
    eb, ci = out["observed"]["e_beta"], out["observed"]["e_beta_ci"]
    zone = lambda x: "H_thermal" if x >= -0.95 else ("H_plain" if x > -1.05 else "H_steep")
    verdict["beta_exponent_zone"] = zone(eb)
    verdict["beta_exponent_ci_zones"] = sorted({zone(ci[0]), zone(ci[1]), zone(eb)})
    verdict["beta_decidable"] = len(verdict["beta_exponent_ci_zones"]) == 1
    out["verdict"] = verdict
    p_out = HERE / f"score_{stem}__{Path(fz).stem}.json"
    p_out.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(dict(observed=out["observed"], verdict=verdict, n=out["n_columns"], finite=out["n_finite"], right=out["n_right"]), indent=1, default=float))
    for m, x in ms.items():
        print(f"  {m:13s} level err {x['level_error']:+.3f} (PI90 ok {x['level_within_pi90']}) NLL {x['nll_per_col']:.3f} "
              f"e_v {x['e_v_pred']:+.2f} (err {x['e_v_err']:+.2f}) e_b {x['e_beta_pred']:+.2f} (err {x['e_beta_err']:+.2f})")


if __name__ == "__main__":
    main()
