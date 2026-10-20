"""Label-free load transform vs window-axis exponents (plan: README.md). Claude, 2026-10-20 (post hoc)."""
import csv, json, sys
from pathlib import Path
import numpy as np
H = Path.home(); sys.path.insert(0, str(H / "violin-rne/novelty/lower_law"))
import onset_axis as OA  # noqa: E402  (RAMP paths, TAG map, session loader)
SV = OA.SV
HERE = Path(__file__).resolve().parent
SESS = ["A1", "A2", "C1", "D1", "A_T2", "B_T2s1", "C_T2", "D_T2", "A3", "A4", "D2"]


def transform(sess):
    rm = {}
    for r in csv.DictReader(open(OA.RAMP[sess])):
        if not r["error"] and r["force_at_onset"] not in ("", "nan"):
            rm[(r["folder"], int(r["trial"]))] = float(r["force_at_onset"])
    d = dict(SV.R.load(OA.TAG.get(sess, sess)))
    m = np.isclose(d["t"], 0.2)
    fon = np.array([rm.get((d["folders"][fo], int(tr)), np.nan) for fo, tr in zip(d["folder"][m], d["trial"][m])])
    fw = d["force"][m]; v = d["speed"][m]; b = d["beta"][m]
    k = np.isfinite(fon) & (fon >= 0.05) & (fon <= 1.5) & (fw > 0.005)
    X = np.column_stack([np.ones(k.sum()), np.log(fon[k]), np.log(v[k]), np.log(b[k])])
    coef = np.linalg.lstsq(X, np.log(fw[k]), rcond=None)[0]
    return dict(alpha=float(coef[0]), one_plus_c=float(coef[1]), d_v=float(coef[2]), d_beta=float(coef[3]), n=int(k.sum()))


def main():
    on = json.load(open(H / "violin-rne/verify_lower_law/axis/axis_results.json"))["sessions"]
    out = {}
    for s in SESS:
        tf = transform(s)
        fz = on[s]["fits"]; oo = fz["onset"]["legacy_fit"]; ww = fz["window"].get("legacy_fit") if fz.get("window") else None
        o = [None, oo["b_v"], oo["b_beta"]]; w = [None, ww["b_v"], ww["b_beta"]] if ww else None
        pred_v = tf["one_plus_c"] * o[1] + tf["d_v"]; pred_b = tf["one_plus_c"] * o[2] + tf["d_beta"]
        out[s] = dict(transform=tf, onset=dict(b_v=o[1], b_beta=o[2]), window_obs=dict(b_v=w[1], b_beta=w[2]) if w else None,
                      window_pred=dict(b_v=pred_v, b_beta=pred_b))
    ok = [s for s in SESS if out[s]["window_obs"]]
    ob = np.array([out[s]["window_obs"]["b_beta"] for s in ok]); pb = np.array([out[s]["window_pred"]["b_beta"] for s in ok]); onb = np.array([out[s]["onset"]["b_beta"] for s in ok])
    ov = np.array([out[s]["window_obs"]["b_v"] for s in ok]); pv = np.array([out[s]["window_pred"]["b_v"] for s in ok]); onv = np.array([out[s]["onset"]["b_v"] for s in ok])
    summ = dict(n=len(ok), beta=dict(mae=float(np.mean(np.abs(ob - pb))), r=float(np.corrcoef(ob, pb)[0, 1]), shift_obs=float(np.mean(ob - onb)), shift_pred=float(np.mean(pb - onb)),
                                     sd_onset=float(np.std(onb, ddof=1)), sd_window_obs=float(np.std(ob, ddof=1)), sd_window_pred=float(np.std(pb, ddof=1))),
                v=dict(mae=float(np.mean(np.abs(ov - pv))), r=float(np.corrcoef(ov, pv)[0, 1]), shift_obs=float(np.mean(ov - onv)), shift_pred=float(np.mean(pv - onv))))
    summ["verdict"] = "mechanism supported" if abs(summ["beta"]["shift_pred"] - summ["beta"]["shift_obs"]) < 0.07 and summ["beta"]["r"] > 0.7 else "not supported"
    out["_summary"] = summ
    (HERE / "results.json").write_text(json.dumps(out, indent=1, default=float))
    for s in ok:
        x = out[s]
        print(f"{s:7s} 1+c {x['transform']['one_plus_c']:.3f} d_v {x['transform']['d_v']:+.3f} d_β {x['transform']['d_beta']:+.3f} (n {x['transform']['n']}) | β on {x['onset']['b_beta']:+.2f} → pred {x['window_pred']['b_beta']:+.2f} obs {x['window_obs']['b_beta']:+.2f} | v on {x['onset']['b_v']:+.2f} → pred {x['window_pred']['b_v']:+.2f} obs {x['window_obs']['b_v']:+.2f}")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
