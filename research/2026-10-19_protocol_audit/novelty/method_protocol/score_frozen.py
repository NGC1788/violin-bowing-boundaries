"""Stage B: score a frozen controls prediction ONCE. Creates a single-use claim before any label is read.
Usage: score_frozen.py --prediction frozen/B_T2s2_controls_prediction.json --out frozen/B_T2s2_score.json [--frozen frozen]
Claude, 2026-10-19."""
import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

import common as C
import model as M
from predict_from_controls import verify


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frozen", default="frozen")
    a = ap.parse_args()
    fdir = C.HERE / a.frozen
    verify(fdir)
    ppath = Path(a.prediction) if Path(a.prediction).is_absolute() else C.HERE / a.prediction
    PR = json.loads(ppath.read_text())
    F = json.loads((fdir / "frozen_params.json").read_text())
    if PR["freeze_receipt_sha256"] != C.sha(fdir / "freeze_receipt.json"):
        raise RuntimeError("prediction was made with a different freeze")
    feat = Path(PR["features_path"])
    if C.sha(feat) != PR["input_sha256"]:
        raise RuntimeError("feature file changed since the controls prediction")
    claim = ppath.with_suffix(".score_claim.json")
    with open(claim, "x") as fh:     # single use: exists -> refuse
        json.dump(dict(created_kst=dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(),
                       prediction_sha256=C.sha(ppath)), fh)
    d = C.labels_after_claim(feat, PR["rep"], claim)
    rows = C.intervals(d, "F_ev", "H2", "delta", 0.01)
    legacy = C.intervals(d, "F_ev", "H2", "none")
    n_zero = int(sum(1 for r in legacy if not r["lo"] < r["hi"]))
    scores = {}
    for name, pr in PR["predictions"].items():
        s = M.score(rows, pr)
        s.pop("fin_errors")
        scores[name] = s
    ref = F["reference_shape_c_a_b"]
    mu = np.array([ref[0] + ref[1] * r["lv"] + ref[2] * r["lb"] for r in rows])
    obs_level = C.fit_offset(rows, mu)
    lvp = PR["level_vs_reference"]
    pp = F["property_only_predictions"].get(PR["target"], {})
    level_cmp = {n: abs(obs_level["level"] - v) for n, v in lvp.items()} if obs_level.get("success") else None
    if level_cmp is not None:
        level_cmp.update({f"{n} (frozen without controls)": abs(obs_level["level"] - v["level_vs_reference"]) for n, v in pp.items()})
    S = lambda m, k: scores[m][k]
    dec = {}
    dd = S("PACE*", "interval_distance") - S("lookup_clip", "interval_distance")
    dec["H1_primary"] = dict(diff=dd, verdict="PACE* better" if dd <= -.02 else "PACE* worse" if dd >= .02 else "indistinguishable")
    dl = S("PACE*", "log_score") - S("lookup_clip", "log_score")
    dec["H1b"] = dict(diff=dl, verdict="PACE* better" if dl >= .1 else "PACE* worse" if dl <= -.1 else "indistinguishable")
    dm = S("PACE*", "col_median_abs_err") - S("lookup_clip", "col_median_abs_err")
    dec["H2_expected_loss"] = dict(diff=dm, verdict="PACE* worse" if dm >= .03 else "not worse")
    d3 = S("PACE*", "interval_distance") - S("props_clip", "interval_distance")
    d3b = S("props_clip", "interval_distance") - S("mean_clip", "interval_distance")
    dec["H3_protocol_vs_properties"] = dict(protocol_minus_props=d3, props_minus_mean=d3b,
                                            protocol_adds=bool(d3 <= -.02), properties_add=bool(d3b <= -.02))
    d4 = S("mean_clip", "interval_distance") - S("mean_only", "interval_distance")
    dec["H4_floor"] = dict(diff=d4, floor_helps=bool(d4 <= -.02))
    le = S("PACE*", "level_error")
    dec["H5_level"] = dict(level_error=le, within_0_23=(bool(le <= .23) if np.isfinite(le) else "검증 불가"))
    out = dict(stage="score_complete", created_kst=dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(),
               target=PR["target"], prediction_sha256=C.sha(ppath), n_zero_width_legacy=n_zero,
               observed_level_vs_reference=obs_level, level_abs_error_vs_reference=level_cmp,
               scores=scores, decisions=dec, rules=F["decision_rules"],
               n_no_band=int(sum(1 for r in rows if r["band"] is False)),
               n_left_censored=int(sum(1 for r in rows if not np.isfinite(r["lo"]))))
    with open(Path(a.out) if Path(a.out).is_absolute() else C.HERE / a.out, "x") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(json.dumps(dict(decisions=dec, observed_level=obs_level.get("level")), default=float, indent=1))


if __name__ == "__main__":
    main()
