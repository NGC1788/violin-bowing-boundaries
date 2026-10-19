"""Stage A (label free): frozen PACE* + baselines -> column predictions for one target archive.
Reads ONLY control columns (folder, trial, beta, window_s, t, force, speed, error). Exclusive-create output.
Usage:
  predict_from_controls.py --target B_T2s2 --features ~/violin-rne/timeres/features/2024-07-03_TypeB_T2_sample2.csv.gz \
      [--rep rep_1] [--frozen frozen] --out frozen/B_T2s2_controls_prediction.json
Claude, 2026-10-19."""
import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

import common as C
import model as M


def verify(frozen_dir):
    rc = json.loads((frozen_dir / "freeze_receipt.json").read_text())
    if C.sha(frozen_dir / "frozen_params.json") != rc["frozen_params_sha256"]:
        raise RuntimeError("frozen_params.json changed after freeze")
    for f, h in rc["code_sha256"].items():
        if C.sha(C.HERE / f) != h:
            raise RuntimeError(f"{f} changed after freeze")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--rep", default=None)
    ap.add_argument("--frozen", default="frozen")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fdir = C.HERE / a.frozen
    rc = verify(fdir)
    F = json.loads((fdir / "frozen_params.json").read_text())
    if a.target not in F["targets"]:
        raise ValueError("target not frozen")
    group = F["targets"][a.target]["group"]
    feat = Path(a.features).expanduser()
    d = C.controls_only(None, a.rep, feat)                       # control columns only
    preds, info = {}, {}
    for name, P in F["models"].items():
        pr, inf = M.predict_from_params(P, d, group)
        preds[name] = pr
        info[name] = dict(extrapolated=inf["extrapolated"], session_intercepts=inf["session_intercepts"])
    for name, lk in F["lookups"].items():
        mod = M.LookupLevel(lk["keys"], lk["clip"]); mod.L = lk["L"]
        preds[name], _ = mod.predict(d, group)
    ref = F["reference_shape_c_a_b"]
    geo = C.intervals(d, "F_ev", None)
    refcol = {(r["folder"], r["column"]): ref[0] + ref[1] * r["lv"] + ref[2] * r["lb"] for r in geo}
    level_vs_ref = {n: float(np.nanmedian([p["point"] - refcol[(p["folder"], p["column"])] for p in pr])) for n, pr in preds.items()}
    cov = C.session_covariates(d)
    out = dict(stage="controls_prediction_frozen", created_kst=dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(),
               target=a.target, group=group, features_path=str(feat), rep=a.rep, input_sha256=C.sha(feat),
               parsed_fields=C.CTRL_FIELDS, outcomes_parsed=False, freeze_receipt_sha256=C.sha(fdir / "freeze_receipt.json"),
               meta=d["meta"], covariates=cov, model_info=info, level_vs_reference=level_vs_ref,
               floors_log=[r["floor1"] for r in geo],
               predictions={n: [{k: p[k] for k in ("folder", "column", "lb", "point", "sd")} for p in pr] for n, pr in preds.items()})
    with open(Path(a.out) if Path(a.out).is_absolute() else C.HERE / a.out, "x") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(json.dumps(dict(target=a.target, covariates=cov["diagram"], rank_reversed=d["meta"]["rank_reversed"],
                          level_vs_reference={k: round(v, 4) for k, v in level_vs_ref.items()},
                          extrapolated=info["PACE*"]["extrapolated"]), default=float))


if __name__ == "__main__":
    main()
