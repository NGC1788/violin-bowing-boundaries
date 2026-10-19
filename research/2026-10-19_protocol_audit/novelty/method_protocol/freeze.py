"""Freeze the PACE* estimator, its baselines, the decision rules and the property-only predictions for the
UNSEEN archives (A4, D2, B_T2s2) BEFORE they are processed.  Claude, 2026-10-19.
Usage: freeze.py [--exclude TAG --outdir dryrun]   (exclude/outdir only for the end-to-end dry run on an exposed diagram)
Selection history (all post hoc on the 11 exposed diagrams): round 1 a-priori primary 'full|m3|clip1' (linear beta)
lost to the model-free lookup; round 2 selected 'protocol|m1|clip1|quad_s' by LODO/LOTO interval distance and
log score.  The unseen archives are the first test of that selection."""
import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

import common as C
import model as M

FROZEN_MODELS = {   # name -> constructor
    "PACE*": lambda: M.PACE("protocol", 1, True, shape="quad_s"),
    "full_clip": lambda: M.PACE("full", 1, True, shape="quad_s"),
    "mean_clip": lambda: M.PACE("mean", 1, True, shape="quad_s"),
    "props_clip": lambda: M.PACE("props", 1, True, shape="quad_s"),
    "props_only": lambda: M.PACE("props", 1, False, shape="quad_s"),
    "mean_only": lambda: M.PACE("mean", 1, False, shape="quad_s"),
    "props_only_lin": lambda: M.PACE("props", 1, False, shape="lin"),
    "protocol_noclip": lambda: M.PACE("protocol", 1, False, shape="quad_s"),
}
LOOKUPS = {"lookup_clip": lambda: M.LookupLevel((), True), "lookup": lambda: M.LookupLevel((), False)}
REF = json.loads((C.BASE / "astra_20261019/d_t2_freeze/frozen_parameters.json").read_text())["lower_reference_shape_c_v_beta_logsigma"][:3]
RULES = {
    "scoring_intervals": "lower boundary, measured force axis, 0.2-s window at t=0.2 s, K=2 different strokes, legacy single band, "
                         "delta=0.01 N resolution widening (post hoc fix; ends <= 1 mN -> -inf); no-band columns right-censored",
    "metrics": "interval_distance (all columns, log units), log_score (Gaussian interval probability with model sd), "
               "col_median_abs_err (finite intervals only, midpoint), hit_rate, level_error (|censored-likelihood offset of the "
               "target intervals around the model's own column points|), level_vs_reference (A1+A2 frozen reference shape)",
    "H1_primary": "PACE* vs lookup_clip on interval_distance: diff <= -0.02 -> PACE* better; >= +0.02 -> worse; else indistinguishable",
    "H1b": "PACE* vs lookup_clip log_score: diff >= +0.10 -> PACE* better; <= -0.10 -> worse",
    "H2_expected_loss": "PACE* vs lookup_clip col_median_abs_err: development showed PACE* WORSE (+0.034); diff >= +0.03 counts as worse",
    "H3_protocol_vs_properties": "PACE* vs props_clip interval_distance: diff <= -0.02 -> protocol covariates add beyond properties; "
                                 "props_clip vs mean_clip: diff <= -0.02 -> properties add beyond the mean",
    "H4_floor": "mean_clip vs mean_only interval_distance: diff <= -0.02 -> the label-free floor clip helps",
    "H5_level": "PACE* level_error <= 0.23 (practical tolerance, same as the D_T2 plan; not an equivalence test)",
    "scope": "one diagram per target; practical margins, not significance; failures/unidentifiable fits reported as such",
}


def nominal_geometry():
    """Median log(beta) per (speed folder, column) over the 11 exposed diagrams (fixed design grid)."""
    g = {}
    for t in C.TAGS:
        for r in C.intervals(C.load_exposed(t), "F_ev", None):
            g.setdefault((r["folder"], r["column"]), []).append(r["lb"])
    return [dict(folder=f, column=c, lb=float(np.median(v)), lb_spread=float(np.ptp(v))) for (f, c), v in sorted(g.items())]


def props_prediction(P, group, geom):
    """Prediction that needs no target controls: only valid for models whose covariates are properties."""
    assert all(k in M.PROP_KEYS for k in P["keys"]) and not P["clip"]
    z, ei = C.PROPS[group]
    beta, cov = np.array(P["beta"]), np.array(P["cov"])
    out = []
    for r in geom:
        x = np.r_[np.eye(3)[r["folder"]], [{"logZ": np.log(z), "logEI": np.log(ei)}[k] for k in P["keys"]]]
        mu = float(x @ beta + M.shape_features(P["shape"], r) @ np.array(P["theta"]))
        lv = np.log([0.05, 0.10, 0.20][r["folder"]])
        out.append(dict(folder=r["folder"], column=r["column"], lb=r["lb"], point=mu,
                        sd=float(np.sqrt(P["sigma"] ** 2 + P["tau"] ** 2 + P["omega"] ** 2 + x @ cov @ x)),
                        ref=float(REF[0] + REF[1] * lv + REF[2] * r["lb"])))
    level = float(np.median([o["point"] - o["ref"] for o in out]))
    return out, level


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", default=None)
    ap.add_argument("--outdir", default="frozen")
    a = ap.parse_args()
    outdir = C.HERE / a.outdir
    outdir.mkdir(exist_ok=a.outdir != "frozen")
    tags = [t for t in C.TAGS if t != a.exclude]
    data = {t: (C.load_exposed(t), C.GROUP[t]) for t in tags}
    params = {}
    for name, mk in FROZEN_MODELS.items():
        params[name] = mk().fit(data).params()
    looks = {}
    for name, mk in LOOKUPS.items():
        lk = mk().fit(data)
        looks[name] = dict(clip=lk.clip, keys=list(lk.keys), L=lk.L)
    geom = nominal_geometry()
    targets = {"B_T2s2": "B_T2", "A4": "A_T1", "D2": "D_T1"} if a.exclude is None else {a.exclude: C.GROUP[a.exclude]}
    prop_preds = {}
    for tgt, grp in targets.items():
        prop_preds[tgt] = {}
        for name in ("props_only", "mean_only", "props_only_lin"):
            cols, lev = props_prediction(params[name], grp, geom)
            prop_preds[tgt][name] = dict(level_vs_reference=lev, columns=cols)
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=9)))
    doc = dict(created_kst=now.isoformat(), primary="PACE*", training_tags=tags, reference_shape_c_a_b=REF,
               status=("FROZEN before processing of the target archives; development (post hoc) on the training tags only"
                       if a.exclude is None else f"DRY RUN: {a.exclude} excluded and used as pseudo-target (exposed; not a test)"),
               decision_rules=RULES, models=params, lookups=looks, nominal_geometry=geom,
               targets={t: dict(group=g, stem=(C.FUTURE[t][0] if t in C.FUTURE else C.EXPOSED[t][0]),
                                published_properties_assumed=C.PROPS[g]) for t, g in targets.items()},
               property_only_predictions=prop_preds,
               excluded_from_prospective_claims={"A3": "2024-04-22_TypeA_sample3 finished processing at 10:39 KST, before this "
                                                       "freeze; not opened by this agent, but no prospective claim is made"})
    path = outdir / "frozen_params.json"
    with open(path, "x") as fh:
        json.dump(doc, fh, indent=1, default=float)
    files = ["common.py", "model.py", "freeze.py", "predict_from_controls.py", "score_frozen.py", "validate.py", "validate2.py"]
    receipt = dict(created_kst=now.isoformat(), frozen_params_sha256=C.sha(path),
                   code_sha256={f: C.sha(C.HERE / f) for f in files},
                   development_results_sha256={f: C.sha(C.HERE / f) for f in ("validation_v1_results.json", "validation_v2_results.json")},
                   classifier_sha256={"claude_extra/verify_astra_boundaries.py": C.sha(C.BASE / "claude_extra/verify_astra_boundaries.py")},
                   note="local freeze (not an external registry); targets unseen at freeze time except where stated")
    with open(outdir / "freeze_receipt.json", "x") as fh:
        json.dump(receipt, fh, indent=1)
    print(json.dumps({t: {m: round(v["level_vs_reference"], 4) for m, v in p.items()} for t, p in prop_preds.items()}))
    print("receipt", C.sha(outdir / "freeze_receipt.json"))


if __name__ == "__main__":
    main()
