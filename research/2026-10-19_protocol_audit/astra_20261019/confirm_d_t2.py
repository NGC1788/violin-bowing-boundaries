"""Two-stage D_T2 confirmation runner, prepared before target access.

No feature path is guessed. freeze-controls parses only control columns; score
requires the resulting exclusive receipt and verifies every locked dependency.
Import has no filesystem side effects. CLI never downloads or signals processes.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pyarrow.csv as pc
from scipy.optimize import minimize
from scipy.special import log_ndtr

BASE = Path(__file__).resolve().parents[1]
REPO = Path.home()/"violin-bowing-boundaries"
STEM = "2024-06-17_TypeD_T2_sample1"
CONTROL_FIELDS = ("folder", "trial", "window_s", "t", "force", "speed", "beta", "error")
OUTCOME_FIELDS = ("amplitude_std", "periodicity", "f0_hz", "flybacks_per_period")
TIMES = (0., .1, .2, .3)
SEED, N_BOOT = 20261019, 2000
STEP = np.log(40.)/49


class ConfirmationError(ValueError):
    pass


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def json_value(x):
    if isinstance(x, dict): return {str(k):json_value(v) for k,v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)): return [json_value(v) for v in x]
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.bool_,)): return bool(x)
    if isinstance(x, (np.floating, float)):
        if np.isnan(x): return None
        if np.isposinf(x): return "+inf"
        if np.isneginf(x): return "-inf"
        return float(x)
    return x


def encode(x):
    return json.dumps(json_value(x), ensure_ascii=False, indent=2, allow_nan=False)+"\n"


def exclusive_json(path, value):
    with Path(path).open("x") as f:
        f.write(encode(value))


def selected_function(path, name):
    """Load only a reviewed function AST: no source module top-level execution."""
    tree = ast.parse(Path(path).read_text())
    matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(matches) != 1 or matches[0].decorator_list:
        raise ConfirmationError(f"Expected one undecorated function: {name}")
    code = compile(ast.Module(body=matches, type_ignores=[]), str(path), "exec")
    namespace = {"np":np, "math":math, "defaultdict":defaultdict}
    exec(code, namespace)
    return namespace[name]


def verify_original_freeze(base=BASE):
    base = Path(base)
    bundle = base/"astra_20261019/d_t2_freeze"
    receipt_path, params_path = bundle/"freeze_receipt.json", bundle/"frozen_parameters.json"
    receipt = json.loads(receipt_path.read_text())
    plan = base/"final/CONFIRM_PLAN_D_T2.md"
    if sha(plan) != receipt["plan_sha256"] or sha(params_path) != receipt["parameters_sha256"]:
        raise ConfirmationError("Original frozen plan or parameters changed")
    params = json.loads(params_path.read_text())
    for relative, expected in params["source_sha256"].items():
        path = (base/relative).resolve()
        if not path.is_relative_to(base.resolve()) or sha(path) != expected:
            raise ConfirmationError(f"Original dependency hash mismatch: {relative}")
    return params, {str(plan.resolve()):sha(plan), str(params_path.resolve()):sha(params_path),
                    str(receipt_path.resolve()):sha(receipt_path),
                    **{str((base/p).resolve()):h for p,h in params["source_sha256"].items()}}


def runtime_dependencies(base=BASE, repo=REPO):
    """Extend the original non-transitive manifest using source-only artifacts."""
    summaries = list((Path(repo)/"reports/research_20261013/lockbox_A2_preprocess/2024-04-19_TypeA_sample2/regime_map").glob("*/summary.json"))
    if len(summaries) != 1:
        raise ConfirmationError("Classifier threshold summary must resolve uniquely")
    paths = {"descriptor":Path(base)/"astra_20261019/descriptor_only.py",
             "columns":Path(base)/"claude_extra/verify_astra_boundaries.py",
             "classifier":Path(repo)/"scripts/regime_map.py", "thresholds":summaries[0],
             "runner":Path(__file__)}
    return {name:{"path":str(p.resolve()), "sha256":sha(p)} for name,p in paths.items()}


def verify_manifest(manifest):
    for path, expected in manifest.items():
        if sha(path) != expected:
            raise ConfirmationError(f"Dependency changed after controls freeze: {path}")


def read_columns(path, fields):
    table = pc.read_csv(Path(path), convert_options=pc.ConvertOptions(include_columns=list(fields)))
    return {f:table[f].to_numpy(zero_copy_only=False) for f in fields}


def control_digest(d):
    """Hash parsed controls including row order, independent of outcome fields."""
    h = hashlib.sha256()
    for k in CONTROL_FIELDS:
        h.update(k.encode()+b"\0")
        h.update(json.dumps(json_value(d[k]), separators=(",",":"),allow_nan=False).encode())
    return h.hexdigest()


def control_inventory(d, columns_function, *, expected_stem=STEM):
    if any(x not in (None, "") for x in d["error"]):
        raise ConfirmationError("Feature error rows exist; no silent exclusion permitted")
    sizes = {len(d[k]) for k in CONTROL_FIELDS}
    if len(sizes) != 1:
        raise ConfirmationError("Unequal control field lengths")
    for key in ("trial", "window_s", "t", "force", "speed", "beta"):
        if not np.all(np.isfinite(d[key])):
            raise ConfirmationError(f"Nonfinite controls: {key}")
    if np.any(d["beta"] <= 0) or np.any(d["speed"] <= 0):
        raise ConfirmationError("Nonpositive beta or speed")
    if not np.all(d["trial"] == np.asarray(d["trial"],int)):
        raise ConfirmationError("Noninteger trial ID")
    m = np.isclose(d["window_s"], .2, atol=1e-10, rtol=0)
    folders = sorted(set(d["folder"][m]))
    if len(folders) != 3 or any(expected_stem not in f for f in folders):
        raise ConfirmationError("Unexpected target identity or number of speed folders")
    beta, snapshots = {}, defaultdict(dict)
    for i in np.flatnonzero(m):
        key = (str(d["folder"][i]), int(d["trial"][i]))
        # Exactly the final/data.py last-valid-.2-window beta convention.
        beta[key] = float(d["beta"][i])
        t = round(float(d["t"][i]), 2)
        if t in TIMES:
            if t in snapshots[key]:
                raise ConfirmationError("Duplicate snapshot trial/time")
            snapshots[key][t] = int(i)
    for folder in folders:
        keys = [k for k in snapshots if k[0] == folder]
        if sorted(k[1] for k in keys) != list(range(1,2001)):
            raise ConfirmationError("Expected 2,000 distinct acquired trials per folder")
        if any(set(snapshots[k]) != set(TIMES) for k in keys):
            raise ConfirmationError("Missing one of four frozen snapshot times")
    columns = columns_function(beta)
    folder_speed = {f:float(np.median([d["speed"][i] for k in snapshots if k[0]==f for i in snapshots[k].values()])) for f in folders}
    if not np.allclose(sorted(folder_speed.values()), [.05,.1,.2], atol=1e-10,rtol=0):
        raise ConfirmationError("Unexpected nominal speed inventory")
    rows = []
    for f in folders:
        for c in range(40):
            keys = sorted(k for k in snapshots if k[0]==f and columns[k]==c)
            if len(keys) != 50:
                raise ConfirmationError("Expected 40 beta columns of 50 strokes")
            idx = [snapshots[k][.2] for k in keys]
            rows.append({"folder":f,"column":c,"lv":float(np.log(folder_speed[f])),
                         "lb":float(np.log(np.median([beta[k] for k in keys]))),
                         "trials":[k[1] for k in keys], "indices":idx})
    return rows, {"folders":folders,"speed_by_folder":folder_speed,"n_strokes":6000,
                  "n_beta_columns":40,"n_boundary_rows":120,"n_four_time_snapshots":24000,
                  "n_nonpositive_force_rows":int(np.sum(d["force"]<=0)),
                  "column_rule":"frozen sort-beta into 40 groups of 50; same ordinal beta column paired across speeds"}


def frozen_predictions(params, descriptor, rows):
    z = params["D_T2_published_properties"]["Z"]
    p = params["protocol_model"]
    dec = descriptor["dec5_pl_N"]
    if not np.isfinite(dec): raise ConfirmationError("Nonfinite descriptor")
    levels = {"protocol":p["intercept"]+p["slope_per_N"]*dec,
              "mean":params["lower_mean_only_prediction"],
              **{name:v["prediction_D_T2"] for name,v in params["property_models"].items()}}
    lo,hi = p["training_descriptor_range_N"]
    shape = params["lower_reference_shape_c_v_beta_logsigma"]
    emp = params["upper_empirical_reference"]
    lower_base = [shape[0]+shape[1]*r["lv"]+shape[2]*r["lb"] for r in rows]
    strict = [params["upper_strict_log_c"]+np.log(z)+r["lv"]-r["lb"] for r in rows]
    empirical = [params["upper_empirical_log_c"]+emp["c"]+emp["a_v"]*r["lv"]+emp["b_beta"]*r["lb"]+np.log(z) for r in rows]
    return {"lower_levels":levels,"descriptor_extrapolation":bool(not lo<=dec<=hi),
            "lower_reference":lower_base,"lower_point_predictions":{k:(np.array(lower_base)+v).tolist() for k,v in levels.items()},
            "upper_strict":strict,"upper_strict_no_Z":(np.array(strict)-np.log(z)).tolist(),
            "upper_empirical_secondary":empirical,"Z":z}


def seal_path(path):
    return Path(str(path)+".seal.json")


def freeze_controls(features, output, *, base=BASE, repo=REPO):
    if Path(output).exists(): raise FileExistsError(output)
    if seal_path(output).exists(): raise FileExistsError(seal_path(output))
    params, original = verify_original_freeze(base)
    deps = runtime_dependencies(base, repo)
    d = read_columns(features, CONTROL_FIELDS)  # no H / bridge / outcome field requested
    columns = selected_function(deps["columns"]["path"], "columns")
    rows, inventory = control_inventory(d, columns)
    descriptor_fn = selected_function(deps["descriptor"]["path"], "from_columns")
    descriptor = descriptor_fn({k:d[k] for k in ("folder","trial","window_s","t","force","speed","error")})
    pred = frozen_predictions(params, descriptor, rows)
    record = {"stage":"controls_frozen","created_utc":datetime.now(timezone.utc).isoformat(),
              "features_path":str(Path(features).resolve()),"input_sha256":sha(features),
              "control_sha256":control_digest(d),"parsed_fields":list(CONTROL_FIELDS),
              "outcomes_parsed":False,"original_manifest":original,"runtime_dependencies":deps,
              "thresholds":json.loads(Path(deps["thresholds"]["path"]).read_text())["thresholds_applied"],
              "inventory":inventory,"descriptor":descriptor,"rows":rows,"predictions":pred,
              "bootstrap":{"count":N_BOOT,"seed":SEED,"cluster":"40 beta positions, all three speeds jointly",
                           "conditioning":"fixed prediction; excludes training coefficients, descriptor and new-session uncertainty"},
              "execution_choices":{"likelihood":"stable normal interval likelihood; single Nelder-Mead start [0,log(.5)], legacy defaults",
                  "identification":"reject one-sided-only, overlapping-all-residual intervals, optimizer failure/nonfinite/sigma collapse",
                  "primary_only":True,"secondary_variants":"K3,t=.1/.3,nominal axis deferred; cannot reverse primary verdict",
                  "original_plan_unchanged":True,"hash_note":"input SHA256 reads opaque bytes only; no outcome column is parsed"}}
    verify_manifest(original)
    verify_manifest({v["path"]:v["sha256"] for v in deps.values()})
    exclusive_json(output,record)
    exclusive_json(seal_path(output), {"controls_receipt_sha256":sha(output),
                   "runner_sha256":sha(__file__), "original_manifest":original})
    return record


def log_interval_probability(lo, hi, mu, sigma):
    a,b = (lo-mu)/sigma,(hi-mu)/sigma
    left = log_ndtr(np.where(a>0,-a,b)); right = log_ndtr(np.where(a>0,-b,a))
    with np.errstate(divide="ignore",invalid="ignore",over="ignore"):
        return left+np.log(-np.expm1(np.minimum(right-left,0)))


def fit_level(intervals, offset):
    intervals = np.asarray(intervals,float); offset = np.asarray(offset,float)
    valid = ~np.isnan(intervals).any(axis=1)
    iv, off = intervals[valid], offset[valid]
    common = {"n_total":len(intervals),"n_used":int(valid.sum()),"n_unavailable":int((~valid).sum())}
    def failed(reason, **extra): return {**common,"success":False,"reason":reason,**extra}
    if not len(iv): return failed("no_available_intervals")
    lo,hi = iv.T
    if np.any(lo>=hi) or not np.all(np.isfinite(off)):
        return failed("invalid_or_zero_width_interval")
    if not np.isfinite(lo).any() or not np.isfinite(hi).any():
        return failed("one_sided_location_not_identified")
    if np.max(lo-off) <= np.min(hi-off):
        return failed("common_interval_sigma_collapse_location_not_identified")
    def nll(p):
        if not np.all(np.isfinite(p)) or not -50 < p[1] < 50: return 1e100
        lp = log_interval_probability(lo,hi,off+p[0],np.exp(p[1]))
        return -float(lp.sum()) if np.all(np.isfinite(lp)) else 1e100
    # Same single start/default options as frozen confirm_A_T2.offset_fit.
    r = minimize(nll,[0.,np.log(.5)],method="Nelder-Mead")
    extra={"optimizer_success":bool(r.success),"message":str(r.message),"nll":float(r.fun),"nit":int(r.nit)}
    if not r.success or not np.all(np.isfinite(r.x)) or r.fun >= 1e90:
        return failed("optimizer_failed",**extra)
    if r.x[1] <= -49 or r.x[1] >= 49:
        return failed("sigma_unidentified_or_numerical_boundary",**extra)
    return {**common,"success":True,"level":float(r.x[0]),"sigma":float(np.exp(r.x[1])),**extra}


def observed_boundaries(d, rows, classifier, thresholds):
    result, counts = [], Counter()
    for r in rows:
        ii = sorted(r["indices"],key=lambda i:(d["force"][i],i))
        force = np.log(np.maximum(d["force"][ii],1e-3))
        h=[]
        for i in ii:
            try:
                fpp=d["flybacks_per_period"][i]
                fpp=None if fpp in (None,"","nan","None") or not np.isfinite(float(fpp)) else int(float(fpp))
                lab=classifier(float(d["amplitude_std"][i]),float(d["periodicity"][i]),
                               float(d["f0_hz"][i]),fpp,98.08,thresholds)
                counts[lab]+=1;h.append(lab=="helmholtz")
            except (TypeError,ValueError,OverflowError) as exc:
                raise ConfirmationError(f"Classifier failed at a primary window: {type(exc).__name__}") from exc
        runs=[j for j in range(len(h)-1) if h[j] and h[j+1]]
        if runs:
            s,e=runs[0],runs[-1]+1
            low=(-np.inf,force[0]) if s==0 else (force[s-1],force[s])
            up=(force[-1],np.inf) if e==len(force)-1 else (force[e],force[e+1])
        else: low,up=(force[-1],np.inf),(np.nan,np.nan)
        result.append({"lower":low,"upper":up,"column":r["column"],"folder":r["folder"],
                       "no_detected_band":not bool(runs)})
    return result,dict(counts)


def boundary_score(intervals, prediction):
    counts=Counter();errors=[];hits=0
    for (lo,hi),p in zip(intervals,prediction):
        if np.isnan(lo) or np.isnan(hi):kind="unavailable"
        elif np.isfinite(lo) and np.isfinite(hi):kind="finite"
        elif np.isneginf(lo):kind="left_censored"
        else:kind="right_censored"
        counts[kind]+=1
        if not np.isfinite(p): counts["prediction_failures"]+=1;continue
        if kind=="unavailable":continue
        counts["n_scored"]+=1
        hits+=int(lo-STEP<=p<=hi+STEP)
        if kind=="finite":errors.append(abs(p-(lo+hi)/2))
    for k in ("finite","left_censored","right_censored","unavailable","prediction_failures","n_scored"):
        counts.setdefault(k,0)
    return {"counts":dict(counts),"hits":hits,"hit_rate":hits/counts["n_scored"] if counts["n_scored"] else None,
            "n_finite_errors":len(errors),"median_abs_log_error":float(np.median(errors)) if errors else None}


def decisions(level_fits, predictions, thresholds):
    lower = level_fits["lower"]
    if not lower["success"]:
        lo={"verdict":"검증 불가","reason":lower["reason"]}
    else:
        e={name:abs(lower["level"]-p) for name,p in predictions["lower_levels"].items()}
        gain=e["log_Z_EI"]-e["protocol"]
        if e["protocol"]<=thresholds["lower_tolerance"] and gain>=thresholds["lower_comparison_margin"]:
            verdict="이 도표에서 절차 기술자 예측 지지"
        elif -gain>=thresholds["lower_comparison_margin"]:verdict="절차 기술자 예측 반증 방향"
        else:verdict="구별 불충분"
        lo={"verdict":verdict,"absolute_log_errors":e,"primary_gain_over_properties":gain,
            "gain_over_mean":e["mean"]-e["protocol"],"descriptor_extrapolation":predictions["descriptor_extrapolation"]}
    up=level_fits["upper_strict"]
    if not up["success"]: upper={"verdict":"검증 불가","reason":up["reason"]}
    else:
        error=abs(up["level"]); noz=abs(up["level"]+np.log(predictions["Z"]))
        gain=noz-error
        upper={"verdict":"허용오차 안" if error<=thresholds["upper_tolerance"] else "허용오차 밖",
               "absolute_log_error":error,"no_Z_absolute_log_error":noz,"gain_over_no_Z":gain,
               "Z_comparison":"Z 식 우세 방향" if gain>=thresholds["upper_comparison_margin"] else
                               ("Z 없는 식 우세 방향" if -gain>=thresholds["upper_comparison_margin"] else "구별 불충분")}
    return {"lower":lo,"upper_strict":upper,"scope":"한 도표의 동결 실용 기준; 인과/보편법칙/통계적 동등성 판정 아님"}


def cluster_resamples(rows, *, n_boot=N_BOOT, seed=SEED):
    groups={c:[i for i,r in enumerate(rows) if r["column"]==c] for c in range(40)}
    if any(len(ii)!=3 or len({rows[i]["folder"] for i in ii})!=3 for ii in groups.values()):
        raise ConfirmationError("Bootstrap requires each beta column paired across all three speeds")
    rng=np.random.default_rng(seed)
    for _ in range(n_boot):
        selected=rng.integers(0,40,40)
        yield np.concatenate([groups[int(c)] for c in selected])


def summarize_bootstrap(values, failures, n_boot):
    values=np.asarray(values,float)
    return {"requested":n_boot,"successful":len(values),"failed":sum(failures.values()),
            "failure_reasons":dict(failures),"percentile_95_conditional_successes":np.percentile(values,[2.5,97.5]).tolist() if len(values) else None,
            "warning":"조건부 성공 재표집의 구간; 실패 수를 무시한 무조건 coverage 주장 불가"}


def calculate_scores(observed, rows, pred, params, *, n_boot=N_BOOT):
    lower=np.array([r["lower"] for r in observed]);upper=np.array([r["upper"] for r in observed])
    specs={"lower":(lower,np.array(pred["lower_reference"])),
           "upper_strict":(upper,np.array(pred["upper_strict"])),
           "upper_empirical_secondary":(upper,np.array(pred["upper_empirical_secondary"]))}
    fits={name:fit_level(iv,off) for name,(iv,off) in specs.items()}
    boot={name:[] for name in specs}; failures={name:Counter() for name in specs}
    for indices in cluster_resamples(rows,n_boot=n_boot):
        for name,(iv,off) in specs.items():
            r=fit_level(iv[indices],off[indices])
            if r["success"]:boot[name].append(r["level"])
            else:failures[name][r["reason"]]+=1
    secondary={"lower":{name:boundary_score(lower,points) for name,points in pred["lower_point_predictions"].items()},
               "upper":{name:boundary_score(upper,pred[name]) for name in ("upper_strict","upper_strict_no_Z","upper_empirical_secondary")}}
    return {"level_fits":fits,"decisions":decisions(fits,pred,params["thresholds_log"]),"column_scores":secondary,
            "no_detected_band_columns":sum(r["no_detected_band"] for r in observed),
            "bootstrap":{name:summarize_bootstrap(boot[name],failures[name],n_boot) for name in specs},
            "bootstrap_conditioning":"predictions fixed; only 40 beta clusters paired across speeds resampled; no descriptor/training/new-session uncertainty",
            "upper_empirical_secondary_absolute_level_error":abs(fits["upper_empirical_secondary"]["level"]) if fits["upper_empirical_secondary"]["success"] else None,
            "secondary_variants_deferred":["K3","t=.1/.3","nominal force axis"]}


def score(controls_receipt, output, *, base=BASE):
    if Path(output).exists(): raise FileExistsError(output)
    params,original=verify_original_freeze(base)
    seal=json.loads(seal_path(controls_receipt).read_text())
    if seal["controls_receipt_sha256"]!=sha(controls_receipt) or seal["runner_sha256"]!=sha(__file__):
        raise ConfirmationError("Controls receipt or runner changed after sealing")
    if seal["original_manifest"]!=original:raise ConfirmationError("Original manifest differs from seal")
    rec=json.loads(Path(controls_receipt).read_text())
    if rec["stage"]!="controls_frozen" or rec["outcomes_parsed"] is not False or rec["parsed_fields"]!=list(CONTROL_FIELDS):
        raise ConfirmationError("Not a valid controls-only freeze")
    if rec["original_manifest"]!=original:
        raise ConfirmationError("Original frozen manifest differs between stages")
    verify_manifest(rec["original_manifest"])
    verify_manifest({v["path"]:v["sha256"] for v in rec["runtime_dependencies"].values()})
    source=Path(rec["features_path"])
    if sha(source)!=rec["input_sha256"]:
        raise ConfirmationError("Feature input changed after controls freeze")
    # A receipt is single-use even when a caller supplies a different output name.
    exclusive_json(Path(str(controls_receipt)+".score_claim.json"),
                   {"output":str(Path(output).resolve()), "controls_receipt_sha256":sha(controls_receipt),
                    "claimed_utc":datetime.now(timezone.utc).isoformat()})
    # Reserve a result file before opening outcome columns. Every failure after
    # first outcome access is written here; rerunning cannot overwrite it.
    with Path(output).open("x") as f:
        result={"stage":"score_reserved","controls_receipt_sha256":sha(controls_receipt),"created_utc":datetime.now(timezone.utc).isoformat()}
        f.write(encode(result));f.flush()
        try:
            d=read_columns(source,CONTROL_FIELDS+OUTCOME_FIELDS)
            if control_digest(d)!=rec["control_sha256"]:
                raise ConfirmationError("Parsed controls changed after freezing")
            columns=selected_function(rec["runtime_dependencies"]["columns"]["path"],"columns")
            rebuilt_rows,rebuilt_inventory=control_inventory(d,columns)
            if rebuilt_rows!=rec["rows"] or rebuilt_inventory!=rec["inventory"]:
                raise ConfirmationError("Control row inventory differs from frozen predictions")
            expected=frozen_predictions(params,rec["descriptor"],rebuilt_rows)
            if json_value(expected)!=rec["predictions"]:raise ConfirmationError("Predictions differ from frozen formulas")
            classifier=selected_function(rec["runtime_dependencies"]["classifier"]["path"],"classify")
            observed,labels=observed_boundaries(d,rec["rows"],classifier,rec["thresholds"])
            result.update(calculate_scores(observed,rec["rows"],rec["predictions"],params))
            result.update(stage="score_complete",label_counts=labels,n_primary_windows=sum(labels.values()),
                          target=STEM,observed_boundaries=observed,prediction_receipt=str(Path(controls_receipt).resolve()))
            # Post-read integrity checks catch concurrent source edits.
            verify_manifest(original)
            verify_manifest({v["path"]:v["sha256"] for v in rec["runtime_dependencies"].values()})
            if sha(source)!=rec["input_sha256"]:raise ConfirmationError("Feature file changed during scoring")
        except Exception as exc:
            result.update(stage="score_failed",error_type=type(exc).__name__,reason=str(exc),verdict="검증 불가")
            f.seek(0);f.truncate();f.write(encode(result));f.flush()
            raise
        f.seek(0);f.truncate();f.write(encode(result))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    a=sub.add_parser("freeze-controls");a.add_argument("--features",type=Path,required=True);a.add_argument("--out",type=Path,required=True)
    b=sub.add_parser("score");b.add_argument("--controls-receipt",type=Path,required=True);b.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="freeze-controls":freeze_controls(args.features,args.out)
    else:score(args.controls_receipt,args.out)
    print(f"Created {args.out} exclusively")


if __name__=="__main__":main()
