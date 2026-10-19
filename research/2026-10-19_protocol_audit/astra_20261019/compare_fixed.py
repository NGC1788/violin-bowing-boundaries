"""Post-hoc bug-corrected boundary comparison; no new holdout or superiority claim.

Uses only existing A1/A2 and five explicitly listed historical test caches.
Legacy fitting and K=2 observation extraction are preserved. Probability inversion
is support restricted, failure/censor coverage is explicit, and slope calibration
is source-only. D_T2 is neither loaded nor accepted by the command.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import warnings

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ("A1", "A2")
TEST = ("B1", "C1", "D1", "B_T2r1", "B_T2r2")
T = 0.2
STEP = np.log(40.0) / 49
PUB_Z = {"A1": 1.0592, "A2": 1.0592, "B1": 1.0998, "C1": 1.0525,
         "D1": 1.0398, "B_T2r1": 1.1899, "B_T2r2": 1.1899}
EPS = 1e-6


def legacy_module():
    """Import read-only legacy code without its command-line parsing side effect."""
    name = "_compare_20261018_legacy"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "final/compare.py")
    module = importlib.util.module_from_spec(spec)
    argv = sys.argv
    sys.path.insert(0, str(ROOT / "final"))
    try:
        sys.argv = [str(spec.origin)]
        spec.loader.exec_module(module)
    finally:
        sys.argv = argv
        sys.path.pop(0)
    sys.modules[name] = module
    return module


@dataclass(frozen=True)
class Band:
    lower: float
    upper: float
    status: str = "band"  # band, empty (all p<.5 on support), invalid
    support_min: float | None = None
    support_max: float | None = None
    components: int = 1
    reason: str = ""

    def endpoint(self, which):
        """Return point or censor interval; empty/invalid is never a hit."""
        if self.status != "band":
            return self.status, (np.nan, np.nan)
        value = self.lower if which == "lower" else self.upper
        if np.isfinite(value):
            return "point", (value, value)
        if which == "lower" and value == -np.inf and self.support_min is not None:
            return "left_censored", (-np.inf, self.support_min)
        if which == "upper" and value == np.inf and self.support_max is not None:
            return "right_censored", (self.support_max, np.inf)
        return "invalid", (np.nan, np.nan)


def checked_band(lower, upper, **kwargs):
    if np.isnan(lower) or np.isnan(upper):
        return Band(float(lower), float(upper), status="invalid", reason="nonfinite_endpoint", **kwargs)
    if lower == np.inf or upper == -np.inf or lower > upper:
        return Band(float(lower), float(upper), status="invalid", reason="crossed_or_inverted_endpoint", **kwargs)
    return Band(float(lower), float(upper), **kwargs)


def band_from_prob(p_fun, r, t=T, n_grid=400):
    """Hull of p>=.5 only within measured support; edge contact is censored.

    No root/extrapolation is invented beyond the observed force range. Disjoint
    components are counted; the legacy single-band hull convention is retained.
    """
    a, b = float(r["fmin"]), float(r["fmax"])
    if not np.isfinite(a + b) or b <= a or n_grid < 2:
        return Band(np.nan, np.nan, "invalid", reason="invalid_support")
    grid = np.linspace(a, b, n_grid)
    p = np.asarray(p_fun(grid, np.full_like(grid, r["lv"]),
                         np.full_like(grid, r["lb"]), np.full_like(grid, t)), float)
    if p.shape != grid.shape or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        return Band(np.nan, np.nan, "invalid", a, b, reason="invalid_probability")
    above = p >= .5
    if not above.any():
        return Band(np.nan, np.nan, "empty", a, b, 0, "no_band_within_support")
    inds = np.flatnonzero(above)
    ncomp = int(np.sum(above & ~np.r_[False, above[:-1]]))
    lower = -np.inf if above[0] else grid[inds[0]]
    upper = np.inf if above[-1] else grid[inds[-1]]
    return checked_band(lower, upper, support_min=a, support_max=b, components=ncomp)


def band_prob(lf, band, k):
    """One-sided bands are valid; empty means p=0; invalid remains unavailable."""
    x = np.asarray(lf, float)
    if band.status == "empty":
        return np.zeros_like(x)
    if band.status != "band" or not np.isfinite(k) or k <= 0:
        return np.full_like(x, np.nan)
    # checked_band validates orientation. +/-inf give genuine one-sided margins.
    return expit(k * np.minimum(x - band.lower, band.upper - x))


def fit_slope(train_by_tag, rows_by_tag, pred_fun, grid=None):
    """Source-only explicit calibration, with finite losses and exclusion counts.

    No-band predictions contribute fixed zero probabilities and constant loss.
    Invalid predictions are excluded with counts. A completely constant band
    model has no identified slope: return k=None rather than an arbitrary .5.
    """
    if set(train_by_tag) != set(TRAIN):
        raise ValueError("Slope calibration requires exactly A1 and A2; targets are forbidden")
    ks = np.linspace(.5, 15, 59) if grid is None else np.asarray(grid, float)
    if np.any(~np.isfinite(ks)) or np.any(ks <= 0) or len(ks) == 0:
        raise ValueError("Invalid slope grid")
    blocks, counts = [], Counter()
    for tag in TRAIN:
        d = train_by_tag[tag]
        for r in rows_by_tag[tag]:
            if r.get("tag", tag) != tag:
                raise ValueError("Row tag does not match source")
            band = pred_fun(r, tag)
            counts[band.status + "_columns"] += 1
            idx = r["idx"]
            if band.status == "invalid":
                counts["excluded_windows"] += len(idx)
                continue
            f = np.log(np.maximum(d["force"][idx], 1e-3))
            y = np.asarray(d["H"][idx], float)
            if not np.all(np.isfinite(f)) or not np.all(np.isin(y, [0, 1])):
                raise ValueError("Invalid source forces or binary labels")
            blocks.append((f, y, band))
            counts["used_windows"] += len(idx)
    counts = dict(counts)
    if not blocks:
        return {"k": None, "identified": False, "losses": [], "counts": counts,
                "source_tags": list(TRAIN), "reason": "no_valid_predictions"}
    losses = []
    for k in ks:
        loss = 0.0
        for f, y, band in blocks:
            p = np.clip(band_prob(f, band, k), EPS, 1-EPS)
            if not np.all(np.isfinite(p)):
                raise ValueError("Nonfinite probability escaped calibration validation")
            loss -= float(np.sum(y*np.log(p)+(1-y)*np.log1p(-p)))
        losses.append(loss / counts["used_windows"])
    identified = np.ptp(losses) > 1e-12
    return {"k": float(ks[np.argmin(losses)]) if identified else None,
            "identified": bool(identified), "losses": [float(v) for v in losses],
            "grid": ks.tolist(), "counts": counts, "source_tags": list(TRAIN),
            "reason": "minimum_source_logloss" if identified else "constant_loss_not_identified"}


def observation_kind(row, which):
    lo, hi = row[which]
    if np.isnan(lo) or np.isnan(hi):
        return "unavailable"
    if np.isfinite(lo) and np.isfinite(hi):
        return "finite_interval"
    return "left_censored" if np.isneginf(lo) else "right_censored"


def score_boundaries(rows, preds, masks=None):
    """Conditional point errors + full denominators; censor compatibility is separate."""
    if len(rows) != len(preds):
        raise ValueError("Prediction length mismatch")
    out = {}
    for which in ("lower", "upper"):
        counts = Counter()
        errors, common_errors = [], []
        for i, (r, pr) in enumerate(zip(rows, preds)):
            obs = observation_kind(r, which)
            kind, piv = pr.endpoint(which)
            counts["observation_" + obs] += 1
            counts["prediction_" + kind] += 1
            if not r.get("has_detected_band", True):
                counts["no_detected_observed_band"] += 1
            if obs == "unavailable":
                continue
            counts["observations_available"] += 1
            lo, hi = r[which]
            lo, hi = lo-STEP, hi+STEP  # exact historical one-grid-step tolerance
            if kind == "point":
                counts["point_predictions_available"] += 1
                hit = lo <= piv[0] <= hi
                counts["point_hits"] += int(hit)
                if obs == "finite_interval":
                    e = float(abs(piv[0] - np.mean(r[which])))
                    errors.append(e)
                    if masks is not None and masks[which][i]:
                        common_errors.append(e)
            elif kind in ("left_censored", "right_censored"):
                counts["support_censored_predictions_available"] += 1
                counts["censor_interval_overlaps"] += int(max(piv[0], lo) <= min(piv[1], hi))
                counts["censor_interval_contained"] += int(piv[0] >= lo and piv[1] <= hi)
            else:
                counts["prediction_failures_available"] += 1
        n = counts["observations_available"]
        npnt = counts["point_predictions_available"]
        # All fields exist even for zero counts: absent events never masquerade as missing data.
        for key in ("observation_finite_interval", "observation_left_censored", "observation_right_censored",
                    "observation_unavailable", "prediction_point", "prediction_left_censored",
                    "prediction_right_censored", "prediction_empty", "prediction_invalid", "point_hits",
                    "support_censored_predictions_available", "censor_interval_overlaps", "censor_interval_contained",
                    "prediction_failures_available", "no_detected_observed_band"):
            counts.setdefault(key, 0)
        out[which] = {"median_abs_log_error_conditional": float(np.median(errors)) if errors else None,
                      "n_finite_observation_and_point_prediction": len(errors),
                      "point_hit_rate_all_available": counts["point_hits"]/n if n else None,
                      "point_hit_rate_point_predictions": counts["point_hits"]/npnt if npnt else None,
                      "common_point_median_abs_log_error": float(np.median(common_errors)) if common_errors else None,
                      "n_common_point_errors": len(common_errors), "counts": dict(counts)}
    out["bands"] = {"statuses": dict(Counter(p.status for p in preds)),
                    "multi_component_hulls": sum(p.components > 1 for p in preds),
                    "reasons": dict(Counter(p.reason for p in preds if p.reason))}
    return out


def classification_score(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, float)
    valid = np.isfinite(p) & (p >= 0) & (p <= 1)
    counts = {"n_total": len(y), "n_valid": int(valid.sum()), "n_prediction_failures": int((~valid).sum())}
    if not valid.any():
        return {**counts, "iou_conditional": None, "logloss_conditional": None}
    y, p = y[valid], np.clip(p[valid], 1e-4, 1-1e-4)
    pred = p >= .5
    union = int(np.sum(pred | y))
    return {**counts, "iou_conditional": float(np.sum(pred & y)/max(union, 1)),
            "logloss_conditional": float(-np.mean(y*np.log(p)+(~y)*np.log1p(-p))),
            "n_positive_observed": int(y.sum()), "n_positive_predicted": int(pred.sum()), "union": union}


def triv5_predictor(source_rows, legacy_edges=False):
    """Source-only lookup. Primary is exactly confirm_A_T2's finite-interval Triv-5.

    Diagnostic legacy_edges reproduces robustness/rb_core: nearest column +/-2,
    censored intervals replaced by finite edges, then median within and across A.
    """
    if any(r["tag"] not in TRAIN for r in source_rows):
        raise ValueError("Triv-5 lookup must contain source rows only")
    table = tuple(dict(r) for r in source_rows)
    def predict(r, tag):
        vals = {"lower": [], "upper": []}
        for tg in TRAIN:
            base = [x for x in table if x["tag"] == tg and abs(x["lv"]-r["lv"]) < 1e-6]
            if legacy_edges:
                base.sort(key=lambda x: x["col"])
                if not base:
                    continue
                j = int(np.argmin([abs(x["lb"]-r["lb"]) for x in base]))
                near = base[max(0, j-2):j+3]
            for which in vals:
                if legacy_edges:
                    pool = []
                    for x in near:
                        lo, hi = x[which]
                        if np.isfinite(lo) and np.isfinite(hi):
                            pool.append((lo+hi)/2)
                        elif np.isfinite(lo):
                            pool.append(lo)
                        elif np.isfinite(hi):
                            pool.append(hi)
                    if pool:
                        vals[which].append(float(np.median(pool)))
                else:
                    cand = [x for x in base if np.all(np.isfinite(x[which]))]
                    cand.sort(key=lambda x: abs(x["lb"]-r["lb"]))
                    vals[which].extend(float(np.mean(x[which])) for x in cand[:5])
        return checked_band(*(float(np.median(vals[w])) if vals[w] else np.nan for w in ("lower", "upper")))
    return predict


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_json(x):
    if isinstance(x, dict):
        return {str(k): safe_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [safe_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return safe_json(x.tolist())
    if isinstance(x, (float, np.floating)):
        if np.isnan(x): return None
        if np.isposinf(x): return "+inf"
        if np.isneginf(x): return "-inf"
        return float(x)
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.bool_): return bool(x)
    return x


def write_json(path, x):
    path.write_text(json.dumps(safe_json(x), ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def run(out):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing result directory")
    out.mkdir(parents=True, exist_ok=True)
    C = legacy_module()
    inputs = {tag: ROOT/"final/cache"/(tag+".npz") for tag in TRAIN+TEST}
    # Existing-cache-only: never call data.load(), which may build or alter caches.
    dat = {}
    for tag, path in inputs.items():
        with np.load(path, allow_pickle=False) as z:
            dat[tag] = {k: z[k] for k in z.files}
    rows = {}
    for tag, d in dat.items():
        rows[tag] = C.intervals(d, T)
        for r in rows[tag]:
            r["tag"] = tag
            r["col"] = int(d["column"][r["idx"][0]])
            h = d["H"][r["idx"]]
            r["has_detected_band"] = bool(np.any(h[:-1] & h[1:]))
        if len(rows[tag]) != 120 or sum(len(r["idx"]) for r in rows[tag]) != 6000:
            raise ValueError(f"Unexpected support for {tag}")
    source_rows = rows["A1"]+rows["A2"]
    z_old = {tag: float(d["Z"][0]) for tag, d in dat.items()}
    models, params, fits = {}, {}, []
    # Same optimizer/objective as legacy, now recording convergence explicitly.
    def monitored_fit(rs, design, which):
        X = np.array([design(r)[0] for r in rs]); off = np.array([design(r)[1] for r in rs])
        lo = np.array([r[which][0] for r in rs]); hi = np.array([r[which][1] for r in rs])
        ok = np.isfinite(lo) | np.isfinite(hi)
        X, off, lo, hi = X[ok], off[ok], lo[ok], hi[ok]
        def nll(p):
            mu = X@p[:-1]+off; s = np.exp(p[-1])
            pr = np.where(np.isfinite(hi), C.norm.cdf((hi-mu)/s), 1.)-np.where(np.isfinite(lo), C.norm.cdf((lo-mu)/s), 0.)
            return -np.sum(np.log(np.clip(pr, 1e-300, None)))
        res = C.minimize(nll, np.r_[np.zeros(X.shape[1]), np.log(.5)], method="Nelder-Mead",
                         options={"maxiter":40000,"xatol":1e-8,"fatol":1e-10})
        fits.append({"which":which,"success":bool(res.success),"message":str(res.message),
                     "nit":int(res.nit),"objective":float(res.fun),"params":res.x.tolist()})
        if not res.success or not np.all(np.isfinite(res.x)) or not np.isfinite(res.fun):
            raise RuntimeError("Legacy physics optimizer failed; refusing silent scoring")
        return res.x
    original_fit = C.censored_fit
    C.censored_fit = monitored_fit
    try:
        for provenance, z in (("publishedZ", PUB_Z), ("targetDerivedZ-diagnostic", z_old)):
            fitted = C.physics_models(source_rows, z)
            for name, (pf, par) in fitted.items():
                key = name+"-"+provenance
                models[key] = lambda r, tag, fun=pf: checked_band(*fun(r, tag))
                params[key] = {"lower":par[0].tolist(),"upper":par[1].tolist()}
    finally:
        C.censored_fit = original_fit
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        native = C.data_models([dat[tag] for tag in TRAIN])
    for name, pf in native.items():
        models[name] = lambda r, tag, fun=pf: band_from_prob(fun, r, T)
    models["Triv-5-finite"] = triv5_predictor(source_rows)
    models["Triv-5-legacy-edge-diagnostic"] = triv5_predictor(source_rows, legacy_edges=True)
    # Calibration accesses only A1/A2. No target label, Z, or row enters this call.
    slopes = {name: fit_slope({tag:dat[tag] for tag in TRAIN}, {tag:rows[tag] for tag in TRAIN}, pf)
              for name,pf in models.items()}
    pred = {tag:{name:[pf(r,tag) for r in rows[tag]] for name,pf in models.items()} for tag in TEST}
    native_p = {tag:{} for tag in TEST}
    for tag in TEST:
        for name, pf in native.items():
            native_p[tag][name] = np.concatenate([pf(np.log(np.maximum(dat[tag]["force"][r["idx"]],1e-3)),
                           np.full(len(r["idx"]), r["lv"]), np.full(len(r["idx"]),r["lb"]),
                           np.full(len(r["idx"]),T)) for r in rows[tag]])
    # Written BEFORE computing any target score. Post-hoc development freeze,
    # not a claim of previously unseen target status.
    frozen = {"t_eval":T,"source_tags":list(TRAIN),"test_tags":list(TEST),"posthoc":True,
              "script_sha256":sha(__file__),"legacy_compare_sha256":sha(ROOT/"final/compare.py"),
              "input_sha256":{t:sha(p) for t,p in inputs.items()},"published_Z":PUB_Z,"legacy_Z":z_old,
              "physics_params":params,"optimizer_diagnostics":fits,"source_slopes":slopes,
              "training_warnings":[str(w.message) for w in caught],
              "fewshot":"deferred: original invalid-band substitution and calibration/evaluation overlap not reused"}
    write_json(out/"frozen_before_scoring.json", frozen)
    write_json(out/"predictions.json", {tag:{name:[asdict(p) for p in ps] for name,ps in per.items()} for tag,per in pred.items()})
    np.savez_compressed(out/"native_probabilities.npz", **{tag+"__"+name:p for tag,per in native_p.items() for name,p in per.items()})
    results = {}
    for tag in TEST:
        masks = {w: [observation_kind(r,w)=="finite_interval" and all(pred[tag][name][i].endpoint(w)[0]=="point" for name in models)
                     for i,r in enumerate(rows[tag])] for w in ("lower","upper")}
        y = np.concatenate([dat[tag]["H"][r["idx"]] for r in rows[tag]])
        results[tag] = {}
        for name, ps in pred[tag].items():
            result = score_boundaries(rows[tag], ps, masks)
            k = slopes[name]["k"]
            # Identifiability failure is explicit; empty/constant models still have
            # their known probabilities, without implying an estimated slope.
            proxy = np.concatenate([band_prob(np.log(np.maximum(dat[tag]["force"][r["idx"]],1e-3)),p,k if k is not None else 1.)
                                    for r,p in zip(rows[tag],ps)])
            result["band_probability_score"] = classification_score(y,proxy)
            result["band_probability_score"]["slope_identified"] = slopes[name]["identified"]
            if name in native:
                result["native_probability_score"] = classification_score(y,native_p[tag][name])
            results[tag][name] = result
            lo,up = result["lower"],result["upper"]
            print(tag,name,"L",lo["median_abs_log_error_conditional"],lo["n_finite_observation_and_point_prediction"],
                  "U",up["median_abs_log_error_conditional"],up["n_finite_observation_and_point_prediction"],flush=True)
    write_json(out/"results.json", {"results":results,"score_contract":{
        "boundary_error":"median |predicted point - observed interval midpoint| on finite observed intervals and point predictions only",
        "point_hit":"point lies within observation interval plus historical one nominal log-force step tolerance; empty/invalid/censored predictions are not point hits",
        "censor_compatibility":"separate intersection/containment counts, never point accuracy",
        "no_observed_band":"legacy lower right-censor convention preserved and counted; not proof the physical band lies above support",
        "native_probability":"original model probabilities; not changed by boundary extraction",
        "fewshot":"not rerun; old results cannot establish an unbiased adaptation comparison"}})
    write_report(out, results, frozen)
    return results


def write_report(out, results, frozen):
    names = list(next(iter(results.values())))
    lines = ["# 기존 경계 비교 코드의 사후 수정 검증 (2026-10-19)", "",
             "A1+A2로만 학습·slope 교정하고 기존에 이미 본 B1, C1, D1, B_T2r1/r2를 t=0.2 s에서 재평가했다. D_T2는 읽지 않았다. 이 결과는 새 확증 실험이나 성능 우위의 증거가 아니다.", "",
             "## 수정 내용", "",
             "- 확률모델의 0.5 band는 각 열의 실제 측정 force 최솟값~최댓값 안에서만 추출했다. support 끝까지 양성이면 경계값을 만들어 넣지 않고 좌/우 검열로 기록했다. 여러 양성 구간은 기존처럼 hull로 묶되 개수를 남겼다.",
             "- NaN·빈 band를 slope 계산에 섞지 않는다. 빈 band는 측정 범위에서 p=0, invalid band는 제외 수를 공개한다. source loss가 일정하면 slope 미식별로 표시한다. 기존의 임의 3.0 log-error 대입과 비유한 경계 자동 적중을 없앴다.",
             "- 오차는 유한 관측구간과 point prediction이 함께 있는 조건부 값이다. 전체 적중 수·실패 수·관측 검열·예측 검열 수를 별도로 공개한다. 검열구간이 겹치는 것은 point 적중으로 세지 않는다.",
             "- 공개 Z 버전과 target H 파형에서 유도한 과거 Z 버전을 분리했다. 후자는 엄밀한 무표적 정보 예측으로 부를 수 없는 진단용이다.",
             "- Triv-5-finite는 confirm_A_T2.py의 동일 속도·source별 β 최근접 5개 유한구간 중점 정의다. Triv-5-legacy-edge-diagnostic은 robustness의 β 이웃 ±2와 검열 edge 대입 정의를 재현한다. 서로 다른 baseline이다.",
             "- 기존 few-shot 결과는 재사용하지 않았다. 비유한 경계의 임의 대입과 교정에 쓴 stroke의 평가 중복을 함께 수정해야 한다.", "",
             "## 평균 조건부 경계 오차와 평가 가능 수", "",
             "오차는 표본별 median 다섯 개의 평균, N은 다섯 표본의 평가 가능한 열 수 합계(최대 600)다. 서로 다른 N의 숫자만 비교해 우위를 주장하면 안 된다.", "",
             "| 모델 | 아래 오차 | 아래 N | 위 오차 | 위 N | 예측 검열 L/U | 빈/invalid band |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in names:
        rs = [results[tag][name] for tag in TEST]
        avg = lambda w: np.mean([r[w]["median_abs_log_error_conditional"] for r in rs if r[w]["median_abs_log_error_conditional"] is not None])
        nl = sum(r["lower"]["n_finite_observation_and_point_prediction"] for r in rs)
        nu = sum(r["upper"]["n_finite_observation_and_point_prediction"] for r in rs)
        cl = sum(r["lower"]["counts"]["prediction_left_censored"] for r in rs)
        cu = sum(r["upper"]["counts"]["prediction_right_censored"] for r in rs)
        fail = sum(r["bands"]["statuses"].get("empty",0)+r["bands"]["statuses"].get("invalid",0) for r in rs)
        lines.append(f"| {name} | {avg('lower'):.4f} | {nl} | {avg('upper'):.4f} | {nu} | {cl}/{cu} | {fail} |")
    lines += ["", "## 남아 있는 한계", "",
              "기존 K=2 연속 H 셀 규칙, 단일 band hull 가정, 관측 force 1e-3 clamp, 정규 interval likelihood, 4개 시간창으로 데이터 모델 학습/한 시점으로 physics 경계 학습을 유지한 코드 수정이다. H band가 전혀 없는 열의 아래 경계를 오른쪽 검열로 놓는 기존 convention은 물리적으로 유일하게 정당화되지 않으며 `no_detected_observed_band`로 별도 집계했다. likelihood의 새로운 타당성 증명은 아니다.", "",
              "GBM의 기존 내부 무작위 행 validation 및 같은 stroke의 여러 시간창 의존성도 유지했다. 별도 모델 선택·튜닝을 하지 않았다. 연산 optimizer success와 finite 결과를 확인했고 target scoring 전에 source parameters·slope·예측 파일을 저장했지만, 이 다섯 표본의 과거 노출은 사라지지 않는다.", "",
              "검열된 예측을 버리면 조건부 오차가 좋아질 수 있다. 그래서 전체 failure/coverage, 전체 관측 대비 point-hit, 모든 모델이 point를 낸 공통 열의 오차까지 results.json에 함께 기록했다. 공통 열 역시 선택된 subset이므로 일반 성능 주장으로 확대하지 않는다.", "",
              "현 데이터는 제어된 steady-state 최소 힘을 직접 측정한 것이 아니다. force가 진행 중 변하고 speed와 session이 얽히므로 이 비교가 물리적 분기·CSD·보편적인 최소 힘 법칙을 확립하지 않는다.", "",
              "파일: `results.json`(모든 열·적중·검열·실패 수), `predictions.json`, `native_probabilities.npz`, `frozen_before_scoring.json`(코드/캐시 SHA256, source-only slope losses, optimizer 진단)."]
    (out/"README.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("compare_fixed_results"))
    args = parser.parse_args()
    run(args.out)
