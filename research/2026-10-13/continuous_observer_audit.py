#!/usr/bin/env python3
"""Audit already-frozen predictions; no predictor fitting or target adaptation."""
from pathlib import Path
import sys
import json
import time
import hashlib
import platform
import numpy as np
import numba

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_data as bd
from continuous_observer import ContinuousObserver
from set_envelope import family_ious, iou_bounds


def dense_extrema(observer, p):
    """Independent O(m*n) 2-D histogram/cumulative-sum baseline, NOT cubic rescans."""
    return dense_prepared_extrema(prepare_dense(observer), p)


def prepare_dense(observer):
    """Additional dense geometry, outside the repeated solve-only timing."""
    qs = observer.q_states[::-1]
    shape = (len(qs), len(observer.f_states))
    if shape[0]*shape[1] > 8_000_000:
        raise ValueError("dense baseline exceeds fixed laptop comparison budget")
    rows = observer.order
    qi = np.searchsorted(-qs, -observer.periodicity[rows], side="left")
    return shape, rows, qi, observer.starts


def dense_prepared_extrema(prepared, p):
    shape, rows, qi, starts = prepared
    a, b = np.zeros(shape), np.zeros(shape)
    np.add.at(a, (qi, starts), p[rows])
    np.add.at(b, (qi, starts), 1-p[rows])
    np.cumsum(a, axis=0, out=a); np.cumsum(a, axis=1, out=a)
    np.cumsum(b, axis=0, out=b); np.cumsum(b, axis=1, out=b)
    b += p.sum()
    score = np.divide(a, b, out=np.ones_like(a), where=b > 0)
    return float(score.min()), float(score.max())


def timing_summary(values):
    values = np.asarray(values, float)
    return {"median": float(np.median(values)), "min": float(values.min()),
            "max": float(values.max()), "q25": float(np.quantile(values,.25)),
            "q75": float(np.quantile(values,.75)), "repetitions": values.tolist()}


def repeated_timing(ratio, periodicity, p, repetitions=7):
    """Warm, alternating-order comparison with shared geometry counted separately."""
    warm = ContinuousObserver(ratio, periodicity)
    warm.extreme(p); warm.extreme(p, maximum=True)
    dense_extrema(warm, p)
    preparation, dense_preparation, sweep, dense = [], [], [], []
    for repeat in range(repetitions):
        start = time.perf_counter()
        observer = ContinuousObserver(ratio, periodicity)
        preparation.append(time.perf_counter()-start)
        start = time.perf_counter(); prepared = prepare_dense(observer)
        dense_preparation.append(time.perf_counter()-start)
        for method in (("sweep","dense") if repeat % 2 == 0 else ("dense","sweep")):
            start = time.perf_counter()
            if method == "sweep":
                gotlo = observer.extreme(p); gothi = observer.extreme(p, maximum=True)
                sweep.append(time.perf_counter()-start)
            else:
                wantlo, wanthi = dense_prepared_extrema(prepared,p)
                dense.append(time.perf_counter()-start)
        np.testing.assert_allclose([gotlo["iou"],gothi["iou"]],[wantlo,wanthi],atol=1e-11,rtol=0)
    prep, extra, fast, slow = map(np.asarray,(preparation,dense_preparation,sweep,dense))
    return {"n":len(p), "candidate_pairs":gotlo["candidate_pairs"], "repetition_count":repetitions,
            "shared_geometry_seconds":timing_summary(prep),
            "dense_additional_geometry_seconds":timing_summary(extra),
            "sweep_solve_seconds":timing_summary(fast), "dense_solve_seconds":timing_summary(slow),
            "sweep_including_geometry_seconds":timing_summary(prep+fast),
            "dense_including_geometry_seconds":timing_summary(prep+extra+slow),
            "ratio_solve_medians_dense_over_sweep":float(np.median(slow)/np.median(fast)),
            "ratio_including_geometry_medians_dense_over_sweep":float(np.median(prep+extra+slow)/np.median(prep+fast)),
            "iterations_min":gotlo["iterations"],"iterations_max":gothi["iterations"]}


def main():
    d = bd.load_dataset(include_ambiguous=True)
    # Warm numba outside timing. Compilation cost is not hidden in deployment claims.
    toy = ContinuousObserver([.35,.65], [.78,.82]); toy.extreme([.8,.2])
    rows = []
    for source in "ABCD":
        directory = ROOT / "reports/research_20261013" / f"benchmark_{source}"
        saved = np.load(directory / "predictions.npz", allow_pickle=False)
        np.testing.assert_array_equal(saved["ids"], d["ids"])
        results = json.loads((directory / "results.json").read_text())
        keep = ~saved["source_mask"]
        features = {k:v[keep] for k,v in d["output_features"].items()}
        observer = ContinuousObserver.from_features(features, d["reference_f0_hz"])
        family = saved["family"][keep]
        for name, result in results.items():
            if result["historical_diagnostic"] or result["selection"] != "family_cv":
                continue
            hard = (saved[name][keep] >= .5).astype(float)
            begin = time.perf_counter()
            lo = observer.extreme(hard)
            hi = observer.extreme(hard, maximum=True)
            seconds = time.perf_counter()-begin
            grid = family_ious(hard, family)
            # Rectangular relaxation of the CONTINUOUS family uses monotone endpoints.
            lower = observer.labels(.3,.85); upper = observer.labels(.7,.75)
            boxlo, boxhi = iou_bounds(hard, lower, upper)
            if np.any(np.diff([boxlo, lo["iou"], grid.min(), grid.max(), hi["iou"], boxhi]) < -1e-10):
                raise AssertionError("uncertainty set nesting violated")
            for extremum in (lo,hi):
                extremum.pop("label")
            rows.append({"source":source,"model":name,"n":int(keep.sum()),
                         "continuous_min":lo,"continuous_max":hi,
                         "grid_min":float(grid.min()),"grid_max":float(grid.max()),
                         "box_min":boxlo,"box_max":boxhi,"seconds_both_extrema":seconds,
                         "grid_lower_optimism":float(grid.min()-lo["iou"])})
    rng = np.random.default_rng(20261013)
    timing = []
    for n in (250,500,1000,2000):
        ratio, periodicity = rng.uniform(.3,.7,n), rng.uniform(.75,.85,n)
        p = rng.random(n)
        timing.append(repeated_timing(ratio, periodicity, p))
    out = {"scope":"Post-hoc deterministic sensitivity audit, hard pooled IoU on already-frozen retrospective predictions.",
           "not_claimed":"Mechanical truth coverage, macro-ratio optimization, novel fractional-programming primitive, or prediction improvement.",
           "max_observed_grid_optimism":max(r["grid_lower_optimism"] for r in rows),
           "mean_observed_grid_optimism":float(np.mean([r["grid_lower_optimism"] for r in rows])),
           "models_audited":len(rows),"results":rows,"timing":timing,
           "timing_protocol":{"seed":20261013,"repetitions":7,
                              "input_scope":"One synthetic independent-uniform feature/probability realization per N; same input repeated seven times",
                              "warmup":"Numba and each size's algorithms warmed outside timing; no cold-JIT claim",
                              "ordering":"Alternate sweep/dense order across repetitions",
                              "fairness":"Shared ContinuousObserver geometry reported separately; dense extra bin preparation separately; both solve arrays allocated in timing",
                              "not_claimed":"Independent machine/input replications, general deployment speedup, peak-memory measurement or cold-start advantage",
                              "platform":platform.platform(),"machine":platform.machine(),
                              "python":platform.python_version(),"numpy":np.__version__,"numba":numba.__version__},
           "code_sha256":{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                          (Path(__file__),ROOT/"scripts/continuous_observer.py")}}
    path = Path(__file__).with_name("continuous_observer_summary.json")
    path.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({k:v for k,v in out.items() if k != "results"},indent=2))


if __name__ == "__main__":main()
