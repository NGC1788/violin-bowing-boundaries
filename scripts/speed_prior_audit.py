#!/usr/bin/env python3
"""Post-hoc, matched audit: a static 2-D baseline vs declared speed priors.

The earlier raw-log baseline includes a speed feature almost constant in training;
its standardization can magnify out-of-speed extrapolation. This audit removes
that confound. Every model has exactly the same two input coordinates and learner.
Exponents are compared, not selected using target scores. No sample2 input is read.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import benchmark_data as bd
from benchmark_sets import blocked_folds, evaluate_panels, metrics
from set_envelope import label_family, similarity_coordinates


def estimator(kind, c):
    if kind == "quadratic_logistic":
        return make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(),
                             LogisticRegression(C=c, max_iter=1000, random_state=0))
    return make_pipeline(StandardScaler(), SVC(C=c, probability=True, random_state=0))


def run():
    out = bd.ROOT / "research/2026-10-13/speed_prior_audit"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "scope": "Post-hoc retrospective diagnostic motivated by already inspected primary results; no sample2 access.",
        "priors": [0., .5, 1., 1.5], "learners": ["quadratic_logistic", "rbf_svm"],
        "C_grid": [.1, 1., 10.], "selection": "Pooled source-only 5 contiguous beta-column folds; family-minimum IoU; first tie wins.",
        "no_target_selection": "Every exponent is reported. A single training speed cannot identify a speed law.",
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    # Record the declared comparison before reading target outcomes in this run.
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    d = bd.load_dataset(include_ambiguous=True)
    family = label_family(d["output_features"], d["reference_f0_hz"])
    nominal = d["y_fixed"] == 1
    rows, fits = [], 0
    started = time.perf_counter()
    for source in "ABCD":
        keep = (d["string"] == source) & (d["speed"] == .1)
        folds = blocked_folds(d["beta_level"][keep])
        for kind in manifest["learners"]:
            for exponent in manifest["priors"]:
                x = similarity_coordinates(d["X"], True, exponent)
                xs, ys, fs = x[keep], nominal[keep], family[keep]
                validation = []
                for c in manifest["C_grid"]:
                    p = np.zeros(len(xs))
                    for train, valid in folds:
                        model = estimator(kind, c).fit(xs[train], ys[train])
                        p[valid] = model.predict_proba(xs[valid])[:, 1]
                        fits += 1
                    validation.append({"C": c, "metrics": metrics(p, ys, fs)})
                choice = max(validation, key=lambda r: r["metrics"]["family_min_iou"])
                model = estimator(kind, choice["C"]).fit(xs, ys)
                fits += 1
                p = model.predict_proba(x)[:, 1]
                scores = evaluate_panels(p, nominal, family, d["group"], ~keep)
                rows.append({"source": source, "learner": kind, "exponent": exponent,
                             "selection": choice, "all_source_CV": validation,
                             **scores})
    result = {"scope": manifest["scope"], "fit_count": fits,
              "seconds": time.perf_counter() - started, "results": rows}
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    for row in rows:
        print(row["source"], row["learner"], row["exponent"],
              round(row["macro_panels"]["family_min_iou"], 4), flush=True)
    print(f"Completed {fits} fits in {result['seconds']:.2f} seconds.")


if __name__ == "__main__":
    run()
