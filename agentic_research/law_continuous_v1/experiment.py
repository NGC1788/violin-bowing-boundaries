"""Source-only response prior and continuous stopping replay."""
import gzip
import hashlib
import importlib.util
import json
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from agentic_research.engine_v2.harness.data import load_dataset
from agentic_research.law_counterfactual_selection_v1.experiment import (
    outer, response_patterns, shortlist,
)
from agentic_research.law_decision_pilot_v1.pilot import (
    DEV_SESSIONS, DIGEST, SOURCE, certificates, intervals, possible_starts,
)
from agentic_research.law_continuous_v1.certificate import Observations, decide


def anchor_choice(partial, lower, upper, patterns, prior):
    possible = possible_starts(partial)
    decision, resolved, _ = certificates(possible, lower, upper)
    if decision is not None or resolved:
        rank, column = np.argwhere(partial[:, 0].T == -1)[0]
        return int(column), int(rank), True

    left, right = outer(possible, lower, upper)
    live = left.max(0) <= right.min(0)
    column, rank, priority = shortlist(partial, possible, lower, upper, left, right, live)
    hypothetical = np.broadcast_to(
        partial[column, None], (len(column), len(patterns), *partial.shape[1:])
    ).copy()
    hypothetical[np.arange(len(column)), :, :, rank] = patterns
    l_new, r_new = outer(possible_starts(hypothetical), lower[column, None], upper[column, None])
    others = np.arange(len(partial))[None, :] != column[:, None]
    l_other = np.where(others[..., None], left, -np.inf).max(axis=1)
    r_other = np.where(others[..., None], right, np.inf).min(axis=1)
    lo = np.maximum(l_new, l_other[:, None])
    hi = np.minimum(r_new, r_other[:, None])
    remains = (lo <= hi) & live
    width = np.where(remains, np.maximum(hi - lo, 0), 0).sum(-1)
    probability = prior[column, rank]
    counts = (remains.sum(-1) * probability).sum(-1)
    slack = (width * probability).sum(-1)
    best = np.lexsort((np.arange(len(column)), -priority,
                       np.round(slack, 12), np.round(counts, 12)))[0]
    return int(column[best]), int(rank[best]), False


def replay(columns, rules, patterns, lower, upper, prior, expected):
    shape = (len(columns), len(columns[0]["force"]))
    obs = Observations.empty(shape)
    revealed = [np.array([col[key] for col in columns], dtype=dtype) for key, dtype in
                (("fixed", bool), ("guard", bool), ("periodicity", float), ("ratio", float))]
    total_rectangles = 0
    capped_checks = 0
    fallbacks = 0
    started = time.perf_counter()
    for count in range(np.prod(shape) + 1):
        if count % 64 == 0 or count == np.prod(shape):
            cap = None if count == np.prod(shape) else 192
            verdict, visited = decide(obs, lower, upper, cap)
            total_rectangles += visited
            capped_checks += int(cap is not None and visited == cap and verdict is None)
            if verdict is not None:
                if verdict != expected:
                    raise AssertionError("Certificate disagrees with full solver")
                return dict(queries=count, rectangles=total_rectangles, capped_checks=capped_checks,
                            anchor_fallback_queries=fallbacks, seconds=time.perf_counter()-started)
        if count == np.prod(shape):
            break
        if prior is None:
            rank, col = np.argwhere(~obs.known.T)[0]
        else:
            labels = np.stack([obs.labels(u, v) for u, v in rules], axis=1)
            col, rank, fallback = anchor_choice(labels, lower, upper, patterns, prior)
            fallbacks += fallback
        if obs.known[col, rank]:
            raise AssertionError("Repeat query")
        obs.reveal(col, rank, *(a[col, rank] for a in revealed))
    raise AssertionError("Full observation failed to certify")


def run():
    blob = (SOURCE / "input_joint.json.gz").read_bytes()
    if hashlib.sha256(blob).hexdigest() != DIGEST:
        raise ValueError("Unexpected source data")
    train, dev, rules, _ = load_dataset(SOURCE / "input_joint.json.gz", max_eval_columns=480)
    public = {c.id: c.public for c in dev}
    patterns = response_patterns(rules)
    source = np.concatenate([c.labels.T for c in train])
    matching = (source[:, None] == patterns).all(-1)
    if not (matching.sum(-1) == 1).all():
        raise ValueError("Unrecognized source pattern")
    model = HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=15, min_samples_leaf=30,
                                           learning_rate=.08, l2_regularization=1.,
                                           early_stopping=False, random_state=0)
    started = time.perf_counter()
    model.fit(np.concatenate([c.public for c in train]), matching.argmax(-1))
    fit_seconds = time.perf_counter() - started
    groups = {}
    for col in json.loads(gzip.decompress(blob))["columns"]:
        if col["session"] in DEV_SESSIONS:
            groups.setdefault(col["group"], []).append(col)
    if len(groups) != 12:
        raise ValueError("Unexpected group count")
    spec = importlib.util.spec_from_file_location("frozen_solver", SOURCE / "solver.py")
    solver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solver)
    results = []
    for group, columns in sorted(groups.items()):
        eligible = sorted((col for col in columns if col["eligible"]), key=lambda col: col["id"])
        x = np.stack([public[col["id"]] for col in eligible])
        learned = np.zeros((*x.shape[:2], len(patterns)))
        learned[..., model.classes_] = model.predict_proba(x.reshape(-1, 3)).reshape(
            *x.shape[:2], len(model.classes_))
        uniform = np.full_like(learned, 1 / len(patterns))
        low, high = intervals(eligible)
        reference = solver.analyze_group(columns, slopes=(-2,), epsilons=(.2, .4))
        truth = reference["compatible"].any(axis=(0, 1)).ravel()
        for epsilon, expected in zip((.2, .4), truth):
            lower, upper = low - (epsilon - .2), high + (epsilon - .2)
            policies = {}
            for method, prior in (("rank_first", None), ("uniform", uniform), ("learned", learned)):
                policies[method] = replay(eligible, rules, patterns, lower, upper, prior, bool(expected))
            results.append(dict(group=group, epsilon=epsilon, compatible=bool(expected), policies=policies))
    return {"source_sha256": DIGEST, "fit_seconds": fit_seconds, "source_columns": len(train),
            "source_response_classes": len(model.classes_), "runs": results}


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        print(json.dumps(run(), indent=2, allow_nan=False))
