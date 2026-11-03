"""Fixed HGB acquisition comparison with an independent termination certificate."""
import gzip
import hashlib
import json
import time

import numpy as np
from threadpoolctl import threadpool_limits

from agentic_research.engine_v2.harness.data import load_dataset
from agentic_research.engine_v2.harness.models import fit_prior
from agentic_research.law_decision_pilot_v1.pilot import (
    DEV_SESSIONS, DIGEST, SOURCE, certificates, intervals, possible_starts,
)
from agentic_research.law_query_policy_v1.compare import choose

TOLERANCES = (.2, .4, .8, 1.6)


def selection(partial, possible, lower, upper, probabilities):
    valid = possible & np.isfinite(lower)
    left = np.where(valid, lower, np.inf).min(axis=-1)
    right = np.where(valid, upper, -np.inf).max(axis=-1)
    max_left, min_right = left.max(axis=0), right.min(axis=0)
    live = valid.any(axis=-1).all(axis=0) & (max_left <= min_right)
    weights = live * (
        1 + (left == max_left).astype(int) + (right == min_right).astype(int)
    )
    weights *= possible.sum(axis=-1) > 1
    votes = possible * weights[..., None]
    votes[..., -1] = 0
    positive = votes.copy()
    positive[..., 1:] += votes[..., :-1]
    negative = np.zeros_like(votes)
    negative[..., :-1] = votes[..., 1:]
    score = ((1 - probabilities) * positive + probabilities * negative).sum(axis=1)
    score[(partial != -1).any(axis=1)] = -1
    rank, column = np.unravel_index(np.argmax(score.T), score.T.shape)
    return column, rank


def full_decision(labels, lower, upper):
    # Scalar first-run and intersection, independent of the stopping code.
    for rule in labels.transpose(1, 0, 2):
        bounds = []
        for column, bits in enumerate(rule):
            first = next((i for i in range(len(bits) - 1) if bits[i] and bits[i + 1]), None)
            if first is None or not np.isfinite(lower[column, 0, first]):
                break
            bounds.append((lower[column, 0, first], upper[column, 0, first]))
        else:
            if max(lo for lo, _ in bounds) <= min(hi for _, hi in bounds):
                return True
    return False


def replay(labels, lower, upper, expected, prior, check_uniform=False):
    partial = np.full(labels.shape, -1, dtype=np.int8)
    started = time.perf_counter()
    for queries in range(labels.shape[0] * labels.shape[2] + 1):
        possible = possible_starts(partial)
        decision, _, _ = certificates(possible, lower, upper)
        if decision is not None:
            if decision != expected:
                raise AssertionError("Wrong certificate")
            return {"queries": queries, "seconds": time.perf_counter() - started}
        column, rank = selection(partial, possible, lower, upper, prior)
        if check_uniform and (column, rank) != choose(partial, possible, lower, upper, True):
            raise AssertionError("Uniform policy differs from previous decision policy")
        if (partial[column, :, rank] != -1).any():
            raise AssertionError("Repeat query or missing terminal certificate")
        partial[column, :, rank] = labels[column, :, rank]
    raise AssertionError("Uncertified result")


def run():
    path = SOURCE / "input_joint.json.gz"
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != DIGEST:
        raise ValueError("Unexpected input digest")
    train, dev, rules, receipt = load_dataset(path, max_eval_columns=480)
    by_id = {c.id: c for c in dev}
    groups = {}
    for col in json.loads(gzip.decompress(blob))["columns"]:
        if col["session"] in DEV_SESSIONS and col["eligible"]:
            groups.setdefault(col["group"], []).append(col)
    if len(groups) != 12:
        raise ValueError("Unexpected folder count")
    started = time.perf_counter()
    learned, _ = fit_prior({"prior": "hgb", "seed": 0}, train, rules)
    fit_seconds = time.perf_counter() - started
    prevalence, _ = fit_prior({"prior": "constant"}, train, rules)
    rows = []
    for name, columns in sorted(groups.items()):
        columns.sort(key=lambda col: col["id"])
        members = [by_id[col["id"]] for col in columns]
        labels = np.stack([c.labels for c in members])
        started = time.perf_counter()
        predictions = np.stack([learned.predict(c.public) for c in members])
        predict_seconds = time.perf_counter() - started
        priors = {
            "decision_uniform": np.full(labels.shape, .5),
            "decision_prevalence": np.stack([prevalence.predict(c.public) for c in members]),
            "decision_hgb": predictions,
        }
        low, high = intervals(columns)
        for epsilon in TOLERANCES:
            lower, upper = low - (epsilon - .2), high + (epsilon - .2)
            expected = full_decision(labels, lower, upper)
            methods = {
                method: replay(labels, lower, upper, expected, prior, method == "decision_uniform")
                for method, prior in priors.items()
            }
            rows.append(dict(group=name, epsilon=epsilon, compatible=expected,
                             hgb_predict_seconds_shared_across_tolerances=predict_seconds,
                             methods=methods))
    return {"source_sha256": receipt["input_sha256"], "train_columns": len(train),
            "evaluation_columns": len(dev), "hgb_fits": 1, "hgb_fit_seconds": fit_seconds,
            "runs": rows}


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        print(json.dumps(run(), indent=2, allow_nan=False))
