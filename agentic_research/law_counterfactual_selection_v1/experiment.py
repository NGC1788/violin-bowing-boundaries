"""Counterfactual response scoring with the existing certificate."""
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from agentic_research.engine_v2.harness.data import load_dataset
from agentic_research.engine_v2.harness.models import fit_prior
from agentic_research.law_decision_pilot_v1.pilot import (
    DEV_SESSIONS, DIGEST, SOURCE, certificates, intervals, possible_starts,
)
from agentic_research.law_learned_selection_v1.experiment import selection

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "law_learned_selection_v1"


def response_patterns(rules):
    u, v = rules.T
    masks = [(u <= a) & (v >= b) for a in np.unique(u) for b in np.unique(v)]
    return np.vstack([np.zeros(len(rules), dtype=bool), masks]).astype(np.int8)


def outer(possible, lower, upper):
    valid = possible & np.isfinite(lower)
    left = np.where(valid, lower, np.inf).min(axis=-1)
    right = np.where(valid, upper, -np.inf).max(axis=-1)
    return left, right


def shortlist(partial, possible, lower, upper, left, right, live):
    unobserved = partial[:, 0] == -1
    weights = live * (1 + (left == left.max(0)).astype(int)
                      + (right == right.min(0)).astype(int))
    weights *= possible.sum(-1) > 1
    starts = (possible * weights[..., None]).sum(axis=1)
    starts[:, -1] = 0
    score = starts.copy()
    score[:, 1:] += starts[:, :-1]
    score[:, :-1] += starts[:, 1:]
    score[~unobserved] = -1
    order = np.argsort(-score.T.ravel(), kind="stable")
    top = order[:min(6, int(unobserved.sum()))]
    flat = list(top) + [int(np.flatnonzero(unobserved.T)[0])]
    for endpoints, intervals, bound, select in (
        (left, lower, left.max(0), np.argmin),
        (right, upper, right.min(0), np.argmax),
    ):
        for column, rule in np.argwhere((endpoints == bound) & live & (weights > 0)):
            valid = possible[column, rule] & np.isfinite(intervals[column, 0])
            fill = np.inf if select is np.argmin else -np.inf
            start = int(select(np.where(valid, intervals[column, 0], fill)))
            for rank in range(max(0, start - 1), start + 2):
                if unobserved[column, rank]:
                    flat.append(rank * len(partial) + column)
                    break
    rank, column = np.unravel_index(np.unique(flat), score.T.shape)
    return column, rank, score[column, rank]


def choose(partial, possible, lower, upper, patterns, probabilities):
    left, right = outer(possible, lower, upper)
    max_left, min_right = left.max(0), right.min(0)
    live = max_left <= min_right
    column, rank, priority = shortlist(partial, possible, lower, upper, left, right, live)
    hypothetical = np.broadcast_to(
        partial[column, None], (len(column), len(patterns), *partial.shape[1:])
    ).copy()
    hypothetical[np.arange(len(column)), :, :, rank] = patterns
    l_new, r_new = outer(possible_starts(hypothetical), lower[column, None], upper[column, None])
    others = np.arange(len(partial))[None, :] != column[:, None]
    l_other = np.where(others[..., None], left, -np.inf).max(axis=1)
    r_other = np.where(others[..., None], right, np.inf).min(axis=1)
    l_total = np.maximum(l_new, l_other[:, None])
    r_total = np.minimum(r_new, r_other[:, None])
    remaining = (l_total <= r_total) & live
    width = np.where(remaining, np.maximum(r_total - l_total, 0), 0).sum(axis=-1)
    rules = remaining.sum(axis=-1) @ probabilities
    slack = width @ probabilities
    # The shortlist is already ordered by rank, then column.
    selected = np.lexsort((np.arange(len(column)), -priority,
                           np.round(slack, 12), np.round(rules, 12)))[0]
    return int(column[selected]), int(rank[selected])


def trace(labels, lower, upper, expected, policy, prior, patterns, probabilities, checkpoint):
    partial = np.full(labels.shape, -1, dtype=np.int8)
    queries = []
    at_checkpoint = None
    started = time.perf_counter()
    for count in range(labels.shape[0] * labels.shape[-1] + 1):
        possible = possible_starts(partial)
        if count == checkpoint:
            left, right = outer(possible, lower, upper)
            at_checkpoint = int((left.max(0) <= right.min(0)).sum())
        decision, _, _ = certificates(possible, lower, upper)
        if decision is not None:
            if decision != expected:
                raise AssertionError("Certificate disagrees with full decision")
            return queries, {
                "queries": count,
                "live_rules_at_baseline_budget": at_checkpoint,
                "mean_rank_one_based": float(np.mean([r + 1 for _, r in queries])) if queries else None,
                "first_10_rank_queries": int(sum(r < 10 for _, r in queries)),
                "seconds": time.perf_counter() - started,
            }
        if policy == "counterfactual":
            col, rank = choose(partial, possible, lower, upper, patterns, probabilities)
        else:
            col, rank = selection(partial, possible, lower, upper, prior)
        if (partial[col, :, rank] != -1).any():
            raise AssertionError("Repeated query")
        queries.append((col, rank))
        partial[col, :, rank] = labels[col, :, rank]
    raise AssertionError("Missing terminal result")


def run():
    path = SOURCE / "input_joint.json.gz"
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != DIGEST:
        raise ValueError("Source mismatch")
    train, dev, rules, _ = load_dataset(path, max_eval_columns=480)
    by_id = {c.id: c for c in dev}
    groups = {}
    for col in json.loads(gzip.decompress(blob))["columns"]:
        if col["session"] in DEV_SESSIONS and col["eligible"]:
            groups.setdefault(col["group"], []).append(col)
    if len(groups) != 12:
        raise ValueError("Unexpected groups")
    saved = {(r["group"], r["epsilon"]): r for r in json.loads(
        (PARENT / "RESULTS.json").read_text()
    )["runs"]}
    started = time.perf_counter()
    hgb, _ = fit_prior({"prior": "hgb", "seed": 0}, train, rules)
    fit_seconds = time.perf_counter() - started
    patterns = response_patterns(rules)
    source = np.concatenate([c.labels.T for c in train], axis=0)
    matches = (source[:, None] == patterns).all(axis=-1)
    if not (matches.sum(axis=1) == 1).all():
        raise ValueError("Source response outside declared rule patterns")
    frequencies = matches.sum(axis=0) + 1
    frequencies = frequencies / frequencies.sum()
    uniform_patterns = np.full(len(patterns), 1 / len(patterns))
    results = []
    for group, columns in sorted(groups.items()):
        columns.sort(key=lambda col: col["id"])
        members = [by_id[col["id"]] for col in columns]
        labels = np.stack([c.labels for c in members])
        if not (labels.transpose(0, 2, 1)[..., None, :] == patterns).all(-1).any(-1).all():
            raise ValueError("Evaluation response outside declared patterns")
        learned = np.stack([hgb.predict(c.public) for c in members])
        baseline = np.full(labels.shape, .5)
        low, high = intervals(columns)
        for epsilon in (.2, .4):
            reference = saved[group, epsilon]
            expected = reference["compatible"]
            checkpoint = reference["methods"]["decision_uniform"]["queries"]
            lower, upper = low - (epsilon - .2), high + (epsilon - .2)
            methods = {}
            old_path = None
            first_divergence = None
            for name, policy, prior, pattern_prior in (
                ("baseline", "original", baseline, None),
                ("hgb", "original", learned, None),
                ("counterfactual_source", "counterfactual", None, frequencies),
                ("counterfactual_uniform", "counterfactual", None, uniform_patterns),
            ):
                if epsilon != .2 and name == "hgb":
                    continue
                path, result = trace(labels, lower, upper, expected, policy, prior,
                                     patterns, pattern_prior, checkpoint)
                if name == "baseline":
                    if len(path) != checkpoint:
                        raise AssertionError("Baseline replay mismatch")
                    old_path = path
                if name == "hgb":
                    if len(path) != reference["methods"]["decision_hgb"]["queries"]:
                        raise AssertionError("HGB replay mismatch")
                    for i, (original, changed) in enumerate(zip(old_path, path)):
                        if original != changed:
                            first_divergence = {
                                "query": i + 1, "baseline_rank": int(original[1]) + 1,
                                "hgb_rank": int(changed[1]) + 1,
                            }
                            break
                methods[name] = result
            results.append(dict(group=group, epsilon=epsilon, compatible=expected,
                                methods=methods, first_divergence=first_divergence))
    return {"source_sha256": DIGEST, "hgb_refits": 1, "hgb_fit_seconds": fit_seconds,
            "training_pattern_probabilities": frequencies.tolist(), "runs": results}


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        print(json.dumps(run(), indent=2, allow_nan=False))
