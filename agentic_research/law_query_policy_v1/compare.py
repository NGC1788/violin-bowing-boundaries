"""Compare fixed selection rules behind the same decision certificate."""
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from agentic_research.law_decision_pilot_v1.pilot import (
    DEV_SESSIONS, DIGEST, SOURCE, certificates, intervals, possible_starts,
    rule_bank, waveform_labels,
)

HERE = Path(__file__).resolve().parent


def choose(partial, possible, lower, upper, decision_aware):
    # Each possible start depends on the pair and, except at rank zero,
    # its predecessor. Ignore rules already excluded by the outer bounds.
    starts = possible.copy()
    starts[..., -1] = False
    weights = np.ones(starts.shape[:2], dtype=int)
    if decision_aware:
        valid = possible & np.isfinite(lower)
        left = np.where(valid, lower, np.inf).min(axis=-1)
        right = np.where(valid, upper, -np.inf).max(axis=-1)
        max_left, min_right = left.max(axis=0), right.min(axis=0)
        live = valid.any(axis=-1).all(axis=0) & (max_left <= min_right)
        weights = live * (
            1 + (left == max_left).astype(int) + (right == min_right).astype(int)
        )
    # A resolved rule/column cannot benefit from another query.
    weights = weights * (possible.sum(axis=-1) > 1)
    weighted = (starts * weights[..., None]).sum(axis=1)
    score = weighted.copy()
    score[:, 1:] += weighted[:, :-1]
    score[:, :-1] += weighted[:, 1:]
    score[(partial != -1).any(axis=1)] = -1
    rank, column = np.unravel_index(np.argmax(score.T), score.T.shape)
    return column, rank


def play(labels, lower, upper, expected, policy, order):
    partial = np.full(labels.shape, -1, dtype=np.int8)
    for count in range(len(order) + 1):
        possible = possible_starts(partial)
        decision, _, _ = certificates(possible, lower, upper)
        if decision is not None:
            if decision != expected:
                raise AssertionError("Wrong decision")
            return count
        if count == len(order):
            break
        if policy in {"boundary", "decision"}:
            column, rank = choose(partial, possible, lower, upper, policy == "decision")
        else:
            column, rank = order[count]
        if (partial[column, :, rank] != -1).any():
            raise AssertionError("Duplicate query")
        partial[column, :, rank] = labels[column, :, rank]
    raise AssertionError("Missing terminal decision")


def measure(name, labels, lower, upper, expected, seed):
    order = [(c, i) for i in range(labels.shape[-1]) for c in range(len(labels))]
    shuffled = [order[i] for i in np.random.default_rng(seed).permutation(len(order))]
    counts = {}
    for policy in ("rank_first", "random", "boundary", "decision"):
        counts[policy] = play(
            labels, lower, upper, expected, policy,
            shuffled if policy == "random" else order,
        )
    return {"case": name, "compatible": expected, "queries": counts}


def real_cases():
    path = SOURCE / "input_joint.json.gz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != DIGEST:
        raise ValueError("Unexpected source file")
    parent = json.loads((HERE.parent / "law_decision_pilot_v1/RESULTS.json").read_text())
    if parent["source_sha256"] != DIGEST:
        raise ValueError("Unexpected parent source")
    prior = {(r["group"], r["policy"]): r for r in parent["runs"]}
    with gzip.open(path, "rt") as stream:
        groups = {}
        for column in json.load(stream)["columns"]:
            if column["session"] in DEV_SESSIONS:
                groups.setdefault(column["group"], []).append(column)
    if len(groups) != 12:
        raise ValueError("Unexpected group count")
    for index, (group, columns) in enumerate(sorted(groups.items())):
        eligible = sorted((c for c in columns if c["eligible"]), key=lambda c: c["id"])
        labels = waveform_labels(eligible, rule_bank("nine"))
        result = measure(group, labels, *intervals(eligible),
                         prior[group, "rank_first"]["expected"], 20261103 + index)
        for policy in ("rank_first", "random"):
            if result["queries"][policy] != prior[group, policy]["direct_queries"]:
                raise AssertionError("First pilot replay changed")
        yield result


def synthetic_cases():
    ranks = np.arange(24)
    columns = [{"force": np.exp(ranks * .1), "beta": 1.0} for _ in range(6)]
    lower, upper = intervals(columns)
    for seed in range(8):
        start = int(np.random.default_rng(seed).integers(7, 12))
        for expected in (True, False):
            starts = np.full(6, start)
            if not expected:
                starts[-1] += 7
            lows = [math.log(c["force"][j - 1]) - .2 for c, j in zip(columns, starts)]
            highs = [math.log(c["force"][j]) + .2 for c, j in zip(columns, starts)]
            if (max(lows) <= min(highs)) != expected:
                raise AssertionError("Synthetic construction incorrect")
            labels = np.repeat((ranks >= starts[:, None])[:, None, :], 9, axis=1)
            yield measure(f"synthetic_{seed}_{expected}", labels, lower, upper,
                          expected, 20261103 + seed)


if __name__ == "__main__":
    print(json.dumps({"real": list(real_cases()), "synthetic": list(synthetic_cases())},
                     indent=2, allow_nan=False))
