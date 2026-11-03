"""Replay a decision certificate on nine common waveform rules."""
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path

import numpy as np

from agentic_research.engine_v2.harness.data import DEV_SESSIONS, rule_bank

SOURCE = Path(__file__).resolve().parents[2] / (
    "research/2026-10-19_protocol_audit/novelty/joint_rule"
)
DIGEST = "85ab696054670f80bc9690873291c39bb119b56670b4513da59bae994a2b0d70"


def possible_starts(partial):
    """First adjacent positive pair, with final slot for no run. Unknown is -1."""
    forced = (partial[..., :-1] == 1) & (partial[..., 1:] == 1)
    earlier = np.cumsum(forced, axis=-1) - forced
    possible = (partial[..., :-1] != 0) & (partial[..., 1:] != 0) & (earlier == 0)
    possible[..., 1:] &= partial[..., :-2] != 1
    return np.concatenate([possible, ~forced.any(axis=-1, keepdims=True)], axis=-1)


def certificates(possible, lower, upper):
    """Return decision, whether all boundaries are known, and geometric rejection."""
    valid = np.isfinite(lower)
    candidates = possible & valid
    outer_low = np.where(candidates, lower, np.inf).min(axis=-1).max(axis=0)
    outer_high = np.where(candidates, upper, -np.inf).max(axis=-1).min(axis=0)
    can_be_valid = candidates.any(axis=-1).all(axis=0)
    could_pass = can_be_valid & (outer_low <= outer_high)
    all_valid = ~(possible & ~valid).any(axis=(0, 2))
    inner_low = np.where(candidates, lower, -np.inf).max(axis=-1).max(axis=0)
    inner_high = np.where(candidates, upper, np.inf).min(axis=-1).min(axis=0)
    guaranteed_pass = all_valid & (inner_low <= inner_high)
    decision = True if guaranteed_pass.any() else False if not could_pass.any() else None
    geometric_only = bool((can_be_valid & (outer_low > outer_high)).all())
    return decision, bool((possible.sum(axis=-1) == 1).all()), geometric_only


def intervals(columns):
    force = np.array([col["force"] for col in columns], dtype=float)
    log_beta = np.array([math.log(col["beta"]) for col in columns])
    lower = np.full(force.shape, np.nan)
    upper = lower.copy()
    valid = (force[:, :-2] > 0) & (force[:, 1:-1] > force[:, :-2])
    rows, previous = np.where(valid)
    start = previous + 1
    lower[rows, start] = [math.log(force[i, j - 1]) + 2 * log_beta[i] - .2 for i, j in zip(rows, start)]
    upper[rows, start] = [math.log(force[i, j]) + 2 * log_beta[i] + .2 for i, j in zip(rows, start)]
    return lower[:, None, :], upper[:, None, :]


def waveform_labels(columns, rules):
    fixed = np.array([col["fixed"] for col in columns], dtype=bool)[:, None, :]
    guard = np.array([col["guard"] for col in columns], dtype=bool)[:, None, :]
    periodicity = np.array([col["periodicity"] for col in columns], dtype=float)[:, None, :]
    ratio = np.array([col["ratio"] for col in columns], dtype=float)[:, None, :]
    return fixed | (guard & (periodicity >= rules[None, :, 0, None]) &
                    (ratio < rules[None, :, 1, None]))


def replay(labels, lower, upper, order, expected):
    partial = np.full(labels.shape, -1, dtype=np.int8)
    direct_stop = None
    reason = None
    geometry_stop = None
    for count, query in enumerate(itertools.chain([None], order), start=0):
        if query is not None:
            col, rank = query
            partial[col, :, rank] = labels[col, :, rank]
        decision, resolved, geometry = certificates(possible_starts(partial), lower, upper)
        if decision is not None and decision != expected:
            raise AssertionError("Partial certificate disagrees with full solver")
        if decision is not None and direct_stop is None:
            direct_stop = count
            reason = "positive" if decision else "geometric" if geometry else "includes_invalid"
        if geometry and geometry_stop is None:
            geometry_stop = count
        if resolved:
            if decision != expected or direct_stop is None:
                raise AssertionError("Full boundary certificate failed")
            return dict(direct_queries=direct_stop, boundary_queries=count, reason=reason,
                        geometric_only_queries=geometry_stop)
    raise AssertionError("No terminal certificate")


def run():
    path = SOURCE / "input_joint.json.gz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != DIGEST:
        raise ValueError("Source hash mismatch")
    with gzip.open(path, "rt") as stream:
        groups = {}
        for col in json.load(stream)["columns"]:
            if col["session"] in DEV_SESSIONS:
                groups.setdefault(col["group"], []).append(col)
    if len(groups) != 12:
        raise ValueError("Unexpected folder count")
    spec = importlib.util.spec_from_file_location("frozen_solver", SOURCE / "solver.py")
    solver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solver)
    rules = rule_bank("nine")
    results = []
    for index, (group, columns) in enumerate(sorted(groups.items())):
        columns.sort(key=lambda col: col["id"])
        eligible = [col for col in columns if col["eligible"]]
        labels = waveform_labels(eligible, rules)
        lower, upper = intervals(eligible)
        # Existing solver, run at one exact threshold pair at a time.
        expected = any(
            bool(solver.analyze_group(columns, u_domain=(u, u), v_domain=(v, v),
                                      slopes=(-2,), epsilons=(.2,))["compatible"].item())
            for u, v in rules
        )
        if certificates(possible_starts(labels.astype(np.int8)), lower, upper)[0] != expected:
            raise AssertionError("Full data disagrees with frozen solver")
        order = [(col, rank) for rank in range(labels.shape[-1]) for col in range(len(eligible))]
        permutation = np.random.default_rng(20261103 + index).permutation(len(order))
        for policy, queries in (("rank_first", order), ("random", [order[i] for i in permutation])):
            results.append(dict(group=group, policy=policy, expected=expected, eligible=len(eligible),
                                **replay(labels, lower, upper, queries, expected)))
    return {"source_sha256": DIGEST, "scope": "retrospective nine-rule decision", "runs": results}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
