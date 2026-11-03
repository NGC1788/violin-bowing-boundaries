"""Task A: order waveform queries so the fixed nine-rule law certificate stops early.

The certificate is the frozen one from law_decision_pilot_v1, so any policy is
correct by construction; a policy only changes how many waveforms are read.
Evolution sees A/B folders only. C/D folders (used in the 2026-11-03 report)
are the sealed holdout and are scored only by final_a.py.
"""
import gzip
import hashlib
import json
import math

import numpy as np

from agentic_research.engine_v2.harness.data import DEV_SESSIONS, rule_bank
from agentic_research.law_decision_pilot_v1.pilot import (
    DIGEST, SOURCE, certificates, possible_starts, waveform_labels,
)

EPSILONS = (.2, .4)
FEATURES = ("periodicity", "ratio", "fixed", "guard")


def intervals(columns, epsilon):
    force = np.array([c["force"] for c in columns], dtype=float)
    log_beta = np.array([math.log(c["beta"]) for c in columns])
    lower = np.full(force.shape, np.nan)
    upper = lower.copy()
    valid = (force[:, :-2] > 0) & (force[:, 1:-1] > force[:, :-2])
    rows, previous = np.where(valid)
    start = previous + 1
    lower[rows, start] = np.log(force[rows, start - 1]) + 2 * log_beta[rows] - epsilon
    upper[rows, start] = np.log(force[rows, start]) + 2 * log_beta[rows] + epsilon
    return lower[:, None, :], upper[:, None, :]


def load_cases(split):
    """split: 'search' (A/B, used by evolution) or 'holdout' (C/D, sealed)."""
    path = SOURCE / "input_joint.json.gz"
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != DIGEST:
        raise ValueError("Source hash mismatch")
    groups = {}
    for col in json.loads(gzip.decompress(blob))["columns"]:
        if (col["session"] in DEV_SESSIONS) == (split == "holdout"):
            groups.setdefault(col["group"], []).append(col)
    rules = rule_bank("nine")
    cases = []
    for group, columns in sorted(groups.items()):
        columns.sort(key=lambda c: c["id"])
        eligible = [c for c in columns if c["eligible"]]
        if not eligible:
            continue
        labels = waveform_labels(eligible, rules).astype(np.int8)
        public = dict(
            force=np.array([c["force"] for c in eligible], dtype=float),
            log_beta=np.array([math.log(c["beta"]) for c in eligible]),
            rules=rules.astype(float),
        )
        hidden = {k: np.array([c[k] for c in eligible], dtype=float) for k in FEATURES}
        for epsilon in EPSILONS:
            lower, upper = intervals(eligible, epsilon)
            expected, _, _ = certificates(possible_starts(labels), lower, upper)
            if expected is None:
                raise AssertionError("Full observation must decide")
            cases.append(dict(group=group, epsilon=epsilon, labels=labels, lower=lower,
                              upper=upper, expected=expected, public=public, hidden=hidden))
    return cases


class View:
    """Everything a policy may see. Arrays are copies; mutating them has no effect."""

    def __init__(self, partial, possible, case, revealed, count):
        self.partial = partial.copy()            # (columns, 9 rules, 50 ranks): -1 unknown, 0/1 seen
        self.possible = possible.copy()          # possible first-run starts, last slot = no run
        self.lower = case["lower"].copy()        # (columns, 1, 50) intercept interval ends, NaN invalid
        self.upper = case["upper"].copy()
        self.force = case["public"]["force"].copy()        # (columns, 50) measured force, sorted
        self.log_beta = case["public"]["log_beta"].copy()  # (columns,)
        self.rules = case["public"]["rules"].copy()        # (9, 2) periodicity floor, ratio ceiling
        self.epsilon = case["epsilon"]
        self.revealed = {k: v.copy() for k, v in revealed.items()}  # NaN where unqueried
        self.queries_so_far = count


def replay(case, choose):
    labels, lower, upper = case["labels"], case["lower"], case["upper"]
    partial = np.full(labels.shape, -1, dtype=np.int8)
    revealed = {k: np.full(v.shape, np.nan) for k, v in case["hidden"].items()}
    columns, _, ranks = labels.shape
    for count in range(columns * ranks + 1):
        possible = possible_starts(partial)
        decision, _, _ = certificates(possible, lower, upper)
        if decision is not None:
            if decision != case["expected"]:
                raise AssertionError("Certificate disagrees with full decision")
            return count
        col, rank = choose(View(partial, possible, case, revealed, count))
        col, rank = int(col), int(rank)
        if not (0 <= col < columns and 0 <= rank < ranks) or partial[col, 0, rank] != -1:
            raise ValueError(f"Invalid or repeated query {(col, rank)}")
        partial[col, :, rank] = labels[col, :, rank]
        for key in revealed:
            revealed[key][col, rank] = case["hidden"][key][col, rank]
    raise AssertionError("No terminal certificate")
