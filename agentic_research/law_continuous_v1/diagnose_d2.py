"""Trace live-rule loss under the two frozen D2 finite-bank policies."""
import gzip
import hashlib
import json

import numpy as np

from agentic_research.engine_v2.harness.data import rule_bank
from agentic_research.law_counterfactual_selection_v1.experiment import choose, outer, response_patterns
from agentic_research.law_decision_pilot_v1.pilot import DIGEST, SOURCE, intervals, possible_starts, waveform_labels
from agentic_research.law_query_policy_v1.compare import choose as baseline_choice


def trace():
    path = SOURCE / "input_joint.json.gz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != DIGEST:
        raise ValueError("Unexpected data")
    with gzip.open(path, "rt") as stream:
        columns = sorted((c for c in json.load(stream)["columns"] if c["session"] == "D2"
                          and c["group"].endswith("/v3") and c["eligible"]), key=lambda c: c["id"])
    rules = rule_bank("nine")
    patterns = response_patterns(rules)
    uniform = np.full(len(patterns), 1 / len(patterns))
    truth = waveform_labels(columns, rules)
    lower, upper = intervals(columns)
    out = {}
    for policy, expected_count in (("baseline", 303), ("counterfactual", 339)):
        partial = np.full(truth.shape, -1, dtype=np.int8)
        eliminated = [None] * len(rules)
        snapshots = {}
        inert = 0
        last_width = None
        for count in range(truth.shape[0] * truth.shape[-1] + 1):
            possible = possible_starts(partial)
            left, right = outer(possible, lower, upper)
            lo, hi = left.max(0), right.min(0)
            live = lo <= hi
            width = float(np.where(live, np.maximum(hi - lo, 0), 0).sum())
            if width == last_width:
                inert += 1
            last_width = width
            for i in np.flatnonzero(~live):
                if eliminated[i] is None:
                    eliminated[i] = count
            if count in (0, 64, 128, 256, 303, 339):
                snapshots[str(count)] = {"live_count": int(live.sum()), "width_sum": width}
            if not live.any():
                if count != expected_count:
                    raise AssertionError("Did not reproduce D2 cost")
                break
            if policy == "baseline":
                col, rank = baseline_choice(partial, possible, lower, upper, True)
            else:
                col, rank = choose(partial, possible, lower, upper, patterns, uniform)
            partial[col, :, rank] = truth[col, :, rank]
        out[policy] = dict(queries=count, first_rule_elimination=min(eliminated),
                           per_rule_elimination=eliminated, snapshots=snapshots,
                           queries_without_width_change=inert)
    return out


if __name__ == "__main__":
    print(json.dumps(trace(), indent=2))
