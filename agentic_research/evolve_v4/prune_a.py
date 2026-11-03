"""Layer-0 headroom v2: hindsight-pruned certificate size.

Replay the seed policy, then remove queried waveforms (latest first) whenever the
remaining revealed set still certifies the same decision. The pruned size is an
upper bound on the minimum certificate (an oracle target, not deployable).
"""
import json
import sys

import numpy as np

from agentic_research.evolve_v4.helpers import certificates, possible_starts
from agentic_research.evolve_v4.task_a import View, load_cases
from agentic_research.evolve_v4.seed_a import choose


def seed_queries(case):
    labels, lower, upper = case["labels"], case["lower"], case["upper"]
    partial = np.full(labels.shape, -1, dtype=np.int8)
    revealed = {k: np.full(v.shape, np.nan) for k, v in case["hidden"].items()}
    queries = []
    while certificates(possible_starts(partial), lower, upper)[0] is None:
        col, rank = choose(View(partial, possible_starts(partial), case, revealed, len(queries)))
        partial[col, :, rank] = labels[col, :, rank]
        queries.append((col, rank))
    return queries


def prune(case, queries):
    labels, lower, upper = case["labels"], case["lower"], case["upper"]
    keep = list(queries)
    for q in reversed(queries):
        trial = [x for x in keep if x != q]
        partial = np.full(labels.shape, -1, dtype=np.int8)
        for col, rank in trial:
            partial[col, :, rank] = labels[col, :, rank]
        if certificates(possible_starts(partial), lower, upper)[0] == case["expected"]:
            keep = trial
    return keep


if __name__ == "__main__":
    out = []
    for case in load_cases(sys.argv[1]):
        q = seed_queries(case)
        out.append(dict(group=case["group"], epsilon=case["epsilon"], compatible=bool(case["expected"]),
                        seed=len(q), pruned=len(prune(case, q))))
    tot = {}
    for r in out:
        t = tot.setdefault(r["epsilon"], [0, 0]); t[0] += r["seed"]; t[1] += r["pruned"]
    print(json.dumps(dict(split=sys.argv[1], seed_vs_pruned=tot, per_case=out)))
