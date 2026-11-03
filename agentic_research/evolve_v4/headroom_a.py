"""Layer-0 headroom: a hindsight policy that knows every label (not deployable).

At each step it reads the waveform whose TRUE response most reduces the number of
live rules, then the summed outer-interval width. Its cost is an achievable target
for a perfect predictor under this greedy rule, not a proven lower bound.
"""
import json
import sys

import numpy as np

from agentic_research.evolve_v4.helpers import outer, possible_starts
from agentic_research.evolve_v4.task_a import load_cases, replay


def make_hindsight(case):
    labels = case["labels"]

    def choose(view):
        partial, lower, upper = view.partial, view.lower, view.upper
        left, right = outer(view.possible, lower, upper)
        col, rank = np.nonzero(partial[:, 0, :] == -1)
        hyp = partial[col].copy()
        hyp[np.arange(len(col)), :, rank] = labels[col, :, rank]
        l_new, r_new = outer(possible_starts(hyp), lower[col], upper[col])
        others = np.arange(len(partial))[None, :] != col[:, None]
        l_tot = np.maximum(l_new, np.where(others[..., None], left, -np.inf).max(1))
        r_tot = np.minimum(r_new, np.where(others[..., None], right, np.inf).min(1))
        live = l_tot <= r_tot
        width = np.where(live, np.maximum(r_tot - l_tot, 0), 0).sum(-1)
        best = np.lexsort((rank, width, live.sum(-1)))[0]
        return int(col[best]), int(rank[best])
    return choose


if __name__ == "__main__":
    split = sys.argv[1]
    out = []
    for case in load_cases(split):
        out.append(dict(group=case["group"], epsilon=case["epsilon"],
                        compatible=bool(case["expected"]), queries=replay(case, make_hindsight(case))))
    tot = {}
    for r in out:
        tot[r["epsilon"]] = tot.get(r["epsilon"], 0) + r["queries"]
    print(json.dumps(dict(split=split, totals=tot, compatible=sum(r["queries"] for r in out if r["compatible"]),
                          per_case=out)))
