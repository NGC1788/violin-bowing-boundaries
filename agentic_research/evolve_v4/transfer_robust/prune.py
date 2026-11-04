"""Hindsight-pruned certificate sizes for the robust search split (same method as prune_a.py)."""
import json

from agentic_research.evolve_v4.prune_a import prune, seed_queries
from agentic_research.evolve_v4.transfer_robust.build import OUT, load_cases

if __name__ == "__main__":
    out = []
    for case in load_cases("search"):
        q = seed_queries(case)
        out.append(dict(group=case["group"], epsilon=case["epsilon"], compatible=bool(case["expected"]),
                        seed=len(q), pruned=len(prune(case, q))))
    tot = {}
    for r in out:
        t = tot.setdefault(r["epsilon"], [0, 0]); t[0] += r["seed"]; t[1] += r["pruned"]
    (OUT / "prune_search.json").write_text(json.dumps(dict(split="search", seed_vs_pruned=tot, per_case=out)))
    print(tot, sum(t[1] for t in tot.values()))
