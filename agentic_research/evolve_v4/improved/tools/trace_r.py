"""Trace a policy on search cases: queries, hindsight-pruned subset. Analysis only (uses labels offline)."""
import sys, json, importlib.util
import numpy as np
import os; sys.path.insert(0, os.environ.get('BUNDLE', '/home/user/violin-bowing-boundaries/agentic_research/evolve_v4/colab/bundle'))
from agentic_research.evolve_v4.colab_task import load_cases
from agentic_research.evolve_v4.core import possible_starts, certificates, View

def load(path):
    spec = importlib.util.spec_from_file_location("cand", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def run(case, choose):
    labels, lower, upper = case["labels"], case["lower"], case["upper"]
    partial = np.full(labels.shape, -1, dtype=np.int8)
    revealed = {k: np.full(v.shape, np.nan) for k, v in case["hidden"].items()}
    qs = []
    while True:
        possible = possible_starts(partial)
        d, _, _ = certificates(possible, lower, upper)
        if d is not None:
            return qs
        c, r = choose(View(partial, possible, case, revealed, len(qs)))
        c, r = int(c), int(r)
        assert partial[c, 0, r] == -1
        partial[c, :, r] = labels[c, :, r]
        qs.append((c, r))

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
    m = load(sys.argv[1])
    sel = sys.argv[2] if len(sys.argv) > 2 else None
    cases = load_cases()
    out = []
    for i, case in enumerate(cases):
        if sel and str(i) not in sel.split(','):
            continue
        qs = run(case, m.choose)
        pr = prune(case, qs) if qs else []
        out.append((i, case['group'].split('|')[0] + case['group'][-3:], case['epsilon'], case['expected'], len(qs), len(pr)))
        print(*out[-1], flush=True)
    print('total', sum(o[4] for o in out), 'pruned', sum(o[5] for o in out))
