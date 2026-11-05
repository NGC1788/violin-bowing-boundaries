"""Analysis only: replay a policy on one bundle's search cases and log per-read branch tags.
usage: python tr.py BUNDLE POLICY [case_ids]"""
import sys, importlib.util, os
import numpy as np
B = sys.argv[1]
sys.path.insert(0, B)
from agentic_research.evolve_v4.colab_task import load_cases
from agentic_research.evolve_v4.core import possible_starts, certificates, View

def load(path):
    spec = importlib.util.spec_from_file_location("cand", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def run(case, m):
    labels, lower, upper = case["labels"], case["lower"], case["upper"]
    partial = np.full(labels.shape, -1, dtype=np.int8)
    revealed = {k: np.full(v.shape, np.nan) for k, v in case["hidden"].items()}
    qs = []
    while True:
        possible = possible_starts(partial)
        d, _, _ = certificates(possible, lower, upper)
        if d is not None:
            return qs
        if hasattr(m, 'TAG'): m.TAG[0] = '?'
        c, r = m.choose(View(partial, possible, case, revealed, len(qs)))
        c, r = int(c), int(r)
        partial[c, :, r] = labels[c, :, r]
        qs.append((c, r, getattr(m, 'TAG', ['-'])[0], tuple(labels[c, :, r])))

if __name__ == "__main__":
    m = load(sys.argv[2])
    sel = sys.argv[3].split(',') if len(sys.argv) > 3 else None
    cases = load_cases()
    for i, case in enumerate(cases):
        if sel and str(i) not in sel:
            continue
        qs = run(case, m)
        from collections import Counter
        print(i, case['group'][:20], case['epsilon'], case['expected'], len(qs), dict(Counter(q[2] for q in qs)))
        if os.environ.get('V'):
            for k, q in enumerate(qs):
                print('  ', k, q[0], q[1], q[2], ''.join(str(x) for x in q[3]))
