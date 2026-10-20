"""Post hoc check of the bootstrap unit in session_variance.py (refutation pass, Claude 2026-10-20)."""
import sys; from pathlib import Path; from collections import defaultdict
import numpy as np
sys.path.insert(0, str(Path.home()/"violin-rne/novelty/session_variance"))
import session_variance as S
for s, tags in S.SESSIONS.items():
    for t in tags:
        rs = S.rows_of(t)
        cl = defaultdict(set); colid = defaultdict(set)
        d = S.R.load(t)
        for r in rs:
            cl[round(r["lb"], 2)].add(int(d["column"][r["idx"][0]]))
        sizes = [len(v) for v in cl.values()]
        # how many columns (by column index) are split across >1 cluster
        inv = defaultdict(set)
        for k, v in cl.items():
            for c in v: inv[c].add(k)
        nsplit = sum(len(v) > 1 for v in inv.values())
        ncol = len({int(d['column'][r['idx'][0]]) for r in rs})
        print(f"{s:7s} {t[:12]:12s} rows {len(rs)} distinct beta-index {ncol} bootstrap clusters {len(cl)} split beta-index {nsplit} max columns/cluster {max(sizes)}")
