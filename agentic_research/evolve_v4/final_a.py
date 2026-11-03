"""Score the pre-registered finalists ONCE on the sealed holdout (C/D folders).

  .venv-mac/bin/python -m agentic_research.evolve_v4.final_a --campaign a1 [--extra p12 p30]
Finalist = best search score in the ledger (fixed rule, decided before any holdout look),
plus any explicitly listed extras, plus the seed. Refuses to run twice for a campaign.
"""
import argparse
import json
import os
import sqlite3
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    folder = HERE / "runs" / a.campaign
    out = folder / "final_holdout.json"
    if out.exists():
        raise SystemExit(f"holdout already scored for {a.campaign}: {out}")
    db = sqlite3.connect(folder / "ledger.sqlite")
    best = db.execute("SELECT id FROM programs WHERE status='ok' AND task IS NOT NULL ORDER BY score, id LIMIT 1").fetchone()
    ids = [best[0]] + [int(x.lstrip("p")) for x in a.extra]
    rows = []
    for pid in [None] + ids:
        if pid is None:
            path, search, idea = HERE / "seed_a.py", 9158, "seed"
        else:
            code, search, idea, task = db.execute("SELECT code,score,idea,task FROM programs WHERE id=?", (pid,)).fetchone()
            path = folder / "programs" / f"final_p{pid}.py"
            path.write_text(code)
        proc = subprocess.run([str(REPO / ".venv-mac/bin/python"), "-m", "agentic_research.evolve_v4.evaluate_a",
                               str(path), "holdout"], cwd=REPO, capture_output=True, text=True, timeout=900,
                              env={**os.environ, "EVOLVE_FINAL": "1"})
        metrics = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.returncode == 0 else None
        rows.append(dict(program=pid, idea=idea, search_total=search,
                         holdout=metrics, error=None if metrics else proc.stderr[-800:]))
        print(pid, idea[:80], search, metrics and (metrics["total"], metrics["totals"], metrics["compatible_total"]))
    out.write_text(json.dumps(dict(created=time.time(), rows=rows), indent=1))
