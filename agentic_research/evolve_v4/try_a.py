"""Brief search-split score for a worker: totals plus the cases with most wasted reads.

  .venv-mac/bin/python -m agentic_research.evolve_v4.try_a path/to/candidate.py
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PRUNED = {(r["group"], r["epsilon"]): r["pruned"]
          for r in json.loads((HERE / "prune_search.json").read_text())["per_case"]}
SEED = {(r["group"], r["epsilon"]): r["seed"]
        for r in json.loads((HERE / "prune_search.json").read_text())["per_case"]}

if __name__ == "__main__":
    proc = subprocess.run([str(REPO / ".venv-mac/bin/python"), "-m", "agentic_research.evolve_v4.evaluate_a",
                           sys.argv[1], "search"], cwd=REPO, capture_output=True, text=True, timeout=300)
    if proc.returncode:
        print("FAILED:", (proc.stderr or proc.stdout).strip()[-1500:])
        sys.exit(1)
    m = json.loads(proc.stdout.strip().splitlines()[-1])
    print(f"total={m['total']} (seed 9158) eps0.2={m['totals']['eps0.2']} (seed 3858) "
          f"eps0.4={m['totals']['eps0.4']} (seed 5300) compatible={m['compatible_total']} (seed 1985) "
          f"max_case_seconds={m['max_case_seconds']:.2f} wall={m['seconds']:.1f}s")
    rows = sorted(m["per_case"], key=lambda r: PRUNED[(r["group"], r["epsilon"])] - r["queries"])[:8]
    print("most wasted (reads / seed reads / hindsight minimum):")
    for r in rows:
        key = (r["group"], r["epsilon"])
        print(f"  {r['group']} eps{r['epsilon']} compatible={r['compatible']}: "
              f"{r['queries']} / {SEED[key]} / {PRUNED[key]}")
