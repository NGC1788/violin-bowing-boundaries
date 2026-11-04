"""Score violin-evolved policies, unchanged, on the robustness-audit transfer cases.

  .venv-mac/bin/python -m agentic_research.evolve_v4.transfer_robust.score [search] [policy.py ...]

Same isolation as evaluate_a: every case runs in a forked child that imports the policy fresh.
Default policies: the violin seed, the distilled principle, the Mac record holders and the three
literature baselines. The holdout split is sealed (EVOLVE_FINAL=1, once).
"""
import json
import os
import sys
from pathlib import Path

from threadpoolctl import threadpool_limits

from agentic_research.evolve_v4 import guard
from agentic_research.evolve_v4.evaluate_a import collect, run_child
from agentic_research.evolve_v4.transfer_robust.build import load_cases

EVOLVE = Path(__file__).resolve().parents[1]
POP = EVOLVE / "colab" / "bundle" / "population"
DEFAULT = [EVOLVE / "seed_a.py", EVOLVE / "distilled_a.py", POP / "p10_t6.py", POP / "p12_t9.py",
           POP / "p25_t21.py", EVOLVE / "baselines" / "adaptive_greedy.py",
           EVOLVE / "baselines" / "proof_number.py", EVOLVE / "baselines" / "direct_ec.py"]
PARALLEL = int(os.environ.get("EVOLVE_PARALLEL", "8"))


def score(path, cases):
    guard.check(Path(path).read_text())
    results, running, queue = [None] * len(cases), {}, list(enumerate(cases))
    with threadpool_limits(1):
        while queue or running:
            while queue and len(running) < PARALLEL:
                index, case = queue.pop(0)
                read_fd, write_fd = os.pipe()
                pid = os.fork()
                if pid == 0:
                    os.close(read_fd)
                    run_child(str(path), case, write_fd)
                os.close(write_fd)
                running[pid] = (index, read_fd)
            pid, (index, read_fd) = next(iter(running.items()))
            results[index] = collect(pid, read_fd)
            del running[pid]
    errors = [r["error"] for r in results if "error" in r]
    if errors:
        return dict(policy=Path(path).stem, error=errors[0], failed=len(errors))
    totals = {}
    for case, r in zip(cases, results):
        totals[case["epsilon"]] = totals.get(case["epsilon"], 0) + r["queries"]
    return dict(policy=Path(path).stem, total=sum(totals.values()), totals=totals,
                per_case=[r["queries"] for r in results])


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].endswith(".py") else "search"
    paths = [a for a in sys.argv[1:] if a.endswith(".py")] or DEFAULT
    cases = load_cases(split)
    everything = sum(c["labels"].shape[0] * c["labels"].shape[2] for c in cases)
    print(json.dumps(dict(split=split, cases=len(cases), read_everything=everything)))
    for path in paths:
        print(json.dumps({k: v for k, v in score(path, cases).items() if k != "per_case"}), flush=True)
