"""Score one candidate policy on the shipped search cases. Prints one JSON line.

Same protocol as evaluate_a.py: every case runs in its own forked child that imports
the candidate fresh, and the frozen certificate must reach the full-data decision.
"""
import importlib.util
import json
import os
import pickle
import sys
import time

from threadpoolctl import threadpool_limits

from agentic_research.evolve_v4 import guard
from agentic_research.evolve_v4.colab_task import load_cases
from agentic_research.evolve_v4.core import replay

PARALLEL = int(os.environ.get("EVOLVE_PARALLEL", "4"))


def run_child(path, case, write_fd):
    try:
        spec = importlib.util.spec_from_file_location("candidate", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        started = time.perf_counter()
        result = dict(queries=replay(case, module.choose), seconds=time.perf_counter() - started)
    except BaseException as exc:  # report every failure, including assertion errors
        result = dict(error=f"{type(exc).__name__}: {exc}"[:400])
    os.write(write_fd, pickle.dumps(result))
    os.close(write_fd)
    os._exit(0)


def collect(pid, read_fd):
    chunks = []
    while chunk := os.read(read_fd, 65536):
        chunks.append(chunk)
    os.close(read_fd)
    os.waitpid(pid, 0)
    return pickle.loads(b"".join(chunks)) if chunks else dict(error="child died")


def main(path):
    guard.check(open(path).read())
    cases = load_cases()
    started = time.perf_counter()
    results, running, queue = [None] * len(cases), {}, list(enumerate(cases))
    with threadpool_limits(1):
        while queue or running:
            while queue and len(running) < PARALLEL:
                index, case = queue.pop(0)
                read_fd, write_fd = os.pipe()
                pid = os.fork()
                if pid == 0:
                    os.close(read_fd)
                    run_child(path, case, write_fd)
                os.close(write_fd)
                running[index] = (pid, read_fd)
            index = next(iter(running))
            results[index] = collect(*running.pop(index))
            if "error" in results[index]:
                for pid, fd in running.values():
                    os.kill(pid, 9)
                    collect(pid, fd)
                raise RuntimeError(f"case {index}: {results[index]['error']}")
    per_case = [[c["group"], c["epsilon"], r["queries"]] for c, r in zip(cases, results)]
    totals = {}
    for _, epsilon, queries in per_case:
        totals[f"eps{epsilon}"] = totals.get(f"eps{epsilon}", 0) + queries
    print(json.dumps(dict(ok=True, split="search", total=sum(q for _, _, q in per_case), totals=totals,
                          compatible_total=sum(r["queries"] for c, r in zip(cases, results) if c["expected"]),
                          max_case_seconds=max(r["seconds"] for r in results),
                          seconds=time.perf_counter() - started, per_case=per_case)))


if __name__ == "__main__":
    main(sys.argv[1])
