"""Score one candidate policy file. Prints one JSON line. Run as a subprocess.

Every case is replayed in its own forked child that imports the candidate fresh,
so a policy cannot carry labels from one case (or epsilon) into the next.
"""
import importlib.util
import json
import os
import pickle
import sys
import time

from threadpoolctl import threadpool_limits

from agentic_research.evolve_v4 import guard
from agentic_research.evolve_v4.task_a import load_cases, replay

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


def main(path, split, limit):
    if split == "holdout" and os.environ.get("EVOLVE_FINAL") != "1":
        raise SystemExit("holdout is sealed: only final_a.py scores it, once")
    guard.check(open(path).read())
    cases = load_cases(split)
    if limit:
        cases = cases[:limit]
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
    per_case = [dict(group=c["group"], epsilon=c["epsilon"], compatible=bool(c["expected"]),
                     queries=r["queries"], seconds=round(r["seconds"], 3))
                for c, r in zip(cases, results)]
    totals = {}
    for row in per_case:
        key = f"eps{row['epsilon']}"
        totals[key] = totals.get(key, 0) + row["queries"]
    print(json.dumps(dict(ok=True, split=split, total=sum(r["queries"] for r in per_case),
                          totals=totals, cases=len(per_case),
                          compatible_total=sum(r["queries"] for r in per_case if r["compatible"]),
                          compatible_cases=sum(r["compatible"] for r in per_case),
                          max_case_seconds=max(r["seconds"] for r in per_case),
                          seconds=time.perf_counter() - started, per_case=per_case)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "search",
         int(sys.argv[3]) if len(sys.argv) > 3 else 0)
