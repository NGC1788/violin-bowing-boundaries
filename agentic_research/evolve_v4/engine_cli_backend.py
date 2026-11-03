"""evolve_v4 engine: LLM code evolution with islands, a diagnosing scientist and a ledger.

Usage (from repo root):
  .venv-mac/bin/python -m agentic_research.evolve_v4.engine --campaign a1 --max-calls 40
Touch runs/<campaign>/STOP to stop after in-flight jobs. Only the search split is
ever evaluated here; the sealed holdout is scored separately by final_a.py.
"""
import argparse
import inspect
import json
import random
import re
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PYTHON = str(REPO / ".venv-mac/bin/python")
ISLANDS = 4
DIAGNOSE_EVERY = 12

TASK_A = """You are evolving a QUERY POLICY for an exact certification problem.

Setting. A folder has C columns (bow positions beta). Each column has 50 waveforms sorted by
measured bow force. Reading one waveform reveals its label under 9 threshold rules at once
(rules = (periodicity floor u, ratio ceiling v); label = fixed OR (guard AND periodicity>=u AND
ratio<v)). For each rule and column, the "boundary" is the first adjacent pair of positive labels
along force rank. A fixed, exact certificate (already implemented, you cannot change it) decides
whether SOME rule has boundaries whose intercept intervals [lower, upper] (one per possible start,
slope -2 law in log force, tolerance epsilon) share a common point across all columns. It stops as
soon as every completion of the unread labels gives the same answer. Unknown labels make several
first-run starts possible; view.possible marks them (last slot = "no run").

Your job: write choose(view) -> (column, rank) of an UNREAD waveform so the certificate stops after
as FEW reads as possible. Score = total reads over 54 cases (27 folders x eps 0.2/0.4), lower is
better. The current seed scores 9158. Hindsight analysis shows the certificates actually need only
about 6-25% of what the seed reads, so most reads are wasted: the gain is in predicting WHICH
waveforms matter (e.g. those that can kill the binding intercept endpoints of live rules, that
resolve the tightest columns, that exploit label monotonicity in force and in the rule thresholds,
or that use revealed periodicity/ratio values of neighbours to predict unread labels).

view fields (numpy copies): partial (C,9,50) int8 -1 unknown/0/1; possible (C,9,50) bool;
lower, upper (C,1,50) float, NaN = invalid start; force (C,50); log_beta (C,); rules (9,2);
epsilon; revealed: dict periodicity/ratio/fixed/guard (C,50) float, NaN where unread;
queries_so_far.
You may import only numpy, math, itertools, functools, heapq, collections and
agentic_research.evolve_v4.helpers, which provides: possible_starts(partial),
certificates(possible, lower, upper) -> (decision or None, resolved, geometric),
outer(possible, lower, upper) -> (left, right) per (column, rule),
response_patterns(rules) -> (10,9) feasible joint label vectors, shortlist(...),
counterfactual_choose(partial, possible, lower, upper, patterns, probabilities) (the seed), UNIFORM10.
No file, network, process or introspection access. Runtime per case must stay under ~3 s.
A wrong or repeated query is an error and scores nothing."""

PRUNED = {(r["group"], r["epsilon"]): r["pruned"]
          for r in json.loads((HERE / "prune_search.json").read_text())["per_case"]}


def library_source():
    from agentic_research.evolve_v4 import helpers
    parts = [inspect.getsource(f) for f in (helpers.possible_starts, helpers.certificates, helpers.outer,
                                             helpers.shortlist, helpers.counterfactual_choose,
                                             helpers.response_patterns)]
    return "\n".join(parts).replace("def choose(", "def counterfactual_choose(")


FORMAT = """Reply with exactly three parts:
IDEA: one sentence describing the mechanism you are testing.
NOVELTY: one sentence on how it differs from the parent and from standard
value-of-information / uncertainty sampling.
```python
<the complete new program file defining choose(view)>
```"""


class Ledger:
    def __init__(self, path):
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS programs(id INTEGER PRIMARY KEY, parent INT, island INT,
          code TEXT, idea TEXT, novelty TEXT, status TEXT, score REAL, metrics TEXT,
          created REAL, llm_in INT, llm_out INT, cost REAL, error TEXT);
        CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY, created REAL, text TEXT,
          llm_in INT, llm_out INT, cost REAL);""")

    def q(self, sql, args=()):
        with self.lock:
            cur = self.db.execute(sql, args)
            self.db.commit()
            return cur.fetchall(), cur.lastrowid


def call_claude(prompt, model, timeout=420):
    cmd = ["claude", "-p", "--model", model, "--output-format", "json", "--tools", "",
           "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}', "--no-session-persistence",
           "--setting-sources", "", "--permission-mode", "dontAsk", "--disable-slash-commands"]
    proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
    payload = json.loads(proc.stdout)
    if payload.get("is_error"):
        raise RuntimeError(str(payload.get("result"))[:300])
    usage = payload.get("usage", {})
    tokens_in = sum(usage.get(k, 0) for k in ("input_tokens", "cache_creation_input_tokens",
                                              "cache_read_input_tokens"))
    return payload["result"], tokens_in, usage.get("output_tokens", 0), payload.get("total_cost_usd", 0.0)


def evaluate(code, path, timeout=240):
    path.write_text(code)
    try:
        from agentic_research.evolve_v4 import guard
        guard.check(code)
        proc = subprocess.run([PYTHON, "-m", "agentic_research.evolve_v4.evaluate_a", str(path), "search"],
                              cwd=REPO, capture_output=True, text=True, timeout=timeout)
        if proc.returncode:
            return None, None, proc.stderr.strip().splitlines()[-1][:400] if proc.stderr else "failed"
        metrics = json.loads(proc.stdout.strip().splitlines()[-1])
        metrics["per_case"] = [[r["group"], r["epsilon"], r["queries"]] for r in metrics["per_case"]]
        return float(metrics["total"]), metrics, None
    except subprocess.TimeoutExpired:
        return None, None, f"timeout after {timeout}s"
    except Exception as exc:  # guard or parse failure
        return None, None, f"{type(exc).__name__}: {exc}"[:400]


class Engine:
    def __init__(self, campaign, model, diag_model):
        self.dir = HERE / "runs" / campaign
        (self.dir / "programs").mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger(self.dir / "ledger.sqlite")
        self.model, self.diag_model = model, diag_model
        self.log = open(self.dir / "engine.log", "a")
        if not self.ledger.q("SELECT COUNT(*) FROM programs")[0][0][0]:
            code = (HERE / "seed_a.py").read_text()
            score, metrics, error = evaluate(code, self.dir / "programs" / "seed.py")
            for island in range(ISLANDS):
                self.ledger.q("INSERT INTO programs(parent,island,code,idea,novelty,status,score,metrics,created)"
                              " VALUES(NULL,?,?,?,?,?,?,?,?)", (island, code, "seed: published counterfactual "
                              "lookahead", "", "ok", score, json.dumps(metrics), time.time()))
        self.since_diag = 0
        self.library = library_source()

    def say(self, text):
        line = time.strftime("%H:%M:%S ") + text
        print(line, flush=True)
        self.log.write(line + "\n")
        self.log.flush()

    def pick(self, island):
        rows = self.ledger.q("SELECT id,code,idea,score,metrics FROM programs WHERE island=? AND status='ok'"
                             " ORDER BY score LIMIT 8", (island,))[0]
        parent = min(random.sample(rows, min(3, len(rows))), key=lambda r: r[3])
        others = self.ledger.q("SELECT id,code,idea,score FROM programs WHERE island!=? AND status='ok'"
                               " ORDER BY score LIMIT 6", (island,))[0]
        return parent, random.sample(others, min(2, len(others)))

    def prompt(self, parent, inspirations):
        notes = self.ledger.q("SELECT text FROM notes ORDER BY id DESC LIMIT 1")[0]
        fails = self.ledger.q("SELECT idea,error FROM programs WHERE status!='ok' ORDER BY id DESC LIMIT 5")[0]
        best = self.ledger.q("SELECT score FROM programs WHERE status='ok' ORDER BY score LIMIT 1")[0][0][0]
        metrics = json.loads(parent[4])
        per_case = metrics.pop("per_case")
        gaps = sorted(per_case, key=lambda r: PRUNED[(r[0], r[1])] - r[2])[:8]
        parts = [TASK_A, "\nLIBRARY SOURCE (importable from agentic_research.evolve_v4.helpers):\n```python\n"
                 + self.library + "\n```",
                 f"\nBest score so far: {best:.0f}. Parent score: {parent[3]:.0f}. Parent metrics: {metrics}",
                 "\nWORST CASES FOR PARENT (folder, epsilon, parent reads, hindsight-pruned certificate size):\n"
                 + "\n".join(f"- {g} eps{e}: {q} reads vs {PRUNED[(g, e)]}" for g, e, q in gaps),
                 f"\nPARENT IDEA: {parent[2]}\nPARENT PROGRAM:\n```python\n"
                 f"{parent[1]}\n```"]
        for ins in inspirations:
            parts.append(f"\nINSPIRATION (score {ins[3]:.0f}): {ins[2]}\n```python\n{ins[1][:3000]}\n```")
        if notes:
            parts.append("\nSCIENTIST DIRECTIONS (pick one or combine):\n" + notes[0][0][:3000])
        if fails:
            parts.append("\nRECENT FAILURES TO AVOID:\n" + "\n".join(f"- {i}: {e}" for i, e in fails))
        parts.append("\n" + FORMAT)
        return "\n".join(parts)

    def mutate(self, island):
        parent, inspirations = self.pick(island)
        try:
            text, tin, tout, cost = call_claude(self.prompt(parent, inspirations), self.model)
        except Exception as exc:
            self.ledger.q("INSERT INTO programs(parent,island,status,error,created) VALUES(?,?,?,?,?)",
                          (parent[0], island, "llm_error", str(exc)[:400], time.time()))
            return "llm_error", None
        match = re.search(r"```python\n(.*?)```", text, re.S)
        idea = (re.search(r"IDEA:\s*(.*)", text) or [None, ""])[1].strip()[:400]
        novelty = (re.search(r"NOVELTY:\s*(.*)", text) or [None, ""])[1].strip()[:400]
        _, pid = self.ledger.q("INSERT INTO programs(parent,island,idea,novelty,status,created,llm_in,llm_out,cost)"
                               " VALUES(?,?,?,?,?,?,?,?,?)", (parent[0], island, idea, novelty, "evaluating",
                                                              time.time(), tin, tout, cost))
        if not match:
            self.ledger.q("UPDATE programs SET status='no_code', error='no python block' WHERE id=?", (pid,))
            return "no_code", pid
        code = match.group(1)
        score, metrics, error = evaluate(code, self.dir / "programs" / f"p{pid}.py")
        status = "ok" if error is None else "eval_error"
        self.ledger.q("UPDATE programs SET code=?,status=?,score=?,metrics=?,error=? WHERE id=?",
                      (code, status, score, json.dumps(metrics), error, pid))
        self.say(f"p{pid} island{island} parent{parent[0]} {status} score={score} tokens={tin}/{tout} "
                 f"| {idea[:120]}" + (f" | ERR {error[:120]}" if error else ""))
        return status, pid

    def diagnose(self):
        top = self.ledger.q("SELECT id,score,idea,novelty FROM programs WHERE status='ok' ORDER BY score LIMIT 10")[0]
        recent = self.ledger.q("SELECT id,status,score,idea,error FROM programs ORDER BY id DESC LIMIT 15")[0]
        best = self.ledger.q("SELECT code FROM programs WHERE status='ok' ORDER BY score LIMIT 1")[0][0][0]
        prompt = (TASK_A + "\n\nYou are the SCIENTIST overseeing an evolutionary search. Diagnose why the"
                  " search is or is not improving and propose 3 concrete, testable mechanisms not yet tried."
                  " For each: the mechanism, why it should cut reads given the certificate's structure, the"
                  " closest known method (e.g. EC2, value of information, adaptive submodularity, branch and"
                  " bound) and what would make it genuinely different. Be specific and terse.\n\nTOP:\n"
                  + "\n".join(f"p{i} {s:.0f} | {d} | {n}" for i, s, d, n in top)
                  + "\n\nRECENT:\n" + "\n".join(f"p{i} {st} {s} | {d} | {e}" for i, st, s, d, e in recent)
                  + f"\n\nBEST PROGRAM:\n```python\n{best}\n```\n\nLIBRARY:\n```python\n{self.library}\n```")
        try:
            text, tin, tout, cost = call_claude(prompt, self.diag_model, timeout=600)
            self.ledger.q("INSERT INTO notes(created,text,llm_in,llm_out,cost) VALUES(?,?,?,?,?)",
                          (time.time(), text, tin, tout, cost))
            self.say(f"SCIENTIST note tokens={tin}/{tout}: {text[:200]!r}")
        except Exception as exc:
            self.say(f"SCIENTIST failed: {exc}")

    def run(self, max_calls, workers):
        calls = 0
        with ThreadPoolExecutor(workers) as pool:
            futures = set()
            while calls < max_calls or futures:
                stop = (self.dir / "STOP").exists()
                while not stop and calls < max_calls and len(futures) < workers:
                    futures.add(pool.submit(self.mutate, calls % ISLANDS))
                    calls += 1
                if not futures:
                    break
                done = next(as_completed(futures))
                futures.discard(done)
                self.since_diag += 1
                if self.since_diag >= DIAGNOSE_EVERY and not stop:
                    self.since_diag = 0
                    self.diagnose()
                    calls += 1
        best = self.ledger.q("SELECT id,score,idea FROM programs WHERE status='ok' ORDER BY score LIMIT 3")[0]
        self.say(f"DONE calls={calls} best={best}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--max-calls", type=int, default=30)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--diag-model", default="opus")
    a = ap.parse_args()
    Engine(a.campaign, a.model, a.diag_model).run(a.max_calls, a.workers)
    sys.exit(0)
