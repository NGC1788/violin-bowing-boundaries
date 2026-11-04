"""evolve_v4 Colab engine: island code evolution driven by an open-weight model on localhost vLLM.

Everything runs inside one Colab runtime: the model is served on localhost only, candidates
are scored by colab_eval.py on the shipped SEARCH folders, and the ledger lives in run_dir
(local disk, copied to --backup-dir on Google Drive every 5 min). The sealed C/D holdout is not in the bundle; it is scored on the Mac later.

  python -m agentic_research.evolve_v4.colab_engine --run-dir DIR --model openai/gpt-oss-120b \
      --hours 10 --seed 1
Touch DIR/STOP to finish after the jobs in flight.
"""
import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[2]
DATA = BUNDLE / "cases"
PRUNED = {(r["group"], r["epsilon"]): r["pruned"]
          for r in json.loads((DATA / "prune_search.json").read_text())["per_case"]}
ISLANDS = 4
LIBRARY = (BUNDLE / "agentic_research/evolve_v4/core.py").read_text()

TASK = """THE PROBLEM. You are evolving a QUERY POLICY for an exact certification problem.

A folder has C columns (bow positions beta). Each column has 50 waveforms sorted by measured bow
force. Reading one waveform reveals its label under 9 threshold rules at once (rules = (periodicity
floor u, ratio ceiling v); label = fixed OR (guard AND periodicity>=u AND ratio<v)). For each rule and
column the "boundary" is the first adjacent pair of positive labels along force rank. A fixed exact
certificate (you cannot change it) asks whether SOME rule has boundaries whose intercept intervals
[lower, upper] (slope -2 law in log force, tolerance epsilon; one interval per possible start) share a
common point across all columns. It stops as soon as every completion of the unread labels gives the
same answer. Unknown labels make several first-run starts possible; possible marks them (last slot =
"no run in this column").

choose(view) must return (column, rank) of an UNREAD waveform so the certificate stops after as FEW
reads as possible. Score = total reads over 54 search cases (27 folders x epsilon 0.2/0.4), lower is
better. The published one-step counterfactual lookahead scores 9158. Hindsight pruning of its reads
shows certificates exist with about 1554 reads in total, so there is still a lot of room.
50 of 54 cases are INCOMPATIBLE (answer "no rule fits"): proving that needs, for every one of the 9
rules, either a column with no valid start or two columns whose intervals cannot overlap. Labels are
monotone in the thresholds (positive for a strict rule => positive for every looser rule; zero for a
loose rule => zero for every stricter rule), and neighbouring ranks are correlated.

What is known so far (ablation of the best programs): the gain comes from COMMITTING to a separating
column pair per rule and VERIFYING it (zero-sweep of the high column's early starts + one forced
positive pair in the low column), with bisection-style probing of where labels flip. Replacing
bisection by a linear scan more than doubles the reads; dropping the verification phase is worse than
the published baseline.

view fields (numpy copies): partial (C,9,50) int8 -1 unknown/0/1; possible (C,9,51) bool;
lower, upper (C,1,50) float, NaN = invalid start; force (C,50); log_beta (C,); rules (9,2);
epsilon; queries_so_far. (LABELS ONLY: view.revealed exists but is forbidden; the guard rejects it.)
Each case runs in a fresh process: no memory between cases.
Allowed imports: numpy, math, itertools, functools, heapq, collections,
agentic_research.evolve_v4.helpers (exports possible_starts, certificates, outer, shortlist,
counterfactual_choose, response_patterns, UNIFORM10; source below). No file, network, process or
introspection access (an AST guard rejects open/exec/eval/getattr/__dict__/np.load etc.). Keep each
case under ~3 s. A wrong or repeated query is an error and scores nothing. Do not hard-code folder
names or per-case constants: the program must generalise to unseen folders."""

FORMAT = """Reply with exactly three parts and nothing else:
IDEA: one sentence describing the mechanism you are testing.
NOVELTY: one sentence on how it differs from the parent and from value-of-information /
uncertainty sampling / greedy information gain.
```python
<the complete new program file defining choose(view)>
```"""

DIRECTIONS = [
    "Joint multi-rule witness planning: exploit nested rules so one zero-sweep or one forced pair serves "
    "several rules; choose one shared set of witness pairs at minimum total predicted reads.",
    "Epsilon-aware witness choice: at eps 0.4 intervals are 0.8 wider; only pursue pairs whose best-case "
    "separation exceeds the widening, abandon a pair as soon as it cannot separate, prefer the "
    "no-valid-start route when cheaper.",
    "Better pass certificates for the 4 compatible cases: pick the most plausible rule and common "
    "intercept window early and pin every column's first run inside it by bisection.",
    "Robustness: find the cases where the parent is much worse than the hindsight minimum and add a cheap "
    "guard (re-plan, switch witness, or fall back) without hurting the other cases.",
    "Exploit label monotonicity across rules to infer unread labels exactly (implied zeros/ones) and "
    "never spend a read whose outcome is already implied for every live rule.",
    "Order reads fail-first: read the element of the chosen proof most likely to break it first, so a "
    "wrong commitment is discovered after one or two reads.",
]


def post_chat(base_url, model, prompt, max_tokens, effort, timeout=1800):
    body = dict(model=model, messages=[{"role": "user", "content": prompt}],
                temperature=1.0, top_p=1.0, max_tokens=max_tokens)
    if "gpt-oss" in model:
        body["reasoning_effort"] = effort
    request = urllib.request.Request(base_url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    message = payload["choices"][0]["message"]
    usage = payload.get("usage") or {}
    return message.get("content") or "", usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


def extract(text):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    code = max(blocks, key=len) if blocks else None
    idea = (re.search(r"IDEA:\s*(.*)", text) or [None, ""])[1].strip()[:400]
    novelty = (re.search(r"NOVELTY:\s*(.*)", text) or [None, ""])[1].strip()[:400]
    if code and not code.startswith("# IDEA"):
        code = f"# IDEA: {idea}\n# NOVELTY: {novelty}\n" + code
    return code, idea, novelty


def evaluate(code, path, timeout=300):
    path.write_text(code)
    try:
        from agentic_research.evolve_v4 import guard
        guard.check(code)
        proc = subprocess.run([sys.executable, "-m", "agentic_research.evolve_v4.colab_eval", str(path)],
                              cwd=BUNDLE, capture_output=True, text=True, timeout=timeout,
                              env={**os.environ, "EVOLVE_PARALLEL": os.environ.get("EVOLVE_PARALLEL", "2")})
        if proc.returncode:
            return None, None, (proc.stderr or proc.stdout).strip().splitlines()[-1][:400]
        metrics = json.loads(proc.stdout.strip().splitlines()[-1])
        return float(metrics["total"]), metrics, None
    except subprocess.TimeoutExpired:
        return None, None, f"timeout after {timeout}s"
    except Exception as exc:  # guard or parse failure
        return None, None, f"{type(exc).__name__}: {exc}"[:400]


class Engine:
    def __init__(self, a):
        self.a = a
        self.dir = Path(a.run_dir)
        (self.dir / "programs").mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(self.dir / "ledger.sqlite", check_same_thread=False)
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS programs(id INTEGER PRIMARY KEY, parent INT, island INT, code TEXT,
          code_hash TEXT, idea TEXT, novelty TEXT, status TEXT, score REAL, totals TEXT, per_case TEXT,
          created REAL, model TEXT, prompt_tokens INT, completion_tokens INT, seconds REAL, error TEXT,
          response TEXT, direction TEXT);
        CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY, created REAL, text TEXT);""")
        self.eval_slots = threading.Semaphore(max(1, (os.cpu_count() or 2) // 2))
        self.rng = random.Random(a.seed)
        self.log = open(self.dir / "engine.log", "a")
        if not self.q("SELECT COUNT(*) FROM programs")[0][0]:
            self.load_population()

    def q(self, sql, args=()):
        with self.lock:
            cur = self.db.execute(sql, args)
            self.db.commit()
            return cur.fetchall() if sql.lstrip().upper().startswith("SELECT") else cur.lastrowid

    def say(self, text):
        line = time.strftime("%m-%d %H:%M:%S ") + text
        print(line, flush=True)
        self.log.write(line + "\n")
        self.log.flush()

    def load_population(self):
        files = sorted((BUNDLE / "population").glob("*.py"))
        for k, path in enumerate(files):
            code = path.read_text()
            score, metrics, error = evaluate(code, self.dir / "programs" / f"init_{path.stem}.py")
            idea = (re.search(r"#\s*IDEA:\s*(.*)", code) or [None, path.stem])[1].strip()[:400]
            islands = range(ISLANDS) if path.stem.startswith("seed") else [k % ISLANDS]
            for island in islands:
                self.q("INSERT INTO programs(parent,island,code,code_hash,idea,status,score,totals,per_case,created,"
                       "model,error) VALUES(NULL,?,?,?,?,?,?,?,?,?,?,?)",
                       (island, code, hashlib.sha256(code.encode()).hexdigest(), idea,
                        "ok" if error is None else "eval_error", score, json.dumps(metrics and metrics["totals"]),
                        json.dumps(metrics and metrics["per_case"]), time.time(), "init:" + path.stem, error))
            self.say(f"init {path.stem}: score={score} {error or ''}")

    def prompt(self, island, direction):
        rows = self.q("SELECT id,code,idea,score,per_case,totals FROM programs WHERE island=? AND status='ok'"
                      " ORDER BY score LIMIT 8", (island,))
        if self.rng.random() < .15:
            parent = self.rng.choice(rows)
        else:
            parent = min(self.rng.sample(rows, min(3, len(rows))), key=lambda r: r[3])
        others = self.q("SELECT id,code,idea,score FROM programs WHERE status='ok' AND id!=? ORDER BY score LIMIT 10",
                        (parent[0],))
        inspirations = self.rng.sample(others, min(2, len(others)))
        note = self.q("SELECT text FROM notes ORDER BY id DESC LIMIT 1")
        tried = self.q("SELECT score,idea FROM programs WHERE model NOT LIKE 'init:%' AND idea!='' "
                       "ORDER BY id DESC LIMIT 20")
        fails = self.q("SELECT idea,error FROM programs WHERE status IN ('eval_error','no_code') "
                       "ORDER BY id DESC LIMIT 5")
        best = self.q("SELECT MIN(score) FROM programs WHERE status='ok'")[0][0]
        gaps = sorted(json.loads(parent[4]), key=lambda r: PRUNED[(r[0], r[1])] - r[2])[:6]
        parts = [TASK, "\n## Library source (agentic_research.evolve_v4.helpers re-exports these)\n```python\n"
                 + LIBRARY + "\n```",
                 f"\n## Parent p{parent[0]} (score {parent[3]:.0f}; best in population {best:.0f}; "
                 f"totals {parent[5]})\nIdea: {parent[2]}\n```python\n{parent[1]}\n```",
                 "\nParent's most wasted cases (folder, epsilon: parent reads vs hindsight minimum):\n"
                 + "\n".join(f"- {g} eps{e}: {r} vs {PRUNED[(g, e)]}" for g, e, r in gaps)]
        for ins in inspirations:
            parts.append(f"\n## Inspiration p{ins[0]} (score {ins[3]:.0f})\nIdea: {ins[2]}\n```python\n"
                         f"{ins[1][:7000]}\n```")
        parts.append(f"\n## Direction for this attempt\n{direction}")
        if note:
            parts.append("\n## Scientist note\n" + note[0][0][:4000])
        if tried:
            parts.append("\n## Recent attempts (score, idea) - do not repeat\n"
                         + "\n".join(f"- {s if s is None else round(s)}: {i[:160]}" for s, i in tried))
        if fails:
            parts.append("\n## Recent failures\n" + "\n".join(f"- {i[:120]}: {e}" for i, e in fails))
        parts.append("\nWrite a program that scores clearly LOWER than the parent by a real mechanism, not a "
                     "constant tweak.\n" + FORMAT)
        return parent[0], "\n".join(parts)

    def job(self, number):
        island = number % ISLANDS
        direction = DIRECTIONS[(number // ISLANDS + self.a.seed) % len(DIRECTIONS)]
        parent, prompt = self.prompt(island, direction)
        started = time.time()
        try:
            text, tin, tout = post_chat(self.a.base_url, self.a.model, prompt, self.a.max_tokens, self.a.effort)
        except Exception as exc:
            self.q("INSERT INTO programs(parent,island,status,error,created,model,direction) VALUES(?,?,?,?,?,?,?)",
                   (parent, island, "llm_error", str(exc)[:400], time.time(), self.a.model, direction))
            return "llm_error"
        code, idea, novelty = extract(text)
        row = dict(parent=parent, island=island, idea=idea, novelty=novelty, model=self.a.model,
                   prompt_tokens=tin, completion_tokens=tout, response=text[-20000:], direction=direction)
        if not code:
            status, score, metrics, error, digest = "no_code", None, None, "no python block", None
        else:
            digest = hashlib.sha256(code.encode()).hexdigest()
            if self.q("SELECT id FROM programs WHERE code_hash=?", (digest,)):
                status, score, metrics, error = "duplicate", None, None, None
            else:
                with self.eval_slots:
                    score, metrics, error = evaluate(code, self.dir / "programs" / f"job{number}.py")
                status = "ok" if error is None else "eval_error"
        self.q("INSERT INTO programs(parent,island,code,code_hash,idea,novelty,status,score,totals,per_case,created,"
               "model,prompt_tokens,completion_tokens,seconds,error,response,direction)"
               " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
               (parent, island, code, digest, idea, novelty, status, score, json.dumps(metrics and metrics["totals"]),
                json.dumps(metrics and metrics["per_case"]), time.time(), self.a.model, tin, tout,
                time.time() - started, error, row["response"], direction))
        self.say(f"job{number} island{island} parent p{parent} {status} score={score} tokens={tin}/{tout} "
                 f"| {idea[:110]}" + (f" | ERR {error[:100]}" if error else ""))
        return status

    def diagnose(self):
        top = self.q("SELECT id,score,idea FROM programs WHERE status='ok' ORDER BY score LIMIT 8")
        recent = self.q("SELECT id,status,score,idea,error FROM programs ORDER BY id DESC LIMIT 25")
        best = self.q("SELECT code FROM programs WHERE status='ok' ORDER BY score LIMIT 1")[0][0]
        prompt = (TASK + "\n\nYou are the SCIENTIST overseeing this evolutionary search. Diagnose why it is or is "
                  "not improving, and propose 3 concrete mechanisms that have NOT been tried, each with why it "
                  "should cut reads given the certificate's structure. Be terse (under 350 words).\n\nTOP:\n"
                  + "\n".join(f"p{i} {s:.0f} | {d[:200]}" for i, s, d in top) + "\n\nRECENT:\n"
                  + "\n".join(f"p{i} {st} {s} | {(d or '')[:150]} | {(e or '')[:80]}" for i, st, s, d, e in recent)
                  + f"\n\nBEST PROGRAM:\n```python\n{best}\n```")
        try:
            text, _, _ = post_chat(self.a.base_url, self.a.model, prompt, self.a.max_tokens, "high")
            self.q("INSERT INTO notes(created,text) VALUES(?,?)", (time.time(), text))
            self.say("SCIENTIST: " + text[:300].replace("\n", " "))
        except Exception as exc:
            self.say(f"SCIENTIST failed: {exc}")

    def migrate(self):
        for island in range(ISLANDS):
            best = self.q("SELECT id,code,code_hash,idea,score,totals,per_case FROM programs WHERE island=? AND "
                          "status='ok' ORDER BY score LIMIT 1", (island,))
            target = (island + 1) % ISLANDS
            if best and not self.q("SELECT id FROM programs WHERE island=? AND code_hash=?", (target, best[0][2])):
                b = best[0]
                self.q("INSERT INTO programs(parent,island,code,code_hash,idea,status,score,totals,per_case,created,"
                       "model) VALUES(?,?,?,?,?,?,?,?,?,?,?)", (b[0], target, b[1], b[2], b[3], "ok", b[4], b[5], b[6],
                                                               time.time(), "migrant"))

    def backup(self):
        if not self.a.backup_dir:
            return
        target = Path(self.a.backup_dir)
        target.mkdir(parents=True, exist_ok=True)
        with self.lock:
            out = sqlite3.connect(target / "ledger.tmp")
            self.db.backup(out)
            out.close()
        (target / "ledger.tmp").replace(target / "ledger.sqlite")
        shutil.copy(self.dir / "engine.log", target / "engine.log")

    def run(self):
        last_backup = time.time()
        deadline = time.time() + self.a.hours * 3600
        start_count = self.q("SELECT COUNT(*) FROM programs")[0][0]
        number, done, inflight = start_count, 0, set()
        next_diagnose, next_migrate, scientist = self.a.diagnose_every, self.a.migrate_every, None
        with ThreadPoolExecutor(self.a.concurrency) as pool, ThreadPoolExecutor(1) as side:
            while True:
                stop = any((Path(d) / "STOP").exists() for d in (self.dir, self.a.backup_dir) if d) \
                    or time.time() > deadline
                if time.time() - last_backup > 300:
                    self.backup()
                    last_backup = time.time()
                while not stop and len(inflight) < self.a.concurrency:
                    inflight.add(pool.submit(self.job, number))
                    number += 1
                if not inflight:
                    break
                finished, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                done += len(finished)
                if not stop and done >= next_diagnose and (scientist is None or scientist.done()):
                    next_diagnose = done + self.a.diagnose_every
                    scientist = side.submit(self.diagnose)
                if done >= next_migrate:
                    next_migrate = done + self.a.migrate_every
                    self.migrate()
        best = self.q("SELECT id,score,idea FROM programs WHERE status='ok' ORDER BY score LIMIT 5")
        self.backup()
        self.say(f"DONE jobs={done} best={[(i, s) for i, s, _ in best]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--backup-dir", default="")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--hours", type=float, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=24000)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--diagnose-every", type=int, default=40)
    ap.add_argument("--migrate-every", type=int, default=60)
    Engine(ap.parse_args()).run()
