"""evolve_v4 engine: island-based LLM code evolution with a scientist note and a ledger.

Workers are LLM agents. Each gets one brief (task, library source, parent program,
inspirations from other islands, scientist directions, recent failures), may score
its drafts on the SEARCH split with try_a.py, and leaves one program in the inbox.
The engine re-scores every inbox program itself and records everything.
(engine_cli_backend.py is the same loop driven by headless `claude -p` calls.)

  python -m agentic_research.evolve_v4.engine dispatch --campaign a1 --n 4 [--direction "..."]
  python -m agentic_research.evolve_v4.engine ingest   --campaign a1
  python -m agentic_research.evolve_v4.engine note     --campaign a1 --file note.md
  python -m agentic_research.evolve_v4.engine status   --campaign a1
The sealed holdout is never touched here (final_a.py scores it once).
"""
import argparse
import inspect
import json
import random
import re
import sqlite3
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PYTHON = str(REPO / ".venv-mac/bin/python")
ISLANDS = 4
PRUNED = {(r["group"], r["epsilon"]): r["pruned"]
          for r in json.loads((HERE / "prune_search.json").read_text())["per_case"]}

TASK_A = """THE PROBLEM. You are evolving a QUERY POLICY for an exact certification problem.

A folder has C columns (bow positions beta). Each column has 50 waveforms sorted by measured bow
force. Reading one waveform reveals its label under 9 threshold rules at once (rules = (periodicity
floor u, ratio ceiling v); label = fixed OR (guard AND periodicity>=u AND ratio<v)), plus its raw
periodicity / ratio / fixed / guard values. For each rule and column the "boundary" is the first
adjacent pair of positive labels along force rank. A fixed exact certificate (you cannot change it)
asks whether SOME rule has boundaries whose intercept intervals [lower, upper] (slope -2 law in log
force, tolerance epsilon; one interval per possible start) share a common point across all columns.
It stops as soon as every completion of the unread labels gives the same answer. Unknown labels make
several first-run starts possible; possible marks them (last slot = "no run in this column").

choose(view) must return (column, rank) of an UNREAD waveform so the certificate stops after as FEW
reads as possible. Score = total reads over 54 search cases (27 folders x epsilon 0.2/0.4), lower is
better. The seed (published counterfactual one-step lookahead) scores 9158. Hindsight pruning shows
the certificates need only about 6-25% of what the seed reads (e.g. 735 reads where 26 suffice), so
most reads are wasted. 50 of 54 cases are INCOMPATIBLE (the answer is "no rule fits"): proving that
needs, for every one of the 9 rules, either a column with no valid start or two columns whose
intervals cannot overlap. Labels are monotone in the rule thresholds (a waveform positive for a
strict rule is positive for every looser rule), and neighbouring ranks are correlated.

view fields (numpy copies): partial (C,9,50) int8 -1 unknown/0/1; possible (C,9,51) bool;
lower, upper (C,1,50) float, NaN = invalid start; force (C,50); log_beta (C,); rules (9,2);
epsilon; revealed: dict periodicity/ratio/fixed/guard (C,50) float, NaN where unread;
queries_so_far. Each case runs in a fresh process: no memory between cases.
Allowed imports: numpy, math, itertools, functools, heapq, collections,
agentic_research.evolve_v4.helpers (source below). No file, network, process or introspection
access (an AST guard rejects open/exec/eval/getattr/__dict__/np.load etc.). Keep each case under
~3 s. A wrong or repeated query is an error."""


def library_source():
    from agentic_research.evolve_v4 import helpers
    parts = [inspect.getsource(f) for f in (helpers.possible_starts, helpers.certificates, helpers.outer,
                                             helpers.shortlist, helpers.counterfactual_choose,
                                             helpers.response_patterns)]
    return "\n".join(parts).replace("def choose(", "def counterfactual_choose(") + "\nUNIFORM10 = np.full(10, .1)\n"


def connect(campaign):
    folder = HERE / "runs" / campaign
    for sub in ("programs", "tasks", "work", "inbox"):
        (folder / sub).mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(folder / "ledger.sqlite")
    db.executescript("""
    CREATE TABLE IF NOT EXISTS programs(id INTEGER PRIMARY KEY, task INT, parent INT, island INT,
      code TEXT, idea TEXT, novelty TEXT, status TEXT, score REAL, metrics TEXT, created REAL, error TEXT);
    CREATE TABLE IF NOT EXISTS tasks(id INTEGER PRIMARY KEY, island INT, parent INT, direction TEXT,
      created REAL, ingested INT DEFAULT 0);
    CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY, created REAL, text TEXT);""")
    if not db.execute("SELECT COUNT(*) FROM programs").fetchone()[0]:
        code = (HERE / "seed_a.py").read_text()
        score, metrics, error = evaluate(code, folder / "programs" / "seed.py")
        for island in range(ISLANDS):
            db.execute("INSERT INTO programs(task,parent,island,code,idea,novelty,status,score,metrics,created)"
                       " VALUES(NULL,NULL,?,?,?,?,?,?,?,?)", (island, code, "seed: published counterfactual"
                       " one-step lookahead", "", "ok", score, json.dumps(metrics), time.time()))
        db.commit()
    return folder, db


def evaluate(code, path, timeout=400):
    path.write_text(code)
    try:
        from agentic_research.evolve_v4 import guard
        guard.check(code)
        proc = subprocess.run([PYTHON, "-m", "agentic_research.evolve_v4.evaluate_a", str(path), "search"],
                              cwd=REPO, capture_output=True, text=True, timeout=timeout)
        if proc.returncode:
            return None, None, (proc.stderr or proc.stdout).strip().splitlines()[-1][:400]
        metrics = json.loads(proc.stdout.strip().splitlines()[-1])
        metrics["per_case"] = [[r["group"], r["epsilon"], r["queries"]] for r in metrics["per_case"]]
        return float(metrics["total"]), metrics, None
    except subprocess.TimeoutExpired:
        return None, None, f"timeout after {timeout}s"
    except Exception as exc:  # guard or parse failure
        return None, None, f"{type(exc).__name__}: {exc}"[:400]


def brief(db, folder, task_id, island, direction):
    rows = db.execute("SELECT id,code,idea,score,metrics FROM programs WHERE island=? AND status='ok'"
                      " ORDER BY score LIMIT 8", (island,)).fetchall()
    parent = min(random.sample(rows, min(3, len(rows))), key=lambda r: r[3])
    others = db.execute("SELECT id,code,idea,score FROM programs WHERE island!=? AND status='ok'"
                        " AND task IS NOT NULL ORDER BY score LIMIT 6", (island,)).fetchall()
    inspirations = random.sample(others, min(2, len(others)))
    note = db.execute("SELECT text FROM notes ORDER BY id DESC LIMIT 1").fetchone()
    fails = db.execute("SELECT idea,error FROM programs WHERE status!='ok' ORDER BY id DESC LIMIT 6").fetchall()
    tried = db.execute("SELECT score,idea FROM programs WHERE task IS NOT NULL ORDER BY id DESC LIMIT 25").fetchall()
    best = db.execute("SELECT score FROM programs WHERE status='ok' ORDER BY score LIMIT 1").fetchone()[0]
    metrics = json.loads(parent[4])
    per_case = metrics.pop("per_case")
    gaps = sorted(per_case, key=lambda r: PRUNED[(r[0], r[1])] - r[2])[:8]
    work, inbox = folder / "work" / f"t{task_id}", folder / "inbox" / f"t{task_id}.py"
    work.mkdir(exist_ok=True)
    out = [f"# Worker brief t{task_id} (island {island}, parent p{parent[0]})\n",
           "You are one mutation worker in an evolutionary program search (AlphaEvolve-style). Your goal:"
           " a program that scores clearly LOWER than its parent by a NEW mechanism, not a tweak.\n",
           TASK_A,
           "\n## Library source (import from agentic_research.evolve_v4.helpers)\n```python\n"
           + library_source() + "```",
           f"\n## Parent p{parent[0]} (score {parent[3]:.0f}; best in population {best:.0f})\nIdea: {parent[2]}\n"
           f"Totals: {metrics.get('totals')}\n```python\n{parent[1]}\n```",
           "\nParent's most wasted cases (folder, epsilon: parent reads vs hindsight minimum):\n"
           + "\n".join(f"- {g} eps{e}: {q} vs {PRUNED[(g, e)]}" for g, e, q in gaps)]
    for ins in inspirations:
        out.append(f"\n## Inspiration p{ins[0]} from another island (score {ins[3]:.0f})\nIdea: {ins[2]}\n"
                   f"```python\n{ins[1][:6000]}\n```")
    if direction:
        out.append(f"\n## ASSIGNED DIRECTION (follow this)\n{direction}")
    if note:
        out.append("\n## Scientist directions\n" + note[0][:5000])
    if tried:
        out.append("\n## Already tried (score, idea) — do not repeat these\n"
                   + "\n".join(f"- {s if s is None else round(s)}: {i}" for s, i in tried))
    if fails:
        out.append("\n## Recent failures\n" + "\n".join(f"- {i}: {e}" for i, e in fails))
    out.append(f"""
## How to work (strict)
1. Write drafts only under {work}/ . Score a draft with (from {REPO}):
   .venv-mac/bin/python -m agentic_research.evolve_v4.try_a {work}/draft.py
   It reports the SEARCH split only. Use at most 6 scorings. Debug errors it prints.
2. Do NOT read or run anything that touches the holdout (it is sealed), do not read data files
   (*.json.gz, *.npz), other runs, ledgers, or git history. Do not edit any file outside {work}/
   and {inbox}. Do not hard-code folder names or per-case constants: the program must generalise to
   unseen folders.
3. Save your single best program to {inbox} whose FIRST TWO LINES are
   # IDEA: <one sentence: the mechanism>
   # NOVELTY: <one sentence: how it differs from the parent and from standard value-of-information /
   #          uncertainty sampling / greedy information gain>
   Save it even if it did not beat the parent (failures are recorded and useful).
4. Final reply: ONE line only: "t{task_id} score=<your best search total> idea=<short>".
""")
    path = folder / "tasks" / f"t{task_id}.md"
    path.write_text("\n".join(out))
    return parent[0], path


def dispatch(campaign, n, direction):
    folder, db = connect(campaign)
    island_counts = dict(db.execute("SELECT island, COUNT(*) FROM tasks GROUP BY island").fetchall())
    for _ in range(n):
        island = min(range(ISLANDS), key=lambda i: island_counts.get(i, 0))
        island_counts[island] = island_counts.get(island, 0) + 1
        cur = db.execute("INSERT INTO tasks(island,parent,direction,created) VALUES(?,?,?,?)",
                         (island, None, direction, time.time()))
        parent, path = brief(db, folder, cur.lastrowid, island, direction)
        db.execute("UPDATE tasks SET parent=? WHERE id=?", (parent, cur.lastrowid))
        db.commit()
        print(f"t{cur.lastrowid} island{island} parent p{parent}: {path}")


def ingest(campaign, only=None):
    folder, db = connect(campaign)
    for task_id, island, parent in db.execute("SELECT id,island,parent FROM tasks WHERE ingested=0").fetchall():
        if only and task_id not in only:
            continue
        path = folder / "inbox" / f"t{task_id}.py"
        if not path.exists():
            continue
        code = path.read_text()
        idea = (re.search(r"#\s*IDEA:\s*(.*)", code) or [None, ""])[1].strip()[:400]
        novelty = (re.search(r"#\s*NOVELTY:\s*(.*)", code) or [None, ""])[1].strip()[:400]
        score, metrics, error = evaluate(code, folder / "programs" / f"t{task_id}.py")
        db.execute("INSERT INTO programs(task,parent,island,code,idea,novelty,status,score,metrics,created,error)"
                   " VALUES(?,?,?,?,?,?,?,?,?,?,?)", (task_id, parent, island, code, idea, novelty,
                   "ok" if error is None else "eval_error", score, json.dumps(metrics), time.time(), error))
        db.execute("UPDATE tasks SET ingested=1 WHERE id=?", (task_id,))
        db.commit()
        totals = metrics["totals"] if metrics else None
        print(f"t{task_id} island{island} parent p{parent}: score={score} {totals} {error or ''} | {idea[:150]}")


def status(campaign):
    folder, db = connect(campaign)
    for row in db.execute("SELECT id,task,island,score,metrics,idea FROM programs WHERE status='ok'"
                          " ORDER BY score LIMIT 8"):
        print(f"p{row[0]} t{row[1]} island{row[2]} {row[3]:.0f} {json.loads(row[4])['totals']} | {row[5][:110]}")
    counts = db.execute("SELECT status, COUNT(*) FROM programs WHERE task IS NOT NULL GROUP BY status").fetchall()
    pending = db.execute("SELECT COUNT(*) FROM tasks WHERE ingested=0").fetchone()[0]
    print(f"programs by status {counts}; pending tasks {pending}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("dispatch", "ingest", "note", "status"))
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--direction", default="")
    ap.add_argument("--file")
    ap.add_argument("--task", type=int, nargs="*")
    a = ap.parse_args()
    if a.mode == "dispatch":
        dispatch(a.campaign, a.n, a.direction)
    elif a.mode == "ingest":
        ingest(a.campaign, a.task)
    elif a.mode == "note":
        folder, db = connect(a.campaign)
        db.execute("INSERT INTO notes(created,text) VALUES(?,?)", (time.time(), Path(a.file).read_text()))
        db.commit()
    else:
        status(a.campaign)
