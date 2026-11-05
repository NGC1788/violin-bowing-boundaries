"""Build the Colab bundle: search folders only, frozen certificate code copied verbatim.

  .venv-mac/bin/python -m agentic_research.evolve_v4.colab.build_bundle
writes agentic_research/evolve_v4/colab/bundle/ (committed; the notebook sparse-clones only
this folder). The sealed C/D holdout is never exported, so a Colab run cannot score or see it.
"""
import ast
import gzip
import hashlib
import inspect
import json
import pickle
import shutil
import sqlite3
from pathlib import Path

from agentic_research.evolve_v4 import guard, task_a
from agentic_research.law_counterfactual_selection_v1 import experiment
from agentic_research.law_decision_pilot_v1 import pilot

HERE = Path(__file__).resolve().parent
EVOLVE = HERE.parent
OUT = HERE / "bundle"
ROBUST = EVOLVE / "transfer_robust" / "cases"
ROBUST_TASK = """THE PROBLEM. You are evolving a QUERY POLICY for an exact certification problem (AI robustness audit).

A folder is a group of C correctly classified MNIST test images (one column each) and a trained CNN.
Each column has 50 Gaussian-noise levels sigma (shared grid, sorted ascending; view.force). Reading one
cell (column, rank) runs the network on 64 noisy copies at that sigma and reveals its label under 9
nested failure rules at once (rules = (fail-rate floor u, true-class-probability ceiling v);
label = fail_rate>=u AND mean true-class probability<v, i.e. "broken"). For each rule and column the
"boundary" (break point) is the first adjacent pair of positive labels along the noise rank. A fixed
exact certificate (you cannot change it) asks whether SOME rule has break points that all agree with
the local-linear robustness law sigma* = C * margin/||grad margin|| within a factor exp(epsilon):
view.log_beta = -0.5*log(margin/grad) per image, so the intercept log sigma + 2*log_beta must share a
common point across all columns ([lower, upper] per possible start, tolerance epsilon). It stops as
soon as every completion of the unread labels gives the same answer. Unknown labels make several
first-run starts possible; possible marks them (last slot = "no break point in this column").

choose(view) must return (column, rank) of an UNREAD cell so the certificate stops after as FEW reads
as possible. Score = total reads over 40 search cases (20 folders x epsilon 0.2/0.4), lower is
better. Reading every cell costs 48000. The one-step counterfactual lookahead (seed) scores 11182;
bisecting every image's break point scores 9914. A policy evolved on a different benchmark with the
same interface (violin playability law) already scores about 3560-3700 here unchanged. Hindsight
pruning of the seed's reads shows certificates exist with about 1842 reads in total.
37 of 40 cases are INCOMPATIBLE (answer "the law fails under every rule"): proving that needs, for
every one of the 9 rules, either a column with no valid start or two columns whose intervals cannot
overlap. Labels are monotone in the thresholds, and neighbouring ranks are correlated (labels are
noisy, not strictly monotone in sigma). NOTE: one read reveals all 9 labels of that cell at once, so
cross-rule monotonicity can NEVER infer labels of an unread cell; use it only to reason about which
rules can still pass.

What is known (ablation of the best programs on the violin benchmark): the gain comes from COMMITTING
to a separating column pair per rule and VERIFYING it (zero-sweep of the high column's early starts +
one forced positive pair in the low column), with bisection-style probing of where labels flip, and
a small exact lookahead at the start and the end.

view fields (numpy copies): partial (C,9,50) int8 -1 unknown/0/1; possible (C,9,51) bool;
lower, upper (C,1,50) float, NaN = invalid start; force (C,50) = sigma; log_beta (C,); rules (9,2);
epsilon; queries_so_far. (LABELS ONLY: view.revealed exists but is forbidden; the guard rejects it.)
Each case runs in a fresh process: no memory between cases.
Allowed imports: numpy, math, itertools, functools, heapq, collections,
agentic_research.evolve_v4.helpers (exports possible_starts, certificates, outer, shortlist,
counterfactual_choose, response_patterns, UNIFORM10; source below). No file, network, process or
introspection access (an AST guard rejects open/exec/eval/getattr/__dict__/np.load etc.). Keep each
case under ~3 s. A wrong or repeated query is an error and scores nothing. Do not hard-code folder
names or per-case constants: the program must generalise to unseen folders."""
NOEPS_NOTE = """

THIS TRACK (no epsilon). The two epsilon values (0.2, 0.4) are the same in the search and the sealed
holdout, so a branch on epsilon is a hand-set constant in disguise; earlier records that switched on
epsilon >= 0.3 got better when the switch was removed. Here the policy cannot read epsilon at all. Do
not reconstruct it from interval widths either: decide from the labels read so far and the geometry.
The current best (about 2238) uses no epsilon: (1) an A column is ruled out only by a CONFIRMED
positive pair below the threshold, its cost is a greedy zero-cover count that steps over isolated
positives; (2) rules are attacked loosest first; (3) a pass-proof mode pins every column inside a
common window, switched on by a ski-rental rule (reads so far >= optimistic remaining pass cost).
A variant (about 2349) switches from sweeps to witness bisection while every column read so far is
monotone along force (no zero above a positive), which generalises to clean monotone labels."""
POPULATION = {"seed": None, "p25_t21": 25, "p12_t9": 12, "p10_t6": 10, "p5_t2": 5, "p9_t5": 9, "p7_t4": 7, "p8_t1": 8,
              # 2026-11-05 cloud improvement session (files, not ledger rows): violin search 2238 / 2349
              "c1_best": "improved/best_policy.py", "c2_general": "improved/general_policy.py"}


def uses(code, name="revealed"):
    return any(isinstance(n, ast.Attribute) and n.attr == name or isinstance(n, ast.Name) and n.id == name
               for n in ast.walk(ast.parse(code)))


def noeps_task():
    """The violin task text with epsilon removed from the view (bundle_noeps, a generalisation track)."""
    tree = ast.parse((HERE / "src" / "colab_engine.py").read_text())
    task = next(n.value.value for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "TASK")
    old = "epsilon; queries_so_far. (LABELS ONLY: view.revealed exists but is forbidden; the guard rejects it.)"
    assert old in task
    task = task.replace(old, "queries_so_far. (LABELS ONLY: view.revealed exists but is forbidden; the guard rejects it.\n"
                             "NO-EPSILON TRACK: view.epsilon is forbidden too and the guard rejects it.)")
    return task + NOEPS_NOTE


def core_source():
    parts = ['"""Frozen certificate, counterfactual baseline and replay, copied verbatim from the repo."""',
             "import numpy as np\n"]
    for f in (pilot.possible_starts, pilot.certificates, experiment.response_patterns, experiment.outer,
              experiment.shortlist):
        parts.append(inspect.getsource(f))
    parts.append(inspect.getsource(experiment.choose).replace("def choose(", "def counterfactual_choose(", 1))
    parts += [inspect.getsource(task_a.View), inspect.getsource(task_a.replay)]
    return "\n\n".join(parts)


if __name__ == "__main__":
    import sys
    robust = sys.argv[1:] == ["robust"]  # bundle_robust: the AI-robustness audit transfer benchmark
    noeps = sys.argv[1:] == ["noeps"]    # bundle_noeps: violin search, view.epsilon forbidden
    if robust:
        OUT = HERE / "bundle_robust"
    if noeps:
        OUT = HERE / "bundle_noeps"
    if OUT.exists():
        shutil.rmtree(OUT)
    pkg = OUT / "agentic_research" / "evolve_v4"
    (OUT / "cases").mkdir(parents=True)
    (OUT / "population").mkdir()
    pkg.mkdir(parents=True)
    (OUT / "agentic_research" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "core.py").write_text(core_source())
    (pkg / "helpers.py").write_text(
        '"""Building blocks a candidate policy may import (same names as on the Mac)."""\n'
        "import numpy as np\n\nfrom agentic_research.evolve_v4.core import (  # noqa: F401\n"
        "    certificates, counterfactual_choose, outer, possible_starts, response_patterns, shortlist,\n)\n\n"
        "UNIFORM10 = np.full(10, .1)\n")
    source = inspect.getsource(guard).replace(
        '"load", "loadtxt"', '"revealed", "epsilon", "load", "loadtxt"' if noeps else '"revealed", "load", "loadtxt"')
    assert '"revealed"' in source and (not noeps or '"epsilon"' in source)  # labels-only (and no-epsilon) track
    (pkg / "guard.py").write_text(source)
    for name in ("colab_task.py", "colab_eval.py", "colab_engine.py", "colab_serve.py"):
        shutil.copy(HERE / "src" / name, pkg / name)
    if robust:  # search split only; the robust holdout stays on the Mac, sealed
        blob = (ROBUST / "cases_search.pkl.gz").read_bytes()
        shutil.copy(ROBUST / "prune_search.json", OUT / "cases" / "prune_search.json")
        (OUT / "cases" / "task.txt").write_text(ROBUST_TASK)
        info = dict(split="robust search (digit groups 0-1; groups 2-3 sealed on the Mac)", cases=40,
                    source_sha256=json.loads((ROBUST / "manifest.json").read_text())["model_sha256"])
    else:
        blob = gzip.compress(pickle.dumps(task_a.load_cases("search"), protocol=4), mtime=0)
        shutil.copy(EVOLVE / "prune_search.json", OUT / "cases" / "prune_search.json")
        info = dict(split="search (A/B folders only; C/D holdout not included)", cases=54,
                    source_sha256=pilot.DIGEST)
        if noeps:
            (OUT / "cases" / "task.txt").write_text(noeps_task())
            info["track"] = "no-epsilon (guard rejects view.epsilon)"
    (OUT / "cases" / "cases_search.pkl.gz").write_bytes(blob)
    (OUT / "cases" / "manifest.json").write_text(json.dumps(dict(info, cases_sha256=hashlib.sha256(blob).hexdigest()),
                                                            indent=1))
    db = sqlite3.connect(EVOLVE / "runs" / "a1" / "ledger.sqlite")
    for name, pid in POPULATION.items():
        if pid is None:
            code = (EVOLVE / "seed_a.py").read_text()
        elif isinstance(pid, str):
            code = (EVOLVE / pid).read_text()
        else:
            code = db.execute("SELECT code FROM programs WHERE id=?", (pid,)).fetchone()[0]
        assert not uses(code), name  # labels-only track (comments may mention the word)
        if noeps and uses(code, "epsilon"):
            continue  # would fail this track's guard
        (OUT / "population" / f"{name}.py").write_text(code)
    files = [p for p in OUT.rglob("*") if p.is_file()]
    print(OUT, len(files), "files", sum(p.stat().st_size for p in files), "bytes")
