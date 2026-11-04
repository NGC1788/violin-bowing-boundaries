"""Build the Colab bundle: search folders only, frozen certificate code copied verbatim.

  .venv-mac/bin/python -m agentic_research.evolve_v4.colab.build_bundle
writes agentic_research/evolve_v4/colab/bundle/ (committed; the notebook sparse-clones only
this folder). The sealed C/D holdout is never exported, so a Colab run cannot score or see it.
"""
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
POPULATION = {"seed": None, "p10_t6": 10, "p5_t2": 5, "p9_t5": 9, "p7_t4": 7, "p8_t1": 8}


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
        '"load", "loadtxt"', '"revealed", "load", "loadtxt"')  # labels-only track
    assert '"revealed"' in source
    (pkg / "guard.py").write_text(source)
    for name in ("colab_task.py", "colab_eval.py", "colab_engine.py"):
        shutil.copy(HERE / "src" / name, pkg / name)
    blob = gzip.compress(pickle.dumps(task_a.load_cases("search"), protocol=4), mtime=0)
    (OUT / "cases" / "cases_search.pkl.gz").write_bytes(blob)
    shutil.copy(EVOLVE / "prune_search.json", OUT / "cases" / "prune_search.json")
    (OUT / "cases" / "manifest.json").write_text(json.dumps(dict(
        split="search (A/B folders only; C/D holdout not included)", cases=54,
        cases_sha256=hashlib.sha256(blob).hexdigest(), source_sha256=pilot.DIGEST), indent=1))
    db = sqlite3.connect(EVOLVE / "runs" / "a1" / "ledger.sqlite")
    for name, pid in POPULATION.items():
        code = (EVOLVE / "seed_a.py").read_text() if pid is None else \
            db.execute("SELECT code FROM programs WHERE id=?", (pid,)).fetchone()[0]
        assert "revealed" not in code, name
        (OUT / "population" / f"{name}.py").write_text(code)
    files = [p for p in OUT.rglob("*") if p.is_file()]
    print(OUT, len(files), "files", sum(p.stat().st_size for p in files), "bytes")
