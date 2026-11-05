"""Run ShinkaEvolve (SakanaAI, ICLR 2026) on the violin query-policy task, matched to the in-house engine.

  python run_shinka.py --model Qwen/Qwen3.8-27B-FP8 [--run-dir D] [--backup-dir DRIVE]

Matched to colab_engine for the comparison (pre-registered 2026-11-05, before any Shinka score):
  same served model (local vLLM, temperature 1.0), same 54 search cases and scorer, same start
  (c1_best 2238 as the initial program; c2_general 2349 is not loaded because Shinka takes one
  initial program), same task text and library source, same per-case "most wasted" hint
  (text feedback), 8 proposals in flight, 45000 max tokens.  Compare best score at equal proposal counts.
  Shinka sends only temperature; vLLM fills top_p / top_k from the model's generation_config, which is
  where colab_engine's explicit 0.95 / 20 come from.
Shinka keeps its own machinery: weighted parent sampling, archive and top-k inspirations, diff /
full / crossover patches, embedding + LLM novelty rejection, meta recommendations, 4 islands.
Embeddings come from embed_server.py on CPU (the GPU stays with the proposal model).
The novelty judge samples at temperature 1.0 like the proposals (greedy decoding loops in thinking mode).
Shinka keeps its database at <run-dir>/results/programs.sqlite; failed proposals go to its attempt_log table.
"""
import argparse
import ast
import fcntl
import json
import os
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def task_text(bundle):
    tree = ast.parse((bundle / "agentic_research/evolve_v4/colab_engine.py").read_text())
    task = next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", "") == "TASK")
    if (bundle / "cases" / "task.txt").exists():
        task = (bundle / "cases" / "task.txt").read_text()
    library = (bundle / "agentic_research/evolve_v4/core.py").read_text()
    return (task + "\n\n## Library source (agentic_research.evolve_v4.helpers re-exports these)\n```python\n"
            + library + "\n```\n\nThe whole program lies between EVOLVE-BLOCK-START and EVOLVE-BLOCK-END and must "
            "keep defining choose(view). Lower reads are better: combined_score = -(total reads).")


def port_open(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def backup_once(db, target):
    if db.exists():
        try:
            src, dst = sqlite3.connect(db, timeout=30), sqlite3.connect(target)
            src.backup(dst)
            src.close(), dst.close()
        except Exception as exc:  # keep running; next round retries
            print("backup failed:", exc, flush=True)


def backup_loop(db, target, every=300):
    time.sleep(min(60, every))  # first copy early, then every `every` seconds
    while True:
        backup_once(db, target)
        time.sleep(every)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bundle", default=os.environ.get("EVOLVE_BUNDLE", "/content/evolve"))
    ap.add_argument("--run-dir", default="/content/run/acct2_shinka")
    ap.add_argument("--backup-dir", default=None)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--embed-port", type=int, default=8001)
    ap.add_argument("--generations", type=int, default=100000)
    ap.add_argument("--proposals", type=int, default=8)
    ap.add_argument("--evals", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=45000)  # colab_engine default
    ap.add_argument("--initial", default="c1_best")
    ap.add_argument("--patch-probs", default="0.6,0.3,0.1", help="diff,full,cross")
    ap.add_argument("--backup-every", type=int, default=300)
    ap.add_argument("--eval-python", default=sys.executable,
                    help="interpreter for evaluate.py; on Colab the system one, where the bundle scorer was checked")
    ap.add_argument("--embed-python", default=sys.executable)
    a = ap.parse_args()

    bundle, run = Path(a.bundle), Path(a.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    lock = open(run / "run.lock", "w")  # one Shinka per run dir: a second launch would share the DB
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(f"already running in {run} (stop it first: pkill -f run_shinka.py)")
    lock.write(str(os.getpid()))
    lock.flush()
    os.environ["EVOLVE_BUNDLE"] = str(bundle)
    os.environ.setdefault("SHINKA_PRICING_MODE", "offline")
    os.environ.setdefault("LOCAL_OPENAI_API_KEY", "local")
    if not port_open(a.embed_port):
        subprocess.Popen([a.embed_python, str(HERE / "embed_server.py"), str(a.embed_port)],
                         stdout=open(run / "embed.log", "a"), stderr=subprocess.STDOUT)
        for _ in range(120):
            if port_open(a.embed_port):
                break
            time.sleep(1)
    initial = run / "initial.py"
    if not initial.exists():
        code = (bundle / "population" / f"{a.initial}.py").read_text()
        initial.write_text("# EVOLVE-BLOCK-START\n" + code.rstrip() + "\n# EVOLVE-BLOCK-END\n")

    pre = run / "preflight"  # score the initial program once with the eval interpreter before any LLM call
    subprocess.run([a.eval_python, str(HERE / "evaluate.py"), "--program_path", str(initial), "--results_dir", str(pre)],
                   check=False)
    check = json.loads((pre / "metrics.json").read_text()) if (pre / "metrics.json").exists() else {}
    if check.get("public", {}).get("total_reads") is None:
        raise SystemExit(f"preflight failed: {check.get('text_feedback', 'no metrics written')}")
    print("preflight: initial program", check["public"]["total_reads"], "reads", flush=True)

    from shinka.core import EvolutionConfig, ShinkaEvolveRunner
    from shinka.database import DatabaseConfig
    from shinka.launch import LocalJobConfig

    llm = f"local/{a.model}@{a.base_url}"
    evo = EvolutionConfig(
        task_sys_msg=task_text(bundle), language="python", init_program_path=str(initial),
        results_dir=str(run / "results"), num_generations=a.generations, job_type="local",
        patch_types=["diff", "full", "cross"], patch_type_probs=[float(x) for x in a.patch_probs.split(",")],
        max_patch_resamples=3, max_patch_attempts=1,
        llm_models=[llm], llm_dynamic_selection="fixed",
        llm_kwargs={"temperatures": [1.0], "max_tokens": a.max_tokens},
        meta_rec_interval=10, meta_llm_models=[llm], meta_llm_kwargs={"temperatures": [1.0], "max_tokens": 16000},
        embedding_model=f"local/all-MiniLM-L6-v2@http://127.0.0.1:{a.embed_port}/v1",
        code_embed_sim_threshold=0.99, max_novelty_attempts=3,
        novelty_llm_models=[llm], novelty_llm_kwargs={"temperatures": [1.0], "max_tokens": 16000},
        use_text_feedback=True, max_api_costs=None)
    db = DatabaseConfig(
        num_islands=4, archive_size=40, elite_selection_ratio=0.3,
        num_archive_inspirations=1, num_top_k_inspirations=1, migration_interval=10, migration_rate=0.1,
        island_elitism=True, parent_selection_strategy="weighted", parent_selection_lambda=10)
    job = LocalJobConfig(eval_program_path=str(HERE / "evaluate.py"), time="00:10:00",
                         python_executable=a.eval_python)
    src, dst = run / "results" / "programs.sqlite", Path(a.backup_dir or run) / "programs.sqlite"
    if a.backup_dir:
        dst.parent.mkdir(parents=True, exist_ok=True)
        threading.Thread(target=backup_loop, args=(src, dst, a.backup_every), daemon=True).start()
    try:
        ShinkaEvolveRunner(evo_config=evo, job_config=job, db_config=db, max_evaluation_jobs=a.evals,
                           max_proposal_jobs=a.proposals, max_db_workers=4, verbose=True).run()
    finally:
        if a.backup_dir:
            backup_once(src, dst)


if __name__ == "__main__":
    main()
