"""ShinkaEvolve evaluator for the violin query-policy task (same scorer as colab_engine.evaluate).

  python evaluate.py --program_path P --results_dir D
Writes D/metrics.json (combined_score = -total reads, higher is better) and D/correct.json.
The bundle (EVOLVE_BUNDLE, default /content/evolve) holds the 54 search cases only.
text_feedback gives the same per-case hint the in-house engine shows: the parent's most wasted cases
(reads vs the hindsight-pruned minimum).
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

BUNDLE = Path(os.environ.get("EVOLVE_BUNDLE", "/content/evolve"))
FAIL = -1e6


def pruned():
    rows = json.loads((BUNDLE / "cases" / "prune_search.json").read_text())["per_case"]
    return {(r["group"], r["epsilon"]): r["pruned"] for r in rows}


def score(program_path, timeout=300):
    """Return (metrics dict from colab_eval, None) or (None, error line); same call as colab_engine.evaluate."""
    sys.path.insert(0, str(BUNDLE))
    try:
        from agentic_research.evolve_v4 import guard
        guard.check(Path(program_path).read_text())
        proc = subprocess.run([sys.executable, "-m", "agentic_research.evolve_v4.colab_eval",
                               str(Path(program_path).resolve())], cwd=BUNDLE, capture_output=True, text=True,
                              timeout=timeout,
                              env={**os.environ, "EVOLVE_PARALLEL": os.environ.get("EVOLVE_PARALLEL", "2")})
        if proc.returncode:
            return None, (proc.stderr or proc.stdout).strip().splitlines()[-1][:400]
        return json.loads(proc.stdout.strip().splitlines()[-1]), None
    except subprocess.TimeoutExpired:
        return None, f"timeout after {timeout}s"
    except Exception as exc:  # guard or parse failure
        return None, f"{type(exc).__name__}: {exc}"[:400]


def main(program_path, results_dir):
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics, error = score(program_path)
    if error is not None:
        result = dict(combined_score=FAIL, public=dict(total_reads=None), private=dict(error=error),
                      text_feedback=f"Evaluation failed: {error}")
    else:
        target = pruned()
        gaps = sorted(metrics["per_case"], key=lambda r: target[(r[0], r[1])] - r[2])[:6]
        feedback = ("Most wasted cases (folder, epsilon: reads vs hindsight minimum):\n"
                    + "\n".join(f"- {g} eps{e}: {r} vs {target[(g, e)]}" for g, e, r in gaps))
        result = dict(combined_score=-float(metrics["total"]),
                      public=dict(total_reads=metrics["total"], **metrics["totals"],
                                  compatible_reads=metrics["compatible_total"]),
                      private=dict(per_case=metrics["per_case"], max_case_seconds=metrics["max_case_seconds"]),
                      text_feedback=feedback)
    (out / "metrics.json").write_text(json.dumps(result))
    (out / "correct.json").write_text(json.dumps(dict(correct=error is None, error=error or "")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--program_path", required=True)
    ap.add_argument("--results_dir", required=True)
    a = ap.parse_args()
    main(a.program_path, a.results_dir)
