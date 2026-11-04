"""Bring Colab ledgers back to the Mac: re-score the top programs here, then build a campaign.

  .venv-mac/bin/python -m agentic_research.evolve_v4.colab.merge_runs --campaign colab1 \
      runs_colab/acct1/ledger.sqlite runs_colab/acct2/ledger.sqlite ...
Every candidate is re-scored on the Mac search split (must equal the Colab score). The new
campaign can then go to final_a.py once; report it as the SECOND look at the C/D holdout
(the first was campaign a1).
"""
import argparse
import hashlib
import json
import sqlite3
import time

from agentic_research.evolve_v4.engine import connect, evaluate

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("ledgers", nargs="+")
    a = ap.parse_args()
    pool = {}
    for path in a.ledgers:
        db = sqlite3.connect(path)
        for code, score, idea, novelty, model in db.execute(
                "SELECT code,score,idea,novelty,model FROM programs WHERE status='ok' AND model NOT LIKE 'init:%'"
                " AND model!='migrant'"):
            digest = hashlib.sha256(code.encode()).hexdigest()
            if digest not in pool or score < pool[digest][1]:
                pool[digest] = (code, score, idea, novelty, f"{path}:{model}")
    ranked = sorted(pool.values(), key=lambda r: r[1])[:a.top]
    folder, db = connect(a.campaign)
    for k, (code, colab_score, idea, novelty, origin) in enumerate(ranked):
        score, metrics, error = evaluate(code, folder / "programs" / f"colab{k}.py")
        agree = score == colab_score
        db.execute("INSERT INTO programs(task,parent,island,code,idea,novelty,status,score,metrics,created,error)"
                   " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                   (1000 + k, None, k % 4, code, idea, novelty, "ok" if error is None and agree else "mismatch",
                    score, json.dumps(metrics), time.time(), error or (None if agree else f"colab {colab_score}")))
        db.commit()
        print(f"colab{k} colab={colab_score} mac={score} {'OK' if agree else 'MISMATCH'} | {origin} | {idea[:100]}")
