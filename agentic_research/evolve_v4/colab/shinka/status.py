"""Progress of a ShinkaEvolve run, counted the way the pre-registered comparison counts it.

  python status.py [programs.sqlite]     (default /content/run/acct2_shinka/results/programs.sqlite)
A proposal is one Shinka generation (its patch resamples included): evaluated programs plus terminal
pre-eval failures in attempt_log. Reads = -combined_score; failed evaluations score -1e6 and are skipped.
"""
import sqlite3
import sys

DB = sys.argv[1] if len(sys.argv) > 1 else "/content/run/acct2_shinka/results/programs.sqlite"
db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
rows = db.execute("SELECT generation, combined_score, correct, timestamp, json_extract(metadata, '$.patch_type'),"
                  " json_extract(metadata, '$.patch_name') FROM programs WHERE generation > 0").fetchall()
failed = {g for (g,) in db.execute("SELECT generation FROM attempt_log WHERE status = 'failed'"
                                   " AND json_extract(details, '$.node_kind') = 'failed_proposal'")}
start = db.execute("SELECT MIN(timestamp), MIN(-combined_score) FROM programs WHERE generation = 0").fetchone()
ok = [r for r in rows if r[2] and r[1] > -1e5]
gens = {r[0] for r in rows} | failed
print(f"proposals {len(gens)}: evaluated {len(rows)} (passed {len(ok)}, eval failed {len(rows) - len(ok)}),"
      f" failed before eval {len(failed)}")
if start[0] and rows:
    print(f"elapsed {(max(r[3] for r in rows) - start[0]) / 3600:.2f} h since the initial program ({start[1]:.0f})")
if ok:
    print("best reads:", int(min(-r[1] for r in ok)))
    for n in (100, 200, 300):  # best among the first n proposals (Shinka numbers generations in start order)
        if len(gens) >= n:
            first = sorted(gens)[:n]
            print(f"  first {n}: best {int(min([-r[1] for r in ok if r[0] <= first[-1]] or [start[1]]))}")
    for g, s, _, _, kind, name in sorted(ok, key=lambda r: -r[1])[:5]:
        print(f"  gen {g:4d} {kind:5s} {int(-s):5d}  {name}")
