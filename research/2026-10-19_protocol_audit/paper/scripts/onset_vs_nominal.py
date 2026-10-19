"""Open-loop force setting: ratio of the measured force at bow onset to the nominal grid force, per speed folder,
for A1 and A2 (reads claude_extra ramp_meta CSVs only). Nominal grid: 50 log-spaced forces 0.1-4 N (POMA 2025).
A1 trials run in reverse force order (rank = 51 - ((trial-1) % 50 + 1)), as in claude_extra/drift_tests.py."""
import csv
from pathlib import Path
import numpy as np

H = Path.home() / "violin-rne/claude_extra"
SRC = {"A1": (H / "ramp/ramp_meta_2024-03-25_TypeA_sample1.csv", True), "A2": (H / "ramp_meta_A2.csv", False)}
grid = np.geomspace(0.1, 4.0, 50)
for tag, (path, rev) in SRC.items():
    rows = [r for r in csv.DictReader(path.open()) if not r["error"]]
    for fol in sorted({r["folder"] for r in rows}):
        rs = [r for r in rows if r["folder"] == fol]
        rank = np.array([(int(r["trial"]) - 1) % 50 + 1 for r in rs]); rank = 51 - rank if rev else rank
        on = np.array([float(r["force_at_onset"]) for r in rs]); ratio = on / grid[rank - 1]
        lo = rank <= 20; hi = rank >= 31
        print(f"{tag} {fol.split('/')[-1]}: n {len(rs)} | onset/nominal median ranks 1-20 {np.median(ratio[lo]):.2f} "
              f"(IQR {np.percentile(ratio[lo],25):.2f}-{np.percentile(ratio[lo],75):.2f}) | ranks 31-50 {np.median(ratio[hi]):.2f}")
