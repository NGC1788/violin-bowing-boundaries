"""Pick the A2 strokes around each lower boundary and extract only those from the archive.

For every (speed, beta column) the lower boundary from the dataset's own 0.4 s
window is located (first Helmholtz run of length >= 2). The strokes from six
force steps below to six above it are taken: these are the strokes whose label
can change as the motion settles. Left-censored columns take their lowest
eight forces.
"""

import csv
import gzip
import subprocess
from pathlib import Path

REPO = Path.home() / "violin-bowing-boundaries"
ARCHIVE = REPO / "data/raw/zenodo/17782542/2024-04-19_TypeA_sample2.7z"
SEVEN = REPO / ".local-tools/7zip-26.03/7zz"
OUT = Path.home() / "violin-rne/timeres"
EXTRACT = OUT / "raw"
SPAN = 6


def main() -> None:
    rows = list(csv.DictReader(gzip.open(REPO / "research/2026-10-14/lockbox_A2/regimes.csv.gz", "rt")))
    columns: dict = {}
    for r in rows:
        columns.setdefault((r["condition"], int(r["beta_level"])), []).append(r)
    chosen = []
    for (condition, level), cells in sorted(columns.items()):
        cells.sort(key=lambda r: float(r["c1_window_mean"]))
        h = [r["label"] == "helmholtz" for r in cells]
        start = next((i for i in range(len(h) - 1) if h[i] and h[i + 1]), None)
        if start is None:
            continue
        lo, hi = (0, 8) if start == 0 else (max(0, start - SPAN), min(len(cells), start + SPAN + 1))
        chosen += [(condition, int(r["trial"])) for r in cells[lo:hi]]
    chosen = sorted(set(chosen))
    names = [f"2024-04-19_TypeA_sample2/{c}/whole_{n}.csv" for c, n in chosen]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "chosen.csv").write_text("condition,trial\n" + "".join(f"{c},{n}\n" for c, n in chosen))
    listfile = OUT / "members.txt"
    listfile.write_text("\n".join(names) + "\n")
    print(f"선택한 긋기 {len(chosen)}개: " + ", ".join(f"{c} {sum(1 for x in chosen if x[0] == c)}" for c in ("v1", "v2", "v3")))
    EXTRACT.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(SEVEN), "x", str(ARCHIVE), f"-o{EXTRACT}", f"@{listfile}", "-y", "-bso0", "-bsp0"], check=True)
    got = list(EXTRACT.rglob("whole_*.csv"))
    print(f"추출 완료 {len(got)}개, {sum(p.stat().st_size for p in got) / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
