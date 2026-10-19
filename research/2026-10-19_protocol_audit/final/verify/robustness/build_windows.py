"""Build window-length variants (0.1 s, 0.4 s) of final/data.py's cache into this scratch folder (read-only on features).
Same classifier, same columns; Z/f0 copied from the original 0.2 s cache so only the labels change."""
import csv
import gzip
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "violin-rne/claude_extra"))
sys.path.insert(0, str(Path.home() / "violin-rne/final"))
import verify_astra_boundaries as V  # noqa: E402
import data as D  # noqa: E402

OUT = Path(__file__).parent / "cache_w"
OUT.mkdir(exist_ok=True)
WINS = ("0.1", "0.4")


def build_file(stem, reps):
    acc = {(w, rep): dict(rows=[], beta={}) for w in WINS for rep in reps}
    with gzip.open(D.FEAT / f"{stem}.csv.gz", "rt") as fh:
        for r in csv.DictReader(fh):
            w = r["window_s"]
            if r["error"] or w not in WINS:
                continue
            rep = None
            for rp in reps:
                if rp is None or f"/{rp}/" in r["folder"] + "/":
                    rep = rp
                    break
            else:
                continue
            t = round(float(r["t"]), 2)
            k = (r["folder"], int(r["trial"]))
            a = acc[(w, rep)]
            a["beta"][k] = float(r["beta"])
            if t in D.TIMES:
                fpp = r["flybacks_per_period"]
                lab = V.rm.classify(float(r["amplitude_std"]), float(r["periodicity"]), float(r["f0_hz"]),
                                    int(float(fpp)) if fpp not in ("", "nan", "None") else None, 98.08, V.APPLIED)
                a["rows"].append((k, t, float(r["force"]), float(r["speed"]), lab == "helmholtz"))
    return acc


def finish(a, tag):
    rows, beta = a["rows"], a["beta"]
    col = V.columns(beta)
    folders = sorted({k[0] for k, *_ in rows})
    sp = {}
    for k, t, f, v, h in rows:
        sp.setdefault(k[0], []).append(v)
    sp = {f: float(np.median(v)) for f, v in sp.items()}
    orig = D.load(tag)
    return dict(folder=np.array([folders.index(k[0]) for k, *_ in rows]), trial=np.array([k[1] for k, *_ in rows]),
                beta=np.array([beta[k] for k, *_ in rows]), column=np.array([col[k] for k, *_ in rows]),
                speed=np.array([sp[k[0]] for k, *_ in rows]), t=np.array([r[1] for r in rows]),
                force=np.array([r[2] for r in rows]), H=np.array([r[4] for r in rows]), folders=np.array(folders),
                Z=orig["Z"], f0=orig["f0"])


if __name__ == "__main__":
    groups = {}
    for tag, stem in D.STEMS.items():
        if isinstance(stem, tuple):
            groups.setdefault(stem[0], []).append((tag, stem[1]))
        else:
            groups.setdefault(stem, []).append((tag, None))
    for stem, items in groups.items():
        if all((OUT / f"{tag}_w{w}.npz").exists() for tag, _ in items for w in WINS):
            continue
        acc = build_file(stem, [rp for _, rp in items])
        for tag, rp in items:
            for w in WINS:
                d = finish(acc[(w, rp)], tag)
                np.savez_compressed(OUT / f"{tag}_w{w}.npz", **d)
                print(tag, w, len(d["t"]), "H share", round(float(d["H"].mean()), 3), flush=True)
