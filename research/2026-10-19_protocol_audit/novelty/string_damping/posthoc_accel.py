"""POST HOC (decided after seeing that the per-stroke bow acceleration spans 0.03-3.4 m/s^2, which makes Guettler-type
establishment limits of the same order as the measured boundary for the fastest attacks).
Question: do strokes fail to be Helmholtz at t = 0.2 s at forces well above the pluck-loss Schelleng minimum F_S also
when the attack was slow? If non-H at F >= 4 F_S occurs only for fast attacks, the order-of-magnitude excess (T1) would be
an establishment effect; if it occurs at slow attacks too, it is not.
Stroke level, onset force, sessions with ramp metadata (B_T2 s2 skipped: different loader). Claude, 2026-10-21."""
import csv, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import damping_test as DT  # noqa: E402
OA = DT.OA

EDGES = [0, 0.15, 1.0, 99]          # slow / medium / fast attack (m/s^2), fixed before computing rates
BANDS = [(4, 8), (8, 16), (16, 64)]  # F_onset / F_S bands


def main():
    res = json.loads((HERE / "results.json").read_text())
    tab = {}
    for s in ["A1", "A2", "A3", "A4", "B1", "C1", "D1", "D2", "A_T2", "B_T2s1", "C_T2"]:
        t = DT.TYPE[s]; S10 = res["damping"][t]["S10"]
        meta = {}
        for r in csv.DictReader(open(OA.RAMP[s])):
            if not r["error"] and r["force_at_onset"] not in ("", "nan"):
                meta[(r["folder"], int(r["trial"]))] = (float(r["force_at_onset"]), float(r["accel"]))
        d = dict(OA.SV.R.load(OA.TAG.get(s, s)))
        m = np.isclose(d["t"], 0.2)
        for i in np.flatnonzero(m):
            k = (str(d["folders"][d["folder"][i]]), int(d["trial"][i]))
            if k not in meta:
                continue
            f, a = meta[k]
            q = f / DT.F_S(DT.Z[t], S10, v=float(d["speed"][i]), b=float(d["beta"][i]))
            ai = int(np.searchsorted(EDGES, a, side="right") - 1)
            for bi, (lo, hi) in enumerate(BANDS):
                if lo <= q < hi:
                    key = (bi, ai, round(float(d["speed"][i]), 3))
                    n, h = tab.get(key, (0, 0)); tab[key] = (n + 1, h + int(d["H"][i]))
    out = []
    names = ["slow <0.15", "medium 0.15-1", "fast >=1"]
    print("non-Helmholtz share at t = 0.2 s, by F_onset/F_S band x attack acceleration (all speeds pooled; per speed in json)")
    for bi, (lo, hi) in enumerate(BANDS):
        cells = []
        for ai in range(3):
            n = sum(v[0] for k, v in tab.items() if k[0] == bi and k[1] == ai); h = sum(v[1] for k, v in tab.items() if k[0] == bi and k[1] == ai)
            cells.append(f"{names[ai]}: {100 * (1 - h / n):.0f}% of {n}" if n else f"{names[ai]}: -")
            out.append(dict(band=[lo, hi], accel=names[ai], n=n, nonH_share=(1 - h / n) if n else None,
                            per_speed={str(k[2]): dict(n=v[0], nonH_share=1 - v[1] / v[0]) for k, v in tab.items() if k[0] == bi and k[1] == ai}))
        print(f"  {lo:>2}-{hi:<2} x F_S | " + " | ".join(cells))
    (HERE / "posthoc_accel.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
