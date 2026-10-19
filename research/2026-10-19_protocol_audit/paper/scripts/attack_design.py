"""Attack (bow acceleration) design in IWK A2: share of strokes on the constant-ramp-distance floor and a
power-law fit of the remaining accelerations vs onset force and beta. Reads claude_extra/ramp_meta_A2.csv only.
(paper draft support, 2026-10-19)"""
import csv
from pathlib import Path
import numpy as np

META = Path.home() / "violin-rne/claude_extra/ramp_meta_A2.csv"
rows = [r for r in csv.DictReader(META.open()) if not r["error"]]
print(f"strokes {len(rows)}")
for vf in ("v1", "v2", "v3"):
    rs = [r for r in rows if r["folder"].endswith(vf)]
    v = np.median([float(r["v_max"]) for r in rs])
    a = np.array([float(r["accel"]) for r in rs]); F = np.array([float(r["force_at_onset"]) for r in rs])
    b = np.array([float(r["beta"]) for r in rs]); ramp = np.array([float(r["ramp_s"]) for r in rs])
    dist = 0.5 * a * ramp ** 2
    floor = v ** 2 / (2 * 0.048)
    on_floor = np.abs(a / floor - 1) < 0.03
    m = (~on_floor) & (F > 0) & (a > 0)
    X = np.c_[np.ones(m.sum()), np.log(F[m]), np.log(b[m])]
    y = np.log(a[m]); coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - np.sum((y - X @ coef) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"{vf}: v {v:.3f} m/s, n {len(rs)}, floor a=v^2/(2*0.048 m)={floor:.3f} m/s^2, "
          f"on floor {on_floor.mean()*100:.0f}% (ramp distance median {np.median(dist[on_floor])*100:.2f} cm), "
          f"off-floor fit log a = {coef[0]:+.2f} {coef[1]:+.2f} log F_onset {coef[2]:+.2f} log beta (R2 {r2:.2f}, n {m.sum()}), "
          f"max a {a.max():.1f} m/s^2")
