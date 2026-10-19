"""code_audit: measured vs nominal force per specimen (nominal reconstructed from trial rank within a column: 0.1*40^(rank/49))."""
import sys, numpy as np
sys.path.insert(0, "/Users/jeongjihoon/violin-rne/final")
from data import load, STEMS
import compare as C
for tag in STEMS:
    d = load(tag)
    m = np.isclose(d["t"], 0.2)
    idx = np.flatnonzero(m)
    lr = {r: [] for r in range(50)}
    mono = 0; ncol = 0
    for (fo, co) in set(zip(d["folder"][idx], d["column"][idx])):
        j = idx[(d["folder"][idx] == fo) & (d["column"][idx] == co)]
        j = j[np.argsort(d["trial"][j])]
        ncol += 1
        nom = 0.1 * 40 ** (np.arange(50) / 49)
        for r in range(50):
            lr[r].append(np.log(max(d["force"][j[r]], 1e-3) / nom[r]))
    # where are the lower boundaries (finite) in measured force?
    rows = C.intervals(d, 0.2)
    mids = [np.exp(0.5 * sum(r["lower"])) for r in rows if np.isfinite(r["lower"][0]) and np.isfinite(r["lower"][1])]
    print(f"{tag:7s} median log(meas/nom) rank0 {np.median(lr[0]):+.2f} rank5 {np.median(lr[5]):+.2f} rank10 {np.median(lr[10]):+.2f} rank20 {np.median(lr[20]):+.2f} rank49 {np.median(lr[49]):+.2f} | median lower-boundary midpoint {np.median(mids):.3f} N")
