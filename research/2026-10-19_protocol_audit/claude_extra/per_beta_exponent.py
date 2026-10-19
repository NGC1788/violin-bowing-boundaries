"""Per-beta speed exponent of the E and M boundaries from a sim_contact2 log table (Claude, 2026-10-18).
Usage: python per_beta_exponent.py <log> [block_index]"""
import re, sys
import numpy as np
SPEEDS = [0.05, 0.075, 0.10, 0.15, 0.20]; BETAS = [0.06, 0.08, 0.10, 0.12, 0.15, 0.18]
text = open(sys.argv[1]).read()
blocks = [b for b in re.split(r"(?=^rate )", text, flags=re.M) if b.startswith("rate")]
block = blocks[int(sys.argv[2]) if len(sys.argv) > 2 else -1]
print(block.splitlines()[0])
m = re.search(r"β \[([^\]]*)\]", block.splitlines()[0])
if m:
    BETAS = [float(x) for x in m.group(1).split(",")]
for tag in ("E", "M"):
    sec = block.split(f"[{tag} ")[1]
    rows = re.findall(r"v=([\d.]+)\s*:(.*)", sec)[:5]
    T = np.array([[float(c) if re.fullmatch(r"[\d.]+", c) else np.nan for c in r[1].split()] for r in rows])
    out = []
    for j, b in enumerate(BETAS):
        ok = np.isfinite(T[:, j])
        if ok.sum() >= 3:
            s = np.polyfit(np.log(np.array(SPEEDS)[ok]), np.log(T[ok, j]), 1)[0]
            out.append(f"β={b}: {s:+.2f} (n{ok.sum()})")
        else:
            out.append(f"β={b}: -")
    print(f"  {tag} 속도지수(β별):  " + "   ".join(out))
