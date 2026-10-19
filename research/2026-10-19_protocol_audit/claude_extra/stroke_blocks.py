"""Compact per-stroke time series (50 ms blocks) from one IWK archive, for the question
"is the low-force bow-force decay real or a load-cell drift?"  (Claude, 2026-10-18)

One streaming pass (`7zz x -so`, nothing written but the output). Per stroke and 50 ms block:
  speed, position (integrated bow travel), force_dc (median of column 0), force_ac (rms of column 0 after
  removing a 5 ms moving median: dynamic content of the normal-force signal), force_ac_lo (rms of column 0
  minus its 50 ms mean, i.e. everything faster than the block), bridge_dc, nut_dc (medians of columns 2, 3),
  bridge_p2p (5-95 % range of column 2), bridge_edge (99th percentile of |d column2 / dt| in N/s; with
  bridge_p2p gives an edge time = p2p / edge, a corner-rounding proxy), bridge_ac_force_coh (correlation of
  the force-channel fluctuation with the bridge-force fluctuation inside the block).
Usage: python -I stroke_blocks.py <archive.7z> <out.npz>
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

SEVEN = Path.home() / "violin-bowing-boundaries/.local-tools/7zip-26.03/7zz"
RATE = 50_000
BLOCK = 2500  # 50 ms
FIELDS = ["speed", "position", "force_dc", "force_ac", "force_ac_lo", "bridge_dc", "nut_dc", "bridge_p2p",
          "bridge_edge", "coh"]
MAXB = 160    # 8 s


def listing(archive):
    out = subprocess.run([str(SEVEN), "l", "-slt", str(archive)], capture_output=True, text=True, check=True).stdout
    items, path = [], None
    for line in out.splitlines():
        if line.startswith("Path = "):
            path = line[7:].strip()
        elif line.startswith("Size = ") and path is not None:
            if re.search(r"/(whole|beta)_\d+\.csv$", path):
                items.append((path, int(line[7:].strip())))
            path = None
    return items


def moving_median_5ms(x):
    k = 250
    n = x.size // k * k
    med = np.median(x[:n].reshape(-1, k), axis=1)
    return np.repeat(med, k), n


def blocks(sig):
    nb = min(sig.shape[0] // BLOCK, MAXB)
    out = np.full((len(FIELDS), MAXB), np.nan, dtype=np.float32)
    speed = sig[:, 1]
    pos = np.cumsum(speed) / RATE
    f = sig[:, 0]
    base, n5 = moving_median_5ms(f)
    for b in range(nb):
        s = slice(b * BLOCK, (b + 1) * BLOCK)
        fb, br, nu = f[s], sig[s, 2], sig[s, 3]
        lo = b * BLOCK
        hi = min((b + 1) * BLOCK, n5)
        fac = float(np.std(f[lo:hi] - base[lo:hi])) if hi > lo else np.nan
        q5, q95 = np.percentile(br, [5, 95])
        edge = float(np.percentile(np.abs(np.diff(br)), 99)) * RATE
        fz, bz = fb - fb.mean(), br - br.mean()
        den = np.sqrt(np.sum(fz * fz) * np.sum(bz * bz))
        coh = float(np.sum(fz * bz) / den) if den > 0 else np.nan
        out[:, b] = [np.mean(speed[s]), pos[s][-1], np.median(fb), fac, np.std(fz), np.median(br), np.median(nu),
                     q95 - q5, edge, coh]
    return out


def main():
    archive, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    items = listing(archive)
    proc = subprocess.Popen([str(SEVEN), "x", "-so", str(archive), "-ir!*/whole_*.csv", "-ir!*/beta_*.csv"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1 << 24)
    keys, arrays, betas = [], [], {}
    for path, size in items:
        raw = proc.stdout.read(size) if size else b""
        if len(raw) != size:
            raise RuntimeError(f"stream ended early at {path}")
        m = re.search(r"(.*)/(whole|beta)_(\d+)\.csv$", path)
        folder, kind, trial = m.group(1).split("/")[-1], m.group(2), int(m.group(3))
        if kind == "beta":
            try:
                betas[(folder, trial)] = float(re.split(r"[,\s;]+", raw.decode().strip())[0])
            except ValueError:
                betas[(folder, trial)] = np.nan
            continue
        try:
            data = np.fromstring(raw.replace(b"\r", b"").replace(b";", b",").replace(b"\n", b","), sep=",")
            sig = data[: data.size // 4 * 4].reshape(-1, 4)
            arrays.append(blocks(sig))
        except Exception:
            arrays.append(np.full((len(FIELDS), MAXB), np.nan, dtype=np.float32))
        keys.append((folder, trial))
        if len(keys) % 500 == 0:
            print(f"{len(keys)} strokes", flush=True)
    proc.wait()
    np.savez_compressed(out_path, data=np.stack(arrays), fields=np.array(FIELDS),
                        folder=np.array([k[0] for k in keys]), trial=np.array([k[1] for k in keys]),
                        beta=np.array([betas.get(k, np.nan) for k in keys]), block_s=BLOCK / RATE)
    print(f"done: {len(keys)} strokes -> {out_path}")


if __name__ == "__main__":
    main()
