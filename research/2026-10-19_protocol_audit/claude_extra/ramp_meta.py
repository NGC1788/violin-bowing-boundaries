"""Per-stroke bow-motion protocol metadata (ramp, acceleration, plateau) from one IWK archive (Claude, 2026-10-18).

Why: the dataset paper does not tabulate the attack profile. Spot checks on A2 showed that the bow
acceleration before the plateau differs by stroke (0.026 to 1.2 m/s^2), and Guettler-type attack
theory says acceleration moves the establishment boundary. This records it for every stroke.
Streams the archive (`7zz x -so`, one decompression pass, no files on disk) and cuts the stream by
the uncompressed sizes from the listing (7-Zip writes items of a solid block in index order).
Usage: python -I ramp_meta.py <archive.7z> <out.csv>
Columns: folder, trial, beta, v_max, onset_s, plateau_on_s, plateau_off_s, total_s, ramp_s, accel,
         ramp_linearity (R^2 of a line fitted to the ramp), force_plateau, force_at_onset,
         force_cv_plateau, decel_s, error
"""
import csv
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

SEVEN = Path.home() / "violin-bowing-boundaries/.local-tools/7zip-26.03/7zz"
RATE = 50_000
FIELDS = ["folder", "trial", "beta", "v_max", "onset_s", "plateau_on_s", "plateau_off_s", "total_s", "ramp_s",
          "accel", "ramp_linearity", "force_plateau", "force_at_onset", "force_cv_plateau", "decel_s", "error"]


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


def stroke(raw: bytes):
    data = np.fromstring(raw.replace(b"\r", b"").replace(b";", b",").replace(b"\n", b","), sep=",")
    sig = data[: data.size // 4 * 4].reshape(-1, 4)
    speed = np.convolve(sig[:, 1], np.ones(50) / 50, mode="same")
    vmax = float(speed.max())
    if not np.isfinite(vmax) or vmax <= 0:
        return {"error": "no positive speed"}
    on = int(np.argmax(speed > 0.02 * vmax))
    plateau = np.flatnonzero(speed >= 0.98 * vmax)
    full, off = int(plateau[0]), int(plateau[-1])
    after = np.flatnonzero(speed[off:] < 0.02 * vmax)
    stop = off + int(after[0]) if after.size else speed.size - 1
    ramp = speed[on:full + 1]
    t = np.arange(ramp.size) / RATE
    r2 = float(np.corrcoef(t, ramp)[0, 1] ** 2) if ramp.size > 5 else float("nan")
    force = sig[:, 0]
    fp = force[full:off + 1]
    return {"v_max": vmax, "onset_s": on / RATE, "plateau_on_s": full / RATE, "plateau_off_s": off / RATE,
            "total_s": sig.shape[0] / RATE, "ramp_s": (full - on) / RATE,
            "accel": vmax / max((full - on) / RATE, 1 / RATE), "ramp_linearity": r2,
            "force_plateau": float(np.median(fp)), "force_at_onset": float(np.median(force[max(on - 500, 0):on + 1])),
            "force_cv_plateau": float(np.std(fp) / max(abs(np.mean(fp)), 1e-9)), "decel_s": (stop - off) / RATE,
            "error": ""}


def main():
    archive, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    items = listing(archive)
    proc = subprocess.Popen([str(SEVEN), "x", "-so", str(archive), "-ir!*/whole_*.csv", "-ir!*/beta_*.csv"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1 << 24)
    rows, betas = {}, {}
    for i, (path, size) in enumerate(items):
        raw = proc.stdout.read(size) if size else b""
        if len(raw) != size:
            raise RuntimeError(f"stream ended early at {path} ({len(raw)}/{size} bytes)")
        m = re.search(r"(.*)/(whole|beta)_(\d+)\.csv$", path)
        folder, kind, trial = m.group(1), m.group(2), int(m.group(3))
        if kind == "beta":
            try:
                betas[(folder, trial)] = float(re.split(r"[,\s;]+", raw.decode().strip())[0])
            except ValueError:
                betas[(folder, trial)] = float("nan")
            continue
        try:
            rows[(folder, trial)] = stroke(raw)
        except Exception as error:  # keep going, record the failure
            rows[(folder, trial)] = {"error": f"parse: {error}"}
        if len(rows) % 500 == 0:
            print(f"{len(rows)} strokes ({i + 1}/{len(items)} items)", flush=True)
    proc.wait()
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, FIELDS)
        writer.writeheader()
        for (folder, trial), row in sorted(rows.items()):
            writer.writerow({"folder": folder, "trial": trial, "beta": betas.get((folder, trial), float("nan")), **row})
    print(f"done: {len(rows)} strokes -> {out_path}")


if __name__ == "__main__":
    main()
