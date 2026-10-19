"""One streaming pass over the protected A2 archive (read-only, `7zz x -so`, nothing extracted to disk).
Per stroke it keeps (Claude, skeptic_force, 2026-10-18):
  * timestamp text (recording clock)
  * attack window: bridge force at 50 kHz from onset-20 ms to onset+200 ms (float32), force at 1 kHz
  * per 50 ms block (100 ms Hann window, zero-padded FFT): f0 from the bridge harmonic sum, |B_n| (bridge) and
    |F_n| (bow-force channel) at n*f0 for n=1..24, and cos(phase(B_n)-phase(F_n)) for n=1..12, plus the force DC.
Usage: python -I stream_raw.py <archive.7z> <outdir>
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

SEVEN = Path.home() / "violin-bowing-boundaries/.local-tools/7zip-26.03/7zz"
RATE = 50_000
HOP = 2500
WIN = 5000
NFFT = 16384
NH = 24
NC = 12
MAXB = 160
PRE, POST = 1000, 10000
HANN = np.hanning(WIN).astype(np.float64)
FREQ = np.fft.rfftfreq(NFFT, 1 / RATE)
F0GRID = np.arange(85.0, 101.0, 0.05)


def listing(archive):
    out = subprocess.run([str(SEVEN), "l", "-slt", str(archive)], capture_output=True, text=True, check=True).stdout
    items, path = [], None
    for line in out.splitlines():
        if line.startswith("Path = "):
            path = line[7:].strip()
        elif line.startswith("Size = ") and path is not None:
            if re.search(r"/(whole|beta|timestamp)_\d+\.csv$", path):
                items.append((path, int(line[7:].strip())))
            path = None
    return items


def harm(spec, f0, n):
    """magnitude and complex value at n*f0 (nearest bin, parabolic magnitude interpolation)."""
    k = n * f0 / (RATE / NFFT)
    k0 = int(round(k))
    if k0 < 1 or k0 >= spec.size - 1:
        return np.nan, np.nan + 0j
    a, b, c = np.abs(spec[k0 - 1:k0 + 2])
    den = a - 2 * b + c
    p = 0.5 * (a - c) / den if den != 0 else 0.0
    mag = b - 0.25 * (a - c) * p
    return mag, spec[k0]


def analyse(sig):
    n = sig.shape[0]
    speed = sig[:, 1]
    vmax = np.max(speed)
    mv = np.flatnonzero(speed > 0.02 * vmax)
    onset = int(mv[0]) if mv.size else -1
    att_b = np.full(PRE + POST, np.nan, np.float32)
    att_f = np.full((PRE + POST) // 50, np.nan, np.float32)
    if onset >= 0:
        lo, hi = onset - PRE, onset + POST
        seg = np.full((PRE + POST, 4), np.nan)
        a0, a1 = max(lo, 0), min(hi, n)
        seg[a0 - lo:a1 - lo] = sig[a0:a1]
        att_b[:] = seg[:, 2]
        att_f[:] = np.nanmean(seg[:, 0].reshape(-1, 50), axis=1)
    nb = min(n // HOP, MAXB)
    out = np.full((1 + NH + NH + NC + 1, MAXB), np.nan, np.float32)
    for b in range(nb):
        c = b * HOP + HOP // 2
        lo, hi = c - WIN // 2, c + WIN // 2
        if lo < 0 or hi > n:
            continue
        br = sig[lo:hi, 2] - sig[lo:hi, 2].mean()
        fo = sig[lo:hi, 0] - sig[lo:hi, 0].mean()
        SB = np.fft.rfft(br * HANN, NFFT)
        SF = np.fft.rfft(fo * HANN, NFFT)
        mag = np.abs(SB)
        # f0 by harmonic sum (n=1..8) of the bridge magnitude spectrum
        idx = np.rint(np.outer(F0GRID, np.arange(1, 9)) / (RATE / NFFT)).astype(int)
        f0 = F0GRID[np.argmax(mag[idx].sum(axis=1))]
        out[0, b] = f0
        for h in range(1, NH + 1):
            mb, cb = harm(SB, f0, h)
            mf, cf = harm(SF, f0, h)
            out[h, b] = mb
            out[NH + h, b] = mf
            if h <= NC and np.isfinite(mb) and abs(cb) > 0 and abs(cf) > 0:
                out[2 * NH + h, b] = np.real(cb * np.conj(cf)) / (abs(cb) * abs(cf))
        out[-1, b] = np.median(sig[b * HOP:(b + 1) * HOP, 0])
    return onset, att_b, att_f, out


def main():
    archive, outdir = Path(sys.argv[1]), Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    items = listing(archive)
    proc = subprocess.Popen([str(SEVEN), "x", "-so", str(archive), "-ir!*/whole_*.csv", "-ir!*/beta_*.csv",
                             "-ir!*/timestamp_*.csv"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1 << 24)
    cur, store, stamps, betas = None, None, {}, {}

    def flush(folder, st):
        np.savez_compressed(outdir / f"raw_{folder}.npz", trial=np.array(st["trial"]), onset=np.array(st["onset"]),
                            att_b=np.stack(st["att_b"]), att_f=np.stack(st["att_f"]), harm=np.stack(st["harm"]),
                            rows=np.array(["f0"] + [f"B{h}" for h in range(1, NH + 1)] + [f"F{h}" for h in range(1, NH + 1)]
                                          + [f"cos{h}" for h in range(1, NC + 1)] + ["force_dc"]))
        print(f"saved {folder}: {len(st['trial'])} strokes", flush=True)

    count = 0
    for path, size in items:
        raw = proc.stdout.read(size) if size else b""
        if len(raw) != size:
            raise RuntimeError(f"stream ended early at {path}")
        m = re.search(r"(.*)/(whole|beta|timestamp)_(\d+)\.csv$", path)
        folder, kind, trial = m.group(1).split("/")[-1], m.group(2), int(m.group(3))
        if kind == "timestamp":
            stamps[(folder, trial)] = raw.decode(errors="replace").strip()
            continue
        if kind == "beta":
            betas[(folder, trial)] = raw.decode(errors="replace").strip()
            continue
        if folder != cur:
            if store is not None:
                flush(cur, store)
            cur, store = folder, {"trial": [], "onset": [], "att_b": [], "att_f": [], "harm": []}
        try:
            data = np.fromstring(raw.replace(b"\r", b"").replace(b";", b",").replace(b"\n", b","), sep=",")
            sig = data[: data.size // 4 * 4].reshape(-1, 4)
            onset, ab, af, h = analyse(sig)
        except Exception as exc:  # keep going, mark as missing
            print(f"error {path}: {exc}", flush=True)
            onset, ab, af = -1, np.full(PRE + POST, np.nan, np.float32), np.full((PRE + POST) // 50, np.nan, np.float32)
            h = np.full((1 + 2 * NH + NC + 1, MAXB), np.nan, np.float32)
        store["trial"].append(trial); store["onset"].append(onset)
        store["att_b"].append(ab); store["att_f"].append(af); store["harm"].append(h)
        count += 1
        if count % 250 == 0:
            print(f"{count} strokes ({folder})", flush=True)
    proc.wait()
    if store is not None:
        flush(cur, store)
    with open(outdir / "timestamps.csv", "w") as fh:
        fh.write("folder,trial,timestamp,beta\n")
        for (f, t), s in sorted(stamps.items()):
            fh.write(f"{f},{t},{s.replace(',', ';')},{betas.get((f, t), '')}\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
