"""Stream the B1 raw archive member by member (no extraction to disk) and compute, for every stroke, the features of the
t = 0.2 s, 0.2 s window exactly as timeres/replicate.windows_of, plus the flyback duration (corner width).
Plan: PLAN.md. Claude, 2026-10-21."""
import csv, json, math, re, subprocess, sys, time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

H = Path.home(); HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(H / "violin-rne/astra_20261019"))
import stream_features as SF  # noqa: E402  (7zz path, inventory, read_exact)

ARCH = H / "violin-rne/data/raw/zenodo/17749111/2024-04-10_TypeB_sample1.7z"
OUT = HERE / "b1_flyback.csv"
FIELDS = ["folder", "trial", "beta", "t", "plateau_s", "force", "speed", "amplitude_std", "periodicity", "f0_hz",
          "flybacks_per_period", "second_flyback_ratio", "flyback_height", "flyback_ms", "error"]


def work(task):
    folder, trial, beta, raw = task
    sys.path.insert(0, str(H / "violin-bowing-boundaries/scripts"))
    import numpy as np
    import regime_map as rm
    import trial_audit
    base = {"folder": folder, "trial": trial, "beta": beta}
    try:
        sig = trial_audit.parse_signal(raw)
    except Exception as e:  # noqa: BLE001
        return {**base, "error": f"read: {e}"}
    rate = rm.RATE_HZ
    smooth = np.convolve(sig[:, 1], np.ones(50) / 50, mode="same")
    if not np.isfinite(smooth).all() or smooth.max() <= 0:
        return {**base, "error": "no positive speed"}
    above = np.flatnonzero(smooth >= 0.98 * smooth.max())
    on, off = int(above[0]), int(above[-1])
    width, step = int(0.2 * rate), int(0.05 * rate)
    start = on + 4 * step
    if start + width > off:
        return {**base, "error": "no t=0.2 window"}
    seg = sig[start:start + width]; bridge = seg[:, 2]
    periodicity, f0 = rm.dominant_period(bridge)
    period = rate / f0 if math.isfinite(f0) else math.nan
    shape = rm.flyback_structure(bridge, period, 0.5, 3.0)
    return {**base, "t": round((start - on) / rate, 3), "plateau_s": round((off - on) / rate, 3),
            "force": float(np.mean(seg[:, 0])), "speed": float(np.mean(seg[:, 1])), "amplitude_std": float(np.std(bridge)),
            "periodicity": periodicity, "f0_hz": f0, "flybacks_per_period": shape.get("flybacks_per_period"),
            "second_flyback_ratio": shape.get("second_flyback_ratio"), "flyback_height": shape.get("flyback_height"),
            "flyback_ms": shape.get("flyback_ms"), "error": ""}


def main():
    t0 = time.time()
    text = subprocess.run([str(SF.SEVEN), "l", "-slt", str(ARCH)], capture_output=True, text=True, check=True).stdout
    items = SF.inventory(text)
    n_whole = sum(1 for it in items if it[3] == "whole")
    print(f"inventory: {len(items)} members, {n_whole} strokes", flush=True)
    cmd = [str(SF.SEVEN), "x", "-so", str(ARCH), "-ir!*/whole_*.csv", "-ir!*/beta_*.csv"]
    betas, held, pending, done = {}, {}, deque(), 0
    with OUT.open("w", newline="") as fh, ProcessPoolExecutor(max_workers=6) as pool, open(HERE / "seven.log", "wb") as err:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore"); w.writeheader()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, bufsize=1 << 20)

        def finish_one():
            nonlocal done
            w.writerow(pending.popleft().result()); done += 1
            if done % 500 == 0:
                print(f"{done}/{n_whole} strokes, {time.time() - t0:.0f} s", flush=True)

        def submit(key, raw):
            if len(pending) >= 12:
                finish_one()
            pending.append(pool.submit(work, (key[0], key[1], betas[key], raw)))
        try:
            for _, size, folder, kind, trial in items:
                raw = SF.read_exact(proc.stdout, size); key = (folder, trial)
                if kind == "beta":
                    betas[key] = float(re.split(r"[,\s;]+", raw.decode().strip())[0])
                    if key in held:
                        submit(key, held.pop(key))
                    continue
                if key not in betas:
                    held[key] = raw; continue
                submit(key, raw)
            while pending:
                finish_one()
            if proc.wait() != 0:
                raise RuntimeError("7zz failed; see seven.log")
        finally:
            if proc.poll() is None:
                proc.terminate(); proc.wait()
    print(f"done: {done} strokes in {time.time() - t0:.0f} s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
