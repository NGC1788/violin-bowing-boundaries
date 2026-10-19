"""Label every stroke as a function of time since the bow reached full speed.

The same classifier as the dataset analysis (regime_map.classify with the
thresholds that produced the A2 labels) is applied to 0.2 s windows slid in
0.05 s steps across the whole constant-speed plateau. Each output row is one
window of one stroke: its time since plateau onset, its label, and the mean
bow force and speed inside it.
"""

import csv
import gzip
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path.home() / "violin-bowing-boundaries"
sys.path.insert(0, str(REPO / "scripts"))
import regime_map as rm  # noqa: E402
import trial_audit  # noqa: E402

OUT = Path.home() / "violin-rne/timeres"
RAW = OUT / "raw/2024-04-19_TypeA_sample2"
SUMMARY = next((REPO / "reports/research_20261013/lockbox_A2_preprocess/2024-04-19_TypeA_sample2/regime_map").glob("*/summary.json"))
RATE = rm.RATE_HZ
WINDOW_S, STEP_S = 0.2, 0.05


def plateau(speed: np.ndarray) -> tuple[int, int]:
    smooth = np.convolve(speed, np.ones(50) / 50, mode="same")
    level = 0.98 * np.max(smooth)
    above = np.flatnonzero(smooth >= level)
    return int(above[0]), int(above[-1])


def process(task):
    condition, trial, meta, applied, ref_f0 = task
    path = RAW / condition / f"whole_{trial}.csv"
    try:
        sig = trial_audit.parse_signal(trial_audit._read_bounded(path, trial_audit.MAX_WHOLE_BYTES))
    except Exception as error:  # report, do not hide
        return [{"condition": condition, "trial": trial, "error": str(error)}]
    on, off = plateau(sig[:, 1])
    width, step = int(WINDOW_S * RATE), int(STEP_S * RATE)
    rows = []
    start = on
    while start + width <= off:
        seg = sig[start:start + width]
        bridge = seg[:, 2]
        periodicity, f0 = rm.dominant_period(bridge)
        period = RATE / f0 if math.isfinite(f0) else math.nan
        shape = rm.flyback_structure(bridge, period, applied["flyback_frac"], applied["hysteresis_sigma"])
        label = rm.classify(float(np.std(bridge)), periodicity, f0, shape["flybacks_per_period"], ref_f0, applied)
        rows.append({"condition": condition, "trial": trial, **meta, "t": round((start - on) / RATE, 3),
                     "label": label, "force": float(np.mean(seg[:, 0])), "speed": float(np.mean(seg[:, 1])),
                     "plateau_s": round((off - on) / RATE, 3), "error": ""})
        start += step
    return rows


def main() -> None:
    summary = json.loads(SUMMARY.read_text())
    applied = summary["thresholds_applied"]
    ref = {c: info["reference_f0_hz"] for c, info in summary["conditions"].items()}
    regimes = {(r["condition"], int(r["trial"])): r
               for r in csv.DictReader(gzip.open(REPO / "research/2026-10-14/lockbox_A2/regimes.csv.gz", "rt"))}
    chosen = [(r["condition"], int(r["trial"])) for r in csv.DictReader(open(OUT / "chosen.csv"))]
    tasks = []
    for condition, trial in chosen:
        r = regimes[(condition, trial)]
        meta = {"beta": float(r["beta"]), "beta_level": int(r["beta_level"]), "force_rank": int(r["force_rank"]),
                "window_label": r["label"]}
        tasks.append((condition, trial, meta, applied, ref[condition]))
    out_rows, done = [], 0
    with ProcessPoolExecutor(max_workers=6) as pool:
        for rows in pool.map(process, tasks, chunksize=8):
            out_rows += rows
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(tasks)}", flush=True)
    fields = ["condition", "trial", "beta", "beta_level", "force_rank", "window_label", "t", "label",
              "force", "speed", "plateau_s", "error"]
    with gzip.open(OUT / "windows.csv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)
    errors = sum(1 for r in out_rows if r.get("error"))
    print(f"창 {len(out_rows)}개 (오류 {errors}) → {OUT / 'windows.csv.gz'}")


if __name__ == "__main__":
    main()
