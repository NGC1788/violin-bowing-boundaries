"""All 6,000 A2 strokes, one speed folder at a time: extract -> label windows -> delete CSVs."""
import csv, gzip, json, math, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

REPO = Path.home() / "violin-bowing-boundaries"
OUT = Path.home() / "violin-rne/timeres"
ARCHIVE = REPO / "data/raw/zenodo/17782542/2024-04-19_TypeA_sample2.7z"
SEVEN = REPO / ".local-tools/7zip-26.03/7zz"
SUMMARY = next((REPO / "reports/research_20261013/lockbox_A2_preprocess/2024-04-19_TypeA_sample2/regime_map").glob("*/summary.json"))
WINDOW_S, STEP_S = 0.2, 0.05


def process(task):
    sys.path.insert(0, str(REPO / "scripts"))
    import regime_map as rm, trial_audit
    path, condition, trial, meta, applied, ref_f0 = task
    try:
        sig = trial_audit.parse_signal(trial_audit._read_bounded(Path(path), trial_audit.MAX_WHOLE_BYTES))
    except Exception as error:
        return [{"condition": condition, "trial": trial, **meta, "error": str(error)}]
    smooth = np.convolve(sig[:, 1], np.ones(50) / 50, mode="same")
    above = np.flatnonzero(smooth >= 0.98 * smooth.max())
    on, off = int(above[0]), int(above[-1])
    rate = rm.RATE_HZ
    width, step, rows, start = int(WINDOW_S * rate), int(STEP_S * rate), [], on
    while start + width <= off:
        seg = sig[start:start + width]; bridge = seg[:, 2]
        per, f0 = rm.dominant_period(bridge)
        shape = rm.flyback_structure(bridge, rate / f0 if math.isfinite(f0) else math.nan, applied["flyback_frac"], applied["hysteresis_sigma"])
        label = rm.classify(float(np.std(bridge)), per, f0, shape["flybacks_per_period"], ref_f0, applied)
        rows.append({"condition": condition, "trial": trial, **meta, "t": round((start - on) / rate, 3), "label": label,
                     "force": float(np.mean(seg[:, 0])), "speed": float(np.mean(seg[:, 1])),
                     "plateau_s": round((off - on) / rate, 3), "error": ""})
        start += step
    return rows


def main():
    summary = json.loads(SUMMARY.read_text()); applied = summary["thresholds_applied"]
    ref = {c: i["reference_f0_hz"] for c, i in summary["conditions"].items()}
    regimes = list(csv.DictReader(gzip.open(REPO / "research/2026-10-14/lockbox_A2/regimes.csv.gz", "rt")))
    fields = ["condition", "trial", "beta", "beta_level", "force_rank", "window_label", "t", "label", "force", "speed", "plateau_s", "error"]
    out = gzip.open(OUT / "windows_full.csv.gz", "wt", newline="")
    writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore"); writer.writeheader()
    for condition in ("v1", "v2", "v3"):
        rows = [r for r in regimes if r["condition"] == condition]
        raw = OUT / f"raw_{condition}"
        listfile = OUT / f"members_{condition}.txt"
        listfile.write_text("\n".join(f"2024-04-19_TypeA_sample2/{condition}/whole_{r['trial']}.csv" for r in rows) + "\n")
        print(f"[{condition}] 추출 {len(rows)}개 ...", flush=True)
        subprocess.run([str(SEVEN), "x", str(ARCHIVE), f"-o{raw}", f"@{listfile}", "-y", "-bso0", "-bsp0"], check=True)
        tasks = [(str(raw / "2024-04-19_TypeA_sample2" / condition / f"whole_{r['trial']}.csv"), condition, int(r["trial"]),
                  {"beta": float(r["beta"]), "beta_level": int(r["beta_level"]), "force_rank": int(r["force_rank"]), "window_label": r["label"]},
                  applied, ref[condition]) for r in rows]
        n = 0
        with ProcessPoolExecutor(max_workers=7) as pool:
            for result in pool.map(process, tasks, chunksize=8):
                writer.writerows(result); n += len(result)
        out.flush()
        subprocess.run(["rm", "-rf", str(raw)], check=True)
        print(f"[{condition}] 창 {n}개 판정, 원시 CSV 삭제", flush=True)
    out.close()
    print("전체 완료", flush=True)


if __name__ == "__main__":
    main()
