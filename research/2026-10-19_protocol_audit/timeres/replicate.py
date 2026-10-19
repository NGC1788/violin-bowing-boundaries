"""Time-resolved features for every stroke of every archive in the public collection.

Resumable, unattended pipeline. For each archive in QUEUE:
  download (resumable, MD5-checked) -> for each speed folder: extract only that
  folder -> slide 0.1/0.2/0.4 s windows over the whole constant-speed plateau ->
  store classifier *features* per window (not only a label, so any labelling rule
  can be re-applied later without the raw data) -> delete the extracted CSVs ->
  after all folders, delete the downloaded archive (it is re-downloadable).

State lives in STATE (JSON); rerunning skips finished folders/archives.
Outputs: features/<archive>.csv.gz, one row per (stroke, window length, window start).

Run:  nohup caffeinate -i ~/violin-bowing-boundaries/.venv-mac/bin/python replicate.py >> replicate.log 2>&1 &
"""

import csv
import gzip
import json
import math
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path.home() / "violin-bowing-boundaries"
HERE = Path.home() / "violin-rne/timeres"
DATA = Path.home() / "violin-rne/data"            # downloads go here, outside the repository
FEATURES = HERE / "features"
STATE = HERE / "replicate_state.json"
SEVEN = REPO / ".local-tools/7zip-26.03/7zz"
PY = REPO / ".venv-mac/bin/python"
RECORD = {1: "17749110", 2: "17782542", 3: "17782552", 4: "17782565",
          5: "17782538", 6: "17795326", 7: "17822016", 8: "17822037"}
# Priority: the A2 archive already on disk, the four string types, replicates of A and D,
# the tension-2 set (string impedance changes -> Z vs Z^2 test), then high force range and the rest.
QUEUE = [
    (2, "2024-04-19_TypeA_sample2.7z"),
    (1, "2024-03-25_TypeA_sample1.7z"), (1, "2024-04-10_TypeB_sample1.7z"), (1, "2024-04-12_TypeC_sample1.7z"),
    # Part 6 (setup variability: two repeated diagrams) moved up on 2026-10-18 at the user's request: each speed
    # folder is its own recording session, so session-to-session variation must be sized before comparing specimens.
    (6, "2024-10-23_variability_setup.7z"),
    (2, "2024-04-15_TypeD_sample1.7z"),
    # tension-2 set next: string impedance changes -> decisive Z vs Z^2 test between Beigi and Schelleng
    (4, "2024-06-19_TypeA_T2_sample1.7z"), (4, "2024-06-13_TypeB_T2_sample1.7z"), (4, "2024-06-11_TypeC_T2_sample1.7z"),
    (5, "2024-06-17_TypeD_T2_sample1.7z"),
    (2, "2024-04-22_TypeA_sample3.7z"), (3, "2024-04-24_TypeA_sample4.7z"), (3, "2024-04-26_TypeD_sample2.7z"),
    (5, "2024-07-03_TypeB_T2_sample2.7z"),
    (7, "HighBowForceRange_1.7z"), (8, "HighBowForceRange_2.7z"),
    (3, "2024-03-25_TypeA_sample1_coldwarm.7z"),
]
# Archives that must never be deleted by this script (someone else downloaded them).
KEEP = {"2024-04-19_TypeA_sample2.7z": REPO / "data/raw/zenodo/17782542/2024-04-19_TypeA_sample2.7z"}
WINDOWS_S = (0.1, 0.2, 0.4)
STEP_S = 0.05
MIN_FREE_GB = 60
FIELDS = ["folder", "trial", "beta", "window_s", "t", "plateau_s", "force", "speed",
          "amplitude_std", "periodicity", "f0_hz", "flybacks_per_period", "second_flyback_ratio",
          "flyback_height", "error"]


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(state: dict) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(STATE)


def free_gb() -> float:
    return shutil.disk_usage(Path.home()).free / 1e9


def archive_path(part: int, name: str) -> Path:
    return KEEP.get(name, DATA / "raw/zenodo" / RECORD[part] / name)


def download(part: int, name: str) -> Path:
    path = archive_path(part, name)
    if path.exists() and not Path(str(path) + ".part").exists():
        return path
    log(f"다운로드 시작 {name} (record {RECORD[part]})")
    subprocess.run([str(PY), str(REPO / "scripts/zenodo_catalog.py"), "download", "--record", RECORD[part],
                    "--file", name, "--max-gib", "40", "--connections", "16", "--retries", "12",
                    "--data-dir", str(DATA)], check=True, cwd=REPO)
    if not path.exists():
        found = list((DATA / "raw/zenodo").rglob(name))
        if not found:
            raise FileNotFoundError(f"download finished but {name} not found under {DATA}")
        path = found[0]
    return path


def folders_of(archive: Path) -> dict:
    """folder -> sorted trial numbers, from the archive listing."""
    listing = subprocess.run([str(SEVEN), "l", "-slt", str(archive)], capture_output=True, text=True, check=True).stdout
    folders: dict = {}
    for line in listing.splitlines():
        if not line.startswith("Path = "):
            continue
        member = line[7:].strip()
        m = re.search(r"(.*)/whole_(\d+)\.csv$", member)
        if m:
            folders.setdefault(m.group(1), []).append(int(m.group(2)))
    return {k: sorted(v) for k, v in folders.items()}


def windows_of(task):
    """Features of every window of one stroke (all window lengths)."""
    sys.path.insert(0, str(REPO / "scripts"))
    import regime_map as rm
    import trial_audit
    csv_path, beta_path, folder, trial = task
    base = {"folder": folder, "trial": trial}
    try:
        beta_text = Path(beta_path).read_text().strip().replace("\r", "")
        beta = float(re.split(r"[,\s;]+", beta_text)[0])
        sig = trial_audit.parse_signal(trial_audit._read_bounded(Path(csv_path), trial_audit.MAX_WHOLE_BYTES))
    except Exception as error:
        return [{**base, "error": f"read: {error}"}]
    rate = rm.RATE_HZ
    smooth = np.convolve(sig[:, 1], np.ones(50) / 50, mode="same")
    if not np.isfinite(smooth).all() or smooth.max() <= 0:
        return [{**base, "beta": beta, "error": "no positive speed"}]
    above = np.flatnonzero(smooth >= 0.98 * smooth.max())
    on, off = int(above[0]), int(above[-1])
    rows = []
    for length in WINDOWS_S:
        width, step = int(length * rate), int(STEP_S * rate)
        start = on
        while start + width <= off:
            seg = sig[start:start + width]
            bridge = seg[:, 2]
            periodicity, f0 = rm.dominant_period(bridge)
            period = rate / f0 if math.isfinite(f0) else math.nan
            shape = rm.flyback_structure(bridge, period, 0.5, 3.0)
            rows.append({**base, "beta": beta, "window_s": length, "t": round((start - on) / rate, 3),
                         "plateau_s": round((off - on) / rate, 3), "force": float(np.mean(seg[:, 0])),
                         "speed": float(np.mean(seg[:, 1])), "amplitude_std": float(np.std(bridge)),
                         "periodicity": periodicity, "f0_hz": f0,
                         "flybacks_per_period": shape.get("flybacks_per_period"),
                         "second_flyback_ratio": shape.get("second_flyback_ratio"),
                         "flyback_height": shape.get("flyback_height"), "error": ""})
            start += step
    return rows


def process_folder(archive: Path, name: str, folder: str, trials: list[int], writer) -> int:
    scratch = DATA / "extract" / name.replace(".7z", "")
    if scratch.exists():
        shutil.rmtree(scratch)
    log(f"  추출 {folder} ({len(trials)} 긋기) · 여유 {free_gb():.0f} GB")
    subprocess.run([str(SEVEN), "x", str(archive), f"-o{scratch}", f"-ir!{folder}/whole_*.csv",
                    f"-ir!{folder}/beta_*.csv", "-y", "-bso0", "-bsp0"], check=True)
    tasks = [(str(scratch / folder / f"whole_{n}.csv"), str(scratch / folder / f"beta_{n}.csv"), folder, n)
             for n in trials]
    count = 0
    with ProcessPoolExecutor(max_workers=7) as pool:
        for rows in pool.map(windows_of, tasks, chunksize=8):
            writer.writerows(rows)
            count += len(rows)
    shutil.rmtree(scratch)
    log(f"  {folder}: 창 {count}개 저장, 추출본 삭제")
    return count


def main() -> None:
    FEATURES.mkdir(parents=True, exist_ok=True)
    state = load_state()
    for part, name in QUEUE:
        entry = state.setdefault(name, {"folders_done": [], "status": "pending"})
        if entry["status"] == "done":
            continue
        try:
            if free_gb() < MIN_FREE_GB:
                log(f"여유 공간 {free_gb():.0f} GB < {MIN_FREE_GB} GB: 중단 (공간 확보 후 다시 실행)")
                return
            archive = download(part, name)
            folders = folders_of(archive)
            if not folders:  # unexpected member layout: keep the archive, record an error, never delete it
                raise RuntimeError(f"no */whole_<n>.csv members in {archive.name}; archive kept for inspection")
            entry["folders"] = {k: len(v) for k, v in folders.items()}
            save_state(state)
            out = FEATURES / name.replace(".7z", ".csv.gz")
            mode = "at" if out.exists() and entry["folders_done"] else "wt"
            with gzip.open(out, mode, newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
                if mode == "wt":
                    writer.writeheader()
                for folder, trials in sorted(folders.items()):
                    if folder in entry["folders_done"]:
                        continue
                    process_folder(archive, name, folder, trials, writer)
                    handle.flush()
                    entry["folders_done"].append(folder)
                    save_state(state)
            if name not in KEEP:
                archive.unlink(missing_ok=True)
                log(f"{name}: 아카이브 삭제 (재다운로드 가능)")
            entry["status"] = "done"
            save_state(state)
            log(f"{name}: 완료")
        except Exception as error:  # record and move on; a rerun retries
            entry["status"] = f"error: {error}"
            save_state(state)
            log(f"{name}: 오류 — {error}")
    log("대기열 끝")


if __name__ == "__main__":
    main()
