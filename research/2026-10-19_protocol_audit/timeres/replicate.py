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
import fcntl
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
sys.path.insert(0, str(HERE.parent / "astra_20261019"))
import disk_guard as DG
ARCHIVE_FOLDER_BYTES = {}
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
    (2, "2024-04-22_TypeA_sample3.7z"), (3, "2024-04-24_TypeA_sample4.7z"),
    # 2026-10-19: high-force parts moved up (lift the 4 N ceiling of the upper boundary; frozen predictions exist)
    (7, "HighBowForceRange_1.7z"), (8, "HighBowForceRange_2.7z"), (3, "2024-04-26_TypeD_sample2.7z"),
    (5, "2024-07-03_TypeB_T2_sample2.7z"),
    (3, "2024-03-25_TypeA_sample1_coldwarm.7z"),
]
# 2026-10-19 22h (Claude, user-approved): D2 waited 17:05-22:00 for 100.6 GB free to extract one 38.6 GB folder.
# These archives use the bounded one-pass stream (astra_20261019/stream_features.py) instead, but only when their
# listing passes stream_features.inventory (member order/size checks); otherwise the legacy folder path is kept.
STREAM_ARCHIVES = {"2024-04-26_TypeD_sample2.7z", "2024-07-03_TypeB_T2_sample2.7z", "2024-03-25_TypeA_sample1_coldwarm.7z"}  # coldwarm 2026-10-20 (disk stall)
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


def wait_for_budget(required_bytes: int, phase: str) -> None:
    """Stay alive while blocked so the B1 recovery waiter does not start.

    Recheck at each download/extraction boundary. Existing archives and companion
    hard links already count against disk free space and are never deleted here.
    """
    last_log = 0.0
    while shutil.disk_usage(Path.home()).free < required_bytes:
        now = time.monotonic()
        if now - last_log >= 300 or last_log == 0:
            log(f"공간 대기 ({phase}): 필요 {required_bytes / 1e9:.2f} GB, "
                f"현재 {free_gb():.2f} GB. 60 GB 보존 + 2 GB 여유; 파일 삭제 없음")
            last_log = now
        time.sleep(60)


def archive_path(part: int, name: str) -> Path:
    return KEEP.get(name, DATA / "raw/zenodo" / RECORD[part] / name)


def download(part: int, name: str) -> Path:
    path = archive_path(part, name)
    if path.exists() and not Path(str(path) + ".part").exists():
        return path
    # Exact metadata preflight: max-gib alone does not protect the 60 GB reserve.
    sys.path.insert(0, str(REPO / "scripts"))
    import zenodo_catalog as catalog
    metadata = catalog.fetch_record(RECORD[part])
    catalog.file_spec(metadata, name)  # checksum, exact filename and URL validation
    catalog.save_manifest(RECORD[part], metadata, DATA / "manifests")
    actual = DATA / "raw/zenodo" / str(metadata["id"]) / name
    intermediate = Path(str(actual) + ".part")
    if intermediate.is_symlink():
        raise ValueError("Refusing symlink partial in disk budget")
    stat = intermediate.stat() if intermediate.exists() else None
    remaining = DG.remaining_download_bytes(metadata, name,
        allocated_bytes=stat.st_blocks * 512 if stat else 0,
        logical_bytes=stat.st_size if stat else 0)
    wait_for_budget(remaining + DG.RESERVE_BYTES + DG.MARGIN_BYTES, f"다운로드 {name}")
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
    # Keep full folder paths; repeated leaf names (v1, v2, v3) are valid.
    ARCHIVE_FOLDER_BYTES[str(archive.resolve())] = DG.parse_slt_listing(listing)
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
    size = ARCHIVE_FOLDER_BYTES[str(archive.resolve())][folder]
    wait_for_budget(size + DG.RESERVE_BYTES + DG.MARGIN_BYTES, f"추출 {folder}")
    log(f"  공간 확인: 이 폴더 {size / 1e9:.2f} GB; 추출 후 예상 {free_gb() - size / 1e9:.2f} GB")
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


def stream_compatible(archive: Path) -> bool:
    import stream_features
    try:
        listing = subprocess.run([str(SEVEN), "l", "-slt", str(archive)], capture_output=True, text=True, check=True).stdout
        stream_features.inventory(listing)
        return True
    except ValueError as error:
        log(f"{archive.name}: stream listing check failed ({error}); legacy folder extraction kept")
        return False


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
            highforce = name.startswith("HighBowForceRange_")
            if highforce or (name in STREAM_ARCHIVES and stream_compatible(archive)):
                import stream_features
                if not highforce and out.exists() and not entry["folders_done"] and out.stat().st_size == 0:
                    stale = out.with_name(out.name + ".stale_legacy_empty")
                    out.rename(stale)
                    log(f"{name}: empty legacy output moved aside to {stale.name}")
                elif not highforce and out.exists() and entry["folders_done"]:
                    # 2026-10-20: partial legacy output (some folders done) is kept aside, never deleted; the stream redoes all folders
                    stale = out.with_name(out.name + f".legacy_partial_{len(entry['folders_done'])}folders")
                    out.rename(stale)
                    entry["legacy_partial_output"] = str(stale)
                    entry["folders_done"] = []
                    save_state(state)
                    log(f"{name}: partial legacy output moved aside to {stale.name}; stream will process every folder")
                log(f"{name}: bounded one-pass stream; rechecking archive MD5")
                integrity = stream_features.verify_archive(archive)
                receipt = stream_features.run(archive, out, folders, log=log, integrity=integrity)
                # A failed legacy companion cannot represent duplicate v1/v2/v3 leaves.
                # Validated full-path ramp/blocks now replace it; original still exists.
                released = stream_features.release_redundant_pin(archive, receipt) if highforce else False
                if released:
                    log(f"{name}: full-path metadata validated; redundant companion link released")
                entry["folders_done"] = sorted(folders)
                entry["status"] = "done"
                entry["stream_receipt"] = str(HERE.parent / "astra_20261019/highforce_stream" / archive.stem / "receipt.json")
                save_state(state)
                if name not in KEEP:
                    archive.unlink(missing_ok=True)
                log(f"{name}: 완료 (single stream, metadata retained)" + ("" if highforce else " [STREAM_ARCHIVES]"))
                continue
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
    # A second launch exits before downloading, extracting, or touching state.
    with (HERE / "replicate.instance.lock").open("a+") as instance_lock:
        fcntl.flock(instance_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
