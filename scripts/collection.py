#!/usr/bin/env python3
"""Process the whole public robot-bowing collection, one Schelleng diagram at a time.

For each published archive: download it (MD5-verified), list its members, then for every
speed folder extract only that folder, audit it while caching each classification window as
float32 .npy, and delete the extracted CSVs. When every folder of a diagram is cached,
regime-map runs on the cache. State files make each step resumable.

Deletion rules: an extracted folder under data/staging is deleted only after its audit passed
and every window was cached (or after a failure, since the archive still holds it). An
archive is deleted only with --delete-archive, only after all of its diagrams completed, and
never for archives with a pre-existing extraction. A pre-existing extraction is read in place
and never modified.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import uuid

import grid_summary
import regime_map
import trial_audit
import zenodo_catalog


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3
# Processing order: string types first, then setup repeats, string samples, extended force, the rest.
QUEUE = (
    (1, "2024-03-25_TypeA_sample1.7z"), (1, "2024-04-10_TypeB_sample1.7z"), (1, "2024-04-12_TypeC_sample1.7z"),
    (2, "2024-04-15_TypeD_sample1.7z"), (6, "2024-10-23_variability_setup.7z"), (2, "2024-04-19_TypeA_sample2.7z"),
    (2, "2024-04-22_TypeA_sample3.7z"), (7, "HighBowForceRange_1.7z"), (8, "HighBowForceRange_2.7z"),
    (3, "2024-04-24_TypeA_sample4.7z"), (3, "2024-04-26_TypeD_sample2.7z"), (3, "2024-03-25_TypeA_sample1_coldwarm.7z"),
    (4, "2024-06-19_TypeA_T2_sample1.7z"), (4, "2024-06-13_TypeB_T2_sample1.7z"), (4, "2024-06-11_TypeC_T2_sample1.7z"),
    (5, "2024-06-17_TypeD_T2_sample1.7z"), (5, "2024-07-03_TypeB_T2_sample2.7z"),
)
EXTRACT_RESERVE = 10 * GIB
PREFETCH_RESERVE = 20 * GIB


class CollectionError(RuntimeError):
    """A collection step cannot continue safely."""


class ExtractionError(CollectionError):
    """7-Zip did not produce exactly the listed files: a structural problem that would repeat for every archive."""


@dataclass
class Layout:
    data: Path
    reports: Path
    existing: dict  # archive filename -> directory holding an earlier, complete extraction

    @property
    def raw(self) -> Path:
        return self.data / "raw" / "zenodo"

    @property
    def staging(self) -> Path:
        return self.data / "staging"

    @property
    def cache(self) -> Path:
        return self.data / "cache"

    def state_path(self, archive: str) -> Path:
        return self.reports / "state" / f"{archive}.json"


def default_layout() -> Layout:
    return Layout(PROJECT_ROOT / "data", PROJECT_ROOT / "reports" / "collection",
                  {"2024-03-25_TypeA_sample1.7z": PROJECT_ROOT / "data" / "interim" / "typeA_sample1"})


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_state(layout: Layout, archive: str) -> dict:
    path = layout.state_path(archive)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"archive": archive, "status": "pending", "leaves": {}, "diagrams": {}}


# ------------------------------------------------------------------ archive members

def parse_slt(text: str) -> list[dict]:
    """Members from `7zz l -slt` output: path, size, is_dir. Archive-level properties before '----------' are skipped."""
    lines = text.splitlines()
    try:
        body = lines[lines.index("----------") + 1:]
    except ValueError as error:
        raise CollectionError("unexpected 7-Zip listing: no '----------' separator") from error
    members, block = [], {}
    for line in body + [""]:
        if not line.strip():
            if "Path" in block:
                attributes = block.get("Attributes", "")
                members.append({"path": block["Path"].replace("\\", "/"), "size": int(block.get("Size") or 0),
                                "is_dir": block.get("Folder") == "+" or attributes.startswith("D")})
            block = {}
            continue
        key, sep, value = line.partition(" = ")
        if sep:
            block[key.strip()] = value
    return members


def seven_zip() -> str:
    executable = shutil.which("7zz") or shutil.which("7z")
    if not executable:
        raise CollectionError("7zz/7z not found; run `bash scripts/run.sh setup-tools`")
    return executable


def list_members(archive: Path) -> list[dict]:
    result = subprocess.run([seven_zip(), "l", "-slt", "--", str(archive)], capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise CollectionError(f"7-Zip could not list {archive.name}: {result.stderr.strip()[:300]}")
    return parse_slt(result.stdout)


def group_leaves(members: list[dict]) -> dict[str, dict[str, dict]]:
    """{diagram: {leaf: {"files": {name: size}, "bytes": total, "trials": n}}}; a leaf holds whole/beta/timestamp files."""
    leaves: dict[str, dict] = {}
    for m in members:
        if m["is_dir"]:
            continue
        parent, _, name = m["path"].rpartition("/")
        match = trial_audit.TRIAL_FILE.fullmatch(name)
        if not match:
            continue
        leaf = leaves.setdefault(parent, {"files": {}, "bytes": 0, "trials": 0})
        leaf["files"][name] = m["size"]
        leaf["bytes"] += m["size"]
        leaf["trials"] += match.group(1) == "whole"
    diagrams: dict[str, dict] = {}
    for leaf in sorted(leaves):
        diagrams.setdefault(leaf.rpartition("/")[0] or leaf, {})[leaf] = leaves[leaf]
    return diagrams


def diagram_id(diagram: str) -> str:
    return diagram.strip("/").replace("/", "__")


# ------------------------------------------------------------------ steps

def ensure_archive(layout: Layout, part: int, filename: str, retries: int, progress=print) -> Path:
    """Download (or re-verify) one archive with the MD5-checking downloader; returns its path."""
    metadata = zenodo_catalog.fetch_record(zenodo_catalog.RECORDS[part], retries=retries)
    size = zenodo_catalog.file_spec(metadata, filename)[0]
    args = SimpleNamespace(record=zenodo_catalog.RECORDS[part], file=filename, max_gib=size / GIB + 0.5,
                           data_dir=str(layout.data), retries=retries, retry_delay=30)
    progress(f"[download] {filename} ({size / GIB:.1f} GiB)")
    zenodo_catalog.download(args, metadata)
    path = layout.raw / str(metadata["id"]) / filename
    if not path.is_file():
        raise CollectionError(f"downloader finished but {path} is missing")
    return path


def existing_leaf(layout: Layout, archive: str, leaf: str) -> Path | None:
    base = layout.existing.get(archive)
    if not base:
        return None
    for candidate in (base / leaf, base / leaf.partition("/")[2]):
        if candidate.is_dir():
            return candidate
    return None


def extract_leaf(archive: Path, leaf: str, spec: dict, staging: Path) -> Path:
    """Extract one folder of a solid archive and check every trial file's name and size."""
    target = staging / leaf
    if target.exists():  # left by an interrupted run; only our own staging copy
        shutil.rmtree(target)
    staging.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(staging).free
    if free < spec["bytes"] + EXTRACT_RESERVE:
        raise CollectionError(f"extracting {leaf} needs {spec['bytes'] / GIB:.1f} GiB + reserve; {free / GIB:.1f} GiB free")
    run_7zip_extract(archive, leaf + "/*", staging)
    found = {p.name: p.stat().st_size for p in target.iterdir() if trial_audit.TRIAL_FILE.fullmatch(p.name)} \
        if target.is_dir() else {}
    if found != spec["files"]:
        missing = len(set(spec["files"]) - set(found))
        raise ExtractionError(f"{leaf}: extracted files differ from the listing ({missing} missing, "
                              f"{sum(found.get(k) != v for k, v in spec['files'].items())} size mismatches)")
    return target


def run_7zip_extract(archive: Path, pattern: str, output: Path) -> None:
    command = [seven_zip(), "x", "-y", "-bb0", "-bsp0", "-mmt=on", "-o" + str(output), "--", str(archive), pattern]
    result = subprocess.run(command, capture_output=True, text=True, timeout=6 * 3600)
    if result.returncode != 0:
        raise ExtractionError(f"7-Zip failed on {pattern} (exit {result.returncode}): "
                              f"{(result.stderr or result.stdout).strip()[:300]}")


def probe_extraction(archive: Path, staging: Path) -> str:
    """Extract one small companion file through a path wildcard and check its listed size (seconds, a few KB)."""
    diagrams = group_leaves(list_members(archive))
    leaf, spec = next(iter(next(iter(diagrams.values())).items()))
    name = sorted(n for n in spec["files"] if n.startswith("beta_"))[0]
    output = staging / f"probe_{uuid.uuid4().hex[:8]}"
    try:
        run_7zip_extract(archive, f"{leaf}/{name[:-1]}*", output)
        path = output / leaf / name
        if not path.is_file() or path.stat().st_size != spec["files"][name]:
            raise ExtractionError(f"probe: {leaf}/{name} missing or wrong size after wildcard extraction")
        return f"{leaf}/{name} ({spec['files'][name]} bytes) extracted and verified"
    finally:
        shutil.rmtree(output, ignore_errors=True)


def process_leaf(layout: Layout, archive_name: str, archive: Path | None, diagram: str, leaf: str, spec: dict,
                 workers: int, progress=print) -> dict:
    source = existing_leaf(layout, archive_name, leaf)
    extracted = None
    if source is None:
        if archive is None:
            raise CollectionError(f"{leaf}: no archive and no existing extraction")
        progress(f"[extract] {leaf} ({spec['bytes'] / GIB:.1f} GiB)")
        source = extracted = extract_leaf(archive, leaf, spec, layout.staging / Path(archive_name).stem)
    ident = diagram_id(diagram)
    cache_dir = layout.cache / ident
    try:
        code, run_dir = trial_audit.run_audit(source, layout.reports / ident / "audit_parts" / Path(leaf).name, None,
                                              workers, 0.01, "auto", progress, cache_dir=cache_dir)
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        cached = min(len(list((cache_dir / source.name).glob(f"{kind}_*.npy"))) for kind in ("window", "profile"))
        good = summary["trials_audited"] - summary["trials_failed"]
        if code != 0 or cached != good or good != spec["trials"]:
            raise CollectionError(f"{leaf}: audit {summary['status']}, {good}/{spec['trials']} readable, {cached} cached")
        return {"status": "cached", "audit_run": str(run_dir), "trials": good, "condition": source.name}
    finally:
        if extracted is not None:
            shutil.rmtree(extracted, ignore_errors=True)


def merge_audits(layout: Layout, diagram: str, leaf_states: list[dict]) -> Path:
    """One PASS audit table for the whole diagram, pointing at the window cache."""
    ident = diagram_id(diagram)
    rows, fields, conditions, warnings = [], [], [], []
    for state in leaf_states:
        run_dir = Path(state["audit_run"])
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        if summary["status"] != "PASS":
            raise CollectionError(f"audit part {run_dir} is not PASS")
        with (run_dir / "trials.csv").open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields += [f for f in reader.fieldnames or [] if f not in fields]
            rows += list(reader)
        conditions += summary["conditions"]
        warnings += summary["warnings"]
    reports = layout.reports / ident / "trial_audit"
    stamp = now()
    run_dir = reports / f"{stamp}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["condition"], int(r["trial"]))))
    cache = layout.cache / ident
    write_json(run_dir / "summary.json", {
        "status": "PASS", "created_utc": stamp, "root": str(cache), "window_cache": str(cache), "diagram": diagram,
        "merged_from": [s["audit_run"] for s in leaf_states], "trials_audited": len(rows), "trials_failed": 0,
        "conditions": conditions, "warnings": warnings})
    write_json(reports / "latest.json", {"status": "PASS", "run_dir": str(run_dir), "created_utc": stamp})
    return run_dir


def process_archive(layout: Layout, part: int, archive_name: str, archive: Path | None, workers: int,
                    progress=print) -> dict:
    state = load_state(layout, archive_name)
    if state["status"] == "complete":
        return state
    state.update(part=part, status="in_progress", started_utc=state.get("started_utc") or now())
    if archive is not None:
        members = list_members(archive)
    else:
        stem = Path(archive_name).stem  # keep the archive's root folder so speed folders stay one diagram
        members = []
        for path in layout.existing[archive_name].rglob("*.csv"):
            relative = path.relative_to(layout.existing[archive_name]).as_posix()
            members.append({"path": relative if relative.startswith(stem + "/") else f"{stem}/{relative}",
                            "size": path.stat().st_size, "is_dir": False})
    diagrams = group_leaves(members)
    if not diagrams:
        raise CollectionError(f"{archive_name}: no trial folders found")
    state["planned"] = {d: sorted(leaves) for d, leaves in diagrams.items()}
    write_json(layout.state_path(archive_name), state)
    for diagram, leaves in diagrams.items():
        if state["diagrams"].get(diagram, {}).get("status") == "complete":
            continue
        try:
            for leaf, spec in leaves.items():
                if state["leaves"].get(leaf, {}).get("status") == "cached":
                    continue
                state["leaves"][leaf] = process_leaf(layout, archive_name, archive, diagram, leaf, spec, workers, progress)
                write_json(layout.state_path(archive_name), state)
            merged = merge_audits(layout, diagram, [state["leaves"][leaf] for leaf in leaves])
            code, regime_dir = regime_map.run(layout.reports / diagram_id(diagram) / "trial_audit",
                                              layout.reports / diagram_id(diagram) / "regime_map", workers,
                                              dict(regime_map.DEFAULT_THRESHOLDS), None, True, progress)
            state["diagrams"][diagram] = {"status": "complete", "audit_run": str(merged), "regime_run": str(regime_dir),
                                          "finished_utc": now()}
        except ExtractionError:
            write_json(layout.state_path(archive_name), state)
            raise
        except (CollectionError, trial_audit.TrialAuditError, grid_summary.GridError, regime_map.RegimeError,
                OSError, ValueError, KeyError) as error:
            progress(f"[failed] {diagram}: {error}")
            state["diagrams"][diagram] = {"status": "failed", "error": str(error)[:500], "finished_utc": now()}
        write_json(layout.state_path(archive_name), state)
    if all(state["diagrams"].get(d, {}).get("status") == "complete" for d in diagrams):
        state["status"] = "complete"
    else:
        state["status"] = "failed"
    state["finished_utc"] = now()
    write_json(layout.state_path(archive_name), state)
    return state


def run(layout: Layout, queue, workers: int, delete_archive: bool, prefetch: bool, retries: int, progress=print,
        fetch=ensure_archive) -> int:
    failures = 0
    pending = [(part, name) for part, name in queue if load_state(layout, name)["status"] != "complete"]
    progress(f"COLLECTION: {len(pending)} archive(s) to process: {', '.join(n for _, n in pending)}")
    downloads = ThreadPoolExecutor(max_workers=1) if prefetch else None
    futures: dict[str, object] = {}
    try:
        for index, (part, name) in enumerate(pending):
            uses_existing = name in layout.existing and layout.existing[name].is_dir()
            if uses_existing:
                archive = None
            else:
                future = futures.pop(name, None)
                try:
                    archive = future.result() if future is not None else fetch(layout, part, name, retries, progress)
                except Exception as error:  # network or checksum problems: record and move on
                    progress(f"[failed] download {name}: {error}")
                    failures += 1
                    continue
            if downloads is not None and index + 1 < len(pending):
                next_part, next_name = pending[index + 1]
                if next_name not in futures and not (next_name in layout.existing and layout.existing[next_name].is_dir()):
                    metadata_size = _published_size(next_part, next_name, retries)
                    free = shutil.disk_usage(layout.data if layout.data.exists() else layout.data.parent).free
                    if metadata_size is not None and free > metadata_size + 25 * GIB + PREFETCH_RESERVE:
                        futures[next_name] = downloads.submit(fetch, layout, next_part, next_name, retries, progress)
            progress(f"[archive] {name}")
            state = process_archive(layout, part, name, archive, workers, progress)
            if state["status"] != "complete":
                failures += 1
            elif delete_archive and archive is not None and name not in layout.existing:
                if archive.resolve().is_relative_to(layout.raw.resolve()):
                    archive.unlink()
                    state["archive_deleted_utc"] = now()
                    write_json(layout.state_path(name), state)
                    progress(f"[deleted] {archive}")
            progress(f"[done] {name}: {state['status']}")
    finally:
        if downloads is not None:
            downloads.shutdown(wait=True)
    progress(f"COLLECTION: finished with {failures} failure(s)")
    return 0 if failures == 0 else 1


def _published_size(part: int, name: str, retries: int) -> int | None:
    try:
        return zenodo_catalog.file_spec(zenodo_catalog.fetch_record(zenodo_catalog.RECORDS[part], retries=retries), name)[0]
    except Exception:
        return None


# ------------------------------------------------------------------ reporting

def status_lines(layout: Layout, queue=QUEUE) -> list[str]:
    lines = []
    for part, name in queue:
        state = load_state(layout, name)
        lines.append(f"[{state['status']}] part {part} {name}" + (" (archive deleted)" if state.get("archive_deleted_utc") else ""))
        for diagram, d in state.get("diagrams", {}).items():
            if d.get("status") != "complete":
                lines.append(f"  {diagram}: {d.get('status')} {d.get('error', '')}")
                continue
            summary = json.loads((Path(d["regime_run"]) / "summary.json").read_text(encoding="utf-8"))
            law = summary.get("flyback_law", {})
            lines.append(f"  {diagram}: floor {summary['noise_floor']['min_amplitude']:.3g}, law speed "
                         f"{law.get('exponent_speed')}±{law.get('exponent_speed_se')} beta {law.get('exponent_beta')}±"
                         f"{law.get('exponent_beta_se')} Z {law.get('impedance_kg_s', {}).get('median')} (n {law.get('n')})")
            for condition, c in summary["conditions"].items():
                lo, up = c["boundaries"]["lower"], c["boundaries"]["upper"]
                classes = " ".join(f"{k[:4]} {v}" for k, v in c["classes"].items())
                lines.append(f"    {condition} v {c['c2_peak']} f0 {c['reference_f0_hz']} | {classes} | lower "
                             f"{lo['slope']}±{lo['slope_se']} n{lo['n']} | upper {up['slope']}±{up['slope_se']} n{up['n']}")
            for side, fit in summary.get("schelleng_law", {}).items():
                lines.append(f"    Schelleng {side}: speed {fit.get('exponent_speed')}±{fit.get('exponent_speed_se')} "
                             f"beta {fit.get('exponent_beta')}±{fit.get('exponent_beta_se')} (n {fit.get('n')})")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="show the queue and state; with --list, list the trial folders of a local archive")
    plan.add_argument("--list", type=Path, help="local .7z archive to list with 7-Zip (checks the listing parser)")
    plan.add_argument("--probe", action="store_true", help="with --list: also extract one small file by wildcard")
    go = sub.add_parser("run", help="download, extract folder by folder, cache windows, map regimes")
    go.add_argument("--only", action="append", help="archive filename to process (repeatable); default: whole queue")
    go.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    go.add_argument("--delete-archive", action="store_true", help="delete each downloaded archive after all its diagrams completed")
    go.add_argument("--no-prefetch", action="store_true", help="do not download the next archive while processing")
    go.add_argument("--retries", type=int, default=5)
    sub.add_parser("status", help="compact results of processed diagrams")
    args = parser.parse_args(argv)
    layout = default_layout()
    try:
        if args.command == "plan":
            if args.list:
                for diagram, leaves in group_leaves(list_members(args.list)).items():
                    print(f"{diagram} -> {diagram_id(diagram)}")
                    for leaf, spec in leaves.items():
                        print(f"  {leaf}: {spec['trials']} trials, {spec['bytes'] / GIB:.1f} GiB")
                if args.probe:
                    print("Probe:", probe_extraction(args.list, layout.staging))
                return 0
            free = shutil.disk_usage(PROJECT_ROOT).free
            print(f"Free disk: {free / GIB:.1f} GiB; 7-Zip: {shutil.which('7zz') or shutil.which('7z')}")
            for part, name in QUEUE:
                state = load_state(layout, name)
                source = "existing extraction" if name in layout.existing and layout.existing[name].is_dir() else "download"
                print(f"[{state['status']}] part {part} {name} ({source})")
            return 0
        if args.command == "status":
            print("\n".join(status_lines(layout)))
            return 0
        queue = QUEUE if not args.only else [(p, n) for p, n in QUEUE if n in set(args.only)]
        if args.only and len(queue) != len(set(args.only)):
            parser.error(f"unknown archive in --only; choose from: {', '.join(n for _, n in QUEUE)}")
        if args.workers <= 0:
            parser.error("--workers must be positive")
        return run(layout, queue, args.workers, args.delete_archive, not args.no_prefetch, args.retries)
    except CollectionError as error:
        print("COLLECTION: FAIL —", error, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. State is saved; run the same command again to resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
