#!/usr/bin/env python3
"""Prepare the first published archive, optionally extract, then inspect CSV prefixes.

No model training, physical labels, or unverified unit conversion is performed.
The extraction flag is explicit and constrained by the inspected member sizes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import http.client
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

from archive_audit import audit_archive

ROOT = Path(__file__).resolve().parents[1]
RECORD = "17749111"
FILENAME = "2024-03-25_TypeA_sample1.7z"
EXPECTED_SIZE = 14083546513
EXPECTED_MD5 = "2709034cc10344f37f3488376ae3ebcd"
GIB = 1024**3


def pinned_record(record, catalog):
    if not isinstance(record, dict) or not isinstance(record.get("files"), list) or any(
        not isinstance(item, dict) for item in record.get("files", [])
    ):
        raise ValueError("Unexpected pinned record schema")
    if str(record.get("id")) != RECORD or catalog.file_spec(record, FILENAME)[:2] != (EXPECTED_SIZE, EXPECTED_MD5):
        raise ValueError("Metadata does not match the pinned first archive version, size and MD5")


def resolve_metadata(catalog, retries):
    """Reuse a validated snapshot of this fixed version; never resolve 'latest' from cache."""
    manifest_dir = ROOT / "data/manifests"
    source_url = "https://zenodo.org/api/records/" + RECORD
    for path in sorted(manifest_dir.glob("zenodo_" + RECORD + "_*.json"), reverse=True):
        try:
            if path.is_symlink():
                raise ValueError("symlink manifest")
            with path.open("rb") as source:
                raw = source.read(catalog.MAX_METADATA_BYTES + 1)
            if len(raw) > catalog.MAX_METADATA_BYTES:
                raise ValueError("manifest exceeds size bound")
            envelope = json.loads(raw)
            if not isinstance(envelope, dict) or any(str(envelope.get(key)) != RECORD for key in (
                "requested_record_id", "resolved_record_id"
            )) or envelope.get("source_url") != source_url:
                raise ValueError("manifest record/source mismatch")
            if not isinstance(envelope.get("fetched_at_utc"), str) or not envelope["fetched_at_utc"]:
                raise ValueError("manifest has no original retrieval time")
            record = envelope.get("record")
            pinned_record(record, catalog)
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"Skipping incompatible saved metadata {path.name}: {error}", flush=True)
            continue
        print("Using validated saved metadata:", path.relative_to(ROOT), flush=True)
        return record, {"mode": "cached_manifest", "manifest": str(path.relative_to(ROOT)),
            "original_fetched_at_utc": envelope["fetched_at_utc"], "source_url": source_url}
    print("No matching saved metadata; requesting the pinned official record.", flush=True)
    record = catalog.fetch_record(RECORD, retries, 10)
    pinned_record(record, catalog)
    manifest = catalog.save_manifest(RECORD, record, manifest_dir)
    return record, {"mode": "live_api", "manifest": str(manifest.relative_to(ROOT)), "source_url": source_url}


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as out:
            json.dump(payload, out, ensure_ascii=False, indent=2, allow_nan=False)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.link(temporary, path)  # Atomic publication, without replacing existing files.
    finally:
        temporary.unlink(missing_ok=True)


def load_members(audit):
    manifest = Path(audit["report_dir"]) / audit["artifacts"]["manifest"]
    with manifest.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def verify_tree(directory, members):
    """Check exact files and sizes; not a fresh cryptographic test of extracted data."""
    expected = {m["path"]: m["size_bytes"] for m in members if not m["directory"]}
    actual = {}
    for current, dirs, files in os.walk(directory, followlinks=False):
        for name in dirs + files:
            path = Path(current) / name
            if path.is_symlink():
                raise ValueError("Unexpected link in extraction output")
        for name in files:
            path = Path(current) / name
            relative = path.relative_to(directory).as_posix()
            actual[relative] = path.stat().st_size
    if actual != expected:
        raise ValueError("Extracted file names/sizes do not match the inspected archive")


def extract_checked(archive, audit, destination, receipt, max_unpacked_gib, report_dir):
    if audit.get("status") != "pass" or not audit.get("inventory_guards_passed"):
        raise ValueError("Extraction requires a successful guarded archive inventory")
    total = audit["total_uncompressed_bytes"]
    if not math.isfinite(max_unpacked_gib) or max_unpacked_gib <= 0 or total > max_unpacked_gib * GIB:
        raise ValueError("Inspected uncompressed size exceeds the explicit extraction budget")
    members = load_members(audit)
    if destination.is_symlink():
        raise ValueError("Refusing a symlink extraction destination")
    staging = destination.with_name(destination.name + ".partial")
    if receipt.is_symlink():
        raise ValueError("Refusing a symlink extraction receipt")
    if destination.exists() or receipt.exists():
        if not receipt.is_file():
            raise ValueError("Extraction directory already exists without a completion receipt; it will not be overwritten")
        previous = json.loads(receipt.read_text())
        if previous.get("archive_md5") != EXPECTED_MD5 or previous.get("status") != "complete":
            raise ValueError("Existing extraction receipt does not match this archive")
        if not destination.exists():
            if not staging.is_dir() or staging.is_symlink():
                raise ValueError("Completed receipt has no valid extraction output; inspect it first")
            verify_tree(staging, members)
            staging.rename(destination)
            print("Recovered completed extraction after an interrupted publication.", flush=True)
        verify_tree(destination, members)
        print("Existing completed extraction matches inspected names and sizes.", flush=True)
        return {"status": "reused", "directory": str(destination.relative_to(ROOT))}
    destination.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(destination.parent).free
    if free < total + 20 * GIB:
        raise ValueError(f"Extraction needs {total/GIB:.2f} GiB plus 20 GiB reserve; only {free/GIB:.2f} GiB free")
    if staging.exists() or staging.is_symlink():
        raise ValueError(f"An unfinished extraction remains at {staging.name}; preserve and inspect it before another extraction")
    executable = shutil.which("7zz") or shutil.which("7z")
    if not executable:
        raise ValueError("7zz/7z is required")
    before = archive.stat()
    if {"size_bytes": before.st_size, "mtime_ns": before.st_mtime_ns} != audit["archive_fingerprint"]:
        raise ValueError("Archive changed after inventory")
    staging.mkdir()
    log_path = report_dir / "extraction.log"
    command = [executable, "x", "-y", "-bb0", "-bsp0", "-mmt=4", "-sccUTF-8", "-o" + str(staging), "--", str(archive)]
    print(f"Extracting {total/GIB:.2f} GiB into a new directory. This can take time; log: {log_path.relative_to(ROOT)}", flush=True)
    started = time.monotonic()
    next_notice = started + 30
    with log_path.open("xb") as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                now = time.monotonic()
                free_now = shutil.disk_usage(staging).free
                if free_now < 10 * GIB:
                    raise ValueError("Extraction stopped with less than 10 GiB free; partial output preserved")
                if now - started > 4 * 3600:
                    raise ValueError("Extraction exceeded four hours; partial output preserved")
                if log_path.stat().st_size > 8 * 1024**2:
                    raise ValueError("Extraction log exceeded 8 MiB; partial output preserved")
                if now >= next_notice:
                    print(f"  Extracting: {(now-started)/60:.1f} min; disk free {free_now/GIB:.1f} GiB", flush=True)
                    next_notice = now + 30
                time.sleep(1)
            if process.returncode != 0:
                raise ValueError(f"7z extraction failed (exit {process.returncode}); inspect {log_path.relative_to(ROOT)}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    after = archive.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Archive changed during extraction")
    verify_tree(staging, members)
    if destination.exists():
        raise ValueError("Destination appeared during extraction; partial output preserved")
    write_json(receipt, {"status": "complete", "archive_md5": EXPECTED_MD5,
        "total_files": audit["total_files"], "uncompressed_bytes": total,
        "verification": "7z exit 0 and exact member names/sizes; raw archive MD5 checked by downloader"})
    staging.rename(destination)
    return {"status": "extracted", "directory": str(destination.relative_to(ROOT)), "seconds": round(time.monotonic()-started, 2)}


def inspect_prefix(path):
    """Bounded text inspection. A CSV prefix is not a whole-signal quality check."""
    with path.open("rb") as source:
        data = source.read(32769)
    prefix = data[:32768]
    if len(data) > 32768:
        prefix = prefix.rsplit(b"\n", 1)[0]
    text = prefix.decode("utf-8-sig")
    lines = text.splitlines()[:64]
    sample = "\n".join(lines)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        rows = list(csv.reader(lines, dialect))
        delimiter = repr(dialect.delimiter)
    except csv.Error:
        rows = [line.split() for line in lines]
        delimiter = "whitespace_or_single_value"
    finite = []
    for row in rows:
        try:
            finite.append(bool(row) and all(math.isfinite(float(token)) for token in row))
        except ValueError:
            finite.append(False)
    return {"delimiter_guess": delimiter, "prefix_rows_inspected": len(rows),
        "column_count_histogram": dict(Counter(len(row) for row in rows)),
        "numeric_finite_rows": sum(finite), "first_rows": rows[:5],
        "scope": "prefix only; no unit, timestamp convention, state label, or whole-file validation"}


def scout_csv(directory, members):
    groups = defaultdict(list)
    index = {m["path"]: m for m in members if not m["directory"]}
    for name in index:
        match = re.fullmatch(r"whole(?:_(\d+))?\.csv", Path(name).name, re.I)
        if match:
            groups[str(Path(name).parent)].append((int(match[1] or 0), name))
    samples = []
    for parent, entries in sorted(groups.items())[:6]:
        entries.sort()
        for i in sorted({0, len(entries)//2, len(entries)-1}):
            number, name = entries[i]
            selected = [name]
            suffix = "_" + str(number) if number else ""
            for stem in ("beta", "betas", "timestamp", "timestamps"):
                companion = (Path(parent) / (stem + suffix + ".csv")).as_posix()
                if companion in index:
                    selected.append(companion)
            for member in selected:
                try:
                    observation = inspect_prefix(directory / member)
                except (OSError, UnicodeError, ValueError) as error:
                    observation = {"error_type": type(error).__name__}
                samples.append({"member": member, "size_bytes": index[member]["size_bytes"], **observation})
    errors = sum("error_type" in sample for sample in samples)
    empty = sum(sample.get("prefix_rows_inspected") == 0 for sample in samples)
    status = "no_matching_waveform_names" if not samples else ("incomplete" if errors or empty else "scouted")
    return {"status": status, "read_errors": errors, "empty_prefixes": empty,
        "group_counts": {k: len(v) for k,v in groups.items()},
        "samples": samples, "warning": "No physical labels or training splits have been assigned. Folder indices are not interpreted as measured speeds."}


def show_latest():
    """Print saved status and representative prefixes, without starting any work."""
    try:
        summary = json.loads((ROOT / "reports/today/latest.json").read_text())
        output = {"preparation": summary}
        if summary.get("csv_scout"):
            scout = json.loads((ROOT / summary["csv_scout"]).read_text())
            representatives = []
            seen = set()
            for sample in scout["samples"]:
                member = Path(sample["member"])
                key = (str(member.parent), re.sub(r"_\d+$", "", member.stem))
                if key in seen:
                    continue
                seen.add(key)
                entry = dict(sample)
                if "first_rows" in entry:
                    entry["first_rows"] = entry["first_rows"][:2]
                representatives.append(entry)
            output["csv_scout"] = {k: v for k, v in scout.items() if k != "samples"}
            output["csv_scout"]["representative_prefixes"] = representatives[:18]
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError) as error:
        print("Could not read the saved report:", error, file=sys.stderr)
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--extract", action="store_true", help="Explicitly allow guarded extraction of this ONE archive")
    action.add_argument("--show", action="store_true", help="Print the saved summary and representative CSV prefixes; no network or extraction")
    parser.add_argument("--max-unpacked-gib", type=float, default=80)
    parser.add_argument("--retries", type=int, default=8)
    args = parser.parse_args(argv)
    if args.show:
        return show_latest()
    os.chdir(ROOT)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    report_dir = ROOT / "reports/today" / stamp
    report_dir.mkdir(parents=True)
    summary = {"status": "running", "archive_name": FILENAME, "record": RECORD, "extraction_requested": args.extract,
        "research_training_performed": False, "report_dir": str(report_dir.relative_to(ROOT))}
    exit_code = 1
    try:
        if not shutil.which("7zz") and not shutil.which("7z"):
            raise ValueError("Run bash scripts/run.sh setup-tools first (no sudo required).")
        if not math.isfinite(args.max_unpacked_gib) or args.max_unpacked_gib <= 0:
            raise ValueError("--max-unpacked-gib must be positive and finite")
        print("Stage 1/3: download/resume and verify the raw archive MD5.", flush=True)
        import zenodo_catalog as catalog
        catalog.validate_retry_policy(args.retries, 10)
        record, provenance = resolve_metadata(catalog, args.retries)
        summary["metadata"] = provenance
        write_json(report_dir / "metadata_used.json", {"provenance": provenance, "record": record})
        catalog.download(SimpleNamespace(file=FILENAME, max_gib=14, data_dir=str(ROOT / "data"),
            retries=args.retries, retry_delay=10), record)
        archive = ROOT / "data/raw/zenodo" / RECORD / FILENAME
        if archive.stat().st_size != EXPECTED_SIZE:
            raise ValueError("First-archive size differs from the inspected version; review before extracting")
        summary["download_md5_verified"] = True
        print("Stage 2/3: inspect archive names, sizes, blocks, and extraction guards.", flush=True)
        audit = audit_archive(archive, report_dir / "inventory")
        summary["archive_inventory"] = str(Path(audit["summary_path"]).relative_to(ROOT))
        summary["uncompressed_gib"] = round(audit.get("total_uncompressed_bytes", 0)/GIB, 3)
        summary["files"] = audit.get("total_files")
        if audit["status"] != "pass":
            raise ValueError("Archive inventory did not pass; inspect its report before extracting")
        print(f"Inventory PASS: {audit['total_files']} files; {summary['uncompressed_gib']} GiB uncompressed; Solid={audit.get('solid')}", flush=True)
        if args.extract:
            destination = ROOT / "data/interim/typeA_sample1"
            receipt = ROOT / "data/interim/typeA_sample1.receipt.json"
            summary["extraction"] = extract_checked(archive, audit, destination, receipt, args.max_unpacked_gib, report_dir)
            print("Stage 3/3: inspect bounded CSV prefixes and companion metadata.", flush=True)
            scout = scout_csv(destination, load_members(audit))
            write_json(report_dir / "csv_scout.json", scout)
            summary["csv_scout"] = str((report_dir / "csv_scout.json").relative_to(ROOT))
            summary["scout_status"] = scout["status"]
            if scout["status"] != "scouted":
                raise ValueError("CSV prefix inspection was incomplete; inspect csv_scout.json before analysis")
        else:
            print("Stage 3/3 skipped: extraction was not requested.", flush=True)
        summary["status"] = "pass"
        exit_code = 0
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
        summary["error"] = "Interrupted; downloaded/extracted partial files were preserved."
        exit_code = 130
    except (OSError, ValueError, KeyError, TypeError, http.client.HTTPException, subprocess.CalledProcessError) as error:
        summary["status"] = "fail"
        summary["error"] = str(error)
        print("PREPARATION FAILED:", error, file=sys.stderr)
    finally:
        summary["free_disk_gib"] = round(shutil.disk_usage(ROOT).free/GIB, 2)
        write_json(report_dir / "today_summary.json", summary)
        pointer = ROOT / "reports/today" / (".latest_" + uuid.uuid4().hex + ".json")
        write_json(pointer, summary)
        os.replace(pointer, ROOT / "reports/today/latest.json")
    print("TODAY DATA PREPARATION:", summary["status"].upper(), flush=True)
    print("Status: reports/today/latest.json", flush=True)
    if summary.get("csv_scout"):
        print("CSV scout:", summary["csv_scout"], flush=True)
    print("This prepares data only. Physical labels, units and research performance still require validation.", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
