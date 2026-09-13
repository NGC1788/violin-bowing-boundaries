#!/usr/bin/env python3
"""Inventory one local 7z archive without extracting or testing its payload.

Uses only the standard library and an existing 7zz/7z executable. A successful
inventory is not a checksum test, scientific validation, or extraction approval.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import selectors
import shutil
import subprocess
import sys
import time
import unicodedata
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_LISTING_BYTES = 64 * 1024**2
MAX_ERROR_BYTES = 1024**2
MAX_SAMPLES = 30


class AuditError(ValueError):
    """An inventory could not be completed or interpreted reliably."""


def _integer(value: str | None, field: str) -> int:
    if value is None or not re.fullmatch(r"[0-9]+", value):
        raise AuditError(f"Invalid or missing non-negative integer: {field}")
    return int(value)


def _path_flags(name: str) -> tuple[str, list[str]]:
    portable = name.replace("\\", "/")
    parts = portable.split("/")
    flags = []
    if not name or portable.startswith("/") or re.match(r"^[A-Za-z]:", portable):
        flags.append("absolute_or_empty_path")
    if any(part in {"", ".", ".."} for part in parts):
        flags.append("ambiguous_or_traversing_path")
    if any(ord(char) < 32 or ord(char) == 127 for char in name):
        flags.append("control_character_in_path")
    if any(":" in part or part.endswith((".", " ")) for part in parts):
        flags.append("nonportable_path")
    return portable, flags


def _records(listing: Path):
    """Yield (header/member, properties); split only at the first ' = '."""
    section = None
    properties = {}
    saw_members = False
    with listing.open(encoding="utf-8", errors="strict") as source:
        for raw_line in source:
            line = raw_line.rstrip("\r\n")
            if line in {"--", "----------"}:
                if properties:
                    yield section, properties
                    properties = {}
                section = "header" if line == "--" else "member"
                saw_members |= section == "member"
            elif not line:
                if properties:
                    yield section, properties
                    properties = {}
            elif section is not None:
                if " = " not in line:
                    raise AuditError("Unrecognized line in technical listing")
                key, value = line.split(" = ", 1)
                if not key or key in properties:
                    raise AuditError("Missing or duplicate technical property")
                properties[key] = value
        if properties:
            yield section, properties
    if not saw_members:
        raise AuditError("Missing technical member-list separator")


def parse_listing(listing: Path, manifest: Path, *, archive_size: int) -> dict:
    """Validate technical metadata and write one bounded-detail JSONL per member."""
    header = {}
    counts = Counter()
    suffixes = Counter()
    top_names = Counter()
    issue_counts = Counter()
    issue_examples = []
    candidates = {key: {"count": 0, "examples": []} for key in (
        "whole_csv", "betas_csv", "timestamps", "possible_label_names",
    )}
    seen = set()
    portable_seen = set()
    block_ids = set()
    with manifest.open("x", encoding="utf-8") as output:
        for section, properties in _records(listing):
            if section == "header":
                if header:
                    raise AuditError("Multiple archive headers are unsupported")
                header = properties
                continue
            if section != "member" or "Path" not in properties:
                raise AuditError("Member has no path")
            name = properties["Path"]
            normalized, flags = _path_flags(name)
            collision_key = unicodedata.normalize("NFC", normalized).casefold()
            if normalized in seen:
                flags.append("duplicate_member_path")
            elif collision_key in portable_seen:
                flags.append("portable_path_collision")
            seen.add(normalized)
            portable_seen.add(collision_key)
            attributes = properties.get("Attributes", "")
            folder = properties.get("Folder")
            if folder not in {None, "+", "-"}:
                raise AuditError("Unrecognized Folder flag")
            is_directory = folder == "+" or (folder is None and (
                attributes.startswith("D") or bool(re.search(r"(?:^|\s)d[rwx-]{9}", attributes))
            ))
            size = _integer(properties.get("Size", "0" if is_directory else None), "member Size")
            if is_directory and size != 0:
                raise AuditError("Directory has nonzero Size")
            if any("link" in key.lower() or "reparse" in key.lower() for key in properties):
                flags.append("link_or_reparse_metadata")
            if re.search(r"(?:^|\s)l[rwx-]{9}", attributes):
                flags.append("symbolic_link_attributes")
            encrypted = properties.get("Encrypted", "-")
            if encrypted != "-":
                flags.append("encrypted_or_unknown_encryption")
            if properties.get("Anti", "-") != "-":
                flags.append("anti_item")
            block = properties.get("Block", "")
            if block:
                block_ids.add(_integer(block, "member Block"))
            counts["members"] += 1
            counts["directories" if is_directory else "files"] += 1
            counts["uncompressed_bytes"] += size
            top_names[normalized.split("/", 1)[0]] += 1
            if not is_directory:
                suffixes[PurePosixPath(normalized).suffix.lower() or "[no suffix]"] += 1
                basename = PurePosixPath(normalized).name
                candidate_keys = []
                if re.fullmatch(r"whole(?:_\d+)?\.csv", basename, re.I):
                    candidate_keys.append("whole_csv")
                if re.fullmatch(r"betas?(?:_\d+)?\.csv", basename, re.I):
                    candidate_keys.append("betas_csv")
                if re.fullmatch(r"timestamps?(?:_\d+)?(?:\.csv)?", basename, re.I):
                    candidate_keys.append("timestamps")
                if size <= 1024**2 and re.search(r"label|classif|regime|annotation", basename, re.I):
                    candidate_keys.append("possible_label_names")
                for key in candidate_keys:
                    candidates[key]["count"] += 1
                    if len(candidates[key]["examples"]) < MAX_SAMPLES:
                        candidates[key]["examples"].append({"path": name, "size_bytes": size})
            for flag in flags:
                issue_counts[flag] += 1
                if len(issue_examples) < MAX_SAMPLES:
                    issue_examples.append({"path": name, "issue": flag})
            output.write(json.dumps({
                "path": name, "directory": is_directory, "size_bytes": size,
                "block": int(block) if block else None, "flags": flags,
            }, ensure_ascii=False) + "\n")
    if not header or header.get("Type") != "7z":
        raise AuditError("Only one 7z archive is supported")
    if _integer(header.get("Physical Size"), "Physical Size") != archive_size:
        raise AuditError("Archive physical size does not match the local file")
    if any(key in header for key in ("Volumes", "Total Physical Size", "Offset", "Tail Size")):
        raise AuditError("Multipart, embedded, or trailing-data archives are unsupported")
    if header.get("Encrypted", "-") != "-" or "7zAES" in header.get("Method", ""):
        issue_counts["encrypted_archive"] += 1
    for field, count_key in (("Files", "files"), ("Folders", "directories"), ("Size", "uncompressed_bytes")):
        if field in header and _integer(header[field], field) != counts[count_key]:
            raise AuditError(f"Archive header {field} does not match member totals")
    blocks = _integer(header["Blocks"], "Blocks") if "Blocks" in header else None
    if blocks is not None and (len(block_ids) != blocks or any(block >= blocks for block in block_ids)):
        raise AuditError("Archive Blocks does not match listed member blocks")
    if counts["members"] != counts["files"] + counts["directories"]:
        raise AuditError("Member count mismatch")
    return {
        "archive_format": header["Type"], "solid": header.get("Solid"), "blocks": blocks,
        "total_members": counts["members"], "total_files": counts["files"],
        "total_directories": counts["directories"],
        "total_uncompressed_bytes": counts["uncompressed_bytes"],
        "top_level_names": top_names.most_common(MAX_SAMPLES),
        "suffix_counts": suffixes.most_common(MAX_SAMPLES), "candidate_members": candidates,
        "guard_issue_counts": dict(issue_counts), "guard_issue_examples": issue_examples,
        "inventory_guards_passed": not bool(issue_counts),
        "candidate_note": "Names only; file contents and scientific labels have not been inspected.",
    }


def _capture_listing(executable: str, archive: Path, listing: Path, errors: Path,
                     timeout: float, max_listing_bytes: int) -> dict:
    command = [executable, "l", "-slt", "-sccUTF-8", "--", str(archive)]
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    started = time.monotonic()
    byte_counts = {"stdout": 0, "stderr": 0}
    # Technical output is kept locally; redact the caller's archive/home paths.
    redactions = [(str(archive).encode(), b"[archive]/" + archive.name.encode())]
    home = str(Path.home()).encode()
    if len(home) > 1:
        redactions.append((home, b"[home]"))
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    with listing.open("xb") as stdout, errors.open("xb") as stderr:
        handles = {"stdout": stdout, "stderr": stderr}
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, env=env, bufsize=0)
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ, "stdout")
                selector.register(process.stderr, selectors.EVENT_READ, "stderr")
                while selector.get_map():
                    remaining = timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        raise AuditError("7z listing timed out")
                    for key, _ in selector.select(min(remaining, 0.25)):
                        chunk = os.read(key.fd, 65536)
                        stream = key.data
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        byte_counts[stream] += len(chunk)
                        limit = max_listing_bytes if stream == "stdout" else MAX_ERROR_BYTES
                        if byte_counts[stream] > limit:
                            raise AuditError(f"7z {stream} exceeded its byte limit")
                        buffers[stream].extend(chunk)
                        while b"\n" in buffers[stream]:
                            line, _, rest = buffers[stream].partition(b"\n")
                            buffers[stream] = bytearray(rest)
                            for old, replacement in redactions:
                                line = line.replace(old, replacement)
                            handles[stream].write(line + b"\n")
                        if len(buffers[stream]) > 1024**2:
                            raise AuditError("Technical listing line exceeds 1 MiB")
                for stream, buffer in buffers.items():
                    for old, replacement in redactions:
                        buffer = buffer.replace(old, replacement)
                    handles[stream].write(buffer)
                returncode = process.wait(timeout=max(0.01, timeout - (time.monotonic() - started)))
                if returncode != 0:
                    raise AuditError(f"7z listing returned nonzero exit code {returncode}")
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()
            process.stderr.close()
    return {"listing_bytes_received": byte_counts["stdout"], "stderr_bytes_received": byte_counts["stderr"]}


def audit_archive(archive: Path, report_dir: Path | None = None, *, timeout: float = 120,
                  max_listing_bytes: int = MAX_LISTING_BYTES) -> dict:
    """Create a unique report directory and return its summary, including paths."""
    if timeout <= 0 or not 0 < max_listing_bytes <= MAX_LISTING_BYTES:
        raise ValueError("Timeout must be positive; listing limit must be between 1 and 64 MiB")
    archive = Path(archive).expanduser().resolve(strict=True)
    if not archive.is_file():
        raise ValueError("Archive must be a regular file")
    base = Path(report_dir) if report_dir is not None else PROJECT_ROOT / "reports" / "data"
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc)
    destination = base / (stamp.strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:8])
    destination.mkdir()
    listing = destination / "technical_listing.txt"
    errors = destination / "errors.txt"
    manifest = destination / "members.jsonl"
    summary_path = destination / "archive_summary.json"
    before = archive.stat()
    summary = {
        "schema_version": 1, "created_at_utc": stamp.isoformat().replace("+00:00", "Z"),
        "archive_name": archive.name,
        "archive_fingerprint": {"size_bytes": before.st_size, "mtime_ns": before.st_mtime_ns},
        "fingerprint_note": "Size and mtime only; no payload hash or integrity test was run.",
        "operation": "list_only", "extracted": False, "status": "fail",
        "artifacts": {"listing": listing.name, "errors": errors.name, "manifest": manifest.name},
    }
    try:
        free = shutil.disk_usage(destination).free
        summary["free_disk_bytes_before"] = free
        # Reserve enough for listing, manifest, error output and bookkeeping.
        if free < 4 * max_listing_bytes + MAX_ERROR_BYTES:
            raise AuditError("Insufficient free disk space for bounded inventory reports")
        executable = shutil.which("7zz") or shutil.which("7z")
        if executable is None:
            raise AuditError("Install 7zz or 7z before running an archive inventory")
        summary["executable_name"] = Path(executable).name
        summary.update(_capture_listing(executable, archive, listing, errors, timeout, max_listing_bytes))
        after = archive.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise AuditError("Archive changed while its inventory was being read")
        summary.update(parse_listing(listing, manifest, archive_size=before.st_size))
        summary["status"] = "pass" if summary["inventory_guards_passed"] else "rejected"
    except (AuditError, OSError, UnicodeError, subprocess.TimeoutExpired) as error:
        summary["error"] = str(error) if isinstance(error, AuditError) else type(error).__name__
    summary["free_disk_bytes_after"] = shutil.disk_usage(destination).free
    with summary_path.open("x", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")
    # Local paths are returned to the caller, never embedded in the JSON report.
    return {**summary, "summary_path": str(summary_path), "report_dir": str(destination)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--report-dir", type=Path, help="Base directory; a unique UTC child is created.")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args(argv)
    try:
        summary = audit_archive(args.archive, args.report_dir, timeout=args.timeout)
    except (OSError, ValueError) as error:
        print(f"FAIL: could not start inventory ({type(error).__name__}).", file=sys.stderr)
        return 2
    print(f"Archive inventory: {summary['status'].upper()} (no extraction)")
    if "total_files" in summary:
        print(f"Members: {summary['total_members']}; files: {summary['total_files']}; directories: {summary['total_directories']}")
        print(f"Uncompressed bytes: {summary['total_uncompressed_bytes']}")
    if "error" in summary:
        print(f"Reason: {summary['error']}")
    print(f"Summary: {summary['summary_path']}")
    print(f"Artifacts: {summary['report_dir']}")
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
