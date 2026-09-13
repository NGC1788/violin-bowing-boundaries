#!/usr/bin/env python3
"""Inspect the bowed-string Zenodo collection; download ONLY by explicit command.

Python standard library only. Run from the research project directory.
  python scripts/zenodo_catalog.py catalog
  python scripts/zenodo_catalog.py catalog --record 17822016
  python scripts/zenodo_catalog.py download --record 17749110 \
      --file 'EXACT_FILENAME_FROM_CATALOG.7z' --max-gib 16

--max-gib bounds ONE compressed file, not extraction size or accumulated storage.
Nothing is extracted. No labels, train/test splits, or audio are inferred.
"""

import argparse
import datetime as dt
import errno
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import shutil
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


RECORDS = {
    1: "17749110", 2: "17782542", 3: "17782552", 4: "17782565",
    5: "17782538", 6: "17795326", 7: "17822016", 8: "17822037",
}
DEFAULT_RECORDS = [RECORDS[p] for p in (1, 2, 6)]
GIB = 1024 ** 3
CHUNK = 1024 ** 2
MAX_METADATA_BYTES = 10 * CHUNK
USER_AGENT = "violin-research-starter/1.0 (metadata-first; stdlib urllib)"


class PrematureEOF(OSError):
    """A validated response ended before all published bytes arrived."""


def validate_retry_policy(retries, retry_delay):
    if not isinstance(retries, int) or not 0 <= retries <= 20:
        raise ValueError("--retries must be an integer from 0 to 20 (additional attempts)")
    if not math.isfinite(retry_delay) or not 0 <= retry_delay <= 60:
        raise ValueError("--retry-delay must be finite and between 0 and 60 seconds")


def is_transient(error):
    # HTTPError subclasses URLError: check its status before the general network case.
    if isinstance(error, urllib.error.HTTPError):
        return error.code in (408, 429) or 500 <= error.code <= 599
    if isinstance(error, ssl.SSLCertVerificationError):
        return False
    if isinstance(error, urllib.error.URLError):
        return not isinstance(error.reason, ssl.SSLCertVerificationError)
    if isinstance(error, (PrematureEOF, http.client.IncompleteRead, TimeoutError, ConnectionError)):
        return True
    return isinstance(error, OSError) and error.errno in {
        errno.ECONNRESET, errno.ECONNABORTED, errno.EHOSTUNREACH,
        errno.ENETUNREACH, errno.EPIPE, errno.ETIMEDOUT,
    }


def with_retries(operation, retries, retry_delay, label):
    """Retry only transient transport failures, never integrity or local-data errors."""
    validate_retry_policy(retries, retry_delay)
    for attempt in range(retries + 1):
        try:
            return operation()
        except Exception as error:
            if not is_transient(error) or attempt == retries:
                raise
            delay = min(60, retry_delay * (2 ** attempt))
            reason = f"HTTP {error.code}" if isinstance(error, urllib.error.HTTPError) else type(error).__name__
            if isinstance(error, urllib.error.HTTPError):
                error.close()
            print(f"{label}: {reason}; retry {attempt + 1}/{retries} in {delay:g}s", flush=True)
            time.sleep(delay)


def safe_url(url):
    parsed = urllib.parse.urlparse(url)
    if (parsed.scheme != "https" or parsed.hostname != "zenodo.org"
            or parsed.port not in (None, 443) or parsed.username or parsed.password):
        raise ValueError("Only HTTPS URLs on zenodo.org are allowed: " + url)
    return url


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        safe_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_url(url, extra_headers=None):
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    headers.update(extra_headers or {})
    opener = urllib.request.build_opener(SafeRedirect())
    response = opener.open(urllib.request.Request(safe_url(url), headers=headers), timeout=60)
    try:
        safe_url(response.geturl())
    except Exception:
        response.close()
        raise
    return response


def fetch_record(record, retries=3, retry_delay=10):
    if not str(record).isdigit():
        raise ValueError("Record ID must contain digits only")
    def fetch():
        with open_url("https://zenodo.org/api/records/" + str(record)) as response:
            raw = response.read(MAX_METADATA_BYTES + 1)
            length = response.headers.get("Content-Length")
            if length is not None and len(raw) < min(int(length), MAX_METADATA_BYTES + 1):
                raise PrematureEOF("Metadata response ended before its declared length")
            return raw
    raw = with_retries(fetch, retries, retry_delay, "Metadata")
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError("Metadata exceeded the 10 MiB safety bound")
    data = json.loads(raw)
    if not isinstance(data.get("files"), list) or not data.get("id"):
        raise ValueError("Unexpected Zenodo record response")
    return data


def save_manifest(record, data, manifest_dir):
    manifest_dir.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc)
    target = manifest_dir / ("zenodo_" + str(record) + "_" + now.strftime("%Y%m%dT%H%M%S.%fZ") + ".json")
    envelope = {
        "fetched_at_utc": now.isoformat(), "requested_record_id": str(record),
        "resolved_record_id": str(data["id"]),
        "source_url": "https://zenodo.org/api/records/" + str(record),
        "record": data,
    }
    with target.open("x", encoding="utf-8") as handle:
        json.dump(envelope, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return target


def print_record(requested, data, manifest):
    print("\n" + data.get("metadata", {}).get("title", "Untitled record"))
    print(f"Requested ID {requested}; resolved version ID {data['id']}")
    print("Manifest:", manifest)
    total = 0
    for item in data["files"]:
        size = int(item["size"])
        total += size
        print(f"  {item['key']} | {size:,} bytes | {size / GIB:.3f} GiB | {item.get('checksum', 'NO CHECKSUM')}")
    print(f"Compressed total: {total / GIB:.3f} GiB. Extraction requires additional space.")


def md5_file(path):
    digest = hashlib.md5()  # Published data-integrity checksum, not cryptographic authentication.
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def file_spec(data, filename):
    if (not filename or filename in (".", "..") or Path(filename).name != filename
            or "/" in filename or "\\" in filename or any(ord(c) < 32 for c in filename)):
        raise ValueError("File must be an exact plain filename, without directory separators")
    matches = [item for item in data["files"] if item.get("key") == filename]
    if len(matches) != 1:
        raise ValueError("Exact file not found; run catalog and copy its full filename")
    item = matches[0]
    size = int(item["size"])
    if size <= 0:
        raise ValueError("Expected a positive declared file size")
    checksum = item.get("checksum", "")
    if not re.fullmatch(r"md5:[0-9a-fA-F]{32}", checksum):
        raise ValueError("Download requires Zenodo's published MD5 checksum")
    url = item.get("links", {}).get("self")
    if not url:
        raise ValueError("Record has no download URL")
    return size, checksum.split(":", 1)[1].lower(), safe_url(url)


def transfer_file(url, part, size, offset):
    """Append only when the server confirms the exact requested byte range."""
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with open_url(url, headers) as response:
        status = response.status
        encoding = response.headers.get("Content-Encoding", "identity").lower()
        if encoding not in ("", "identity"):
            raise ValueError("Unexpected encoded response; byte-range verification is unsafe")
        if offset:
            expected_range = f"bytes {offset}-{size - 1}/{size}"
            if status != 206 or response.headers.get("Content-Range") != expected_range:
                raise ValueError("Server did not honor the precise resume range; .part was preserved")
        elif status != 200:
            raise ValueError(f"Expected HTTP 200 for fresh download, got {status}")
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) != size - offset:
            raise ValueError("Response size differs from published metadata")
        received = offset
        next_progress = offset + 256 * CHUNK
        mode = "ab" if offset else "xb"
        with part.open(mode) as handle:
            while True:
                interrupted = None
                try:
                    block = response.read(min(CHUNK, size - received + 1))
                except http.client.IncompleteRead as error:
                    # Keep verified-position bytes delivered with the exception before resuming.
                    block = error.partial
                    interrupted = error
                if not block:
                    if interrupted is not None:
                        raise interrupted
                    break
                if received + len(block) > size:
                    raise ValueError("Response exceeded declared size; extra bytes were not saved")
                handle.write(block)
                received += len(block)
                if interrupted is not None:
                    handle.flush()
                    os.fsync(handle.fileno())
                    raise interrupted
                if received >= next_progress:
                    print(f"  {received / GIB:.2f} / {size / GIB:.2f} GiB", flush=True)
                    next_progress = received + 256 * CHUNK
            handle.flush()
            os.fsync(handle.fileno())
        if received != size:
            raise PrematureEOF(f"Incomplete download ({received}/{size} bytes); .part retained for resume")


def download(args, data):
    retries = getattr(args, "retries", 3)
    retry_delay = getattr(args, "retry_delay", 10)
    validate_retry_policy(retries, retry_delay)
    size, checksum, url = file_spec(data, args.file)
    if not math.isfinite(args.max_gib) or args.max_gib <= 0:
        raise ValueError("--max-gib must be a positive finite number")
    if size > args.max_gib * GIB:
        raise ValueError(f"File is {size / GIB:.3f} GiB, above --max-gib {args.max_gib}")
    directory = Path(args.data_dir) / "raw" / "zenodo" / str(data["id"])
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / args.file
    part = directory / (args.file + ".part")
    lock = directory / (args.file + ".download.lock")
    if target.is_symlink() or part.is_symlink():
        raise ValueError("Refusing symlink targets")
    # Exclusive lock prevents two copies of this script from writing the same file.
    try:
        lock_handle = lock.open("x", encoding="utf-8")
    except FileExistsError:
        raise ValueError(f"Download lock exists: {lock}. Check for a running download before manually removing a stale lock.")
    try:
        with lock_handle:
            lock_handle.write(f"pid={os.getpid()}\n")
        if target.exists():
            if target.stat().st_size == size and md5_file(target) == checksum:
                print("Already present and MD5 verified:", target)
                return
            raise ValueError("Existing destination differs; it will not be overwritten: " + str(target))
        def attempt_transfer():
            if part.is_symlink():
                raise ValueError("Refusing symlink intermediate file")
            offset = part.stat().st_size if part.exists() else 0
            if offset > size:
                raise ValueError("Existing .part exceeds published size; inspect or remove it manually")
            if part.exists() and offset == 0:
                part.unlink()  # Only the empty intermediate for this exact target.
            remaining = size - offset
            reserve = max(GIB, int(size * 0.05))
            free = shutil.disk_usage(directory).free
            if free < remaining + reserve:
                raise ValueError(f"Insufficient free space: need {(remaining + reserve) / GIB:.2f} GiB including reserve")
            print(f"Explicit download: {args.file}; {size / GIB:.3f} GiB compressed; resume at {offset / GIB:.3f} GiB", flush=True)
            if remaining:
                transfer_file(url, part, size, offset)
        with_retries(attempt_transfer, retries, retry_delay, "Download")
        print("Checking MD5 (may take several minutes)...", flush=True)
        if part.stat().st_size != size or md5_file(part) != checksum:
            raise ValueError("MD5 mismatch. Untrusted .part retained for inspection; remove it manually before a fresh retry")
        # Hard-link publication is atomic and fails if target unexpectedly exists.
        os.link(part, target)
        part.unlink()
        print("Downloaded and MD5 verified:", target)
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    catalog = sub.add_parser("catalog", help="Fetch small metadata only; defaults to Parts 1, 2, 6")
    catalog.add_argument("--record", action="append", help="Zenodo record ID; repeat to select several")
    catalog.add_argument("--data-dir", default="data", help="Research data directory (default: ./data)")
    get = sub.add_parser("download", help="Download ONE explicitly named file, subject to a hard size bound")
    get.add_argument("--record", required=True)
    get.add_argument("--file", required=True, help="Exact filename printed by catalog")
    get.add_argument("--max-gib", type=float, required=True, help="Maximum allowed compressed file size in GiB")
    get.add_argument("--data-dir", default="data")
    for command in (catalog, get):
        command.add_argument("--retries", type=int, default=3, help="Additional attempts for transient network failures (0-20; default: 3)")
        command.add_argument("--retry-delay", type=float, default=10, help="Initial retry delay in seconds; doubles up to 60s (default: 10)")
    args = parser.parse_args()
    try:
        selected = (args.record or DEFAULT_RECORDS) if args.command == "catalog" else [args.record]
        for record in selected:
            data = fetch_record(record, args.retries, args.retry_delay)
            manifest = save_manifest(record, data, Path(args.data_dir) / "manifests")
            print_record(record, data, manifest)
            if args.command == "download":
                download(args, data)
        if args.command == "catalog":
            print("\nMetadata only: NO dataset files downloaded. Do not download every part.")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted. Any .part file is retained for a later explicit resume.", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, TypeError, http.client.HTTPException) as exc:
        print("ERROR:", exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
