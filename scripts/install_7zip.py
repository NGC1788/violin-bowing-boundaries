#!/usr/bin/env python3
"""Install pinned official 7-Zip in this project, without sudo or shell-profile edits."""
from __future__ import annotations

import hashlib
import http.client
import io
import json
from pathlib import Path
import platform
import re
import ssl
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
VERSION = "26.03"
URL = "https://github.com/ip7z/7zip/releases/download/26.03/7z2603-linux-x64.tar.xz"
PACKAGE_SIZE = 1575072
PACKAGE_SHA256 = "dc99eff5008f1ab79bd7084c68513701547a808a89502bf4133683535ab3c695"
BINARY_SIZE = 2882120
BINARY_SHA256 = "3d52c92deb7e9f1bd059692eefc33f86a144cdc548acc4b6f4c809a4dd7bc369"


def binary_from_package(package):
    if len(package) != PACKAGE_SIZE or hashlib.sha256(package).hexdigest() != PACKAGE_SHA256:
        raise ValueError("Official package size/SHA256 mismatch; nothing will be installed")
    with tarfile.open(fileobj=io.BytesIO(package), mode="r:xz") as archive:
        matches = [entry for entry in archive.getmembers() if entry.name == "7zz"]
        if len(matches) != 1 or not matches[0].isfile() or matches[0].size != BINARY_SIZE:
            raise ValueError("Expected one regular 7zz executable with the pinned size")
        with archive.extractfile(matches[0]) as source:
            binary = source.read(BINARY_SIZE + 1)
    if len(binary) != BINARY_SIZE or hashlib.sha256(binary).hexdigest() != BINARY_SHA256:
        raise ValueError("Official executable size/SHA256 mismatch")
    return binary


def download_package():
    for attempt in range(3):
        try:
            request = urllib.request.Request(URL, headers={"User-Agent": "violin-research-tool-setup/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                package = response.read(PACKAGE_SIZE + 1)
            return package
        except (OSError, http.client.HTTPException) as error:
            retryable = True
            if isinstance(error, urllib.error.HTTPError):
                retryable = error.code in (408, 429) or 500 <= error.code < 600
                error.close()
            if isinstance(error, ssl.SSLCertVerificationError) or (
                isinstance(error, urllib.error.URLError) and isinstance(error.reason, ssl.SSLCertVerificationError)
            ):
                retryable = False
            if not retryable or attempt == 2:
                raise
            print(f"Download interrupted; retry {attempt + 1}/2.", flush=True)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("Unreachable download state")


def check_executable(executable):
    result = subprocess.run([str(executable), "i"], stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20, check=True)
    if not re.search(rb"7-Zip(?: \(z\))? " + re.escape(VERSION.encode()) + rb"(?:\s|$)", result.stdout):
        raise ValueError("7-Zip execution did not report the expected version")


def install():
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise ValueError("Run this installer on the Ubuntu x86_64 server, not the Mac")
    tools_dir = ROOT / ".local-tools"
    destination = tools_dir / ("7zip-" + VERSION)
    executable = destination / "7zz"
    if tools_dir.is_symlink() or destination.is_symlink() or executable.is_symlink():
        raise ValueError("Refusing a symlink tool destination")
    if destination.exists():
        if not executable.is_file() or executable.stat().st_size != BINARY_SIZE:
            raise ValueError("Existing tool directory differs; it will not be overwritten")
        if hashlib.sha256(executable.read_bytes()).hexdigest() != BINARY_SHA256:
            raise ValueError("Existing 7zz SHA256 differs; it will not be executed or overwritten")
        check_executable(executable)
        return executable
    print("Downloading official 7-Zip 26.03 for Linux x86_64 (about 1.5 MiB).", flush=True)
    binary = binary_from_package(download_package())
    tools_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".7zip-", dir=tools_dir) as temporary:
        staging = Path(temporary)
        candidate = staging / "7zz"
        with candidate.open("xb") as output:
            output.write(binary)
        candidate.chmod(0o755)
        check_executable(candidate)
        (staging / "source.json").write_text(json.dumps({"version": VERSION, "url": URL,
            "package_sha256": PACKAGE_SHA256, "binary_sha256": BINARY_SHA256}, indent=2) + "\n")
        if destination.exists():
            raise ValueError("Tool destination appeared during installation; existing files preserved")
        staging.rename(destination)
    return executable


def main():
    if len(sys.argv) > 1:
        if sys.argv[1:] in (["--help"], ["-h"]):
            print(__doc__)
            return 0
        print("This installer takes no arguments.", file=sys.stderr)
        return 2
    try:
        executable = install()
        print("USER TOOL SETUP: PASS")
        print("7-Zip:", executable.relative_to(ROOT))
        print("scripts/run.sh uses this tool automatically. No sudo or tmux installation is needed.")
        return 0
    except (OSError, ValueError, http.client.HTTPException, tarfile.TarError, subprocess.SubprocessError) as error:
        print("USER TOOL SETUP FAILED:", error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
