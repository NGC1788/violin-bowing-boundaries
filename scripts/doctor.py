#!/usr/bin/env python3
"""Read-only environment checks and a tiny optional CUDA execution test.

Preflight uses only Python's standard library and never imports torch. Reports
deliberately exclude hostnames, serial numbers, GPU UUIDs, process information,
environment variables, command output, and authentication material.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import uuid


GIB = 1024**3
MIN_FREE_DISK_GIB = 20
MIN_GLIBC = (2, 28)
MIN_CUDA12_DRIVER = (525, 60, 13)
NATIVE_CUDA126_DRIVER = (560, 35, 5)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGES = (
    "torch", "torchvision", "torchaudio", "numpy", "scipy", "pandas",
    "scikit-learn", "matplotlib", "soundfile", "librosa",
    "opencv-python-headless", "imageio", "imageio-ffmpeg", "gpytorch",
)


def version_tuple(value: str) -> tuple[int, int, int] | None:
    match = re.fullmatch(r"\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?\s*", value)
    if not match:
        return None
    return tuple(int(part or 0) for part in match.groups())


def run_command(arguments: list[str], timeout: int = 15) -> subprocess.CompletedProcess[str] | None:
    """Keep subprocess output private; callers whitelist individual fields."""
    try:
        return subprocess.run(
            arguments, capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def add_check(report: dict, name: str, status: str, message: str, **details) -> None:
    report["checks"].append({
        "name": name, "status": status, "message": message, **details,
    })


def inspect_platform(report: dict, require_cuda: bool) -> None:
    system = platform.system()
    machine = platform.machine()
    information = {"system": system, "machine": machine}
    if system == "Linux":
        try:
            values = {}
            for line in Path("/etc/os-release").read_text().splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    if key in {"ID", "VERSION_ID"}:
                        values[key.lower()] = value.strip('"')
            information["distribution"] = values
        except OSError:
            pass
    report["platform"] = information
    compatible_target = system == "Linux" and machine == "x86_64"
    add_check(
        report, "server_platform", "pass" if compatible_target else ("fail" if require_cuda else "warning"),
        "Linux x86_64 server target confirmed." if compatible_target else
        "Outside the Linux x86_64 server target; local CPU checks do not validate the Ubuntu CUDA environment.",
    )
    if system == "Linux":
        libc_name, libc_version = platform.libc_ver()
        parsed = version_tuple(libc_version)
        report["platform"]["libc"] = {"name": libc_name, "version": libc_version}
        compatible_libc = libc_name == "glibc" and parsed is not None and parsed[:2] >= MIN_GLIBC
        add_check(
            report, "glibc", "pass" if compatible_libc else "fail",
            "Linux wheels require glibc >= 2.28.", minimum="2.28",
        )
    else:
        add_check(report, "glibc", "skip", "glibc check applies only to Linux.")


def inspect_resources(report: dict) -> None:
    minimum_free_gib = MIN_FREE_DISK_GIB if report.get("mode", "preflight") == "preflight" else 1
    ram_bytes = None
    if platform.system() == "Linux":
        try:
            match = re.search(r"^MemTotal:\s+(\d+)\s+kB$", Path("/proc/meminfo").read_text(), re.MULTILINE)
            if match:
                ram_bytes = int(match.group(1)) * 1024
        except OSError:
            pass
    elif platform.system() == "Darwin":
        result = run_command(["sysctl", "-n", "hw.memsize"])
        if result is not None and result.returncode == 0 and result.stdout.strip().isdigit():
            ram_bytes = int(result.stdout.strip())
    add_check(
        report, "system_ram", "pass" if ram_bytes is not None else "warning",
        "RAM inventory only; usable capacity and job-specific memory needs must be checked separately.",
        total_gib=round(ram_bytes / GIB, 2) if ram_bytes is not None else None,
    )
    try:
        usage = shutil.disk_usage(PROJECT_ROOT)
        add_check(
            report, "project_disk_space", "pass" if usage.free >= minimum_free_gib * GIB else "fail",
            f"Free-space floor: {minimum_free_gib} GiB for this check (20 before installation; 1 for tiny post-install checks). Datasets need additional space.",
            free_gib=round(usage.free / GIB, 2), total_gib=round(usage.total / GIB, 2),
            minimum_free_gib=minimum_free_gib,
        )
    except OSError:
        add_check(report, "project_disk_space", "fail", "Could not read free space on the project filesystem.")


def inspect_nvidia(report: dict, require_cuda: bool) -> None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        add_check(
            report, "nvidia_driver", "fail" if require_cuda else "warning",
            "nvidia-smi unavailable. NVIDIA/CUDA server capability is not verified.", gpus=[],
        )
        return
    result = run_command([
        executable,
        "--query-gpu=name,driver_version,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    if result is None or result.returncode != 0:
        add_check(
            report, "nvidia_driver", "fail" if require_cuda else "warning",
            "NVIDIA driver query failed or timed out; raw command output is intentionally omitted.", gpus=[],
        )
        return
    gpus = []
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) != 5:
            continue
        name, driver, total, free, utilization = (item.strip() for item in row)
        parsed_driver = version_tuple(driver)
        if parsed_driver is None:
            continue
        def numeric_or_none(value: str):
            return float(value) if re.fullmatch(r"\d+(?:\.\d+)?", value) else None
        gpus.append({
            "name": name, "driver_version": driver,
            "memory_total_mib": numeric_or_none(total),
            "memory_free_mib": numeric_or_none(free),
            "utilization_percent": numeric_or_none(utilization),
        })
    if not gpus:
        add_check(report, "nvidia_driver", "fail" if require_cuda else "warning", "No valid NVIDIA GPU entries returned.", gpus=[])
        return
    drivers = [version_tuple(gpu["driver_version"]) for gpu in gpus]
    if any(driver < MIN_CUDA12_DRIVER for driver in drivers):
        status = "fail"
        message = "Driver is below CUDA 12.x minor-compatibility minimum 525.60.13; ask the server administrator to inspect it."
    elif any(driver < NATIVE_CUDA126_DRIVER for driver in drivers):
        status = "warning"
        message = "Driver meets CUDA 12.x minor compatibility but is below 560.35.05; eager CUDA execution must pass, and JIT/torch.compile is not validated."
    else:
        status = "pass"
        message = "Driver meets the CUDA 12.6 Update 3 native-driver threshold. Hardware query alone does not prove PyTorch execution."
    add_check(report, "nvidia_driver", status, message, gpus=gpus)


def inspect_packages(report: dict) -> None:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    report["package_versions"] = versions


def inspect_torch(report: dict, require_cuda: bool) -> None:
    try:
        import torch
    except Exception as error:
        add_check(report, "torch_import", "fail", "PyTorch could not be imported. Install/sync the project environment before the full doctor check.", exception_type=type(error).__name__)
        return
    add_check(report, "torch_import", "pass", "PyTorch imported.", version=torch.__version__, compiled_cuda=torch.version.cuda)
    try:
        cuda_available = torch.cuda.is_available()
    except Exception as error:
        add_check(report, "torch_cuda_available", "fail" if require_cuda else "warning", "CUDA availability query failed.", available=False, exception_type=type(error).__name__)
        cuda_available = False
    else:
        add_check(
            report, "torch_cuda_available", "pass" if cuda_available else ("fail" if require_cuda else "warning"),
            "PyTorch sees CUDA; execution is checked separately." if cuda_available else
            "CUDA is unavailable to this Python environment. A CPU test is not an Ubuntu GPU validation.",
            available=cuda_available,
        )
    device = "cuda" if cuda_available else "cpu"
    try:
        # Tiny eager workload, without training, compilation, checkpoints, or data downloads.
        with torch.enable_grad():
            x = torch.linspace(-1, 1, 32 * 16, device=device).reshape(32, 16)
            weight = torch.full((16, 8), 0.125, device=device, requires_grad=True)
            prediction = x @ weight
            loss = prediction.square().mean()
            loss.backward()
            if device == "cuda":
                torch.cuda.synchronize()
            if weight.grad is None or not torch.isfinite(loss).item() or not torch.isfinite(weight.grad).all().item():
                raise ArithmeticError("Non-finite output or gradient")
            if not (weight.grad.abs().sum() > 0).item():
                raise ArithmeticError("Zero gradient")
            details = {"device": device, "loss": float(loss.item())}
            if device == "cuda":
                details["gpu_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
        add_check(report, "torch_forward_backward", "pass", "Small eager forward/backward test passed; CUDA was synchronized when used.", **details)
    except Exception as error:
        add_check(report, "torch_forward_backward", "fail", "Small forward/backward execution failed; no driver or environment changes were attempted.", device=device, exception_type=type(error).__name__)
    inspect_gpytorch(report, torch)
    inspect_torch_companions(report, torch, device)


def inspect_torch_companions(report: dict, torch, device: str) -> None:
    """Exercise binary companion packages; matching metadata is insufficient."""
    try:
        import torchvision
        boxes = torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.]], device=device)
        scores = torch.tensor([0.9, 0.8], device=device)
        keep = torchvision.ops.nms(boxes, scores, 0.5)
        if device == "cuda":
            torch.cuda.synchronize()
        if keep.tolist() != [0]:
            raise ArithmeticError("Unexpected NMS output")
        add_check(report, "torchvision_ops", "pass", "Small torchvision NMS operator executed.", device=device, version=torchvision.__version__)
    except Exception as error:
        add_check(report, "torchvision_ops", "fail", "Torchvision import/operator failed; inspect the locked wheel and runtime compatibility.", exception_type=type(error).__name__)
    try:
        import torchaudio
        signal = torch.sin(torch.arange(480, dtype=torch.float32) * 0.1)
        result = torchaudio.transforms.Resample(48000, 16000)(signal)
        if result.shape != (160,) or not torch.isfinite(result).all().item():
            raise ArithmeticError("Invalid resampling result")
        add_check(report, "torchaudio_transform", "pass", "Small CPU resampler executed. WAV IO uses soundfile; TorchCodec is not tested.", version=torchaudio.__version__, device="cpu")
    except Exception as error:
        add_check(report, "torchaudio_transform", "fail", "TorchAudio import/transform failed.", exception_type=type(error).__name__)


def inspect_gpytorch(report: dict, torch) -> None:
    try:
        import gpytorch
    except ModuleNotFoundError as error:
        if error.name == "gpytorch":
            add_check(report, "gpytorch_kernel", "skip", "GPyTorch is not installed; the optional kernel sanity check was skipped.")
        else:
            add_check(report, "gpytorch_kernel", "fail", "GPyTorch has a missing dependency.", exception_type=type(error).__name__)
        return
    except Exception as error:
        add_check(report, "gpytorch_kernel", "fail", "GPyTorch import failed.", exception_type=type(error).__name__)
        return
    try:
        with torch.no_grad():
            x = torch.linspace(-1, 1, 5, dtype=torch.float64).unsqueeze(-1)
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).double()
            covariance = kernel(x).to_dense()
            if covariance.shape != (5, 5) or not torch.isfinite(covariance).all().item():
                raise ArithmeticError("Invalid covariance")
            if not torch.allclose(covariance, covariance.T, atol=1e-10, rtol=1e-8):
                raise ArithmeticError("Asymmetric covariance")
            torch.linalg.cholesky(covariance + torch.eye(5, dtype=torch.float64) * 1e-6)
        add_check(report, "gpytorch_kernel", "pass", "A small CPU RBF covariance was finite, symmetric, and positive definite with 1e-6 diagonal jitter.", version=gpytorch.__version__, device="cpu")
    except Exception as error:
        add_check(report, "gpytorch_kernel", "fail", "GPyTorch kernel sanity check failed.", exception_type=type(error).__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="Use only the standard library; do not import torch or run GPU computations.")
    parser.add_argument("--require-cuda", action="store_true", help="Require a Linux x86_64 NVIDIA target; in full mode also require actual CUDA execution.")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports" / "environment", help="JSON destination (default: project reports/environment).")
    args = parser.parse_args(argv)
    timestamp = datetime.now(timezone.utc)
    report = {
        "schema_version": 1,
        "created_at_utc": timestamp.isoformat().replace("+00:00", "Z"),
        "mode": "preflight" if args.preflight else "full",
        "require_cuda": args.require_cuda,
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation()},
        "privacy": "Reports omit hostnames, serials, GPU UUIDs, PIDs, environment variables, tokens, and raw subprocess output.",
        "checks": [],
    }
    inspect_platform(report, args.require_cuda)
    inspect_resources(report)
    inspect_nvidia(report, args.require_cuda)
    if not args.preflight:
        inspect_packages(report)
        inspect_torch(report, args.require_cuda)
    else:
        add_check(report, "torch_execution", "skip", "Preflight does not import PyTorch and does not validate CUDA execution.")
    statuses = {item["status"] for item in report["checks"]}
    report["status"] = "fail" if "fail" in statuses else ("warning" if "warning" in statuses else "pass")
    report["cuda_execution_verified"] = any(
        item["name"] == "torch_forward_backward" and item["status"] == "pass" and item.get("device") == "cuda"
        for item in report["checks"]
    )
    filename = f"{timestamp.strftime('%Y%m%dT%H%M%S_%fZ')}_{uuid.uuid4().hex[:8]}.json"
    try:
        args.report_dir.mkdir(parents=True, exist_ok=True)
        report_path = args.report_dir / filename
        with report_path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    except (OSError, ValueError) as error:
        print(f"FAIL: could not save the environment report ({type(error).__name__}).", file=sys.stderr)
        return 2
    for item in report["checks"]:
        print(f"{item['status'].upper():7} {item['name']}: {item['message']}")
        if "free_gib" in item:
            print(f"        Disk free: {item['free_gib']} GiB")
        if item["name"] == "system_ram":
            print(f"        RAM total: {item.get('total_gib')} GiB")
        for gpu in item.get("gpus", []):
            print(f"        GPU: {gpu['name']}; driver: {gpu['driver_version']}; VRAM free/total: {gpu['memory_free_mib']}/{gpu['memory_total_mib']} MiB")
    print(f"Overall: {report['status'].upper()}; CUDA execution verified: {report['cuda_execution_verified']}")
    print(f"Report: {report_path}")
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
