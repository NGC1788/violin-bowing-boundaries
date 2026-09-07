"""Portable doctor regression checks; no GPU or third-party package required."""

import builtins
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "doctor.py"
SPEC = importlib.util.spec_from_file_location("violin_doctor_under_test", SCRIPT)
doctor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(doctor)


def named_check(report, name):
    return next(item for item in report["checks"] if item["name"] == name)


@contextlib.contextmanager
def passing_hardware():
    """Make report/CLI tests independent of the machine running this suite."""
    def platform_check(report, require_cuda):
        doctor.add_check(report, "server_platform", "pass", "Test Linux target")

    def resources_check(report):
        doctor.add_check(report, "project_disk_space", "pass", "Test free space")

    def nvidia_check(report, require_cuda):
        doctor.add_check(report, "nvidia_driver", "pass", "Test driver")

    with (
        mock.patch.object(doctor, "inspect_platform", side_effect=platform_check),
        mock.patch.object(doctor, "inspect_resources", side_effect=resources_check),
        mock.patch.object(doctor, "inspect_nvidia", side_effect=nvidia_check),
    ):
        yield


class NvidiaDriverTests(unittest.TestCase):
    def inspect(self, stdout, returncode=0, stderr=""):
        report = {"checks": []}
        result = subprocess.CompletedProcess([], returncode, stdout, stderr)
        with (
            mock.patch.object(doctor.shutil, "which", return_value="nvidia-smi"),
            mock.patch.object(doctor, "run_command", return_value=result) as command,
        ):
            doctor.inspect_nvidia(report, require_cuda=True)
        return named_check(report, "nvidia_driver"), command.call_args.args[0]

    def test_driver_compatibility_boundaries(self):
        cases = [
            ("525.60.12", "fail"),
            ("525.60.13", "warning"),
            ("550.54.14", "warning"),
            ("560.35.04", "warning"),
            ("560.35.05", "pass"),
            ("580.95.05", "pass"),
        ]
        for version, expected in cases:
            with self.subTest(version=version):
                check, _ = self.inspect(
                    f"NVIDIA RTX A5000, {version}, 24564, 20000, 3\n"
                )
                self.assertEqual(check["status"], expected)
                self.assertEqual(check["gpus"][0]["driver_version"], version)

    def test_query_omits_device_identifiers_and_raw_output(self):
        canary = "PRIVATE_DIAGNOSTIC_CANARY"
        check, command = self.inspect(
            "NVIDIA RTX A5000, 560.35.05, 24564, 20000, 3\n",
            stderr=canary,
        )
        self.assertNotIn(canary, json.dumps(check))
        query = next(argument for argument in command if argument.startswith("--query-gpu="))
        fields = query.removeprefix("--query-gpu=").split(",")
        self.assertEqual(set(fields), {
            "name", "driver_version", "memory.total", "memory.free", "utilization.gpu",
        })

    def test_empty_malformed_or_failed_query_is_not_gpu_validation(self):
        cases = [
            ("", 0),
            ("malformed output\n", 0),
            ("NVIDIA RTX A5000, unknown, 24564, 20000, 3\n", 0),
            ("NVIDIA RTX A5000, 580.95.05, 24564, 20000, 3\n", 1),
        ]
        for stdout, returncode in cases:
            with self.subTest(stdout=stdout, returncode=returncode):
                check, _ = self.inspect(stdout, returncode)
                self.assertEqual(check["status"], "fail")

    def test_unavailable_utilization_and_free_memory_are_nullable(self):
        check, _ = self.inspect(
            "NVIDIA RTX A5000, 560.35.05, 24564, N/A, [Not Supported]\n"
        )
        self.assertEqual(check["status"], "pass")
        self.assertIsNone(check["gpus"][0]["memory_free_mib"])
        self.assertIsNone(check["gpus"][0]["utilization_percent"])


class PlatformAndDiskTests(unittest.TestCase):
    def test_glibc_minimum_and_unsupported_libc(self):
        cases = [
            (("glibc", "2.27"), "fail"),
            (("glibc", "2.28"), "pass"),
            (("glibc", "2.39"), "pass"),
            (("musl", "1.2.5"), "fail"),
            (("", ""), "fail"),
        ]
        for libc, expected in cases:
            with self.subTest(libc=libc):
                report = {"checks": []}
                with (
                    mock.patch.object(doctor.platform, "system", return_value="Linux"),
                    mock.patch.object(doctor.platform, "machine", return_value="x86_64"),
                    mock.patch.object(doctor.platform, "libc_ver", return_value=libc),
                    mock.patch.object(Path, "read_text", return_value='ID=ubuntu\nVERSION_ID="24.04"\n'),
                ):
                    doctor.inspect_platform(report, require_cuda=True)
                self.assertEqual(named_check(report, "glibc")["status"], expected)

    def test_non_server_platform_can_check_cpu_but_cannot_pass_required_cuda(self):
        for require_cuda, expected in [(False, "warning"), (True, "fail")]:
            with self.subTest(require_cuda=require_cuda):
                report = {"checks": []}
                with (
                    mock.patch.object(doctor.platform, "system", return_value="Darwin"),
                    mock.patch.object(doctor.platform, "machine", return_value="arm64"),
                ):
                    doctor.inspect_platform(report, require_cuda)
                self.assertEqual(named_check(report, "server_platform")["status"], expected)

    def test_twenty_gibibyte_free_disk_floor(self):
        for free, expected in [(20 * doctor.GIB - 1, "fail"), (20 * doctor.GIB, "pass")]:
            with self.subTest(free=free):
                report = {"checks": []}
                usage = SimpleNamespace(free=free, total=100 * doctor.GIB)
                with (
                    mock.patch.object(doctor.platform, "system", return_value="TestOS"),
                    mock.patch.object(doctor.shutil, "disk_usage", return_value=usage),
                ):
                    doctor.inspect_resources(report)
                check = named_check(report, "project_disk_space")
                self.assertEqual(check["status"], expected)
                self.assertEqual(check["minimum_free_gib"], 20)


class PreflightAndReportTests(unittest.TestCase):
    def test_post_install_space_floor_does_not_reapply_installation_reserve(self):
        for free, expected in [(doctor.GIB - 1, "fail"), (doctor.GIB, "pass"), (10 * doctor.GIB, "pass")]:
            with self.subTest(free=free):
                report = {"mode": "full", "checks": []}
                with mock.patch.object(doctor.platform, "system", return_value="TestOS"), mock.patch.object(
                    doctor.shutil, "disk_usage", return_value=SimpleNamespace(free=free, total=100 * doctor.GIB)
                ):
                    doctor.inspect_resources(report)
                check = named_check(report, "project_disk_space")
                self.assertEqual(check["status"], expected)
                self.assertEqual(check["minimum_free_gib"], 1)

    def test_preflight_never_imports_ml_packages_and_writes_unique_reports(self):
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "torchvision", "torchaudio", "gpytorch"}:
                raise AssertionError("Preflight attempted an ML import")
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            with (
                passing_hardware(),
                mock.patch("builtins.__import__", side_effect=guarded_import),
                mock.patch.object(doctor, "inspect_torch", side_effect=AssertionError("Preflight attempted torch inspection")),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                for _ in range(2):
                    code = doctor.main([
                        "--preflight", "--require-cuda", "--report-dir", folder,
                    ])
                    self.assertEqual(code, 0)
            paths = list(Path(folder).glob("*.json"))
            self.assertEqual(len(paths), 2)
            for path in paths:
                report = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(report["mode"], "preflight")
                self.assertFalse(report["cuda_execution_verified"])
                self.assertTrue(report["created_at_utc"].endswith("Z"))
                self.assertEqual(named_check(report, "torch_execution")["status"], "skip")

    def test_preflight_failure_returns_one_and_saves_failed_report(self):
        def failing_driver(report, require_cuda):
            doctor.add_check(report, "nvidia_driver", "fail", "Test unavailable driver")

        with tempfile.TemporaryDirectory() as folder:
            with (
                passing_hardware(),
                mock.patch.object(doctor, "inspect_nvidia", side_effect=failing_driver),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                code = doctor.main(["--preflight", "--require-cuda", "--report-dir", folder])
            self.assertEqual(code, 1)
            report = json.loads(next(Path(folder).glob("*.json")).read_text())
            self.assertEqual(report["status"], "fail")
            self.assertFalse(report["cuda_execution_verified"])

    def test_report_write_failure_returns_two_without_overwriting_file(self):
        with tempfile.TemporaryDirectory() as folder:
            occupied = Path(folder) / "existing-file"
            occupied.write_text("preserve this content", encoding="utf-8")
            with (
                passing_hardware(),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                code = doctor.main(["--preflight", "--report-dir", str(occupied)])
            self.assertEqual(code, 2)
            self.assertEqual(occupied.read_text(), "preserve this content")


if __name__ == "__main__":
    unittest.main()
