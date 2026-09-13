import hashlib
import importlib.util
import io
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("install_7zip", Path(__file__).resolve().parents[1] / "scripts/install_7zip.py")
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


class ToolInstallationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binary = b"fixture executable"
        for name, value in [("ROOT", self.root), ("BINARY_SIZE", len(self.binary)),
                            ("BINARY_SHA256", hashlib.sha256(self.binary).hexdigest())]:
            patch = mock.patch.object(installer, name, value)
            patch.start()
            self.addCleanup(patch.stop)

    def package(self, *, duplicate=False, symlink=False):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:xz") as archive:
            entry = tarfile.TarInfo("7zz")
            entry.size = len(self.binary)
            if symlink:
                entry.type = tarfile.SYMTYPE
                entry.linkname = "../../outside"
                entry.size = 0
            archive.addfile(entry, None if symlink else io.BytesIO(self.binary))
            if duplicate:
                archive.addfile(entry, io.BytesIO(self.binary))
            unrelated = tarfile.TarInfo("../../should-not-be-extracted")
            archive.addfile(unrelated, io.BytesIO(b""))
        package = buffer.getvalue()
        for name, value in [("PACKAGE_SIZE", len(package)), ("PACKAGE_SHA256", hashlib.sha256(package).hexdigest())]:
            patch = mock.patch.object(installer, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        return package

    def linux(self):
        for name, value in [("system", "Linux"), ("machine", "x86_64")]:
            patch = mock.patch.object(installer.platform, name, return_value=value)
            patch.start()
            self.addCleanup(patch.stop)

    def test_only_verified_regular_binary_is_read(self):
        self.assertEqual(installer.binary_from_package(self.package()), self.binary)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_bad_package_and_binary_hashes_reject(self):
        package = self.package()
        with self.assertRaisesRegex(ValueError, "package size/SHA256"):
            installer.binary_from_package(package + b"x")
        with mock.patch.object(installer, "BINARY_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "executable size/SHA256"):
                installer.binary_from_package(package)

    def test_duplicate_and_symlink_binary_reject(self):
        for options in ({"duplicate": True}, {"symlink": True}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "one regular"):
                installer.binary_from_package(self.package(**options))

    def test_mac_platform_rejected_before_network_or_write(self):
        with mock.patch.object(installer.platform, "system", return_value="Darwin"), mock.patch.object(installer, "download_package") as download:
            with self.assertRaisesRegex(ValueError, "Ubuntu"):
                installer.install()
            download.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_install_then_reuse_without_network_and_reject_tampering(self):
        self.linux()
        package = self.package()
        with mock.patch.object(installer, "download_package", return_value=package) as download, mock.patch.object(
            installer.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, b"7-Zip (z) 26.03 (x64)\n")
        ) as run:
            executable = installer.install()
            self.assertEqual(executable.read_bytes(), self.binary)
            self.assertEqual(executable.stat().st_mode & 0o777, 0o755)
            self.assertTrue((executable.parent / "source.json").is_file())
            self.assertEqual(installer.install(), executable)
            self.assertEqual(download.call_count, 1)
            self.assertEqual(run.call_count, 2)
            executable.write_bytes(b"x" * len(self.binary))
            with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                installer.install()
            self.assertEqual(run.call_count, 2)
            self.assertEqual(executable.read_bytes(), b"x" * len(self.binary))

    def test_failed_execution_does_not_publish_install(self):
        self.linux()
        package = self.package()
        with mock.patch.object(installer, "download_package", return_value=package), mock.patch.object(
            installer.subprocess, "run", side_effect=OSError("Cannot execute")
        ):
            with self.assertRaises(OSError):
                installer.install()
        self.assertEqual(list((self.root / ".local-tools").iterdir()), [])

    def test_wrong_reported_version_rejects(self):
        with mock.patch.object(installer.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, b"7-Zip (z) 26.030 (x64)")):
            with self.assertRaisesRegex(ValueError, "expected version"):
                installer.check_executable(self.root / "7zz")

    def test_symlink_destination_preserved_before_network(self):
        self.linux()
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        (self.root / ".local-tools").symlink_to(elsewhere, target_is_directory=True)
        with mock.patch.object(installer, "download_package") as download:
            with self.assertRaisesRegex(ValueError, "symlink"):
                installer.install()
            download.assert_not_called()
        self.assertEqual(list(elsewhere.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
