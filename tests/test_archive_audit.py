"""Portable technical-listing and list-only subprocess regression tests."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "archive_audit.py"
SPEC = importlib.util.spec_from_file_location("archive_audit_under_test", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


HEADER = """7-Zip (z) 25.01 : Copyright

Scanning the drive for archives:
1 file, 42 bytes (1 KiB)

Listing archive: fixture.7z

--
Path = fixture.7z
Type = 7z
Physical Size = 42
Headers Size = 20
Method = LZMA2:12
Solid = +
Blocks = 1

----------
"""


def member(path, size="12", extra="", folder=False):
    return (f"Path = {path}\nSize = {size}\nPacked Size = \n"
            f"Modified = 2025-01-01 00:00:00.0000000\n"
            f"Attributes = {'D' if folder else 'A'}\nFolder = {'+' if folder else '-'}\n"
            f"Encrypted = -\nCRC = \nMethod = LZMA2:12\nBlock = {'' if folder else '0'}\n"
            f"{extra}\n")


class ListingTests(unittest.TestCase):
    def parse(self, text):
        with tempfile.TemporaryDirectory() as folder:
            listing = Path(folder) / "listing.txt"
            manifest = Path(folder) / "members.jsonl"
            listing.write_text(text, encoding="utf-8")
            result = audit.parse_listing(listing, manifest, archive_size=42)
            return result, [json.loads(line) for line in manifest.read_text().splitlines()]

    def test_unicode_equals_dirs_counts_and_candidates(self):
        result, rows = self.parse(HEADER + member("측정 = 1", "0", folder=True)
            + member("측정 = 1/whole.csv", "13") + member("측정 = 1/betas.csv", "7")
            + member("측정 = 1/timestamps", "3") + member("측정 = 1/regime_labels.json", "5"))
        self.assertEqual((result["total_members"], result["total_files"], result["total_directories"]), (5, 4, 1))
        self.assertEqual(result["total_uncompressed_bytes"], 28)
        self.assertEqual(rows[1]["path"], "측정 = 1/whole.csv")
        self.assertTrue(result["inventory_guards_passed"])
        self.assertEqual(result["candidate_members"]["possible_label_names"]["count"], 1)

    def test_candidate_examples_bounded_and_names_exact(self):
        text = HEADER + "".join(member(f"{n}/whole.csv") for n in range(50)) + member("Whole.csv")
        result, _ = self.parse(text)
        candidate = result["candidate_members"]["whole_csv"]
        self.assertEqual(candidate["count"], 51)
        self.assertEqual(len(candidate["examples"]), 30)

    def test_numbered_published_archive_names(self):
        result, _ = self.parse(HEADER + member("v1/whole_2000.csv")
            + member("v1/beta_2000.csv") + member("v1/timestamp_2000.csv")
            + member("v1/not_whole_2000.csv"))
        for key in ("whole_csv", "betas_csv", "timestamps"):
            self.assertEqual(result["candidate_members"][key]["count"], 1)

    def test_paths_duplicates_and_links_rejected(self):
        for name, extra in [("../x", ""), ("/etc/x", ""), ("C:\\x", ""),
                            ("a\\..\\x", ""), ("a//x", ""),
                            ("link", "Symbolic Link = ../../target\n"),
                            ("link", "Hard Link = victim\n")]:
            with self.subTest(name=name, extra=extra):
                result, _ = self.parse(HEADER + member(name, extra=extra))
                self.assertFalse(result["inventory_guards_passed"])
        for names in [("a", "a"), ("A", "a"), ("é", "e\u0301")]:
            result, _ = self.parse(HEADER + member(names[0]) + member(names[1]))
            self.assertFalse(result["inventory_guards_passed"])

    def test_encryption_and_unix_link_attributes_rejected(self):
        for text in [member("secret").replace("Encrypted = -", "Encrypted = +"),
                     member("link").replace("Attributes = A", "Attributes = A_ lrwxrwxrwx")]:
            result, _ = self.parse(HEADER + text)
            self.assertFalse(result["inventory_guards_passed"])

    def test_malformed_sizes_blocks_headers_and_truncated_listing_fail(self):
        cases = [HEADER + member("x", "-1"), HEADER + member("x", "12.0"),
                 HEADER + member("x").replace("Size = 12\n", ""),
                 HEADER + member("x", "1", folder=True),
                 HEADER.replace("Physical Size = 42", "Physical Size = 43") + member("x"),
                 HEADER.replace("Blocks = 1", "Blocks = 2") + member("x"),
                 HEADER.replace("Solid = +", "Solid = +\nFiles = 2") + member("x"),
                 HEADER.replace("Type = 7z", "Type = zip") + member("x"),
                 HEADER + member("x", extra="Size = 7\n"),
                 HEADER.replace("----------\n", "")]
        for text in cases:
            with self.subTest(text=text):
                with self.assertRaises(audit.AuditError):
                    self.parse(text)


class AuditTests(unittest.TestCase):
    def test_inventory_reports_fingerprint_and_unique_paths_without_payload_access(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            archive = base / "fixture.7z"
            archive.write_bytes(b"x" * 42)
            def listing_only(executable, path, listing, errors, timeout, max_listing_bytes):
                self.assertEqual(path, archive.resolve())
                listing.write_text(HEADER + member("whole.csv"), encoding="utf-8")
                errors.write_text("")
                return {"listing_bytes_received": listing.stat().st_size}
            with mock.patch.object(audit.shutil, "which", return_value="7zz"), mock.patch.object(
                audit, "_capture_listing", side_effect=listing_only,
            ):
                first = audit.audit_archive(archive, base / "reports")
                second = audit.audit_archive(archive, base / "reports")
            self.assertEqual(first["status"], "pass")
            self.assertFalse(first["extracted"])
            self.assertNotEqual(first["report_dir"], second["report_dir"])
            disk_json = Path(first["summary_path"]).read_text()
            self.assertNotIn(str(base), disk_json)
            self.assertEqual(first["archive_fingerprint"]["size_bytes"], 42)
            self.assertEqual(archive.read_bytes(), b"x" * 42)

    def test_missing_executable_persists_failed_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "fixture.7z"
            archive.write_bytes(b"x" * 42)
            with mock.patch.object(audit.shutil, "which", return_value=None):
                result = audit.audit_archive(archive, Path(folder) / "reports")
            self.assertEqual(result["status"], "fail")
            self.assertTrue(Path(result["summary_path"]).exists())

    def test_actual_subprocess_only_lists_and_redacts_archive_path(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            archive = base / "fixture.7z"
            archive.write_bytes(b"x" * 42)
            fake = base / "fake7zz"
            fake.write_text("#!/usr/bin/env python3\nimport sys\n"
                "assert sys.argv[1:5] == ['l', '-slt', '-sccUTF-8', '--']\n"
                "assert len(sys.argv) == 6\n"
                f"print({(HEADER + member('whole.csv'))!r}.replace('fixture.7z', sys.argv[5]))\n")
            fake.chmod(0o700)
            with mock.patch.object(audit.shutil, "which", return_value=str(fake)):
                result = audit.audit_archive(archive, base / "reports")
            self.assertEqual(result["status"], "pass")
            self.assertNotIn(str(archive), (Path(result["report_dir"]) / "technical_listing.txt").read_text())

    def test_stream_limit_and_timeout_kill_process_and_save_failure(self):
        for body, kwargs in [("import sys\nsys.stdout.write('x'*5000)\n", {"max_listing_bytes": 100}),
                             ("import time\ntime.sleep(5)\n", {"timeout": 0.1})]:
            with self.subTest(body=body), tempfile.TemporaryDirectory() as folder:
                base = Path(folder)
                archive = base / "fixture.7z"
                archive.write_bytes(b"x" * 42)
                fake = base / "fake7zz"
                fake.write_text("#!/usr/bin/env python3\n" + body)
                fake.chmod(0o700)
                with mock.patch.object(audit.shutil, "which", return_value=str(fake)):
                    result = audit.audit_archive(archive, base / "reports", **kwargs)
                self.assertEqual(result["status"], "fail")
                self.assertTrue(Path(result["summary_path"]).exists())


if __name__ == "__main__":
    unittest.main()
