"""Extraction preservation, bounded scouting, and optional real 7-Zip roundtrip."""

import importlib.util
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
from contextlib import redirect_stderr, redirect_stdout


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_first.py"
SPEC = importlib.util.spec_from_file_location("prepare_first_under_test", SCRIPT)
prepare = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SCRIPT.parent))
try:
    SPEC.loader.exec_module(prepare)
finally:
    sys.path.pop(0)


def member(path, data=b"1,2\n", directory=False):
    return {"path": path, "directory": directory, "size_bytes": 0 if directory else len(data)}


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(prepare, "ROOT", self.root).start()
        mock.patch.object(prepare.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * prepare.GIB)).start()
        mock.patch.object(prepare.shutil, "which", return_value="7zz").start()
        self.process = mock.patch.object(prepare.subprocess, "Popen").start()
        self.archive = self.root / "fixture.7z"
        self.archive.write_bytes(b"placeholder archive")
        self.report_dir = self.root / "reports"
        self.report_dir.mkdir()
        self.destination = self.root / "data" / "complete"
        self.receipt = self.destination.with_suffix(".receipt.json")
        self.staging = self.destination.with_name("complete.partial")
        self.members = [member("run/whole_1.csv")]
        (self.report_dir / "members.jsonl").write_text("\n".join(json.dumps(row) for row in self.members) + "\n")
        info = self.archive.stat()
        self.audit = {"status": "pass", "inventory_guards_passed": True,
            "report_dir": str(self.report_dir), "artifacts": {"manifest": "members.jsonl"},
            "total_uncompressed_bytes": 4, "total_files": 1,
            "archive_fingerprint": {"size_bytes": info.st_size, "mtime_ns": info.st_mtime_ns}}

    def extract(self, **overrides):
        options = dict(archive=self.archive, audit=self.audit, destination=self.destination,
            receipt=self.receipt, max_unpacked_gib=1, report_dir=self.report_dir)
        options.update(overrides)
        return prepare.extract_checked(**options)

    def write_output(self):
        path = self.destination / "run/whole_1.csv"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"1,2\n")
        return path

    def write_receipt(self, **overrides):
        payload = {"status": "complete", "archive_md5": prepare.EXPECTED_MD5}
        payload.update(overrides)
        self.receipt.write_text(json.dumps(payload))

    def test_rejected_inventory_or_excess_budget_never_starts_extraction(self):
        for changed in [{"status": "fail"}, {"inventory_guards_passed": False}]:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.extract(audit={**self.audit, **changed})
        for budget in [0, -1, float("nan"), float("inf"), 1 / prepare.GIB]:
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                self.extract(max_unpacked_gib=budget)
        self.process.assert_not_called()
        self.assertFalse(self.staging.exists())

    def test_insufficient_space_preserves_existing_files(self):
        sentinel = self.root / "unrelated.txt"
        sentinel.write_text("keep")
        with mock.patch.object(prepare.shutil, "disk_usage", return_value=SimpleNamespace(free=20 * prepare.GIB)):
            with self.assertRaisesRegex(ValueError, "reserve"):
                self.extract()
        self.process.assert_not_called()
        self.assertEqual(sentinel.read_text(), "keep")
        self.assertFalse(self.staging.exists())

    def test_missing_binary_and_changed_archive_leave_no_staging(self):
        with mock.patch.object(prepare.shutil, "which", return_value=None):
            with self.assertRaisesRegex(ValueError, "7zz/7z"):
                self.extract()
        self.assertFalse(self.staging.exists())
        self.archive.write_bytes(b"changed archive")
        with self.assertRaisesRegex(ValueError, "changed after inventory"):
            self.extract()
        self.assertFalse(self.staging.exists())
        self.process.assert_not_called()

    def test_existing_destination_without_valid_receipt_is_preserved(self):
        output = self.write_output()
        with self.assertRaisesRegex(ValueError, "without a completion receipt"):
            self.extract()
        for change in [{"archive_md5": "wrong"}, {"status": "running"}]:
            self.write_receipt(**change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "does not match"):
                self.extract()
        self.process.assert_not_called()
        self.assertEqual(output.read_bytes(), b"1,2\n")

    def test_orphan_receipt_is_preserved_before_extraction_starts(self):
        self.receipt.parent.mkdir(parents=True)
        self.receipt.write_text("existing receipt must survive")
        with self.assertRaises((ValueError, FileExistsError)):
            self.extract()
        self.process.assert_not_called()
        self.assertFalse(self.staging.exists())
        self.assertEqual(self.receipt.read_text(), "existing receipt must survive")

    def test_completed_extraction_reused_only_when_names_and_sizes_match(self):
        output = self.write_output()
        self.write_receipt()
        self.assertEqual(self.extract()["status"], "reused")
        output.write_bytes(b"truncated")
        with self.assertRaisesRegex(ValueError, "names/sizes"):
            self.extract()
        self.process.assert_not_called()
        self.assertEqual(output.read_bytes(), b"truncated")

    def test_existing_partial_is_preserved(self):
        self.staging.mkdir(parents=True)
        unfinished = self.staging / "unfinished.csv"
        unfinished.write_text("partial data")
        with self.assertRaisesRegex(ValueError, "unfinished extraction"):
            self.extract()
        self.process.assert_not_called()
        self.assertEqual(unfinished.read_text(), "partial data")
        self.assertFalse(self.destination.exists())
        self.assertFalse(self.receipt.exists())

    def test_completed_receipt_recovers_verified_partial_without_reextracting(self):
        output = self.staging / "run/whole_1.csv"
        output.parent.mkdir(parents=True)
        output.write_bytes(b"1,2\n")
        self.write_receipt()
        self.assertEqual(self.extract()["status"], "reused")
        self.assertEqual((self.destination / "run/whole_1.csv").read_bytes(), b"1,2\n")
        self.assertFalse(self.staging.exists())
        self.process.assert_not_called()

    def test_completed_receipt_cannot_publish_damaged_partial(self):
        output = self.staging / "run/whole_1.csv"
        output.parent.mkdir(parents=True)
        output.write_bytes(b"1,2")
        self.write_receipt()
        with self.assertRaisesRegex(ValueError, "names/sizes"):
            self.extract()
        self.assertEqual(output.read_bytes(), b"1,2")
        self.assertFalse(self.destination.exists())
        self.process.assert_not_called()

    def test_symlink_destination_cannot_be_reused(self):
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        self.destination.parent.mkdir(parents=True)
        self.destination.symlink_to(elsewhere, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.extract()
        self.process.assert_not_called()
        self.assertTrue(self.destination.is_symlink())

    def test_failed_extractor_preserves_partial_and_never_issues_receipt(self):
        def fail_extract(command, **kwargs):
            (self.staging / "part.csv").write_text("recoverable prefix")
            return SimpleNamespace(poll=lambda: 2, returncode=2)
        self.process.side_effect = fail_extract
        with self.assertRaisesRegex(ValueError, "exit 2"):
            self.extract()
        self.assertEqual((self.staging / "part.csv").read_text(), "recoverable prefix")
        self.assertFalse(self.destination.exists())
        self.assertFalse(self.receipt.exists())


class TreeAndScoutTests(unittest.TestCase):
    def test_tree_rejects_missing_extra_wrong_size_and_symlink_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            rows = [member("whole_1.csv")]
            target = base / "whole_1.csv"
            with self.assertRaises(ValueError):
                prepare.verify_tree(base, rows)
            target.write_bytes(b"1,2\n")
            prepare.verify_tree(base, rows)
            extra = base / "unexpected.csv"
            extra.write_bytes(b"1,2\n")
            with self.assertRaises(ValueError):
                prepare.verify_tree(base, rows)
            extra.unlink()
            target.write_bytes(b"1,2")
            with self.assertRaises(ValueError):
                prepare.verify_tree(base, rows)
            target.unlink()
            target.symlink_to("missing")
            with self.assertRaisesRegex(ValueError, "link"):
                prepare.verify_tree(base, rows)

    def test_numbered_scout_selects_numeric_first_middle_last_and_companions(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            rows = []
            for group in ["root/r_v1", "root/r_v2"]:
                for number in [1, 2, 3, 10, 11]:
                    for stem, payload in [("whole", b"0.0,1.0\n0.1,2.0\n"), ("beta", b"0.1\n"), ("timestamp", b"0.0\n")]:
                        name = f"{group}/{stem}_{number}.csv"
                        path = base / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(payload)
                        rows.append(member(name, payload))
            result = prepare.scout_csv(base, rows)
            self.assertEqual(result["status"], "scouted")
            self.assertEqual(result["group_counts"], {"root/r_v1": 5, "root/r_v2": 5})
            names = {sample["member"] for sample in result["samples"]}
            self.assertEqual(names, {f"root/r_v{group}/{stem}_{number}.csv"
                for group in [1, 2] for stem in ["whole", "beta", "timestamp"] for number in [1, 3, 11]})
            self.assertTrue(all(sample["numeric_finite_rows"] > 0 for sample in result["samples"]))
            self.assertIn("not interpreted as measured speeds", result["warning"])

    def test_unmatched_and_invalid_text_do_not_invent_measurements(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            self.assertEqual(prepare.scout_csv(base, [member("notes.txt")])["status"], "no_matching_waveform_names")
            path = base / "whole_1.csv"
            path.write_bytes(b"\xff\xfe\x00")
            result = prepare.scout_csv(base, [member("whole_1.csv", path.read_bytes())])
            self.assertEqual(result["status"], "incomplete")
            observation = result["samples"][0]
            self.assertEqual(observation["error_type"], "UnicodeDecodeError")
            self.assertNotIn("first_rows", observation)
            path.write_bytes(b"")
            self.assertEqual(prepare.scout_csv(base, [member("whole_1.csv", b"")])["status"], "incomplete")

    def test_prefix_is_bounded_and_reports_nonfinite_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            path = base / "whole.csv"
            path.write_text("1,2\nNaN,2\nInfinity,2\n\n3,4\n")
            observation = prepare.inspect_prefix(path)
            self.assertEqual(observation["numeric_finite_rows"], 2)
            self.assertIn("prefix only", observation["scope"])
            path.write_bytes(b"1,2\n" * 9000 + b"\xff")
            observation = prepare.inspect_prefix(path)
            self.assertEqual(observation["prefix_rows_inspected"], 64)
            self.assertEqual(observation["numeric_finite_rows"], 64)

    def test_json_publication_is_exclusive_and_failed_serialization_leaves_no_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            path = base / "receipt.json"
            prepare.write_json(path, {"status": "complete"})
            with self.assertRaises(FileExistsError):
                prepare.write_json(path, {"status": "replaced"})
            self.assertEqual(json.loads(path.read_text()), {"status": "complete"})
            invalid = base / "invalid.json"
            with self.assertRaises(ValueError):
                prepare.write_json(invalid, {"invalid": float("nan")})
            self.assertFalse(invalid.exists())
            self.assertEqual(list(base.iterdir()), [path])


class MainOrchestrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.addCleanup(os.chdir, os.getcwd())
        self.root = Path(temporary.name)
        self.enterContext(mock.patch.object(prepare, "ROOT", self.root))
        self.enterContext(mock.patch.object(prepare, "EXPECTED_SIZE", 4))
        self.enterContext(mock.patch.object(prepare.shutil, "which", return_value="7zz"))
        self.enterContext(mock.patch.object(prepare.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * prepare.GIB)))
        self.extractor = self.enterContext(mock.patch.object(prepare, "extract_checked"))
        self.inventory = self.enterContext(mock.patch.object(prepare, "audit_archive"))
        self.catalog = SimpleNamespace(fetch_record=mock.Mock(return_value={"id": prepare.RECORD, "files": []}),
            file_spec=mock.Mock(return_value=(4, prepare.EXPECTED_MD5, "unused")),
            download=mock.Mock(), validate_retry_policy=mock.Mock(), MAX_METADATA_BYTES=10*1024**2,
            save_manifest=mock.Mock(return_value=self.root / "data/manifests/fresh.json"))
        self.download = self.catalog.download
        self.enterContext(mock.patch.dict(sys.modules, {"zenodo_catalog": self.catalog}))
        archive = self.root / "data/raw/zenodo" / prepare.RECORD / prepare.FILENAME
        archive.parent.mkdir(parents=True)
        archive.write_bytes(b"tiny")
        self.latest = self.root / "reports/today/latest.json"

    def run_main(self, arguments):
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            code = prepare.main(arguments)
        return code, output.getvalue(), errors.getvalue()

    def test_exhausted_metadata_read_persists_failure_and_stops_before_inventory(self):
        self.catalog.fetch_record.side_effect = http.client.IncompleteRead(b"partial", 100)
        code, output, errors = self.run_main([])
        self.assertEqual(code, 1)
        self.catalog.fetch_record.assert_called_once_with(prepare.RECORD, 8, 10)
        self.download.assert_not_called()
        self.inventory.assert_not_called()
        self.extractor.assert_not_called()
        summary = json.loads(self.latest.read_text())
        self.assertEqual(summary["status"], "fail")
        self.assertFalse(summary["research_training_performed"])
        self.assertNotIn("download_md5_verified", summary)
        self.assertIn("IncompleteRead", summary["error"])
        self.assertIn("TODAY DATA PREPARATION: FAIL", output)
        self.assertIn("PREPARATION FAILED", errors)

    def test_show_is_read_only_for_existing_report_and_missing_report_returns_one(self):
        code, _, _ = self.run_main(["--show"])
        self.assertEqual(code, 1)
        self.assertFalse(self.latest.parent.exists())
        self.latest.parent.mkdir(parents=True)
        self.latest.write_text(json.dumps({"status": "pass", "csv_scout": "reports/today/scout.json"}))
        (self.latest.parent / "scout.json").write_text(json.dumps({"status": "scouted", "samples": [
            {"member": "run/whole_1.csv", "first_rows": [["1"], ["2"], ["3"]]},
            {"member": "run/whole_2.csv", "first_rows": [["4"]]},
            {"member": "run/beta_1.csv", "first_rows": [["0.1"]]}]}))
        before = {path.relative_to(self.root).as_posix(): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        code, output, errors = self.run_main(["--show"])
        self.assertEqual(code, 0)
        self.assertEqual(errors, "")
        displayed = json.loads(output)
        self.assertEqual(displayed["preparation"]["status"], "pass")
        examples = displayed["csv_scout"]["representative_prefixes"]
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0]["first_rows"], [["1"], ["2"]])
        after = {path.relative_to(self.root).as_posix(): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        self.download.assert_not_called()
        self.catalog.fetch_record.assert_not_called()
        self.inventory.assert_not_called()
        self.extractor.assert_not_called()

    def test_incomplete_scout_preserves_report_and_fails_main_after_extraction(self):
        self.inventory.return_value = {"status": "pass", "summary_path": str(self.root / "inventory.json"),
            "total_uncompressed_bytes": 4, "total_files": 1, "solid": "+"}
        self.extractor.return_value = {"status": "extracted", "directory": "data/interim/typeA_sample1"}
        scout = {"status": "incomplete", "read_errors": 1, "samples": [{"member": "whole_1.csv", "error_type": "UnicodeDecodeError"}]}
        with mock.patch.object(prepare, "load_members", return_value=[]), mock.patch.object(prepare, "scout_csv", return_value=scout):
            code, _, _ = self.run_main(["--extract"])
        self.assertEqual(code, 1)
        self.extractor.assert_called_once()
        summary = json.loads(self.latest.read_text())
        self.assertEqual(summary["status"], "fail")
        self.assertEqual(summary["extraction"]["status"], "extracted")
        self.assertEqual(summary["scout_status"], "incomplete")
        self.assertFalse(summary["research_training_performed"])
        self.assertEqual(json.loads((self.root / summary["csv_scout"]).read_text()), scout)

    def test_successful_download_does_not_fetch_metadata_a_second_time(self):
        self.inventory.return_value = {"status": "pass", "summary_path": str(self.root / "inventory.json"),
            "total_uncompressed_bytes": 4, "total_files": 1, "solid": "+"}
        code, _, _ = self.run_main([])
        self.assertEqual(code, 0)
        self.catalog.fetch_record.assert_called_once_with(prepare.RECORD, 8, 10)
        self.download.assert_called_once()
        self.assertEqual(json.loads(self.latest.read_text())["metadata"]["mode"], "live_api")


@unittest.skipUnless(shutil.which("7zz") or shutil.which("7z"), "real 7-Zip executable not available on PATH")
class RealSevenZipTests(unittest.TestCase):
    def test_real_archive_inventory_extract_scout_and_reuse(self):
        executable = shutil.which("7zz") or shutil.which("7z")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source" / "run_v1"
            source.mkdir(parents=True)
            payloads = {"whole_1.csv": b"0.0,1.0\n0.1,2.0\n", "beta_1.csv": b"0.125\n", "timestamp_1.csv": b"12.0\n"}
            for name, payload in payloads.items():
                (source / name).write_bytes(payload)
            archive = root / "tiny.7z"
            subprocess.run([executable, "a", "-t7z", "-ms=on", str(archive), "run_v1"], cwd=source.parent,
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            report_dir = root / "reports"
            report_dir.mkdir()
            inventory = prepare.audit_archive(archive, report_dir / "inventory")
            self.assertEqual(inventory["status"], "pass")
            self.assertEqual(inventory["total_files"], 3)
            destination = root / "data" / "complete"
            receipt = root / "data" / "complete.receipt.json"
            with mock.patch.object(prepare, "ROOT", root), mock.patch.object(prepare, "EXPECTED_MD5", hashlib.md5(archive.read_bytes()).hexdigest()), mock.patch.object(prepare.shutil, "disk_usage",
                return_value=SimpleNamespace(free=100 * prepare.GIB)):
                result = prepare.extract_checked(archive, inventory, destination, receipt, 1, report_dir)
                self.assertEqual(result["status"], "extracted")
                for name, payload in payloads.items():
                    self.assertEqual((destination / "run_v1" / name).read_bytes(), payload)
                self.assertEqual(json.loads(receipt.read_text())["status"], "complete")
                scout = prepare.scout_csv(destination, prepare.load_members(inventory))
                self.assertEqual(len(scout["samples"]), 3)
                self.assertTrue(all("error_type" not in sample for sample in scout["samples"]))
                self.assertEqual(prepare.extract_checked(archive, inventory, destination, receipt, 1, report_dir)["status"], "reused")
                self.assertFalse(destination.with_name("complete.partial").exists())


if __name__ == "__main__":
    unittest.main()
