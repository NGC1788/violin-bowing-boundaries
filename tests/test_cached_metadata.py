import hashlib
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    import prepare_first as prepare
    import zenodo_catalog as catalog
finally:
    sys.path.pop(0)


class CachedMetadataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.enterContext(mock.patch.object(prepare, "ROOT", self.root))
        self.manifests = self.root / "data/manifests"
        self.manifests.mkdir(parents=True)
        self.record = {"id": int(prepare.RECORD), "files": [{"key": prepare.FILENAME,
            "size": prepare.EXPECTED_SIZE, "checksum": "md5:" + prepare.EXPECTED_MD5,
            "links": {"self": "https://zenodo.org/api/records/" + prepare.RECORD + "/files/" + prepare.FILENAME + "/content"}}]}
        self.envelope = {"requested_record_id": prepare.RECORD, "resolved_record_id": prepare.RECORD,
            "source_url": "https://zenodo.org/api/records/" + prepare.RECORD,
            "fetched_at_utc": "2026-09-07T14:01:27Z", "record": self.record}

    def save(self, envelope=None, suffix="old"):
        path = self.manifests / f"zenodo_{prepare.RECORD}_{suffix}.json"
        path.write_text(json.dumps(envelope or self.envelope))
        return path

    def test_valid_cache_never_requests_metadata_and_preserves_original_time(self):
        path = self.save()
        with mock.patch.object(catalog, "fetch_record", side_effect=AssertionError("API unavailable")) as fetch:
            record, provenance = prepare.resolve_metadata(catalog, 8)
        fetch.assert_not_called()
        self.assertEqual(record, self.record)
        self.assertEqual(provenance["mode"], "cached_manifest")
        self.assertEqual(provenance["original_fetched_at_utc"], "2026-09-07T14:01:27Z")
        self.assertEqual(provenance["manifest"], path.relative_to(self.root).as_posix())

    def test_absent_cache_fetches_once_and_saves_snapshot(self):
        with mock.patch.object(catalog, "fetch_record", return_value=self.record) as fetch:
            record, provenance = prepare.resolve_metadata(catalog, 2)
        fetch.assert_called_once_with(prepare.RECORD, 2, 10)
        self.assertEqual(record, self.record)
        self.assertEqual(provenance["mode"], "live_api")
        self.assertTrue((self.root / provenance["manifest"]).is_file())

    def test_mismatched_source_pin_schema_and_url_cannot_be_reused(self):
        for variant in ("source", "requested", "resolved", "record", "size", "md5", "url", "duplicate", "schema"):
            with self.subTest(variant=variant):
                envelope = json.loads(json.dumps(self.envelope))
                if variant == "source":
                    envelope["source_url"] = "https://other.example/record"
                elif variant in ("requested", "resolved"):
                    envelope[variant + "_record_id"] = "17749110"
                elif variant == "record":
                    envelope["record"]["id"] = "17749110"
                elif variant == "size":
                    envelope["record"]["files"][0]["size"] += 1
                elif variant == "md5":
                    envelope["record"]["files"][0]["checksum"] = "md5:" + "0" * 32
                elif variant == "url":
                    envelope["record"]["files"][0]["links"]["self"] = "https://other.example/file"
                elif variant == "duplicate":
                    envelope["record"]["files"] *= 2
                else:
                    envelope["record"]["files"] = [None]
                path = self.save(envelope)
                with mock.patch.object(catalog, "fetch_record", side_effect=OSError("API unavailable")) as fetch:
                    with self.assertRaises(OSError):
                        prepare.resolve_metadata(catalog, 0)
                fetch.assert_called_once()
                path.unlink()
        self.assertFalse((self.root / "data/raw").exists())

    def test_oversized_and_invalid_json_are_skipped(self):
        for raw in (b"x" * 1025, b"not JSON"):
            self.save().write_bytes(raw)
            with mock.patch.object(catalog, "MAX_METADATA_BYTES", 1024), mock.patch.object(catalog, "fetch_record", side_effect=OSError("API unavailable")):
                with self.assertRaises(OSError):
                    prepare.resolve_metadata(catalog, 0)

    def test_valid_old_cache_can_follow_an_invalid_newer_cache(self):
        self.save(suffix="20260907")
        self.save(suffix="20260914").write_text("bad JSON")
        with mock.patch.object(catalog, "fetch_record", side_effect=AssertionError("Must use valid cache")):
            _, provenance = prepare.resolve_metadata(catalog, 8)
        self.assertTrue(provenance["manifest"].endswith("20260907.json"))

    def test_live_pin_mismatch_is_rejected_before_snapshot_or_download(self):
        self.record["files"][0]["checksum"] = "md5:" + "0" * 32
        with mock.patch.object(catalog, "fetch_record", return_value=self.record), mock.patch.object(catalog, "download") as download:
            with self.assertRaisesRegex(ValueError, "pinned"):
                prepare.resolve_metadata(catalog, 0)
            download.assert_not_called()
        self.assertEqual(list(self.manifests.iterdir()), [])

    def test_cached_pipeline_rechecks_completed_file_md5_without_any_network(self):
        self.addCleanup(os.chdir, os.getcwd())
        payload = b"tiny"
        checksum = hashlib.md5(payload).hexdigest()
        self.record["files"][0].update(size=len(payload), checksum="md5:" + checksum)
        self.save()
        target = self.root / "data/raw/zenodo" / prepare.RECORD / prepare.FILENAME
        target.parent.mkdir(parents=True)
        target.write_bytes(payload)
        inventory = {"status": "pass", "summary_path": str(self.root / "inventory.json"),
            "total_uncompressed_bytes": 4, "total_files": 1, "solid": "+"}
        with mock.patch.object(prepare, "EXPECTED_SIZE", len(payload)), mock.patch.object(prepare, "EXPECTED_MD5", checksum), \
            mock.patch.object(prepare.shutil, "which", return_value="7zz"), \
            mock.patch.object(prepare.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * prepare.GIB)), \
            mock.patch.object(prepare, "audit_archive", return_value=inventory) as audit, \
            mock.patch.object(catalog, "open_url", side_effect=AssertionError("No network permitted")) as network, \
            mock.patch.object(catalog, "md5_file", wraps=catalog.md5_file) as md5, \
            redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(prepare.main([]), 0)
            md5.assert_called_once_with(target)
            network.assert_not_called()
            summary = json.loads((self.root / "reports/today/latest.json").read_text())
            self.assertEqual(summary["metadata"]["mode"], "cached_manifest")
            self.assertTrue(summary["download_md5_verified"])
            self.assertEqual(audit.call_count, 1)
            target.write_bytes(b"evil")  # Same byte count, different contents.
            self.assertEqual(prepare.main([]), 1)
            self.assertEqual(md5.call_count, 2)
            self.assertEqual(audit.call_count, 1)
            network.assert_not_called()
            self.assertEqual(target.read_bytes(), b"evil")


if __name__ == "__main__":
    unittest.main()
