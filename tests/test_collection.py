"""Collection pipeline: listing, folder-by-folder extraction, window cache, resume and deletion rules."""

import csv
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
collection = regime = audit = None
if np is not None:
    sys.path.insert(0, str(SCRIPTS))
    import trial_audit as audit  # noqa: E402
    import regime_map as regime  # noqa: E402
    SPEC = importlib.util.spec_from_file_location("collection_under_test", SCRIPTS / "collection.py")
    collection = importlib.util.module_from_spec(SPEC)
    sys.modules[SPEC.name] = collection
    SPEC.loader.exec_module(collection)
RATE = 50_000

SLT = """
7-Zip (z) 26.03 (x64)
Listing archive: fake.7z

--
Path = fake.7z
Type = 7z
Physical Size = 1234

----------
Path = ARCH/rep_1/v1
Folder = +
Size = 0
Attributes = D

Path = ARCH/rep_1/v1/whole_1.csv
Folder = -
Size = 100
Attributes = A

Path = ARCH/rep_1/v1/beta_1.csv
Folder = -
Size = 10

Path = ARCH/rep_1/v2/whole_1.csv
Size = 120

Path = ARCH/rep_2/v1/timestamp_1.csv
Size = 11

Path = ARCH/cold/v2/whole_7.csv
Size = 5

Path = ARCH/readme.txt
Size = 3
"""


def sawtooth(n, amp=0.8, seed=0):
    t = np.arange(n) / RATE
    x = amp * (np.mod(t * 98.0, 1.0) - 0.5)
    x = np.convolve(x, np.ones(5) / 5, mode="same")
    return x + np.random.default_rng(seed).normal(0, 0.003, n)


def write_diagram(root: Path, leaf: str, seed: int, corrupt: int | None = None) -> None:
    """5 beta levels x 4 forces; force 1.0 Helmholtz, 2.0 aperiodic, <= 0 no contact."""
    folder = root / leaf
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    number = 0
    for beta in (0.2, 0.1, 0.05, 0.035, 0.02):
        for force in (2.0, 1.0, -0.01, -0.02):
            number += 1
            rows = 7000
            c3 = sawtooth(rows, seed=seed + number) if force == 1.0 else rng.normal(0, 0.3 if force > 0 else 0.05, rows)
            i = np.arange(rows)
            signal = np.column_stack([np.full(rows, force), np.where(i < 800, 0.1 * i / 800, 0.1), c3, np.zeros(rows)])
            text = "\r\n".join(",".join(repr(float(v)) for v in row) for row in signal) + "\r\n"
            if number == corrupt:
                text = "not,a,number\r\n"
            (folder / f"whole_{number}.csv").write_text(text, encoding="ascii")
            (folder / f"beta_{number}.csv").write_text(repr(beta + (number % 3 - 1) * 1e-5) + "\r\n", encoding="ascii")
            (folder / f"timestamp_{number}.csv").write_text("1000,6999\r\n", encoding="ascii")


def members_of(root: Path):
    return [{"path": p.relative_to(root).as_posix(), "size": p.stat().st_size, "is_dir": False}
            for p in sorted(root.rglob("*.csv"))]


def expected_labels(force):
    return "helmholtz" if force == 1.0 else ("aperiodic" if force > 0 else "no_oscillation")


@unittest.skipIf(np is None, "numpy not installed")
class ListingTests(unittest.TestCase):
    def test_parse_slt_and_group_into_diagrams(self):
        members = collection.parse_slt(SLT)
        self.assertEqual(len(members), 7)
        self.assertTrue(members[0]["is_dir"])
        diagrams = collection.group_leaves(members)
        self.assertEqual(sorted(diagrams), ["ARCH/cold", "ARCH/rep_1", "ARCH/rep_2"])
        self.assertEqual(sorted(diagrams["ARCH/rep_1"]), ["ARCH/rep_1/v1", "ARCH/rep_1/v2"])
        self.assertEqual(diagrams["ARCH/rep_1"]["ARCH/rep_1/v1"], {"files": {"whole_1.csv": 100, "beta_1.csv": 10},
                                                                   "bytes": 110, "trials": 1})
        self.assertEqual(collection.diagram_id("ARCH/rep_1"), "ARCH__rep_1")
        with self.assertRaises(collection.CollectionError):
            collection.parse_slt("no separator here")


@unittest.skipIf(np is None, "numpy not installed")
class ExtractionTests(unittest.TestCase):
    def fake_7zip(self, source: Path, drop: str | None = None):
        def run(command, **kwargs):
            output = Path(next(a for a in command if a.startswith("-o"))[2:])
            pattern = command[-1]
            leaf = pattern.rsplit("/", 1)[0]
            for path in (source / leaf).iterdir():
                if Path(pattern).match(path.relative_to(source).as_posix()) or pattern.endswith("/*"):
                    if path.name != drop:
                        target = output / leaf / path.name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(path, target)
            return subprocess.CompletedProcess(command, 0, "", "")
        return run

    def test_extract_leaf_verifies_names_and_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, staging = Path(tmp) / "src", Path(tmp) / "staging"
            write_diagram(source, "ARCH/rep_1/v1", 1)
            spec = collection.group_leaves(members_of(source))["ARCH/rep_1"]["ARCH/rep_1/v1"]
            with mock.patch.object(collection, "seven_zip", return_value="7zz"), \
                    mock.patch.object(collection.subprocess, "run", side_effect=self.fake_7zip(source)):
                target = collection.extract_leaf(Path("fake.7z"), "ARCH/rep_1/v1", spec, staging)
            self.assertEqual(len(list(target.iterdir())), 60)
            with mock.patch.object(collection, "seven_zip", return_value="7zz"), \
                    mock.patch.object(collection.subprocess, "run", side_effect=self.fake_7zip(source, drop="whole_3.csv")):
                with self.assertRaises(collection.ExtractionError):
                    collection.extract_leaf(Path("fake.7z"), "ARCH/rep_1/v1", spec, staging)


@unittest.skipIf(np is None, "numpy not installed")
class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.source = base / "src"
        self.layout = collection.Layout(base / "data", base / "reports", {})
        self.fetches = []

    def tearDown(self):
        self.tmp.cleanup()

    def fake_fetch(self, layout, part, name, retries, progress):
        self.fetches.append(name)
        path = layout.raw / "999" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"7z")
        return path

    def fake_extract(self, archive, leaf, spec, staging):
        target = staging / leaf
        shutil.copytree(self.source / leaf, target)
        return target

    def run_pipeline(self, name, **kw):
        lines = []
        with mock.patch.object(collection, "list_members", side_effect=lambda archive: members_of(self.source)), \
                mock.patch.object(collection, "extract_leaf", side_effect=self.fake_extract):
            code = collection.run(self.layout, [(6, name)], 1, kw.get("delete", True), False, 0,
                                  progress=lines.append, fetch=self.fake_fetch)
        return code, lines

    def labels(self, regime_run):
        with (Path(regime_run) / "regimes.csv").open(newline="", encoding="utf-8") as handle:
            return {(r["condition"], int(r["trial"])): r["label"] for r in csv.DictReader(handle)}

    def test_two_repeats_processed_cached_resumable_and_archive_deleted(self):
        write_diagram(self.source, "ARCH/rep_1/v1", 1)
        write_diagram(self.source, "ARCH/rep_2/v1", 2)
        code, lines = self.run_pipeline("fake.7z")
        self.assertEqual(code, 0, "\n".join(lines[-15:]))
        state = collection.load_state(self.layout, "fake.7z")
        self.assertEqual(state["status"], "complete")
        self.assertEqual(sorted(state["diagrams"]), ["ARCH/rep_1", "ARCH/rep_2"])
        self.assertFalse((self.layout.raw / "999" / "fake.7z").exists())  # deleted only after completion
        self.assertIn("archive_deleted_utc", state)
        self.assertEqual(list(self.layout.staging.rglob("*.csv")), [])  # extracted CSVs removed
        cache = self.layout.cache / "ARCH__rep_1" / "v1"
        self.assertEqual(len(list(cache.glob("window_*.npy"))), 20)
        self.assertEqual(len(list(cache.glob("profile_*.npy"))), 20)
        window = np.load(cache / "window_2.npy")
        self.assertEqual((window.shape, window.dtype), ((6000, 4), np.float32))
        self.assertEqual(np.load(cache / "profile_2.npy").shape, (7000 // audit.PROFILE_DECIMATION, 2))
        forces = [2.0, 1.0, -0.01, -0.02] * 5
        got = self.labels(state["diagrams"]["ARCH/rep_1"]["regime_run"])
        self.assertEqual(got, {("v1", n): expected_labels(forces[n - 1]) for n in range(1, 21)})
        summary = json.loads((Path(state["diagrams"]["ARCH/rep_1"]["regime_run"]) / "summary.json").read_text())
        self.assertEqual(summary["window_source"], "cache")
        self.assertEqual(summary["noise_floor"]["controls"], 10)
        self.assertTrue(any("floor" in line for line in collection.status_lines(self.layout, [(6, "fake.7z")])))

        # Cache mode gives the same labels as reading the CSVs directly.
        audit_dir = Path(self.tmp.name) / "direct_audit"
        code, _ = audit.run_audit(self.source / "ARCH/rep_1/v1", audit_dir, None, 1, 0.01, progress=lambda *_: None)
        self.assertEqual(code, 0)
        code, direct = regime.run(audit_dir, Path(self.tmp.name) / "direct_regime", 1, dict(regime.DEFAULT_THRESHOLDS),
                                  None, False, progress=lambda *_: None)
        self.assertEqual(self.labels(direct), got)

        # Resume: nothing is fetched or reprocessed.
        before = len(self.fetches)
        code, _ = self.run_pipeline("fake.7z")
        self.assertEqual((code, len(self.fetches)), (0, before))

    def test_failed_diagram_is_isolated_and_archive_kept(self):
        write_diagram(self.source, "ARCH/rep_1/v1", 1, corrupt=3)
        write_diagram(self.source, "ARCH/rep_2/v1", 2)
        code, lines = self.run_pipeline("fake.7z")
        self.assertEqual(code, 1)
        state = collection.load_state(self.layout, "fake.7z")
        self.assertEqual(state["diagrams"]["ARCH/rep_1"]["status"], "failed")
        self.assertEqual(state["diagrams"]["ARCH/rep_2"]["status"], "complete")
        self.assertEqual(state["status"], "failed")
        self.assertTrue((self.layout.raw / "999" / "fake.7z").exists())
        self.assertEqual(list(self.layout.staging.rglob("*.csv")), [])

    def test_extraction_error_stops_the_run(self):
        write_diagram(self.source, "ARCH/rep_1/v1", 1)
        def broken(*args, **kwargs):
            raise collection.ExtractionError("wildcard matched nothing")
        with mock.patch.object(collection, "list_members", side_effect=lambda archive: members_of(self.source)), \
                mock.patch.object(collection, "extract_leaf", side_effect=broken):
            with self.assertRaises(collection.ExtractionError):
                collection.run(self.layout, [(6, "a.7z"), (6, "b.7z")], 1, True, False, 0,
                               progress=lambda *_: None, fetch=self.fake_fetch)
        self.assertEqual(self.fetches, ["a.7z"])

    def test_existing_extraction_is_read_in_place_and_never_deleted(self):
        existing = Path(self.tmp.name) / "interim"
        write_diagram(existing, "2024-03-27_r_v1", 1)  # speed folders directly under the extraction root
        write_diagram(existing, "2024-03-27_r_v2", 2)
        before = sorted(p.relative_to(existing).as_posix() for p in existing.rglob("*"))
        self.layout.existing = {"old.7z": existing}
        code, lines = collection.run(self.layout, [(1, "old.7z")], 1, True, False, 0, progress=lambda *_: None,
                                     fetch=self.fake_fetch), None
        self.assertEqual(code, 0)
        self.assertEqual(self.fetches, [])
        self.assertEqual(sorted(p.relative_to(existing).as_posix() for p in existing.rglob("*")), before)
        state = collection.load_state(self.layout, "old.7z")
        self.assertEqual(list(state["diagrams"]), ["old"])  # both speed folders form one diagram
        summary = json.loads((Path(state["diagrams"]["old"]["regime_run"]) / "summary.json").read_text())
        self.assertEqual(sorted(summary["conditions"]), ["2024-03-27_r_v1", "2024-03-27_r_v2"])


if __name__ == "__main__":
    unittest.main()
