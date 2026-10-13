"""Input separation, strict joins, training-only nuisance calibration and frozen data."""

import copy
import csv
import gzip
import math
from pathlib import Path
import sys
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_data as bd


def fixture():
    rows = []
    for string, speed in (("A", 0.1), ("B", 0.2)):
        for i in range(1, 5):
            row = dict(id=f"{string}/{i}", string=string, nominal_speed=speed,
                       condition="v2" if string == "A" else "v3", trial=i, beta=0.1,
                       beta_level=0, force_rank=i, c1_window_mean=0.5 + i * 0.1,
                       c2_window_mean=speed, amplitude_std=0.5, periodicity=0.99,
                       f0_hz=98.0, flybacks_per_period=1.0, second_flyback_ratio=0.0,
                       label="helmholtz", window_inside_plateau=True)
            rows.append(row)
    return rows, {r["id"]: copy.deepcopy(r) for r in rows}


class BenchmarkDataTests(unittest.TestCase):
    def test_output_changes_do_not_enter_X_or_train_nuisance(self):
        rows, sims = fixture()
        a = bd.assemble_dataset(rows, sims)
        for row in rows:
            if row["string"] == "B":
                row.update(amplitude_std=999.0, periodicity=0.1, f0_hz=10000.0,
                           flybacks_per_period=15.0, second_flyback_ratio=0.9, label="aperiodic")
        b = bd.assemble_dataset(rows, sims)
        np.testing.assert_array_equal(a["X"], b["X"])
        self.assertEqual(a["manifest"]["classifier_reference"], b["manifest"]["classifier_reference"])
        self.assertFalse(np.array_equal(a["y"], b["y"]))

    def test_trial_id_join_rejects_missing_or_extra_simulation(self):
        rows, sims = fixture()
        sims.pop(rows[-1]["id"])
        with self.assertRaisesRegex(ValueError, "match exactly"):
            bd.assemble_dataset(rows, sims)

    def test_duplicate_measured_ID_rejected(self):
        rows, sims = fixture()
        with self.assertRaisesRegex(ValueError, "duplicate global"):
            bd.assemble_dataset(rows + [rows[0]], sims)

    def test_exclusions_and_ambiguity_are_distinct(self):
        rows, sims = fixture()
        rows[1]["window_inside_plateau"] = False
        rows[2]["c1_window_mean"] = -0.01
        rows[3]["label"] = "ambiguous"
        a = bd.assemble_dataset(rows, sims)
        b = bd.assemble_dataset(rows, sims, include_ambiguous=True)
        self.assertEqual(len(a["y"]), 5)
        self.assertEqual(len(b["y"]), 6)
        self.assertEqual(np.count_nonzero(b["y"] == -1), 1)
        self.assertEqual(a["manifest"]["counts"]["off_plateau"], 1)
        self.assertEqual(a["manifest"]["counts"]["nonpositive_force"], 1)
        self.assertEqual(a["manifest"]["counts"]["ambiguous_eligible"], 1)

    def test_shuffle_preserves_arrays_and_hashes(self):
        rows, sims = fixture()
        a = bd.assemble_dataset(rows, sims)
        b = bd.assemble_dataset(list(reversed(rows)), dict(reversed(list(sims.items()))))
        np.testing.assert_array_equal(a["X"], b["X"])
        np.testing.assert_array_equal(a["ids"], b["ids"])
        self.assertEqual(a["manifest"], b["manifest"])

    def test_one_flyback_ratio_is_zero_and_undefined_ratio_stays_nan(self):
        rows, sims = fixture()
        rows[0]["second_flyback_ratio"] = math.nan
        got = bd.assemble_dataset(rows, sims)
        self.assertTrue(math.isnan(got["output_features"]["second_flyback_ratio"][0]))
        self.assertEqual(got["output_features"]["second_flyback_ratio"][1], 0.0)
        self.assertEqual(got["X"].shape[1], 3)
        self.assertEqual(got["feature_names"], ("beta", "force_N", "velocity_m_s"))

    def test_read_csv_bool_and_duplicate_validation(self):
        rows, _ = fixture()
        row = rows[0]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "regimes.csv"
            def save(items):
                with path.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerows(items)
            save([dict(row, window_inside_plateau=False)])
            self.assertFalse(bd.read_regimes(path)[0]["window_inside_plateau"])
            save([row, row])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                bd.read_regimes(path)
            save([dict(row, window_inside_plateau="no")])
            with self.assertRaisesRegex(ValueError, "boolean"):
                bd.read_regimes(path)
            save([dict(row, beta=math.nan)])
            with self.assertRaisesRegex(ValueError, "non-finite"):
                bd.read_regimes(path)

    @unittest.skipUnless((bd.ROOT / "research/inputs/A_measured_regimes.csv.gz").exists(), "release copies unavailable")
    def test_fresh_clone_release_fallback_and_decompressed_checksum(self):
        reference = bd.load_dataset(include_ambiguous=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            release = root / "research/inputs"
            release.mkdir(parents=True)
            for source in (bd.ROOT / "research/inputs").glob("*.csv.gz"):
                shutil.copyfile(source, release / source.name)
            got = bd.load_dataset(root, include_ambiguous=True)
            np.testing.assert_array_equal(got["X"], reference["X"])
            np.testing.assert_array_equal(got["ids"], reference["ids"])
            self.assertEqual(got["manifest"], reference["manifest"])
            victim = release / "A_measured_regimes.csv.gz"
            victim.write_bytes(gzip.compress(b"changed data", mtime=0))
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                bd.load_dataset(root)

    @unittest.skipUnless((bd.ROOT / "reports/collection/2024-03-25_TypeA_sample1").exists(), "backed-up private artifacts unavailable")
    def test_actual_frozen_sources_counts_and_checksums(self):
        data = bd.load_dataset(include_ambiguous=True)
        self.assertEqual(data["manifest"]["counts"]["raw"], 24000)
        self.assertEqual(data["X"].shape, (23301, 3))
        self.assertEqual(len(set(data["ids"])), 23301)
        self.assertEqual(np.count_nonzero(data["y"] == -1), 211)
        self.assertEqual(data["train_mask"].sum(), 1750)
        self.assertEqual(data["manifest"]["n_binary_sim_changes_with_fixed_reference"], 0)
        with mock.patch.dict(bd.EXPECTED_SHA256, {("A", "measured"): "0" * 64}):
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                bd.load_dataset()


if __name__ == "__main__":
    unittest.main()
