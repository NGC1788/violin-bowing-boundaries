"""Parameter search: cell sampling, scoring, a resumable run on a tiny planted diagram."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent / "scripts"


@unittest.skipIf(np is None, "numpy not installed")
class CalibrateTests(unittest.TestCase):
    def setUp(self):
        sys.path[:0] = [str(SCRIPTS), str(TESTS)]
        import calibrate
        self.calibrate = calibrate

    def test_sample_cells_takes_whole_force_columns(self):
        rows = [{"trial": level * 100 + rank, "beta_level": level, "force_rank": rank}
                for level in range(10) for rank in range(1, 21)]
        picked = self.calibrate.sample_cells(rows, levels=4, ranks=5)
        levels = sorted({r["beta_level"] for r in picked})
        self.assertEqual(len(levels), 4)
        self.assertEqual(len(picked), 20)
        for level in levels:  # a column, not scattered cells
            self.assertEqual(len([r for r in picked if r["beta_level"] == level]), 5)

    def test_score_counts_iou_and_agreement(self):
        cells = {"v1": [{"trial": 1, "label": "helmholtz"}, {"trial": 2, "label": "helmholtz"},
                        {"trial": 3, "label": "multiple_slip"}, {"trial": 4, "label": "aperiodic"}]}
        simulated = {("v1", 1): "helmholtz", ("v1", 2): "multiple_slip", ("v1", 3): "helmholtz", ("v1", 4): "aperiodic"}
        got = self.calibrate.score(cells, simulated)
        self.assertEqual(got["cells"], 4)
        self.assertAlmostEqual(got["helmholtz_iou"], 1 / 3, places=3)   # one shared, three involved (rounded)
        self.assertAlmostEqual(got["agreement"], 0.5)

    def test_draw_keeps_the_friction_curve_falling(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            params = self.calibrate.draw(rng)
            self.assertLess(params["mu_d"], params["mu_s"])
            self.assertTrue(0.01 <= params["v0"] <= 0.4)

    def test_search_writes_evaluations_and_resumes(self):
        import test_collection as tc
        import bowed_string as bs
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            existing = base / "interim"
            tc.write_diagram(existing, "r_v2", 1)
            layout = tc.collection.Layout(base / "data", base / "reports", {"old.7z": existing})
            self.assertEqual(tc.collection.run(layout, [(1, "old.7z")], 1, False, False, 0,
                                               progress=lambda *_: None, fetch=lambda *a: None), 0)
            out = base / "calib.jsonl"
            lines = []
            best = self.calibrate.run("old", layout.reports, layout.cache, out, bs.Backend(), 50_000, 5, 4, 2, 0,
                                      None, False, progress=lines.append)
            saved = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(len(saved), 2)
            self.assertEqual(saved[0]["kind"], "literature")
            self.assertEqual(saved[1]["kind"], "random")
            self.assertEqual(saved[0]["cells"], 20)
            self.assertIn("helmholtz_iou", best)
            self.assertTrue(any("CALIBRATE" in line for line in lines))
            # resume: only the third evaluation is added
            self.calibrate.run("old", layout.reports, layout.cache, out, bs.Backend(), 50_000, 5, 4, 3, 0,
                               None, False, progress=lambda *_: None)
            self.assertEqual(len(out.read_text().splitlines()), 3)
            # the same seed reproduces the same parameter sequence
            second = [json.loads(line)["params"] for line in out.read_text().splitlines()]
            self.assertEqual(second[1], saved[1]["params"])

            # batching parameter sets changes only the wall clock, not the evaluations
            batched = base / "calib_batched.jsonl"
            self.calibrate.run("old", layout.reports, layout.cache, batched, bs.Backend(), 50_000, 5, 4, 3, 0,
                               None, False, progress=lambda *_: None, batch_params=3)
            one_by_one = [json.loads(line) for line in out.read_text().splitlines()]
            together = [json.loads(line) for line in batched.read_text().splitlines()]
            self.assertEqual([r["params"] for r in together], [r["params"] for r in one_by_one])
            self.assertEqual([r["helmholtz_iou"] for r in together], [r["helmholtz_iou"] for r in one_by_one])


if __name__ == "__main__":
    unittest.main()
