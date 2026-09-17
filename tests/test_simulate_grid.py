"""simulate_grid end to end on a tiny processed diagram: windows, audit copy, regime map and comparison."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent / "scripts"


@unittest.skipIf(np is None, "numpy not installed")
class SimulateGridTests(unittest.TestCase):
    def test_batch_window_bookkeeping(self):
        sys.path.insert(0, str(SCRIPTS))
        import simulate_grid as sg
        plan = sg.batch_window([5000, 7000], [9999, 11999], [3000, 4000], 100_000)
        self.assertEqual(plan, {"every": 2, "first_sample": 500, "record_start": 5000,
                                "steps": (11999 + 1 - 500) * 2, "record_from": 4500 * 2})
        with self.assertRaises(ValueError):
            sg.batch_window([1], [2], [0], 75_000)

    def test_simulates_a_processed_diagram_and_compares(self):
        sys.path.insert(0, str(SCRIPTS))
        sys.path.insert(0, str(TESTS))
        import test_collection as tc
        import simulate_grid as sg
        import bowed_string as bs
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            existing = base / "interim"
            tc.write_diagram(existing, "r_v2", 1)
            layout = tc.collection.Layout(base / "data", base / "reports", {"old.7z": existing})
            self.assertEqual(tc.collection.run(layout, [(1, "old.7z")], 1, False, False, 0, progress=lambda *_: None,
                                               fetch=lambda *a: None), 0)
            lines = []
            result, out = sg.run("old", layout.reports, layout.cache, base / "sim", base / "simrep", bs.Backend(),
                                 {"impedance": None, "q1": 2000.0, "corner_hz": 6000.0}, bs.FrictionParams(), 50_000,
                                 8, None, None, 1, False, progress=lines.append)
            self.assertEqual(result["comparison"]["strokes"], 20)
            self.assertEqual(len(list((base / "sim" / "old").rglob("window_*.npy"))), 20)
            saved = json.loads((out / "comparison.json").read_text())
            self.assertIn("r_v2", saved["boundaries"])
            self.assertTrue(any(line.startswith("SIMULATION vs MEASUREMENT") for line in lines))
            limited, _ = sg.run("old", layout.reports, layout.cache, base / "sim", base / "simrep", bs.Backend(),
                                {"impedance": 0.9, "q1": 2000.0, "corner_hz": 6000.0}, bs.FrictionParams(), 50_000,
                                8, 4, 0.0, 1, False, progress=lambda *_: None)
            self.assertEqual(limited["comparison"]["strokes"], 4)
            self.assertEqual(limited["simulation"]["impedance"], 0.9)
            self.assertEqual(limited["simulation"]["observation_noise"], 0.0)  # an explicit value wins
            self.assertGreater(result["simulation"]["observation_noise"], 0.0)  # default: measured no-contact median


if __name__ == "__main__":
    unittest.main()
