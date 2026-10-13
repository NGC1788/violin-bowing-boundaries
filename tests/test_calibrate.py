"""Parameter search: cell sampling, scoring, a resumable run on a tiny planted diagram."""

import importlib.util
import copy
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


@unittest.skipIf(np is None, "numpy not installed")
class SearchBoxTests(unittest.TestCase):
    def setUp(self):
        sys.path[:0] = [str(SCRIPTS), str(TESTS)]
        import calibrate
        self.calibrate = calibrate

    def test_space_override_is_used(self):
        space = self.calibrate.parse_space(["corner_hz=300:900", "mu_s=0.8:0.9"])
        rng = np.random.default_rng(1)
        for _ in range(50):
            params = self.calibrate.draw(rng, space)
            self.assertTrue(300 <= params["corner_hz"] <= 900)
            self.assertTrue(0.8 <= params["mu_s"] <= 0.9)
            self.assertTrue(0.01 <= params["v0"] <= 0.4)  # untouched entries keep their bounds
        for bad in ("nope=1:2", "mu_s=2", "mu_s=3:1", "mu_s=0:2"):
            with self.assertRaises(ValueError):
                self.calibrate.parse_space([bad])

    def test_refinement_stays_near_the_centre(self):
        centre = {"mu_s": 0.9, "mu_d": 0.26, "v0": 0.06, "q1": 3700.0, "corner_hz": 1900.0}
        rng = np.random.default_rng(2)
        for _ in range(100):
            params = self.calibrate.draw(rng, self.calibrate.SPACE, centre, 1.5)
            for name, value in centre.items():
                low, high, _ = self.calibrate.SPACE[name]
                self.assertGreaterEqual(params[name], max(low, value / 1.5) - 1e-9)
                self.assertLessEqual(params[name], min(high, value * 1.5) + 1e-9)

    def test_best_of_picks_the_highest_iou(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.jsonl"
            path.write_text("\n".join(json.dumps({"helmholtz_iou": iou, "params": {"mu_s": iou},
                "model_version": "passive-point-bow-waveguide-v2"})
                                      for iou in (0.2, 0.7, 0.5)) + "\n")
            self.assertEqual(self.calibrate.best_of([path])["params"], {"mu_s": 0.7})
            self.assertIsNone(self.calibrate.best_of([Path(tmp) / "missing.jsonl"]))

    def test_legacy_scores_cannot_seed_corrected_simulator(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.jsonl"
            path.write_text(json.dumps({"helmholtz_iou": .99, "params": {}}) + "\n")
            with self.assertRaisesRegex(ValueError, "legacy or unversioned"):
                self.calibrate.best_of([path])

    def test_nonpassive_candidate_recorded_without_simulation(self):
        import bowed_string as bs
        invalid = {"mu_s": .91246, "mu_d": .26251, "v0": .06066,
                   "q1": 3726.62, "corner_hz": 1896.41}
        valid = {**invalid, "q1": 200., "corner_hz": 6000.}
        settings = {"reference_f0": {"v2": 98.08}}
        cells = {"v2": [{"trial": 1, "label": "helmholtz"}]}
        with mock.patch.object(self.calibrate, "simulate_labels", return_value=[{("v2",1): "helmholtz"}]) as simulator:
            rows = self.calibrate.evaluate(bs.Backend(), [invalid, valid], cells, {}, settings, 50000, 0)
        self.assertEqual(rows[0]["status"], "rejected_nonpassive")
        self.assertEqual(rows[0]["cells"], 0)
        self.assertEqual(rows[1]["helmholtz_iou"], 1.)
        self.assertEqual(simulator.call_args.args[1], [valid])


@unittest.skipIf(np is None, "numpy not installed")
class BoundCalibrationTests(unittest.TestCase):
    def setUp(self):
        sys.path[:0] = [str(SCRIPTS), str(TESTS)]
        import calibrate
        self.c = calibrate
        self.cells = {"v1": [{"trial": "1", "label": "helmholtz"}],
                      "v2": [{"trial": "1", "label": "multiple_slip"}]}
        self.profiles = {("v1", 1): np.ones((2,2)), ("v2", 1): np.ones((2,2))}
        self.settings = {"reference_f0": {"v1": 98., "v2": 98.}, "impedance": .946,
                         "noise": 0., "floor": .02}
        self.backend = type("Backend", (), {"name": "fixture", "dtype": "float64"})()

    def evaluations(self, backend, params_list, cells, profiles, settings, rate, seed):
        return [{"params": p, "status": "evaluated", "cells": 1, "helmholtz_iou": .4,
                 "agreement": .5, "seconds": 0., "simulated_classes": {}} for p in params_list]

    def run_search(self, tmp, **overrides):
        kwargs = dict(diagram="diagram", reports=Path(tmp)/"reports", cache_root=Path(tmp)/"cache",
                      out_path=Path(tmp)/"calibration.jsonl", backend=self.backend, rate=50000,
                      levels=1, ranks=1, iterations=1, seed=0, conditions=["v1"], hold_out=True,
                      progress=lambda line: None)
        kwargs.update(overrides)
        with mock.patch.object(self.c,"load_cells",return_value=(self.cells,self.profiles,self.settings)), \
             mock.patch.object(self.c,"evaluate",side_effect=self.evaluations):
            return self.c.run(**kwargs)

    def test_resume_rejects_changed_configuration_and_selected_inputs(self):
        for change in ("seed", "rate", "conditions", "labels", "profiles", "source_bytes"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                original_cells, original_profiles = copy.deepcopy(self.cells), copy.deepcopy(self.profiles)
                source = Path(tmp)/"reports/diagram/regime_map/retained/summary.json"
                source.parent.mkdir(parents=True);source.write_text('{"version":1}')
                self.run_search(tmp)
                args={}
                if change == "seed": args["seed"]=99
                if change == "rate": args["rate"]=100000
                if change == "conditions": args["conditions"]=["v2"]
                if change == "labels": self.cells["v1"][0]["label"]="aperiodic"
                if change == "profiles": self.profiles[("v1",1)][0,0]+=1e-9
                if change == "source_bytes": source.write_text('{"version":2}')
                with self.assertRaisesRegex(ValueError,"fingerprint mismatch"):
                    self.run_search(tmp, **args)
                self.cells,self.profiles=original_cells,original_profiles

    def test_post_search_holdout_is_saved_and_printed_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            messages=[]
            best=self.run_search(tmp,iterations=3,progress=messages.append)
            summary=json.loads((Path(tmp)/"calibration.jsonl.summary.json").read_text())
            records=[json.loads(line) for line in (Path(tmp)/"calibration.jsonl").read_text().splitlines()]
            self.assertEqual(summary["best"]["hold_out"],best["hold_out"])
            self.assertEqual(best["hold_out"]["status"],"evaluated")
            self.assertEqual(sum("POST-SEARCH HOLD-OUT" in line for line in messages),1)
            self.assertTrue(all("hold_out" not in row for row in records))
            self.assertIn("target-adapted",summary["diagnostic_scope"])
            self.assertEqual(len({row["run_fingerprint"] for row in records}),1)

    def test_same_search_can_extend_budget_and_change_batching(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.run_search(tmp)
            self.run_search(tmp,iterations=3,batch_params=1)
            rows=[json.loads(line) for line in (Path(tmp)/"calibration.jsonl").read_text().splitlines()]
            self.assertEqual([r["index"] for r in rows],[0,1,2])
            self.assertEqual(len({r["run_fingerprint"] for r in rows}),1)

    def test_unversioned_and_unbound_corrected_logs_are_preserved_and_rejected(self):
        for version in (None,"passive-point-bow-waveguide-v2"):
            with self.subTest(version=version),tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/"calibration.jsonl"
                text=json.dumps({"model_version":version,"index":0,"helmholtz_iou":.9})+"\n"
                path.write_text(text)
                with self.assertRaisesRegex(ValueError,"legacy|unversioned|unbound"):
                    self.run_search(tmp)
                self.assertEqual(path.read_text(),text)

    def test_unused_target_pitch_does_not_reject_source_candidate(self):
        import bowed_string as bs
        p={"mu_s":.6725,"mu_d":.346,"v0":.06,"q1":2000.,"corner_hz":6000.}
        settings={"reference_f0":{"v1":98.,"unused_target":1000.}}
        with mock.patch.object(self.c,"simulate_labels",return_value=[{("v1",1):"helmholtz"}]) as simulator:
            result=self.c.evaluate(bs.Backend(),[p],{"v1":self.cells["v1"]},{},settings,50000,0)
        self.assertEqual(result[0]["status"],"evaluated")
        self.assertEqual(simulator.call_count,1)

    def test_refinement_center_is_frozen_on_resume_and_first_kind_is_random(self):
        import bowed_string as bs
        center={"mu_s":bs.FrictionParams.mu_s,"mu_d":bs.FrictionParams.mu_d,"v0":bs.FrictionParams.v0,
                "q1":bs.StringParams.q1,"corner_hz":bs.StringParams.corner_hz}
        with tempfile.TemporaryDirectory() as tmp:
            seed=Path(tmp)/"seed.jsonl"
            seed.write_text(json.dumps({"params":center,"helmholtz_iou":.5,"model_version":self.c.MODEL_VERSION})+"\n")
            self.run_search(tmp,around=1.6,seed_from=[seed])
            self.run_search(tmp,around=1.6,seed_from=[seed],iterations=2)
            rows=[json.loads(line) for line in (Path(tmp)/"calibration.jsonl").read_text().splitlines()]
            rng=np.random.default_rng(0)
            expected=[self.c.draw(rng,self.c.SPACE,center,1.6) for _ in range(2)]
            self.assertEqual([r["params"] for r in rows],expected)
            self.assertEqual(rows[0]["kind"],"random")
