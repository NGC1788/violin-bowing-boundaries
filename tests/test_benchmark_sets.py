"""Blocked evaluation, coherent global rules, and target-blind model selection."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_sets as bs


class BenchmarkSetsTests(unittest.TestCase):
    def test_beta_column_blocks_never_split_or_overlap(self):
        levels = np.repeat(np.arange(40), 3)
        visits = np.zeros(len(levels), int)
        folds = bs.blocked_folds(levels)
        for train, valid in folds:
            self.assertFalse(set(levels[train]) & set(levels[valid]))
            held = np.unique(levels[valid])
            np.testing.assert_array_equal(held, np.arange(held.min(), held.max()+1))
            visits[valid] += 1
        np.testing.assert_array_equal(visits, np.ones(len(levels)))

    def test_global_coherent_rule_is_not_panelwise_adversary(self):
        family = np.array([[True, False], [False, True]])
        got = bs.evaluate_panels(np.ones(2), np.ones(2, bool), family,
                                 np.array(["B/.1", "C/.1"]), np.ones(2, bool))
        self.assertEqual(got["macro_panels"]["family_min_iou"], 0.)
        self.assertEqual(got["global_same_rule"]["family_min_iou"], .5)
        self.assertEqual(got["global_same_rule"]["box_lower_iou"], 0.)

    def test_invalid_probabilities_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid prediction"):
            bs.metrics(np.array([np.nan]), np.array([True]), np.array([[True]]))

    def test_cv_pools_each_source_cell_once_and_preserves_nominal_winner(self):
        levels = np.repeat(np.arange(10), 2)
        X = np.column_stack((np.linspace(.02, .2, 20), np.ones(20), np.full(20,.1)))
        nominal = np.arange(20)%2 == 0
        family = np.column_stack((nominal, np.ones(20, bool)))
        class FixedModel:
            def __init__(self, positive): self.positive = positive
            def predict_proba(self, X):
                p = np.full(len(X), self.positive)
                return np.column_stack((1-p,p))
        def fit(spec, params, X, nominal, family):
            return FixedModel(params["p"]), {"fit_rows": len(X)}
        spec={"name":"fixture","grid":[{"p":.1},{"p":.9}]}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(bs,"fit_model",side_effect=fit):
            path=Path(tmp)/"fits.jsonl"
            cv, primary, alternative = bs.cross_validate(spec,X,nominal,family,levels,np.arange(20),path)
            self.assertEqual(primary,1)
            self.assertEqual(alternative,1)
            self.assertEqual(cv[1]["metrics"]["family_min_iou"],.5)
            records=[json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(records),10)
            self.assertTrue(all(r["fit_rows"]==16 for r in records))

    def test_target_perturbation_does_not_change_selection_inputs(self):
        n=40
        X=np.column_stack((np.tile(np.linspace(.02,.2,20),2),np.ones(n),np.full(n,.1)))
        labels=np.tile(np.arange(20)%2==0,2)
        base={"X":X,"y_fixed":labels.astype(int),"string":np.array(["A"]*20+["B"]*20),
              "speed":np.full(n,.1),"ids":np.array([f"row{i}" for i in range(n)]),
              "group":np.array(["A/0.1"]*20+["B/0.1"]*20),"beta_level":np.tile(np.repeat(np.arange(10),2),2),
              "reference_f0_hz":98.,"manifest":{},"legacy_sim_prediction":labels.astype(int),
              "reclassified_sim_prediction":labels.astype(int),
              "output_features":{"periodicity":np.full(n,.99),"f0_hz":np.full(n,98.),
                                 "flybacks_per_period":np.where(labels,1,2),
                                 "second_flyback_ratio":np.where(labels,0.,.9)}}
        seen=[]
        spec={"name":"test_logistic","kind":"logistic","covariant":True,"grid":[{"C":1.}]}
        def cv(spec,X,nominal,family,levels,ids,path):
            seen.append((X.copy(),nominal.copy(),family.copy(),ids.copy()))
            return [{"params":{"C":1.},"metrics":{}}],0,0
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(bs,"specifications",return_value=[spec]), mock.patch.object(bs,"cross_validate",side_effect=cv):
            with mock.patch.object(bs.bd,"load_dataset",return_value=base):
                bs.run(Path(tmp)/"one")
            changed={**base,"X":base["X"].copy(),"y_fixed":base["y_fixed"].copy(),
                     "output_features":{k:v.copy() for k,v in base["output_features"].items()}}
            changed["X"][20:,1]*=100
            changed["y_fixed"][20:]=1-changed["y_fixed"][20:]
            changed["output_features"]["flybacks_per_period"][20:]=np.where(changed["y_fixed"][20:]==1,1,2)
            changed["output_features"]["second_flyback_ratio"][20:]=np.where(changed["y_fixed"][20:]==1,0.,.9)
            with mock.patch.object(bs.bd,"load_dataset",return_value=changed):
                bs.run(Path(tmp)/"two")
            for first,second in zip(seen[0],seen[1]):
                np.testing.assert_array_equal(first,second)


if __name__ == "__main__":
    unittest.main()
