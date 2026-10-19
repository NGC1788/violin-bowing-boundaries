import unittest
import numpy as np
from history_prediction_audit import unique_map,paired_cluster,metrics,CONFIGS

class HistoryTests(unittest.TestCase):
    def test_duplicate_join_rejected(self):
        with self.assertRaises(ValueError):unique_map([('v1',1),('v1',1)])
        self.assertEqual(unique_map([('v1',1),('v2',1)]),{('v1',1):0,('v2',1):1})

    def test_cluster_sensitivity_preserves_effect_and_zero(self):
        y=np.tile([False,True],40);level=np.repeat(np.arange(40),2)
        p0=np.ones(80)*.5;p1=np.tile([.1,.9],40)
        r=paired_cluster(y,p0,p1,level)
        self.assertLess(r['position_cluster_percentile95'][1],0)
        self.assertAlmostEqual(r['history_minus_clock_log_loss'],np.log(.5/.9))
        z=paired_cluster(y,p1,p1,level)
        self.assertEqual(z['position_cluster_percentile95'],[0.,0.])

    def test_history_is_nested_and_ablations_remove_exact_variable(self):
        self.assertTrue(set(CONFIGS['clock'])<set(CONFIGS['history']))
        for name,number in [('no_onset',4),('no_accel',5),('no_travel',6)]:
            self.assertEqual(set(CONFIGS['history'])-set(CONFIGS[name]),{number})

    def test_perfect_labels_metric_sanity(self):
        r=metrics(np.array([0,1,0,1]),np.array([.1,.9,.2,.8]))
        self.assertEqual(r['IoU'],1.);self.assertEqual(r['balanced_accuracy'],1.)

if __name__=='__main__':unittest.main()
