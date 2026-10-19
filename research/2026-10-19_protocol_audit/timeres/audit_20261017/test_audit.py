"""Regression tests for audit failure modes and numerical likelihood."""
import unittest
import numpy as np
from scipy.stats import norm
import window_audit as wa
import boundary_snapshot_audit as ba


class AuditTests(unittest.TestCase):
    def test_vector_classifier_matches_scalar_edge_cases(self):
        rng=np.random.default_rng(514)
        n=1000
        data={'amplitude_std':rng.choice([np.nan,0,.2,.4,1],n),
              'periodicity':rng.choice([np.nan,.3,.5,.7,.8,.9,1],n),
              'f0_hz':rng.choice([np.nan,48,74,75,100],n),
              'flybacks_per_period':rng.choice([np.nan,0,1,2,3],n),
              'folder':np.full(n,'x')}
        for p in (.7,.8,.9):
            got=wa.vector_labels(data,.4,{'x':100},p)
            thresholds={'min_amplitude':.4,'min_periodicity':p,'subharmonic_ratio':.75,'max_aperiodic':.5}
            expected=[wa.rm.classify(data['amplitude_std'][i],data['periodicity'][i],
                        data['f0_hz'][i],data['flybacks_per_period'][i],100,thresholds)
                      for i in range(n)]
            np.testing.assert_array_equal(got,expected)

    def test_tail_interval_masses(self):
        lo=np.array([-40,-5,-1,0,1,5,39.9,-np.inf,-1,-np.inf])
        hi=np.array([-39.9,-4,0,1,2,6,40,1,np.inf,np.inf])
        got=ba.log_interval_probability(lo,hi,np.zeros(len(lo)),1)
        self.assertTrue(np.isfinite(got).all())
        self.assertAlmostEqual(got[0],got[6],places=10)
        self.assertEqual(got[-1],0)
        for i in (1,2,3,4,7,8):
            self.assertAlmostEqual(got[i],np.log(norm.cdf(hi[i])-norm.cdf(lo[i])),places=8)

    def test_censored_fit_recovers_synthetic_coefficients(self):
        rng=np.random.default_rng(994)
        n=1500
        lv=rng.choice(np.log([.05,.1,.2]),n);lb=rng.uniform(np.log(.02),np.log(.2),n)
        y=-4+1.15*lv-1.8*lb+rng.normal(0,.3,n)
        lo=np.floor(y/.15)*.15;hi=lo+.15
        hi[lo>1.4]=np.inf;lo[lo>1.4]=1.4
        lo[hi < -1.8]=-np.inf;hi[hi < -1.8]=-1.8
        data=[dict(speed=np.exp(v),beta=np.exp(b),lo=l,hi=h) for v,b,l,h in zip(lv,lb,lo,hi)]
        fit=ba.fit(data)
        self.assertTrue(fit['success']);self.assertFalse(fit['sigma_bound_hit'])
        self.assertAlmostEqual(fit['speed_exp'],1.15,delta=.07)
        self.assertAlmostEqual(fit['beta_exp'],-1.8,delta=.07)
        self.assertLess(fit['objective_spread_converged'],1e-4)

    def test_whole_column_support_does_not_depend_on_force_sign(self):
        data={'force':np.array([-.001,.2,.3,.4,.5,.6,.7,.8]),'speed':np.ones(8)*.1,
              'beta':np.ones(8)*.1,'folder_codes':np.zeros(8,int)}
        labels=np.array(['ambiguous']*2+['helmholtz']*6)
        a=ba.extract(data,np.zeros(8,int),labels,np.ones(8,bool))
        data['force'][0]=.001
        b=ba.extract(data,np.zeros(8,int),labels,np.ones(8,bool))
        self.assertEqual(len(ba.common_complete(b,a,1)),1)
        self.assertEqual(a[0]['n_strokes'],8)
        self.assertEqual(a[0]['n_forces'],7)
        self.assertEqual(a[0]['first_run'],1)
        truncated=ba.extract(data,np.zeros(8,int),labels,np.arange(8)!=7)
        self.assertEqual(ba.common_complete(truncated,a,1),[])

    def test_grid_jitter_does_not_create_columns(self):
        rng=np.random.default_rng(4)
        beta=np.repeat(np.geomspace(.02,.2,40),50)+rng.uniform(-1e-6,1e-6,2000)
        data={'first':np.arange(2000),'trial_keys':np.column_stack([np.zeros(2000),np.arange(2000)]),
              'folder_names':np.array(['a']),'beta':beta,'stroke_index':np.arange(2000)}
        levels,info=wa.recover_grid(data)
        np.testing.assert_array_equal(levels,np.repeat(np.arange(40),50))
        self.assertEqual(info['a']['recovered_columns'],40)


if __name__=='__main__': unittest.main()
