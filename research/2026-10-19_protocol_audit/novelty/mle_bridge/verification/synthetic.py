"""Synthetic-only independent likelihood checks; no empirical data reads."""
from pathlib import Path
import argparse, json, time
import numpy as np
from scipy.optimize import minimize
import oracle as O


def run():
    start = time.monotonic()
    cases = [(-np.inf, np.inf), (-np.inf, -40), (-np.inf, 0), (-np.inf, 40),
             (-40, np.inf), (0, np.inf), (40, np.inf), (-40, -39.99),
             (39.99, 40), (-10, -9.999999), (9.999999, 10), (-1e-12, 1e-12),
             (-.01, .001), (-100, 100), (30, 30.1), (-30.1, -30),
             (-2, 2), (-.1, 0), (0, .1), (-100, -99.5), (99.5, 100)]
    rng = np.random.default_rng(20261021)
    for _ in range(180):
        mid = rng.uniform(-40, 40); width = 10 ** rng.uniform(-6, 1)
        cases.append((mid - width / 2, mid + width / 2))
    expected = np.array([O.high_precision_log_mass(a,b) for a,b in cases])
    got = O.log_mass(*np.array(cases).T)
    err = abs(got - expected)
    assert max(err) < 5e-7, (cases[np.argmax(err)], max(err))
    # Symmetry is an exact distributional identity, independent of regressors.
    reflected = O.log_mass(-np.array(cases)[:,1], -np.array(cases)[:,0])
    assert np.allclose(got,reflected,atol=5e-9,rtol=0)
    gradients = []
    for _ in range(30):
        X = np.c_[np.ones(60), rng.normal(size=(60,2))]
        coef = rng.normal(size=3); sig = rng.uniform(.15,1.2)
        yy = X @ coef + rng.normal(0, sig, len(X)); half = rng.uniform(.01,.25,len(X))
        lo,hi = yy-half, yy+half; lo[:7] = -np.inf; hi[7:14] = np.inf
        p = np.r_[coef + rng.normal(0,.1,3),np.log(sig)]
        an,num = O.gradient(p,X,lo,hi),O.numerical_gradient(p,X,lo,hi)
        error = float(max(abs(an-num))); gradients.append(error)
        assert error < 1e-5, (an,num,error)
    X = np.c_[np.ones(6), np.arange(6), np.arange(6)**2]
    valid = O.validate_design(X,np.arange(6)-1,np.arange(6)+1)
    assert valid['valid']
    edge = []
    fixtures = [
        ('zero_width_interval',X,np.arange(6),np.arange(6),None),
        ('reversed_interval',X,np.arange(6)+1,np.arange(6),None),
        ('rank_deficient',np.ones((6,3)),np.arange(6),np.arange(6)+1,None),
        ('all_informative_intervals_left_censored',X,np.full(6,-np.inf),np.arange(6),None),
        ('all_informative_intervals_right_censored',X,np.arange(6),np.full(6,np.inf),None),
        ('no_interval_information',X,np.full(6,-np.inf),np.full(6,np.inf),None),
        ('log_scale_at_or_outside_bound',X,np.arange(6),np.arange(6)+1,-9),
        ('log_scale_at_or_outside_bound',X,np.arange(6),np.arange(6)+1,3.5),
        ('log_scale_at_or_outside_bound',X,np.arange(6),np.arange(6)+1,-10),
    ]
    for code,xx,lo,hi,scale in fixtures:
        found=O.validate_design(xx,lo,hi,scale)
        assert code in found['issues'];edge.append(found)
    rejected=0
    for lo,hi in [(0,0),(1,0),(np.nan,1),(0,np.nan),(np.inf,np.inf),(-np.inf,-np.inf)]:
        try: O.log_mass(lo,hi)
        except ValueError: rejected+=1
        else: raise AssertionError((lo,hi))
    # Dense grid verification of an intercept/scale fit. Synthetic counts only.
    xx=np.ones((12,1)); yy=np.array([-1.1,-.8,-.5,-.2,-.1,0,.2,.4,.6,.8,1.,1.1])
    lo,hi=yy-.09,yy+.09;lo[:2]=-np.inf;hi[-2:]=np.inf
    bounds=[(-2,2),(-3,2)]
    opt=minimize(O.negative_log_likelihood,[0.,0.],args=(xx,lo,hi),method='Nelder-Mead',
                 bounds=bounds,options={'xatol':1e-11,'fatol':1e-11,'maxiter':2000})
    dense=[]
    for mu in np.linspace(-.5,.5,81):
        for ls in np.linspace(-1.5,.5,101):
            dense.append(O.negative_log_likelihood([mu,ls],xx,lo,hi))
    assert opt.success and opt.fun <= min(dense)+1e-9
    assert max(abs(O.numerical_gradient(opt.x,xx,lo,hi))) < 1e-5
    # Equal point residuals have sigma-collapse, despite a successful optimizer.
    flat_lo=np.full(12,-.05);flat_hi=np.full(12,.05)
    assert O.negative_log_likelihood([0,-9],xx,flat_lo,flat_hi) <= O.negative_log_likelihood([0,-2],xx,flat_lo,flat_hi)
    # Any fixed epsilon geometric gap is not equivalent to Gaussian impossible
    # mass: Gaussian sigma gives all positive-width intervals positive mass.
    conflict_lo=np.array([-2.,2.]);conflict_hi=np.array([-1.9,2.1])
    conflict=O.negative_log_likelihood([0.,0.],np.ones((2,1)),conflict_lo,conflict_hi)
    assert np.isfinite(conflict)
    # For a specified scale, a conflict pair does impose a finite likelihood
    # burden. Verify the midpoint sharpness for opposing half-lines and the
    # conservative bound for narrower intervals, over unrelated random pairs.
    pair_checks=0;pair_max_error=0.
    for _ in range(40):
        gap=10**rng.uniform(-2,1);sig=10**rng.uniform(-1,1)
        bound=float(O.separated_pair_nll_bound(gap,sig))
        midpoint_loss=O.negative_log_likelihood([gap/2,np.log(sig)],np.ones((2,1)),
                                              [-np.inf,gap],[0,np.inf])
        pair_max_error=max(pair_max_error,abs(bound-midpoint_loss))
        for mu in gap/2+np.linspace(-3,3,81)*sig:
            half_loss=O.negative_log_likelihood([mu,np.log(sig)],np.ones((2,1)),
                                               [-np.inf,gap],[0,np.inf])
            finite_loss=O.negative_log_likelihood([mu,np.log(sig)],np.ones((2,1)),
                                                 [-2*sig,gap],[0,gap+sig])
            assert half_loss >= bound-1e-9
            assert finite_loss >= half_loss-1e-9
            pair_checks+=2
    assert pair_max_error < 1e-9
    return {'passed':True,'scope':'synthetic; no empirical scores',
            'high_precision_interval_cases':len(cases),'max_log_mass_absolute_error':float(max(err)),
            'gradient_fixtures':len(gradients),'gradient_max_absolute_error':max(gradients),
            'diagnostic_edge_fixtures':len(edge),'invalid_interval_rejections':rejected,
            'dense_grid_objective_evaluations':len(dense),'dense_grid_fit':{'params':opt.x.tolist(),
            'nll':float(opt.fun),'dense_min':float(min(dense))},
            'geometric_contradiction_has_finite_gaussian_nll':float(conflict),
            'fixed_scale_pair_bound_checks':pair_checks,
            'fixed_scale_pair_bound_midpoint_max_error':pair_max_error,
            'seconds':time.monotonic()-start}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path(__file__).with_suffix('.json'));a=p.parse_args()
    out=run();a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
