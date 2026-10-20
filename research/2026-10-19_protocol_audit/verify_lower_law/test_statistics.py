import numpy as np
from scipy.optimize import check_grad
from scipy.stats import norm
from statistics import logprob, objective, fit

def test_extreme_intervals():
    a=np.array([-np.inf,-1.,8.,-9.,2.]);b=np.array([0.,1.,9.,-8.,np.inf])
    lp=logprob(a,b)
    assert np.isfinite(lp).all()
    assert abs(lp[2]-lp[3])<1e-12
    assert abs(lp[1]-np.log(norm.cdf(1)-norm.cdf(-1)))<1e-12

def test_gradient_and_known_parameters():
    rng=np.random.default_rng(123);n=20000
    lv=rng.uniform(-3,-1,n);lb=rng.uniform(-4,-1,n)
    X=np.c_[np.ones(n),lv,lb];true=np.array([-2.,.9,-1.2]);y=X@true+rng.normal(0,.15,n)
    lo=np.floor(y/.08)*.08;hi=lo+.08
    left=hi<=-.48; right=lo>=.96
    lo[left]=-np.inf;hi[left]=-.48;lo[right]=.96;hi[right]=np.inf
    p=np.r_[true,np.log(.15)]
    err=check_grad(lambda p:objective(p,X,lo,hi)[0],lambda p:objective(p,X,lo,hi)[1],p)
    assert err<.05,err
    rows=[{'lv':v,'lb':b,'lower':(l,h)} for v,b,l,h in zip(lv,lb,lo,hi)]
    r=fit(rows)
    assert r['success']
    assert np.max(np.abs(np.array(r['params'][:3])-true))<.03
