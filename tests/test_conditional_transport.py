"""Check the extra conditional-flow mechanism and source isolation."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import affine_transport as af
import conditional_transport as cf


def example():
    b, f, v = np.meshgrid(np.geomspace(.02,.2,10), np.geomspace(.03,5,12), [.05,.1,.2], indexing='ij')
    X = np.column_stack((b.ravel(),f.ravel(),v.ravel()))
    B,U,T = af.raw_coordinates(X)
    z = cf.transport(B,U,T,1.1,.2,.3,-.15)
    return X, abs(z + .8*B + .7)<1.2, np.broadcast_to(np.arange(10)[:,None,None],b.shape).ravel(), v.ravel()


@pytest.mark.parametrize('k0,k1', [(0.,0.), (1e-7,1e-8), (-.5,.4), (.38,-.7)])
def test_inverse_composition_and_known_special_case(k0,k1):
    b,u = np.linspace(-1,1,31), np.linspace(-2,2,31)
    first = cf.transport(b,u,.37,1.2,.3,k0,k1)
    np.testing.assert_allclose(cf.transport(b,first,-.11,1.2,.3,k0,k1),
                               cf.transport(b,u,.26,1.2,.3,k0,k1), atol=2e-14)
    np.testing.assert_allclose(cf.transport(b,first,-.37,1.2,.3,k0,k1), u, atol=2e-14)
    np.testing.assert_allclose(cf.transport(b,u,.37,1.2,.3,k0,0.), af.transport(b,u,.37,1.2,.3,k0), atol=2e-14)


@pytest.mark.parametrize('k0,k1', [(0.,0.), (1e-7,1e-8), (-.38,.7), (.7,-.4)])
def test_analytic_gradient(k0,k1):
    X,y,_,_ = example()
    theta = np.array([.3,-.2,.1,-.4,.15,-.8,1.2,-.3,k0,k1])
    args = (af.raw_coordinates(X),y,af.normalization(X),.01,.03)
    _, analytic = cf.objective(theta,*args)
    numeric = approx_derivative(lambda t: cf.objective(t,*args)[0],theta,method='3-point',abs_step=1e-5).ravel()
    np.testing.assert_allclose(analytic,numeric,atol=2e-8,rtol=1e-6)
    theta[9]=0.
    v,g = cf.objective(theta,*args)
    original,og = af.objective(theta[:9],*args)
    np.testing.assert_allclose(v,original,atol=2e-14)
    np.testing.assert_allclose(g[:9],og,atol=2e-14)


def test_source_fit_reload_and_connected_force_band():
    X,y,_,speed=example()
    model=cf.fit_source(X[speed!=.2],y[speed!=.2],.01,.01)
    chosen=model['starts'][model['selected_start']]
    assert chosen['objective']==min(s['objective'] for s in model['starts'] if s['success'])
    for beta in (.02,.08,.2):
        grid=np.column_stack((np.repeat(beta,1000),np.exp(np.linspace(-8,8,1000)),np.repeat(.2,1000)))
        score=cf.design(np.array(model['theta']),af.raw_coordinates(grid),model['normalization'])
        assert np.max(np.diff(score,n=2))<0
        assert np.count_nonzero(np.diff(np.r_[False,score>=0,False].astype(int))==1)<=1
        np.testing.assert_array_equal(cf.predict_frozen(model,grid),cf.predict_frozen(json.loads(json.dumps(model)),grid))


def test_target_poisoning_and_reference_nonidentifiability(tmp_path):
    X,y,levels,speeds=example()
    source=speeds!=.2
    ids=np.array([f'row{i}' for i in range(len(y))])
    def select(labels,path):
        return cf.select_source(X[source],labels[source],labels[source,None],levels[source],speeds[source],
                                ids[source],path,lambdas=(.01,),warp_lambdas=(.01,))
    first=select(y,tmp_path/'first.jsonl')
    poisoned=y.copy()
    poisoned[~source]=~poisoned[~source]
    second=select(poisoned,tmp_path/'second.jsonl')
    assert first==second
    assert (tmp_path/'first.jsonl').read_bytes()==(tmp_path/'second.jsonl').read_bytes()
    X[:,2]=.1
    _,jac=cf.design(np.asarray(first['model']['theta']),af.raw_coordinates(X),af.normalization(X),derivatives=True)
    np.testing.assert_array_equal(jac[:,6:],0.)
