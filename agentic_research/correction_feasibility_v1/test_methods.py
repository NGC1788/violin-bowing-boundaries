import numpy as np
import pytest
from agentic_research.correction_feasibility_v1.methods import fit_alpha,calibrate,row_loss,person_loss,cap_predictions,math_controls,betting_evalue,hb_pvalue,online_control
from agentic_research.violin_raw_v1.models import patterns


def test_convex_root_and_endpoints():
    y=np.array([[1.]*9,[0.]*9]);b=np.full((2,9),.2);p=np.full((3,2,9),.8)
    fit=fit_alpha(y,b,p,[1,1])
    assert abs(fit['alpha']-.5)<1e-10
    assert fit['derivative_left']<=0<=fit['derivative_right']
    assert fit['bracket_width']<1e-11
    assert fit_alpha(np.ones_like(y),b,p,[1,1])['alpha']==.95
    assert fit_alpha(np.zeros_like(y),b,p,[1,1])['alpha']==0


def test_person_macro_not_record_macro():
    y=np.array([[1.]*9]*3+[[0.]*9]);b=np.full((4,9),.2);p=np.full((3,4,9),.8)
    ids=np.array(['a','a','a','b']);st=np.zeros(4,int)
    _,fits=calibrate(y,b,p,ids,st,np.ones(4,bool))
    assert abs(fits[0]['alpha']-.5)<1e-10
    scores=person_loss(y,p,ids)
    assert scores.shape==(2,)


def test_held_labels_never_enter_calibrator_and_empty_state_falls_back():
    rng=np.random.default_rng(1);y=rng.integers(0,2,(6,9)).astype(float)
    b=np.full((6,9),.3);p=np.full((3,6,9),.6);ids=np.array(['a']*3+['b']*3)
    states=np.array([0,1,2,3,3,3]);mask=ids=='a'
    for statewise in [False,True]:
        q,f=calibrate(y,b,p,ids,states,mask,statewise)
        changed=y.copy();changed[~mask]=np.nan
        q2,f2=calibrate(changed,b,p,ids,states,mask,statewise)
        np.testing.assert_array_equal(q,q2);assert f==f2
        if statewise:np.testing.assert_array_equal(q[:,~mask],np.broadcast_to(b[~mask],(3,3,9)))


def test_no_false_convex_certificate_when_clipping_is_active():
    b=np.full((1,9),2e-6);p=np.zeros((3,1,9));y=np.ones((1,9))
    with pytest.raises(ValueError,match='clipping'):fit_alpha(y,b,p,[1])


def test_logit_span_bounds_all_outcome_scores():
    rng=np.random.default_rng(7);prior=rng.dirichlet(np.ones(10)*2,4)
    residual=rng.normal(size=(3,4,10))*10
    for cap in [.025,.1]:
        q=cap_predictions(prior,residual,cap);p=q@patterns();b=prior@patterns()
        ratio=q/prior[None]
        assert np.max(np.abs(np.log(ratio)))<=cap+1e-12
        for y0 in patterns():
            y=np.tile(y0,(4,1));d=row_loss(y,p)-row_loss(y,np.broadcast_to(b,(3,4,9)))
            assert np.abs(d).max()<=cap+1e-12


def test_betting_and_hb_small_n_are_not_automatically_vacuous():
    r=math_controls()
    assert abs(r['independent_symmetric_null_mean_mixture_evalue']-1)<1e-12
    assert r['iid_best_case_all_gains_one_evalue']>=20
    assert r['iid_best_case_all_gains_one_hb_pvalue']<.05
    assert r['all_equal_symmetric_dependent_null_mean_evalue']>1
    assert r['all_equal_dependent_null_rejection_probability_at_20']==.5
    assert betting_evalue(np.zeros(8))==pytest.approx(1)
    assert hb_pvalue(np.ones(8),1)==1


def test_known_online_bound_is_distinct_from_a_risk_certificate():
    r=online_control()
    assert r['expconcavity_min_gap']>=-1e-12
    assert r['max_cumulative_regret_vs_two_experts']<=np.log(2)+1e-12
    assert not r['novelty']
