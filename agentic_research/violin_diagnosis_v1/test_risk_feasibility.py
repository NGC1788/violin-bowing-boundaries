import numpy as np
from agentic_research.violin_diagnosis_v1.risk_feasibility import outcome_scores,logit_correct,independent_person_hoeffding_width
from agentic_research.violin_raw_v1.models import patterns


def test_span_cap_and_proper_score_identity_all_ten_outcomes():
    b=np.arange(1,11,dtype=float);b/=b.sum()
    residual=np.linspace(-.05,.05,10);q=logit_correct(b,residual)
    difference=outcome_scores(q)-outcome_scores(b)
    assert np.abs(difference).max() <= np.ptp(residual)
    p0=b@patterns();p1=q@patterns()
    kl=np.mean(p0*np.log(p0/p1)+(1-p0)*np.log((1-p0)/(1-p1)))
    np.testing.assert_allclose(b@difference,kl,atol=1e-12)
    assert b@difference>0


def test_probability_mixture_cap_is_for_actual_marginal_score():
    b=np.full(10,.1);r=np.arange(1,11,dtype=float);r/=r.sum()
    alpha=.1;q=(1-alpha)*b+alpha*r
    assert np.max(outcome_scores(q)-outcome_scores(b)) <= -np.log1p(-alpha)
    np.testing.assert_allclose(outcome_scores(b)-outcome_scores(b),0)
    # Actual evaluation clips even for degenerate predictions.
    assert np.isfinite(outcome_scores(np.eye(10)[-1])).all()


def test_independent_cluster_count_controls_width():
    a=independent_person_hoeffding_width(.1,8)
    b=independent_person_hoeffding_width(.1,32)
    np.testing.assert_allclose(a/2,b)
