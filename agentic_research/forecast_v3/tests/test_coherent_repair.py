"""Synthetic-only validation of known coherentization and finite-target fallback."""
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from forecast_v3 import coherent_repair as C, models as M, experiment as E


def test_all_pattern_score_uses_exact_evaluator_operations_and_clipping():
    rng=np.random.default_rng(9);q=rng.random((20,9));q[0]=0;q[1]=1
    scores=C.all_pattern_losses(q)
    for j,y in enumerate(M.PATTERNS):
        np.testing.assert_array_equal(scores[:,j],E.loss_rows(np.tile(y,(len(q),1)),q)[0])


def test_repair_does_not_worsen_any_of_the_ten_targets():
    rng=np.random.default_rng(20261030)
    q=np.vstack([rng.uniform(.01,.99,(60,9)),rng.choice([0.,1.,1e-15,1-1e-15],(20,9))])
    repaired,meta=C.repair(q,max_iterations=64,chunk_size=31)
    assert np.all(C.all_pattern_losses(repaired)<=C.all_pattern_losses(q))
    assert meta['max_certified_delta']<=0
    assert meta['acceptance_score_tolerance']==0
    assert meta['rows']==len(q)
    assert meta['accepted_rows']+meta['fallback_original_rows']==len(q)
    assert 'not rigorous interval' in meta['numerical_scope']
    assert meta['changed_rows']>0


def test_bad_or_nonfinite_candidate_falls_back_to_original():
    q=np.tile(M.PATTERNS[4]*.8+.1,(3,1))
    candidate=np.array([np.zeros(9),np.full(9,np.nan),np.full(9,2.)])
    result,accepted,worst=C.certify_candidate(q,candidate)
    assert not accepted.any()
    np.testing.assert_array_equal(result,q)


def test_positive_roundoff_is_not_accepted_as_nonworsening():
    q=np.full((1,9),.5)
    candidate=np.full((1,9),.5+1e-12)
    result,accepted,worst=C.certify_candidate(q,candidate)
    assert worst[0]>0 and not accepted[0]
    np.testing.assert_array_equal(result,q)
    result,accepted,_=C.certify_candidate(q,q.copy())
    assert accepted[0];np.testing.assert_array_equal(result,q)


def test_coherent_forecasts_are_not_changed_if_candidate_cannot_dominate():
    rng=np.random.default_rng(42);weights=rng.dirichlet(np.ones(10),30);q=weights@M.PATTERNS
    repaired,meta=C.repair(q,max_iterations=16)
    # Exact coherent forecasts cannot have a distinct uniformly better forecast;
    # ordinary floating roundoff is permitted only through the explicit score check.
    np.testing.assert_allclose(repaired,q,atol=1e-12)
    assert np.all(C.all_pattern_losses(repaired)<=C.all_pattern_losses(q))


def test_iteration_limit_and_nonprobability_inputs_fail_closed():
    with pytest.raises(ValueError):C.repair(np.full((1,9),.5),max_iterations=129)
    with pytest.raises(ValueError):C.repair(np.full((1,9),np.nan))
    with pytest.raises(ValueError):C.repair(np.full((1,9),.5),chunk_size=0)
