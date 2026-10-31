import numpy as np
from agentic_research.projection_audit_v2.run_audit import selected_indices,load_allowed,diagnostics
from agentic_research.projection_audit_v2.solver import patterns


def test_selection_order_invariant():
    ids=np.array(['record-'+str(i) for i in range(100)])
    p=np.random.default_rng(42).permutation(100)
    assert ids[selected_indices(ids,12)].tolist()==ids[p][selected_indices(ids[p],12)].tolist()


def test_loader_reads_only_allowed_keys(monkeypatch):
    allowed={'ids':np.array(['x']),'probabilities':np.ones((1,9))*.4,
             'repaired_probabilities':np.ones((1,9))*.4}
    class Guarded:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def __getitem__(self,key):
            assert key in allowed, 'Forbidden observed-label or metric access'
            return allowed[key]
    monkeypatch.setattr(np,'load',lambda *args,**kwargs:Guarded())
    assert set(load_allowed('unused'))==set(allowed)


def test_incoherent_original_not_feasible_comparator():
    q=np.array([.9,.1,.5,.1,.8,.1,.3,.1,.7])
    d=diagnostics(q,q,patterns())
    assert not d['coherent']
    assert d['kl_objective_if_feasible'] is None


def test_same_selection_as_v1():
    from agentic_research.projection_audit_v1.run_audit import selected_indices as old
    ids=np.array(['record-'+str(i) for i in range(100)])
    np.testing.assert_array_equal(selected_indices(ids,64),old(ids,64))
