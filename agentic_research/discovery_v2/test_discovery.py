import numpy as np
import torch
import json
import pytest
from agentic_research.discovery_v2.models import MomentModel
from agentic_research.discovery_v2.runner import build_inputs
from agentic_research.discovery_v1.data import fit_fold,source_folds
from agentic_research.discovery_v1.data import sha
import agentic_research.discovery_v2.runner as runner


def test_delta_anchor_is_exact_when_learned_output_is_zero():
    for attention in [False,True]:
        model=MomentModel(attention,True)
        with torch.no_grad():model.head.weight.zero_();model.head.bias.zero_()
        anchor=torch.tensor([[1.,2.],[3.,4.]])
        raw=model(torch.zeros(2,73),torch.zeros(2,96,4),anchor)
        torch.testing.assert_close(raw[:,2:6],anchor.repeat(1,2))


def test_all_four_shapes_and_gradients_finite():
    torch.manual_seed(0)
    for attention in [False,True]:
        for delta in [False,True]:
            model=MomentModel(attention,delta);out=model(torch.randn(2,73),torch.randn(2,96,4),torch.zeros(2,2))
            assert out.shape==(2,11 if delta else 10)
            out.square().mean().backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_query_is_prefix_only_and_fold_scalers_ignore_held_future():
    rng=np.random.default_rng(3);n=36
    people=np.array([f'p{i}_{c}' for c in ['CG','EG','FG'] for i in range(3) for _ in range(4)])
    x=rng.normal(size=(n,60));x[:,57:59]=rng.uniform(.1,2,(n,2));x[:,59]=1
    seq=rng.normal(size=(n,96,4));target=rng.uniform(.1,4,(n,2));guard=np.ones(n,bool);mask=source_folds(people)!=0
    a=fit_fold(x,seq,target,guard,people,mask);q1=build_inputs(x,seq,a)
    target[~mask]=np.nan;guard[~mask]=False
    b=fit_fold(x,seq,target,guard,people,mask);q2=build_inputs(x,seq,b)
    for p,q in zip(q1,q2):np.testing.assert_array_equal(p,q)
    assert q1[0].shape==(n,73) and np.all(q1[0][:,60:71].sum(1)==1)
    # Both head types receive the same exact moments even when summary clips.
    x[0,57:59]=1e20
    query,_,anchor,_=build_inputs(x,seq,a)
    np.testing.assert_array_equal(query[:,-2:],anchor)
    assert np.max(np.abs(anchor[0]))>8 and np.max(np.abs(query[0,:60]))<=8


def test_parent_artifacts_require_consistent_receipt_and_seal(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'V1',tmp_path)
    frozen={'sources':{'x':'h'},'source_inputs':{'y':'i'}}
    monkeypatch.setattr(runner,'check_v1',lambda:frozen)
    (tmp_path/'local').mkdir();(tmp_path/'local/prepared.npz').write_bytes(b'synthetic bytes,not real records')
    (tmp_path/'FREEZE.json').write_text(json.dumps(frozen))
    receipt={'prepared_sha256':sha(tmp_path/'local/prepared.npz'),'freeze_sha256':sha(tmp_path/'FREEZE.json'),
             'source_inputs':frozen['source_inputs'],'source_hashes':frozen['sources']}
    (tmp_path/'PREPARED.json').write_text(json.dumps(receipt))
    result={'prepared_sha256':receipt['prepared_sha256'],'freeze_sha256':receipt['freeze_sha256'],
            'prepared_receipt_sha256':sha(tmp_path/'PREPARED.json')}
    (tmp_path/'RESULTS.json').write_text(json.dumps(result))
    (tmp_path/'RESULTS_SEAL.json').write_text(json.dumps({'results_sha256':sha(tmp_path/'RESULTS.json')}))
    assert len(runner.inputs())==5
    receipt['prepared_sha256']='wrong';(tmp_path/'PREPARED.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='provenance'):runner.inputs()
