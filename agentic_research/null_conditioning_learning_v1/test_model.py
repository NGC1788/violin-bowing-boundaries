"""Small synthetic functionality only: no fitting/data/model-weight IO."""
import copy
import inspect
import numpy as np
import pytest
import torch
from .model import ConditioningForecaster,null_projector,applied_displacement,match_norm
from .data import fine_to_coarse,fit_fine_cuts,fine_labels
from .runner import contract

torch.set_num_threads(1)


def sample(method='null_conditioned'):
    c=contract()['model_config'];torch.manual_seed(41)
    model=ConditioningForecaster(method,c)
    x=torch.randn((8,73),dtype=torch.float64)
    fine=torch.tensor([0,1,12,15,32,41,63,64],dtype=torch.long)
    coarse=model.fine_map[fine];weights=torch.ones(8,dtype=torch.float64)
    return model,x,fine,coarse,weights


def test_finite_step_projector_global_linear_identity_and_softmax():
    m,x,*_=sample();p,rank,residual=null_projector(m.free_head.weight,1e-10)
    assert rank==9 and residual<1e-12
    torch.testing.assert_close(p@p,p,rtol=0,atol=1e-12)
    delta=p@torch.randn_like(m.adapter.weight)
    # Finite (not infinitesimal), on unrelated hidden probes of two scales.
    u=torch.randn((11,32),dtype=torch.float64)*7
    a=m.free_head.weight.detach();b=m.free_head.bias.detach();w=m.adapter.weight.detach()
    before=torch.softmax(u@w.T@a.T+b,1);after=torch.softmax(u@(w+delta).T@a.T+b,1)
    torch.testing.assert_close(before,after,rtol=0,atol=1e-12)


def test_actual_preconditioned_and_decay_displacement_must_be_projected():
    a=torch.zeros((10,32),dtype=torch.float64);a[0,:2]=1
    p,_,_=null_projector(a,1e-10)
    grad=torch.zeros((32,32),dtype=torch.float64);grad[0,0]=1;grad[1,0]=-1
    assert float(torch.linalg.vector_norm((a-a.mean(0))@(p@grad)))<1e-12
    row_preconditioner=torch.arange(1,33,dtype=torch.float64)[:,None]
    actual=-(p@grad)*row_preconditioner-.01*torch.eye(32,dtype=torch.float64)
    assert float(torch.linalg.vector_norm((a-a.mean(0))@actual))>.1
    assert float(torch.linalg.vector_norm((a-a.mean(0))@(p@actual)))<1e-12


def test_zero_rank_zero_norm_and_zero_random_denominator():
    a=torch.zeros((10,32),dtype=torch.float64);p,rank,res=null_projector(a,1e-10)
    assert rank==0 and res==0 and torch.equal(p,torch.eye(32,dtype=torch.float64))
    proposal=torch.zeros((32,32),dtype=torch.float64)
    for method in contract()['model_config']['methods']:
        delta,target=applied_displacement(proposal,p,p,rank,method)
        assert target==0 and torch.equal(delta,proposal)
    assert torch.equal(match_norm(proposal,torch.tensor(2.)),proposal)


def test_all_arms_have_same_structure_init_and_norm_rule():
    states=[]
    for method in contract()['model_config']['methods']:
        m,x,f,c,w=sample(method);states.append(copy.deepcopy(m.state_dict()))
        assert sum(v.numel() for v in m.parameters())==7583
        diag=m.fine_step(x,f,c,w)
        if method=='main_only':assert diag['applied_norm']==0
        else:assert diag['applied_norm']==pytest.approx(diag['target_norm'],rel=1e-10,abs=1e-15)
        if method=='null_conditioned':assert diag['softmax_drift']<1e-12 and diag['query_drift']<1e-12
    for state in states[1:]:
        for k in state:assert torch.equal(states[0][k],state[k]),k


def test_fine_step_freezes_encoder_coarse_generator_and_changes_fine_head():
    for method in contract()['model_config']['methods']:
        m,x,f,c,w=sample(method);before=copy.deepcopy(m.state_dict());_,g=m(x)
        m.fine_step(x,f,c,w);_,after_g=m(x)
        for key,value in m.state_dict().items():
            if not key.startswith(('adapter.','fine_head.')):assert torch.equal(value,before[key]),key
        for key in g:assert torch.equal(g[key],after_g[key]),key
        assert not torch.equal(before['fine_head.weight'],m.fine_head.weight)
        assert not {id(v) for v in m.fine_head.parameters()}&{id(v) for v in m.main_parameters()}


def test_nonfinite_and_ill_conditioned_steps_fail_without_repair():
    m,x,f,c,w=sample('main_only')
    with torch.no_grad():m.adapter.weight.zero_()
    with pytest.raises(ArithmeticError,match='conditioning'):m.fine_step(x,f,c,w)
    m,x,f,c,w=sample()
    with torch.no_grad():m.fine_head.bias.fill_(float('nan'))
    with pytest.raises(ArithmeticError,match='nonfinite'):m.fine_step(x,f,c,w)


def test_post_adapter_nonlinearity_or_changed_readout_encoder_can_break_proof():
    a=torch.zeros((10,32),dtype=torch.float64);a[0,:2]=1
    w=torch.eye(32,dtype=torch.float64);d=torch.zeros_like(w);d[0,0]=-2;d[1,0]=2
    u=torch.zeros((32,1),dtype=torch.float64);u[0]=1
    assert torch.equal(a@(w+d)@u,a@w@u)
    assert not torch.equal(a@torch.relu((w+d)@u),a@torch.relu(w@u))
    changed=a.clone();changed[1,0]+=2
    assert not torch.equal(torch.softmax(changed@(w+d)@u,0),torch.softmax(a@w@u,0))
    assert not torch.equal(torch.softmax(a@(w+d)@(u*0),0),torch.softmax(a@w@u,0))


def test_exact_65_to_10_fibres_and_boundary_ties_without_dropping_classes():
    mapping=fine_to_coarse();counts=np.bincount(mapping,minlength=10)
    assert sorted(counts.tolist())==[4]*9+[29] and mapping[64]==0
    y=np.column_stack((np.arange(1,33),np.arange(1,33)**1.2));g=np.ones(32,bool)
    coarse=np.quantile(y,[.25,.5,.75],axis=0).T;cuts=fit_fine_cuts(y,g,coarse)
    np.testing.assert_array_equal(cuts[:,[1,3,5]],coarse)
    from ..violin_raw_v1.data import labels_and_states
    for i in range(3):
        target=np.vstack((coarse[:,i],coarse[:,i]+1e-8))
        valid=np.ones(2,bool);ids=fine_labels(target,valid,cuts)
        _,states=labels_and_states(target,valid,coarse)
        np.testing.assert_array_equal(mapping[ids],states)
    # Repeated fine cuts allowed; original coarse cuts remain strict.
    tied=cuts.copy();tied[:,0]=tied[:,1];tied[:,2]=tied[:,3]
    ids=fine_labels(y,g,tied);_,states=labels_and_states(y,g,coarse)
    np.testing.assert_array_equal(mapping[ids],states)
    assert fine_labels(np.array([[np.nan,1.]]),np.array([False]),cuts)[0]==64
    bad=coarse.copy();bad[:,1]=bad[:,0]
    with pytest.raises(ValueError,match='strictly increasing'):fit_fine_cuts(y,g,bad)


def test_training_labels_change_fine_loss_and_gradient_but_are_not_inference_inputs():
    m,x,_,_,w=sample();u=m.encoder(x).detach()
    f=torch.zeros(8,dtype=torch.long);coarse=m.fine_map[f]
    loss0=m.fine_rows(u,f,coarse).mean();grad0=torch.autograd.grad(loss0,m.adapter.weight)[0]
    loss1=m.fine_rows(u,f+1,coarse).mean();grad1=torch.autograd.grad(loss1,m.adapter.weight)[0]
    assert not torch.equal(loss0,loss1) and not torch.equal(grad0,grad1)
    assert list(inspect.signature(m.forward).parameters)==['prefix']
    with pytest.raises(TypeError):m(x,future_mask=torch.ones((8,46)))
    with pytest.raises(ValueError,match='fibre'):m.fine_rows(u,f,torch.zeros_like(coarse))


def test_historical_shared_initialization_coarse_head_and_common_aux_match():
    from ..exact_count_learning_v1.model import Forecaster
    c=contract()['model_config'];old_c=copy.deepcopy(c);old_c['methods']=['joint_aux']
    torch.manual_seed(0);old=Forecaster('joint_aux',old_c)
    torch.manual_seed(0);new=ConditioningForecaster('null_conditioned',c)
    for n,v in old.state_dict().items():
        if n.startswith(('encoder.','generator.','free_head.')):assert torch.equal(v,new.state_dict()[n]),n
    x=torch.randn((4,73),dtype=torch.float64);qo,po=old(x,torch.ones((2,3),dtype=torch.float64));qn,pn=new(x)
    torch.testing.assert_close(qo,qn,rtol=0,atol=0)
    for key in po:assert torch.equal(po[key],pn[key])
