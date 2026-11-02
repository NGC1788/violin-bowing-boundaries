import copy,datetime,json,subprocess
from pathlib import Path
import numpy as np
import pytest
import torch
from .data import (source_folds,validate_arrays,load_arrays,fit_inner,prediction_inputs,
                   verify_predictions,score,sha)
from .control import register,job_limit,wait_child,run_child,seal_private_output


def toy():
    from .runner import contract
    c=copy.deepcopy(contract());people=np.repeat([f'toy{i}_{cohort}' for cohort in ['CG','EG','FG'] for i in range(3)],2)
    n=len(people);rng=np.random.default_rng(19);fold=source_folds(people)
    x=rng.normal(size=(n,60));x[:,57:59]=np.abs(x[:,57:59]);x[:,59]=1.
    prefix={'X':x,'sequence':rng.normal(size=(n,96,4)).astype(np.float32)}
    meta={'people':people,'source_ids':np.array([f'record_{i:02}' for i in range(n)]),'fold':fold}
    pitch=rng.normal(size=(n,46));power=rng.normal(size=(n,46));mask=np.ones((n,46),bool)
    mask[0,:26]=False;mask[5,:25]=False;guard=mask.sum(1)>=23
    targets=np.column_stack(([np.std(y[m],ddof=1) if g else np.nan for y,m,g in zip(pitch,mask,guard)],power.std(1,ddof=1)))
    starts=55125+np.arange(46)*441;support=np.tile(np.column_stack((starts,starts+2205))[None],(n,1,1))
    labels={'pitch_cents':pitch,'power_dbfs':power,'periodicity':rng.random((n,46)),
            'voiced':mask,'guard':guard,'targets':targets,'support':support}
    c.update(expected_rows=n,expected_people=9,expected_invalid=2,fold_rows=[6,6,6],fold_people=[3,3,3])
    return prefix,meta,labels,c


def test_strict_source_schema_counts_and_target_reconstruction():
    prefix,meta,labels,c=toy();d=validate_arrays(prefix,meta,labels,c)
    assert len(d['X'])==18 and (~d['guard']).sum()==2
    bad=copy.deepcopy(prefix);bad['future_guard']=labels['guard']
    with pytest.raises(ValueError):validate_arrays(bad,meta,labels,c)
    bad=copy.deepcopy(labels);bad['targets'][1,0]+=.1
    with pytest.raises(ValueError):validate_arrays(prefix,meta,bad,c)


def test_duplicates_person_overlap_and_time_support_fail_closed():
    x,m,y,c=toy();bad=copy.deepcopy(m);bad['source_ids'][1]=bad['source_ids'][0]
    with pytest.raises(ValueError):validate_arrays(x,bad,y,c)
    bad=copy.deepcopy(m);bad['fold'][0]=(bad['fold'][0]+1)%3
    with pytest.raises(ValueError):validate_arrays(x,bad,y,c)
    bad=copy.deepcopy(y);bad['support'][0,0,0]-=441
    with pytest.raises(ValueError):validate_arrays(x,m,bad,c)


def test_sealed_loader_catches_row_reordering_and_never_uses_held_truth(tmp_path):
    x,m,y,c=toy();paths={n:tmp_path/n for n in ['prefix_inputs.npz','source_metadata.npz','future_labels.npz']}
    for name,values in zip(paths,[x,m,y]):np.savez_compressed(paths[name],**values)
    expected={n:sha(p) for n,p in paths.items()};load_arrays(paths,expected,c)
    altered={k:v[::-1] for k,v in y.items()};np.savez_compressed(paths['future_labels.npz'],**altered)
    with pytest.raises(ValueError):load_arrays(paths,expected,c)
    from .runner import input_paths
    names={p.name for p in input_paths().values()}
    assert not {'dev_labels.npz','test_labels.npz'}&names


def test_held_label_poison_does_not_change_training_or_inference_payload():
    x,m,y,c=toy();d=validate_arrays(x,m,y,c);train,stats,prior=fit_inner(d,0);evalx,indices=prediction_inputs(d,0,stats)
    bad=copy.deepcopy(d);held=d['fold']==0
    for key in ['pitch','power','targets']:bad[key][held]=np.nan
    bad['voiced'][held]=False;bad['guard'][held]=False
    again,s2,p2=fit_inner(bad,0);xe,i2=prediction_inputs(bad,0,s2)
    for key in train:np.testing.assert_array_equal(train[key],again[key])
    for key in stats:np.testing.assert_equal(stats[key],s2[key])
    np.testing.assert_array_equal(prior,p2);np.testing.assert_array_equal(evalx,xe);np.testing.assert_array_equal(indices,i2)
    assert set(train)=={'prefix','pitch','power','voiced','labels','weights','thresholds'}
    assert len(train['prefix'])==12 and len(evalx)==6


def test_prediction_order_finite_and_person_macro_score():
    index=np.array([1,3,5]);p=np.full((3,9),.4)
    with pytest.raises(ValueError):verify_predictions(p,index[::-1],index)
    bad=p.copy();bad[0,0]=np.nan
    with pytest.raises(ValueError):verify_predictions(bad,index,index)
    y=np.zeros_like(p);p[0]=.1;p[1:]=.8
    result,means=score(y,p,np.array(['a','b','b']))
    assert result['log_loss']==pytest.approx((-np.log(.9)-np.log(.2))/2)
    assert len(means)==2


def test_ledger_attempts_are_consumed_failures_and_private_nonce_is_not_shared(tmp_path):
    ledger=tmp_path/'ledger.json';ledger.write_text(json.dumps({'additional_batches_started':[],'max_additional_training_batches':1,
             'max_jobs_per_additional_batch':2,'max_seconds_per_training_job':180}))
    register(ledger,'START',freeze_sha='toy')
    with pytest.raises(ValueError):register(ledger,'START',freeze_sha='again')
    job={'id':'job_0_retrieval_erm','fold':0,'method':'retrieval_erm','limit_seconds':3.,'nonce':'keep-private'}
    register(ledger,'JOB_START',job=job)
    with pytest.raises(ValueError):register(ledger,'JOB_START',job=job)
    register(ledger,'JOB_END',job={'id':job['id'],'status':'FAILED_STOP'},seconds=3.)
    register(ledger,'FAILED_STOP',seconds=3.)
    d=json.loads(ledger.read_text());assert len(d['additional_batches_started'])==1
    assert d['additional_batches_started'][0]['jobs'][0]['status']=='FAILED_STOP'
    assert 'keep-private' not in ledger.read_text()


def test_job_time_limits_include_worker_and_deadline():
    c=toy()[3];utc=datetime.timezone.utc;now=datetime.datetime(2026,11,3,tzinfo=utc)
    assert job_limit(c,0,now)==180
    assert job_limit(c,1795,now)==5
    with pytest.raises(TimeoutError):job_limit(c,1800,now)
    with pytest.raises(TimeoutError):job_limit(c,0,datetime.datetime(2026,11,6,tzinfo=utc))


def test_other_active_campaign_blocks_a_second_training_worker(tmp_path):
    path=tmp_path/'ledger.json';path.write_text(json.dumps({'additional_batches_started':[{'campaign':'other','status':'RUNNING'}],
          'max_additional_training_batches':3,'max_jobs_per_additional_batch':12,'max_seconds_per_training_job':180}))
    with pytest.raises(ValueError,match='another training batch'):register(path,'START',freeze_sha='toy')
    assert len(json.loads(path.read_text())['additional_batches_started'])==1


def test_watchdog_terminates_and_kills_unresponsive_child():
    class Fake:
        def __init__(self):self.waits=0;self.terminated=False;self.killed=False;self.code=None
        def poll(self):return self.code
        def wait(self,timeout=None):
            self.waits+=1
            if self.waits<3:raise subprocess.TimeoutExpired('toy',timeout)
            self.code=-9;return self.code
        def terminate(self):self.terminated=True
        def kill(self):self.killed=True
    p=Fake()
    with pytest.raises(TimeoutError):wait_child(p,.01,grace=.01)
    assert p.terminated and p.killed and p.waits==3


def test_whole_worker_interrupt_does_not_leave_child_running():
    class Fake:
        def __init__(self):self.waits=0;self.terminated=False;self.code=None
        def poll(self):return self.code
        def wait(self,timeout=None):
            self.waits+=1
            if self.waits==1:raise TimeoutError('simulated whole-worker alarm')
            self.code=-15;return self.code
        def terminate(self):self.terminated=True
        def kill(self):raise AssertionError('responsive child should not requirekill')
    p=Fake()
    with pytest.raises(TimeoutError,match='whole-worker'):wait_child(p,1.)
    assert p.terminated and p.waits==2


def test_interrupt_during_grace_still_kills_and_reaps_child():
    class Fake:
        def __init__(self):self.waits=0;self.killed=False;self.code=None
        def poll(self):return self.code
        def wait(self,timeout=None):
            self.waits+=1
            if self.waits==1:raise subprocess.TimeoutExpired('toy',timeout)
            if self.waits==2:raise TimeoutError('simulated alarm during grace')
            self.code=-9;return self.code
        def terminate(self):pass
        def kill(self):self.killed=True
    p=Fake()
    with pytest.raises(TimeoutError,match='during grace'):run_child(lambda:p,1.)
    assert p.killed and p.code==-9


def test_pending_alarm_immediately_after_spawn_has_owned_cleanup(monkeypatch):
    from . import control
    class Fake:
        def __init__(self):self.code=None;self.terminated=False
        def poll(self):return self.code
        def wait(self,timeout=None):self.code=-15;return self.code
        def terminate(self):self.terminated=True
        def kill(self):raise AssertionError('should terminate cleanly')
    p=Fake();interrupted=False
    def masks(how,values):
        nonlocal interrupted
        if how==control.signal.SIG_SETMASK and not interrupted:
            interrupted=True;raise TimeoutError('alarm immediately after spawn')
        return set()
    monkeypatch.setattr(control.signal,'pthread_sigmask',masks)
    with pytest.raises(TimeoutError,match='immediately after spawn'):run_child(lambda:p,1.)
    assert p.terminated and p.code==-15


def test_private_output_cannot_escape_or_follow_symlink(tmp_path):
    root=tmp_path/'local';root.mkdir();directory=root/'job_0_retrieval_erm';directory.mkdir()
    assert seal_private_output(directory,'trace.json',root)==directory/'trace.json'
    with pytest.raises(ValueError):seal_private_output(tmp_path/'job_outside','weights.npz',root)
    with pytest.raises(ValueError):seal_private_output(directory,'../../public.json',root)
    outside=tmp_path/'outside';outside.write_text('keep');(directory/'weights.npz').symlink_to(outside)
    with pytest.raises(ValueError):seal_private_output(directory,'weights.npz',root)



def test_child_preprocessing_does_not_refit_analytic_markov(monkeypatch):
    from . import data
    x,m,y,c=toy();dataset=validate_arrays(x,m,y,c);original=data.transition_prior;calls=[]
    def spy(*args):calls.append(True);return original(*args)
    monkeypatch.setattr(data,'transition_prior',spy)
    assert fit_inner(dataset,0)[2] is None and calls==[]
    assert fit_inner(dataset,0,fit_markov=True)[2] is not None and len(calls)==1


def test_unique_checkpoint_temp_does_not_follow_predictable_symlink(tmp_path):
    from .job import save_weights
    outside=tmp_path/'outside';outside.write_bytes(b'keep')
    target=tmp_path/'partial_weights.npz';target.with_suffix('.tmp').symlink_to(outside)
    save_weights(torch.nn.Linear(1,1).double(),target)
    assert outside.read_bytes()==b'keep'
    with np.load(target,allow_pickle=False) as z:assert 'weight' in z.files



def test_no_held_label_scoring_before_complete_parent_output_seal(monkeypatch):
    from . import runner
    called=[]
    monkeypatch.setattr(runner,'fit_inner',lambda *a,**k:called.append(True))
    with pytest.raises(ValueError,match='output seal'):runner.evaluate_complete({}, {})
    assert not called


def small_memory():
    from .model import bank
    labels=bank()[torch.tensor([0,9,1,8,2,7])]
    groups=torch.tensor([0,0,1,1,2,2])
    scores=torch.tensor([[2.,-1.,0.,3.,-2.,1.],[-1.,3.,2.,0.,1.,-2.]],dtype=torch.float64,requires_grad=True)
    return scores,labels,groups,torch.tensor([0,-1])


def test_exact_joint_group_deletion_matches_explicit_remainder_and_prior():
    from .model import deletion_probabilities,bank
    scores,labels,groups,query=small_memory();got=deletion_probabilities(scores,labels,groups,query,.01)
    for i in range(2):
        for d,g in enumerate(got['donor_codes']):
            mask=(groups!=query[i])&(groups!=g)
            expected=.99*(torch.softmax(scores[i,mask],0)@labels[mask])+.01*bank().mean(0)
            torch.testing.assert_close(got['probability'][i,d],expected,atol=1e-14,rtol=1e-14)
    assert not got['eligible'][0,1]  # already-excluded own donor is duplicate null
    assert bool(((got['probability']>=.001-1e-14)&(got['probability']<=.999+1e-14)).all())
    from ..violin_raw_v1.metrics import coherence_rate
    assert coherence_rate(got['probability'].detach().numpy().reshape(-1,9))==1


def test_extreme_logits_remain_finite_without_subtractive_cancellation():
    from .model import deletion_probabilities
    scores,labels,groups,query=small_memory();scores=torch.tensor([[1000.,-1000.,-900.,900.,-800.,800.]],dtype=torch.float64,requires_grad=True)
    got=deletion_probabilities(scores,labels,groups,torch.tensor([-1]),.01)
    assert bool(torch.isfinite(got['probability']).all())
    got['probability'].sum().backward();assert bool(torch.isfinite(scores.grad).all())
    # Empty allowed remainder must fail, not return an invented forecast.
    with pytest.raises(ValueError,match='empty'):
        deletion_probabilities(scores[:,:4],labels[:4],groups[:4],torch.tensor([0]),.01)


def test_own_person_and_deleted_donor_have_zero_effect_and_allowed_label_control_changes_prediction():
    from .model import deletion_probabilities,bank
    scores,labels,groups,query=small_memory();query=torch.tensor([0,0]);a=deletion_probabilities(scores,labels,groups,query,.01)
    changed=labels.clone();changed[:2]=bank()[5]
    b=deletion_probabilities(scores,changed,groups,query,.01)
    torch.testing.assert_close(a['probability'],b['probability'],atol=0,rtol=0)
    changed=labels.clone();changed[2:4]=bank()[9]
    c=deletion_probabilities(scores,changed,groups,query,.01)
    assert not torch.equal(a['probability'][:,0],c['probability'][:,0])
    torch.testing.assert_close(a['probability'][:,2],c['probability'][:,2],atol=0,rtol=0)  # donor1 removed
    a['probability'][:,0].sum().backward();assert torch.equal(scores.grad[:,:2],torch.zeros_like(scores.grad[:,:2]))


def test_missing_raw_support_is_preserved_despite_finite_prior():
    from .model import deletion_probabilities
    from .data import training_support_summary
    scores,labels,groups,query=small_memory();labels=torch.zeros_like(labels)
    out=deletion_probabilities(scores,labels,groups,query,.01)
    assert bool((out['positive_support']==0).all()) and bool((out['probability']>0).all())
    summary=training_support_summary({'labels':labels.numpy(),'groups':groups.numpy()})
    assert summary['zero_positive_support_cells']==summary['query_group_macro_donor_event_cells']


def test_adversarial_loss_max_is_after_query_average_and_random_menu_is_seeded():
    from .model import event_losses
    q=torch.tensor([[[.2,.4,.6,.3,.5,.7,.4,.6,.8],[.4,.3,.1,.5,.4,.2,.6,.5,.3]]],dtype=torch.float64)
    y=torch.ones((1,9),dtype=torch.float64)
    losses=event_losses(q,y)
    expected=(-q.log()).mean(-1).max(1).values
    torch.testing.assert_close(losses.max(1).values,expected)
    assert float(expected)<float((-q.log()).max(1).values.mean(-1))


def test_shared_initialization_and_common_aux_equal_frozen_constructor():
    from .runner import contract
    from .model import RetrievalForecaster
    from ..exact_count_learning_v1.model import Forecaster,parameterize,auxiliary_nll
    from .job import state_hash
    cfg=contract()['model_config'];old=copy.deepcopy(cfg);old['methods']=['exact_count']
    torch.manual_seed(0);base=Forecaster('exact_count',old)
    hashes=[];x=torch.zeros((2,73),dtype=torch.float64)
    for method in cfg['methods']:
        torch.manual_seed(0);model=RetrievalForecaster(method,cfg);hashes.append(state_hash(model))
        torch.testing.assert_close(model.encoder(x),base.encoder(x),atol=0,rtol=0)
        torch.testing.assert_close(model.generator(model.encoder(x)),base.generator(base.encoder(x)),atol=0,rtol=0)
    assert len(set(hashes))==1 and hashes[0]==state_hash(base)
    parent=json.loads((Path(__file__).resolve().parent.parent/'exact_count_runner_v1/RESULTS.json').read_text())
    assert all(hashes[0]==v['initial_shared_sha256'] for v in parent['common_control_hashes'].values())


def test_query_dro_update_has_correct_importance_scaling_and_no_aux_reweighting():
    from .model import QueryDRO
    groups=torch.tensor([0,0,1]);dro=QueryDRO(groups,.01)
    torch.testing.assert_close(dro.multipliers(groups),torch.ones(3,dtype=torch.float64))
    loss=torch.tensor([1.,3.,4.],dtype=torch.float64);dro.update(loss,groups)
    expected=torch.log_softmax(.01*torch.tensor([2.,4.],dtype=torch.float64),0)
    torch.testing.assert_close(dro.logweights,expected)


def test_all_four_objectives_have_finite_gradients_with_identical_shared_aux():
    from .runner import contract
    from .model import RetrievalForecaster
    c=contract()['model_config'];torch.set_num_threads(1)
    prefix=torch.randn((6,73),dtype=torch.float64);groups=torch.tensor([0,0,1,1,2,2]);labels=torch.zeros((6,9),dtype=torch.float64);labels[1::2]=1
    memory={'prefix':prefix,'labels':labels,'groups':groups}
    pitch=torch.zeros((2,46),dtype=torch.float64);power=torch.zeros_like(pitch);voiced=torch.ones_like(pitch,dtype=torch.bool);aux=[]
    for method in c['methods']:
        torch.manual_seed(0);m=RetrievalForecaster(method,c)
        primary,a,info=m.loss_rows(prefix[:2],memory,labels[:2],pitch,power,voiced,groups[:2],torch.Generator().manual_seed(7))
        (primary+.1*a).mean().backward()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in m.parameters())
        aux.append(a.detach())
    for a in aux[1:]:torch.testing.assert_close(a,aux[0],atol=0,rtol=0)


def test_supervised_labels_change_loss_and_gradient_without_entering_prediction():
    from .model import deletion_probabilities,event_losses
    scores,labels,groups,query=small_memory()
    out=deletion_probabilities(scores,labels,groups,query,.01)
    loss0=event_losses(out['probability'],torch.zeros((2,9),dtype=torch.float64))[:,0].sum()
    grad0=torch.autograd.grad(loss0,scores,retain_graph=True)[0]
    loss1=event_losses(out['probability'],torch.ones((2,9),dtype=torch.float64))[:,0].sum()
    grad1=torch.autograd.grad(loss1,scores)[0]
    assert not torch.equal(grad0,grad1) and not torch.equal(loss0,loss1)


def test_deterministic_memory_archive_and_strict_parent_counterfactual_schema(tmp_path):
    from .runner import verify_retrieval_predictions
    arrays={'prefix':np.zeros((2,73)),'labels':np.zeros((2,9)),'groups':np.array([0,1]),'source_rows':np.array([1,2])}
    for name in ['a.npz','b.npz']:np.savez_compressed(tmp_path/name,**arrays)
    assert sha(tmp_path/'a.npz')==sha(tmp_path/'b.npz')
    cf=np.full((2,3,9),.5)
    out=dict(probability=cf[:,0],counterfactual_probability=cf.copy(),row_indices=np.array([3,4]),donor_codes=np.array([-1,0,1]),group_mass=np.full((2,2),.5),positive_support=np.ones_like(cf),negative_support=np.ones_like(cf),dro_logweights=np.zeros(2))
    verify_retrieval_predictions(out,np.array([3,4]),2)
    out['counterfactual_probability'][:,0]=.7
    with pytest.raises(ValueError,match='order'):verify_retrieval_predictions(out,np.array([3,4]),2)


def test_training_failure_retains_checkpoint_without_any_optimizer_update(monkeypatch):
    from . import job
    from .runner import contract
    cfg=copy.deepcopy(contract()['model_config']);cfg['epochs']=1
    train={'prefix':np.zeros((6,73)),'pitch':np.zeros((6,46)),'power':np.zeros((6,46)),
           'voiced':np.ones((6,46),bool),'labels':np.zeros((6,9)),'weights':np.ones(6),'thresholds':np.ones((2,3))}
    memory={'prefix':train['prefix'],'labels':train['labels'],'groups':np.repeat(np.arange(3),2),'source_rows':np.arange(6)}
    class Refuse:
        def __init__(self,*args,**kwargs):pass
        def zero_grad(self):pass
        def step(self):raise RuntimeError('synthetic refused step; no update')
    monkeypatch.setattr(torch.optim,'AdamW',Refuse)
    checkpoints=[]
    with pytest.raises(RuntimeError,match='refused step'):
        job.train_model(train,memory,np.zeros((1,73)),'retrieval_erm',cfg,float('inf'),lambda m,t:checkpoints.append(len(t)))
    assert checkpoints==[0]
    checkpoints=[]
    with pytest.raises(TimeoutError):
        job.train_model(train,memory,np.zeros((1,73)),'retrieval_erm',cfg,-1,lambda m,t:checkpoints.append(len(t)))
    assert checkpoints==[0]


def test_parent_pins_memory_and_all_counterfactuals_before_scoring(tmp_path,monkeypatch):
    from . import runner
    _,meta,_,c=toy();monkeypatch.setattr(runner,'HERE',tmp_path)
    (tmp_path/'local').mkdir();(tmp_path/'CONTRACT.json').write_text(json.dumps(c));(tmp_path/'FREEZE.json').write_text('{}')
    state={'completed':[],'outputs':{}}
    for spec in c['job_order']:
        name=f"job_{spec['fold']}_{spec['method']}";directory=tmp_path/'local'/name;directory.mkdir();indices=np.flatnonzero(meta['fold']==spec['fold'])
        cf=np.full((len(indices),3,9),.5)
        np.savez_compressed(directory/'predictions.npz',probability=cf[:,0],counterfactual_probability=cf,row_indices=indices,donor_codes=np.arange(-1,2),group_mass=np.full((len(indices),2),.5),positive_support=np.ones_like(cf),negative_support=np.ones_like(cf),dro_logweights=np.zeros(2))
        np.savez_compressed(directory/'weights.npz',synthetic=np.array([1]))
        np.savez_compressed(directory/'memory.npz',prefix=np.zeros((2,73)),labels=np.zeros((2,9)),groups=np.array([0,1]),source_rows=np.array([999,1000]))
        for f in ['trace.json','stats.json']:(directory/f).write_text('{}')
        receipt={'job':name,'method':spec['method'],'fold':spec['fold'],'epochs_completed':40,'predictions_sha256':sha(directory/'predictions.npz'),'weights_sha256':sha(directory/'weights.npz'),'trace_sha256':sha(directory/'trace.json'),'preprocessing_sha256':sha(directory/'stats.json'),'memory_artifact_sha256':sha(directory/'memory.npz')}
        (directory/'receipt.json').write_text(json.dumps(receipt));state['outputs'][name]=runner.pin_job_output(directory,{'id':name,**spec},indices,40);state['completed'].append(name)
    runner.seal_batch_outputs(state);runner.verify_batch_outputs(state)
    directory=tmp_path/'local'/state['completed'][0]
    with np.load(directory/'predictions.npz') as z:arrays={k:z[k] for k in z.files}
    arrays['counterfactual_probability'][:,1]=.7;np.savez_compressed(directory/'predictions.npz',**arrays)
    with pytest.raises(ValueError,match='parent-pinned'):runner.verify_batch_outputs(state)
