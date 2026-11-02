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
    job={'id':'job_0_exact_count','fold':0,'method':'exact_count','limit_seconds':3.,'nonce':'keep-private'}
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
    root=tmp_path/'local';root.mkdir();directory=root/'job_0_exact_count';directory.mkdir()
    assert seal_private_output(directory,'trace.json',root)==directory/'trace.json'
    with pytest.raises(ValueError):seal_private_output(tmp_path/'job_outside','weights.npz',root)
    with pytest.raises(ValueError):seal_private_output(directory,'../../public.json',root)
    outside=tmp_path/'outside';outside.write_text('keep');(directory/'weights.npz').symlink_to(outside)
    with pytest.raises(ValueError):seal_private_output(directory,'weights.npz',root)


def test_checkpoint_is_retained_when_optimizer_refuses_step(monkeypatch):
    from . import job
    from ..exact_count_learning_v1.test_model import config,thresholds
    class Refuse:
        def __init__(self,*a,**kw):pass
        def zero_grad(self):pass
        def step(self):raise RuntimeError('synthetic refused step; no parameter update')
    monkeypatch.setattr(torch.optim,'AdamW',Refuse)
    c=config();c['epochs']=1
    train={'prefix':np.zeros((2,73)),'pitch':np.zeros((2,4)),'power':np.zeros((2,4)),
           'voiced':np.ones((2,4),bool),'labels':np.zeros((2,9)),'weights':np.ones(2),'thresholds':thresholds().numpy()}
    checkpoints=[]
    with pytest.raises(RuntimeError,match='refused step'):
        job.train_model(train,np.zeros((1,73)),'exact_count',c,float('inf'),lambda model,trace:checkpoints.append(len(trace)))
    assert checkpoints==[0]  # finally checkpoint ran; no fitting or real optimizer step occurred


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


def test_parent_output_pin_detects_replaced_prediction_and_self_reported_receipt(tmp_path,monkeypatch):
    from . import runner
    x,m,y,c=toy();monkeypatch.setattr(runner,'HERE',tmp_path)
    (tmp_path/'local').mkdir();(tmp_path/'CONTRACT.json').write_text(json.dumps(c));(tmp_path/'FREEZE.json').write_text('{}')
    state={'completed':[],'outputs':{}}
    for spec in c['job_order']:
        name=f"job_{spec['fold']}_{spec['method']}";directory=tmp_path/'local'/name;directory.mkdir()
        indices=np.flatnonzero(m['fold']==spec['fold'])
        np.savez_compressed(directory/'predictions.npz',probability=np.full((len(indices),9),.5),row_indices=indices)
        np.savez_compressed(directory/'weights.npz',synthetic=np.array([1]))
        for f in ['trace.json','stats.json']:(directory/f).write_text('{}')
        receipt={'job':name,'method':spec['method'],'fold':spec['fold'],'epochs_completed':40,
                 'predictions_sha256':sha(directory/'predictions.npz'),'weights_sha256':sha(directory/'weights.npz'),
                 'trace_sha256':sha(directory/'trace.json'),'preprocessing_sha256':sha(directory/'stats.json')}
        (directory/'receipt.json').write_text(json.dumps(receipt))
        state['outputs'][name]=runner.pin_job_output(directory,{'id':name,**spec},indices,40);state['completed'].append(name)
    runner.seal_batch_outputs(state);runner.verify_batch_outputs(state)
    name=state['completed'][0];directory=tmp_path/'local'/name
    with np.load(directory/'predictions.npz') as z:indices=z['row_indices']
    np.savez_compressed(directory/'predictions.npz',probability=np.full((len(indices),9),.7),row_indices=indices)
    r=json.loads((directory/'receipt.json').read_text());r['predictions_sha256']=sha(directory/'predictions.npz')
    (directory/'receipt.json').write_text(json.dumps(r))
    with pytest.raises(ValueError,match='parent-pinned'):runner.verify_batch_outputs(state)


def test_no_held_label_scoring_before_complete_parent_output_seal(monkeypatch):
    from . import runner
    called=[]
    monkeypatch.setattr(runner,'fit_inner',lambda *a,**k:called.append(True))
    with pytest.raises(ValueError,match='output seal'):runner.evaluate_complete({}, {})
    assert not called
