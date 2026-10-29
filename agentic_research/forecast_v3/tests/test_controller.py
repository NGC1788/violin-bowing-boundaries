"""Real subprocess control tests use tiny synthetic data, never development scores."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import time
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from agentic_research.forecast_v3 import control as C
BASE=Path(__file__).resolve().parents[1]


@pytest.fixture
def campaign(tmp_path):
    root=tmp_path/'forecast_v3';root.mkdir();(root/'candidates').mkdir()
    for name in ('control.py','worker.py'):shutil.copyfile(BASE/name,root/name)
    for name in ('cli.py','models.py','rule_patterns.py','llm_backend.py'):(root/name).write_text('# synthetic fixture\n')
    (root/'experiment.py').write_text('''from . import models
def evaluate(config,dataset,output_dir):
    if config['method'] in ('source_markov','mlp_joint'): raise RuntimeError('intentional ordinary fixture failure')
    if config['method']=='hgb_history': raise AssertionError('synthetic verifier failure')
    if config['method']=='mlp_ind': return {'verification_ok':False,'verification_error_count':1}
    return {'mean_log_loss':.5,'mean_brier':.2,'development_rows':2,'verification_ok':True,'verification_error_count':0}
''')
    (root/'data.py').write_text('''import json
from pathlib import Path
import numpy as np
def load_prepared_dataset(path):
    with np.load(path,allow_pickle=False) as z: return {'metadata':json.loads(str(z['metadata_json']))}
''')
    (root/'candidates/base.json').write_text('{"id":"base","method":"source_frequency"}')
    data=tmp_path/'dataset.npz';metadata={'synthetic':True}
    np.savez_compressed(data,X=np.zeros((2,1)),metadata_json=np.asarray(json.dumps(metadata)))
    data.with_suffix('.json').write_text(json.dumps({'npz_sha256':C.file_sha(data),'metadata':metadata}))
    protocol={'primary_metric':'mean_log_loss','data_scope':'retrospective_landmark_forecast',
        'budget':{'max_trials':12,'max_campaign_seconds':30,'max_trial_seconds':3,'max_consecutive_failures':2,'end_date':'2099-01-01'},
        'planner':{'max_rounds':18,'candidate_dir':'candidates'},
        'allowed':{'methods':sorted(C.METHODS),'seeds':[0,1,2]},
        'model_parameters':{'epochs':8,'hidden':32,'batch_size':512,'learning_rate':.003,'hgb_max_iter':80}}
    proto=tmp_path/'protocol.json';proto.write_text(json.dumps(protocol))
    kw={'root':root,'lock_root':tmp_path/'locks'}
    C.init('synthetic',data,proto,**kw);C.freeze('synthetic',**kw)
    return dict(root=root,data=data,proto=proto,kw=kw,p=C.paths('synthetic',**kw))


def test_init_freeze_receipt_pin_and_same_candidate_alias(campaign):
    f=campaign;m=C.integrity(f['p'])
    assert m['data_receipt_sha']==C.file_sha(f['data'].with_suffix('.json'))
    one=C.enqueue('synthetic',{'id':'first','method':'source_frequency'},**f['kw'])
    two=C.enqueue('synthetic',{'id':'renamed','method':'source_frequency'},**f['kw'])
    assert two['duplicate'] and two['run_id']==one['run_id']
    with pytest.raises(C.ControlError,match='already frozen'):C.freeze('synthetic',**f['kw'])


@pytest.mark.parametrize('target',['data','receipt','source','protocol'])
def test_tamper_blocks_execution_before_worker(campaign,target):
    f=campaign;p={'data':f['data'],'receipt':f['data'].with_suffix('.json'),'source':f['root']/'models.py','protocol':f['proto']}[target]
    old=p.read_bytes();p.write_bytes(old+b'\n ')
    if target=='protocol':p.write_text('{}')
    with pytest.raises(C.ControlError):C.enqueue('synthetic',{},**f['kw'])
    assert not f['p']['runs'].exists()


def test_metadata_receipt_mismatch_fails_initialization(tmp_path):
    p=tmp_path/'bad.npz';np.savez_compressed(p,metadata_json=np.asarray('{"a":1}'))
    p.with_suffix('.json').write_text(json.dumps({'npz_sha256':C.file_sha(p),'metadata':{'a':2}}))
    with pytest.raises(C.ControlError,match='metadata'):C.verify_dataset_receipt(p)


def test_singleton_worker_lock(campaign):
    with C.exclusive_worker(campaign['p']):
        with pytest.raises(C.ControlError,match='another worker'):
            with C.exclusive_worker(campaign['p']):pass


def test_real_package_worker_completed_and_result_tamper_halts(campaign):
    f=campaign;j=C.enqueue('synthetic',{},**f['kw'])
    result=C.run('synthetic',**f['kw'])
    assert result['processed'][0]['status']=='completed'
    assert C.report('synthetic',**f['kw'])['runs'][0]['metrics']['mean_log_loss']==.5
    assert C.run('synthetic',**f['kw'])['processed']==[]
    (f['p']['runs']/j['run_id']/'worker_result.json').write_text('{"mean_log_loss":0}')
    assert C.report('synthetic',**f['kw'])['runs'][0]['status']=='invalid'
    with pytest.raises(C.ControlError,match='safety halt'):C.set_stop('synthetic',False,**f['kw'])


@pytest.mark.parametrize('method',['hgb_history','mlp_ind'])
def test_real_worker_verification_failure_is_permanent(campaign,method):
    f=campaign;C.enqueue('synthetic',{'method':method},**f['kw'])
    C.enqueue('synthetic',{'method':'persistence'},**f['kw'])
    result=C.run('synthetic',max_jobs=2,**f['kw'])
    assert len(result['processed'])==1 and result['processed'][0]['status']=='invalid'
    assert result['stopped'] and C.status('synthetic',**f['kw'])['counts']['queued']==1
    with pytest.raises(C.ControlError,match='safety halt'):C.run('synthetic',**f['kw'])


def test_failure_budget_stops_with_queued_work(campaign):
    f=campaign
    for method in ('source_markov','mlp_joint','persistence'):C.enqueue('synthetic',{'method':method},**f['kw'])
    result=C.run('synthetic',max_jobs=3,**f['kw'])
    assert len(result['processed'])==2 and all(x['status']=='failed' for x in result['processed'])
    assert result['stopped'] and 'failure budget' in result['reason']
    assert C.status('synthetic',**f['kw'])['counts']['queued']==1


def test_runtime_budget_stops_instead_of_repeated_no_progress(campaign):
    f=campaign;rid=C.enqueue('synthetic',{},**f['kw'])['run_id']
    C.enqueue('synthetic',{'method':'persistence'},**f['kw'])
    with C.database(f['p']) as c:c.execute("UPDATE jobs SET status='failed',elapsed=31,ended=? WHERE run_id=?",(time.time(),rid))
    result=C.run('synthetic',**f['kw'])
    assert not result['processed'] and result['stopped'] and 'runtime budget' in result['reason']


def test_closed_twelve_jobs_and_invalid_configs(campaign):
    f=campaign
    for method in sorted(C.METHODS):
        for seed in ([0] if method in {'source_frequency','source_markov','persistence'} else [0,1,2]):
            C.enqueue('synthetic',{'method':method,'seed':seed},**f['kw'])
    assert C.status('synthetic',**f['kw'])['counts']['queued']==12
    assert C.enqueue('synthetic',{'method':'source_frequency','id':'alias'},**f['kw'])['duplicate']
    for bad in ({'method':[]},{'command':'x'},{'id':'../x'},{'seed':True},{'method':'persistence','seed':1},{'hidden':64}):
        with pytest.raises(C.ControlError):C.enqueue('synthetic',bad,**f['kw'])
    with C.database(f['p']) as c:c.execute("UPDATE jobs SET status='completed',ended=?",(time.time(),))
    result=C.run('synthetic',**f['kw'])
    assert result['stopped'] and 'trial budget' in result['reason']
