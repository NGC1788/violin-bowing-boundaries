"""Reviewed source-only12job campaign. Freeze and run are separate actions."""
from pathlib import Path
import argparse,datetime,fcntl,importlib.metadata,json,os,platform,secrets,signal,subprocess,sys,time
import numpy as np
import torch
from .data import sha,load_arrays,verify_predictions,fit_inner,prediction_inputs,markov_prediction,score
from .control import dump,register,job_limit,run_child,seal_private_output
from ..exact_count_learning_v1 import readiness
from ..violin_raw_v1.data import labels_and_states

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
GATE=HERE.parent/'path_source_gate_v1';OLD=HERE.parent/'discovery_v1'
LEDGER=HERE.parent/'discovery_v2/LOOP_BUDGET.json'
NAMES=['AGENTS.md','CONTRACT.json','data.py','control.py','job.py','runner.py','test_runner.py']


def contract():return json.loads((HERE/'CONTRACT.json').read_text())
def versions():return {**readiness.versions(),'sklearn':importlib.metadata.version('scikit-learn'),'threadpoolctl':importlib.metadata.version('threadpoolctl')}
def sources():
    result={str((HERE/n).relative_to(REPO)):sha(HERE/n) for n in NAMES}
    result.update(readiness.sources())
    for n in ['agentic_research/violin_raw_v1/models.py','agentic_research/violin_raw_v1/metrics.py']:
        result[n]=sha(REPO/n)
    return result


def metadata_paths():
    return {'gate_results':GATE/'RESULTS.json','gate_seal':GATE/'RESULTS_SEAL.json','gate_freeze':GATE/'FREEZE.json',
            'old_results':OLD/'RESULTS.json','old_seal':OLD/'RESULTS_SEAL.json','old_freeze':OLD/'FREEZE.json',
            'learning_readiness':HERE.parent/'exact_count_learning_v1/READINESS_FREEZE.json'}


def input_paths():
    d={n:GATE/'local'/n for n in ['prefix_inputs.npz','source_metadata.npz','future_labels.npz']}
    d.update({f'hgb_fold{k}':OLD/'local'/f'summary_hgb_fold{k}.npz' for k in range(3)})
    return d


def metadata_check():
    """Code/public-text provenance only. Safe before actual-input approval."""
    readiness.check();c=contract();paths=metadata_paths()
    g=json.loads(paths['gate_results'].read_text());old=json.loads(paths['old_results'].read_text())
    if sha(paths['gate_results'])!=c['pinned_path_results_sha256'] or sha(paths['old_results'])!=c['pinned_discovery_results_sha256']:
        raise ValueError('canonical prepared/baseline record changed')
    if json.loads(paths['gate_seal'].read_text())['results_sha256']!=sha(paths['gate_results']):raise ValueError('path result seal changed')
    if json.loads(paths['old_seal'].read_text())['results_sha256']!=sha(paths['old_results']):raise ValueError('baseline result seal changed')
    if sha(paths['gate_freeze'])!=g['freeze_sha256'] or sha(paths['old_freeze'])!=old['freeze_sha256']:raise ValueError('parent freeze link changed')
    gf=json.loads(paths['gate_freeze'].read_text())
    if gf['inputs']['source_result']!=sha(paths['old_results']) or gf['inputs']['source_prepared']!=old['prepared_sha256']:
        raise ValueError('parent row-provenance chain broken')
    expected=dict(g['prepared_files'])
    for k in range(3):
        jobs=[j for j in old['jobs'] if j['method']=='summary_hgb' and j['fold']==k]
        if len(jobs)!=1:raise ValueError('missing sealed source HGB reference')
        expected[f'hgb_fold{k}']=jobs[0]['predictions_sha256']
    if set(expected)!=set(input_paths()):raise ValueError('input allowlist changed')
    return expected,{k:sha(p) for k,p in paths.items()}


def verify_baseline_files(data,paths,expected):
    for k in range(3):
        key=f'hgb_fold{k}';p=paths[key]
        if p.is_symlink() or sha(p)!=expected[key]:raise ValueError('HGB reference changed')
        with np.load(p,allow_pickle=False) as z:
            if set(z.files)!={'probability','row_indices'}:raise ValueError('reference schema')
            verify_predictions(z['probability'],z['row_indices'],np.flatnonzero(data['fold']==k))
        if sha(p)!=expected[key]:raise ValueError('reference changed during read')


def freeze():
    if (HERE/'FREEZE.json').exists() or (HERE/'local/state.json').exists():raise ValueError('freeze/start exists')
    r=json.loads((HERE/'REVIEW.json').read_text());u=json.loads((HERE/'UNIT_TESTS.json').read_text())
    if r['verdict']!='APPROVE_TO_FREEZE_SOURCE_INPUTS' or r['blockers'] or r['source_sha256']!=sources():raise ValueError('unapproved runner source')
    if u['failed'] or u['source_sha256']!=sources():raise ValueError('source tests changed')
    c=contract();d=json.loads(LEDGER.read_text())
    if len(d['additional_batches_started'])>=d['max_additional_training_batches']:raise ValueError('training budget exhausted')
    if (c['max_jobs']>d['max_jobs_per_additional_batch'] or c['max_job_seconds']>d['max_seconds_per_training_job']
            or c['max_worker_seconds']>d['max_training_seconds_per_batch'] or c['deadline_kst']!=d['deadline_kst']):raise ValueError('contract exceeds shared limits')
    if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat(c['deadline_kst']):raise TimeoutError('deadline')
    expected,metadata=metadata_check();paths=input_paths()
    # Actual source NPZ access is confined to AFTER the separate code approval.
    data=load_arrays(paths,expected,c);verify_baseline_files(data,paths,expected)
    if any(sha(paths[k])!=h for k,h in expected.items()):raise ValueError('input changed during freeze')
    summary={'rows':len(data['X']),'people':len(np.unique(data['people'])),'invalid_retained':int((~data['guard']).sum()),
             'fold_rows':[int((data['fold']==k).sum()) for k in range(3)],
             'source_ids_order_sha256':__import__('hashlib').sha256(json.dumps(data['source_ids'].tolist()).encode()).hexdigest(),
             'guard_and_continuous_targets_reconstructed':True,'actual_fitting_jobs':0,'old_dev_test_truth_accessed_or_hashed':False}
    dump(HERE/'INPUT_RECEIPT.json',summary)
    value={'status':'FROZEN_READY_FOR_BOUNDED_TRAINING','execution_authorized':True,'sources':sources(),'versions':versions(),
           'inputs':expected,'metadata':metadata,'review_sha256':sha(HERE/'REVIEW.json'),'tests_sha256':sha(HERE/'UNIT_TESTS.json'),
           'input_receipt_sha256':sha(HERE/'INPUT_RECEIPT.json'),'limits':{k:d[k] for k in ['deadline_kst','max_additional_training_batches','max_jobs_per_additional_batch','max_seconds_per_training_job','max_training_seconds_per_batch']}}
    with (HERE/'FREEZE.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps({'status':value['status'],'inputs':len(expected),'actual_fitting_jobs':0,'freeze_sha256':sha(HERE/'FREEZE.json')}))


def check():
    f=json.loads((HERE/'FREEZE.json').read_text());expected,meta=metadata_check()
    if (not f['execution_authorized'] or f['sources']!=sources() or f['versions']!=versions() or f['inputs']!=expected
       or f['metadata']!=meta or f['review_sha256']!=sha(HERE/'REVIEW.json') or f['tests_sha256']!=sha(HERE/'UNIT_TESTS.json')
       or f['input_receipt_sha256']!=sha(HERE/'INPUT_RECEIPT.json')):raise ValueError('execution freeze changed')
    d=json.loads(LEDGER.read_text())
    if any(d[k]!=v for k,v in f['limits'].items()):raise ValueError('shared limits changed')
    for k,h in expected.items():
        p=input_paths()[k]
        if p.is_symlink() or sha(p)!=h:raise ValueError('frozen input changed: '+k)
    return f


def load_frozen(f):return load_arrays(input_paths(),f['inputs'],contract())


def pin_job_output(directory,job,indices,epochs):
    r=json.loads((directory/'receipt.json').read_text())
    if r['job']!=job['id'] or r['method']!=job['method'] or r['fold']!=job['fold'] or r['epochs_completed']!=epochs:
        raise ValueError('job receipt identity/epoch mismatch')
    expected={'predictions.npz':r['predictions_sha256'],'weights.npz':r['weights_sha256'],
              'trace.json':r['trace_sha256'],'stats.json':r['preprocessing_sha256']}
    for n,h in expected.items():
        if (directory/n).is_symlink() or sha(directory/n)!=h:raise ValueError('job output changed before parent pin')
    with np.load(directory/'predictions.npz',allow_pickle=False) as z:
        if set(z.files)!={'probability','row_indices'}:raise ValueError('job prediction schema')
        verify_predictions(z['probability'],z['row_indices'],indices)
    return {'receipt.json':sha(directory/'receipt.json'),**expected}


def seal_batch_outputs(state):
    expected=[f"job_{j['fold']}_{j['method']}" for j in contract()['job_order']]
    if state['completed']!=expected or set(state['outputs'])!=set(expected):raise ValueError('missing ordered output pins')
    value={'freeze_sha256':sha(HERE/'FREEZE.json'),'jobs':state['outputs']}
    with (HERE/'OUTPUT_SEAL.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    state['batch_output_seal_sha256']=sha(HERE/'OUTPUT_SEAL.json')
    dump(HERE/'local/state.json',state)
    verify_batch_outputs(state)


def verify_batch_outputs(state):
    if 'batch_output_seal_sha256' not in state or sha(HERE/'OUTPUT_SEAL.json')!=state['batch_output_seal_sha256']:
        raise ValueError('missing/changed parent batch output seal')
    value=json.loads((HERE/'OUTPUT_SEAL.json').read_text())
    expected={f"job_{j['fold']}_{j['method']}" for j in contract()['job_order']}
    if (value['freeze_sha256']!=sha(HERE/'FREEZE.json') or value['jobs']!=state['outputs'] or set(value['jobs'])!=expected):
        raise ValueError('batch output identity changed')
    for name,outputs in value['jobs'].items():
        for filename,h in outputs.items():
            p=HERE/'local'/name/filename
            if p.is_symlink() or sha(p)!=h:raise ValueError('parent-pinned job artifact changed')


def evaluate_complete(data,state):
    verify_batch_outputs(state)  # Before any held event labels are formed.
    c=contract();methods=c['model_config']['methods'];old=json.loads((OLD/'RESULTS.json').read_text())
    forecasts={m:np.full((len(data['X']),9),np.nan) for m in methods+['markov','summary_hgb']}
    truth=np.zeros((len(data['X']),9));fold_results={};private={};common={}
    for k in range(3):
        train,stats,prior=fit_inner(data,k,fit_markov=True);_,indices=prediction_inputs(data,k,stats)
        yh,_=labels_and_states(data['targets'][indices],data['guard'][indices],stats['thresholds']);truth[indices]=yh
        receipts=[]
        for method in methods:
            directory=HERE/'local'/f'job_{k}_{method}';r=json.loads((directory/'receipt.json').read_text())
            for key,name in [('predictions_sha256','predictions.npz'),('weights_sha256','weights.npz'),('trace_sha256','trace.json')]:
                if sha(directory/name)!=r[key]:raise ValueError('job output seal changed')
            with np.load(directory/'predictions.npz',allow_pickle=False) as z:
                forecasts[method][indices]=verify_predictions(z['probability'],z['row_indices'],indices)
            receipts.append(r)
        for key in ['preprocessing_sha256','initial_shared_sha256','minibatch_order_sha256']:
            if len({r[key] for r in receipts})!=1:raise ValueError('unmatched control preparation/initialization/order')
        common[str(k)]={key:receipts[0][key] for key in ['preprocessing_sha256','initial_shared_sha256','minibatch_order_sha256']}
        forecasts['markov'][indices]=markov_prediction(data,k,stats,prior)
        with np.load(input_paths()[f'hgb_fold{k}'],allow_pickle=False) as z:
            forecasts['summary_hgb'][indices]=verify_predictions(z['probability'],z['row_indices'],indices)
        fold_results[str(k)]={}
        for method,p in forecasts.items():
            result,loss=score(yh,p[indices],data['people'][indices]);fold_results[str(k)][method]=result
            private[f'{k}:{method}:person_loss']=loss.tolist()
        for method in ['markov','summary_hgb']:
            if abs(fold_results[str(k)][method]['log_loss']-old['fold_results'][str(k)][method]['log_loss'])>c['baseline_policy']['expected_score_tolerance']:
                raise ValueError('historical source baseline score not reproduced')
    aggregate={}
    for method,p in forecasts.items():
        aggregate[method],per=score(truth,p,data['people']);private[method+':person_loss']=per.tolist()
        if method in ['markov','summary_hgb'] and abs(aggregate[method]['log_loss']-old['aggregate'][method]['log_loss'])>1e-12:
            raise ValueError('historical aggregate baseline mismatch')
    promoted=[]
    for method in methods:
        gains={b:1-aggregate[method]['log_loss']/aggregate[b]['log_loss'] for b in ['markov','summary_hgb']}
        better=sum(all(fold_results[str(k)][method]['log_loss']<fold_results[str(k)][b]['log_loss'] for b in gains) for k in range(3))
        aggregate[method].update(relative_gain_vs_baselines=gains,folds_better_than_both=better)
        if min(gains.values())>=.02 and better>=2:promoted.append(method)
    dump(HERE/'local/private_scores.json',private)
    np.savez_compressed(HERE/'local/all_predictions.npz',truth=truth,**forecasts)
    verify_batch_outputs(state)
    return {'status':'COMPLETE_STOP_THIS_BATCH','scope':'historically exposed source-person exploratory comparison; not independent confirmation',
            'aggregate':aggregate,'fold_results':fold_results,'promoted_exploratory':promoted,'common_control_hashes':common,
            'neural_job_attempts':len(state['completed']),'analytic_markov_reference_fits':3,'new_hgb_fits':0,
            'batch_output_seal_sha256':state['batch_output_seal_sha256'],
            'invalid_rows_retained':int((~data['guard']).sum()),'actual_worker_seconds':time.monotonic()-state['worker_monotonic_start'],
            'novel_method_established':False,'independent_forecast_improvement_established':False,
            'private_scores_sha256':sha(HERE/'local/private_scores.json'),'all_predictions_sha256':sha(HERE/'local/all_predictions.npz')}


def run():
    f=check();c=contract();local=HERE/'local';local.mkdir(exist_ok=True)
    with (local/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (local/'state.json').exists() or (HERE/'RESULTS.json').exists():raise ValueError('batch already attempted; no rerun')
        if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat(c['deadline_kst']):raise TimeoutError('deadline')
        register(LEDGER,'START',freeze_sha=sha(HERE/'FREEZE.json'));started=time.monotonic()
        state={'status':'RUNNING','worker_monotonic_start':started,'completed':[],'active_job':None,'outputs':{}}
        dump(local/'state.json',state)
        previous_handler=signal.getsignal(signal.SIGALRM)
        def whole_worker_alarm(_signum,_frame):raise TimeoutError('whole-worker wall watchdog reached')
        signal.signal(signal.SIGALRM,whole_worker_alarm)
        total_limit=min(c['max_worker_seconds'],(datetime.datetime.fromisoformat(c['deadline_kst'])-datetime.datetime.now(datetime.timezone.utc)).total_seconds())
        signal.setitimer(signal.ITIMER_REAL,max(.001,total_limit))
        try:
            data=load_frozen(f)
            for j in c['job_order']:
                check();elapsed=time.monotonic()-started;limit=job_limit(c,elapsed,datetime.datetime.now(datetime.timezone.utc))
                job_id=f"job_{j['fold']}_{j['method']}";directory=local/job_id;directory.mkdir()
                active={'id':job_id,'fold':j['fold'],'method':j['method'],'limit_seconds':limit,'nonce':secrets.token_hex(16)}
                register(LEDGER,'JOB_START',job=active);state['active_job']=active;dump(local/'state.json',state)
                job_started=time.monotonic()
                dump(seal_private_output(directory,'request.json'),{**active,'freeze_sha256':sha(HERE/'FREEZE.json'),
                     'parent_pid':os.getpid(),'absolute_monotonic_deadline':job_started+limit,'model_config':c['model_config']})
                try:
                    with seal_private_output(directory,'stdout.log').open('w') as out,seal_private_output(directory,'stderr.log').open('w') as err:
                        spawn=lambda:subprocess.Popen([sys.executable,'-m','agentic_research.exact_count_runner_v1.job',job_id],cwd=REPO,stdout=out,stderr=err)
                        run_child(spawn,limit-(time.monotonic()-job_started))
                    seconds=time.monotonic()-job_started
                    if seconds>limit or time.monotonic()-started>c['max_worker_seconds']:raise TimeoutError('job/worker limit after child')
                    outputs=pin_job_output(directory,active,np.flatnonzero(data['fold']==j['fold']),c['model_config']['epochs'])
                    seconds=time.monotonic()-job_started
                    if seconds>limit or time.monotonic()-started>c['max_worker_seconds']:raise TimeoutError('job/worker limit while pinning output')
                    register(LEDGER,'JOB_END',job={'id':job_id,'status':'COMPLETE'},seconds=seconds)
                    state['outputs'][job_id]=outputs;state['completed'].append(job_id);state['active_job']=None;dump(local/'state.json',state)
                    print(json.dumps({'completed_jobs':len(state['completed']),'job':job_id,'seconds':round(seconds,3)}),flush=True)
                except Exception:
                    register(LEDGER,'JOB_END',job={'id':job_id,'status':'FAILED_STOP'},seconds=time.monotonic()-job_started);raise
            if len(state['completed'])!=12:raise ValueError('incomplete batch')
            seal_batch_outputs(state)
            # Only now form held event labels and evaluate all methods/references.
            result=evaluate_complete(data,state);check()
            job_limit(c,time.monotonic()-started,datetime.datetime.now(datetime.timezone.utc))
            result['freeze_sha256']=sha(HERE/'FREEZE.json');dump(HERE/'RESULTS.json',result)
            dump(HERE/'RESULTS_SEAL.json',{'results_sha256':sha(HERE/'RESULTS.json')})
            state.update(status='COMPLETE_STOP_THIS_BATCH',stopped=True);dump(local/'state.json',state)
            register(LEDGER,state['status'],seconds=time.monotonic()-started)
        except Exception as error:
            state.update(status='FAILED_STOP',stopped=True,error_type=type(error).__name__,reason=str(error));dump(local/'state.json',state)
            register(LEDGER,'FAILED_STOP',seconds=time.monotonic()-started)
            dump(HERE/'FAILURE_SUMMARY.json',{'status':'FAILED_STOP','completed_jobs':len(state['completed']),'error_type':type(error).__name__,
                                             'worker_seconds':time.monotonic()-started,'promotion_allowed':False,'retry_allowed':False})
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous_handler)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','check','run']);a=p.parse_args()
    if a.action=='freeze':freeze()
    elif a.action=='check':check();print('frozen runner intact')
    else:run()
