"""Reviewed source-only12job campaign. Freeze and run are separate actions."""
from pathlib import Path
import argparse,datetime,fcntl,importlib.metadata,json,os,platform,secrets,signal,subprocess,sys,time
import numpy as np
import torch
from .data import sha,load_arrays,verify_predictions,fit_inner,prediction_inputs,score
from .control import dump,register,job_limit,run_child,seal_private_output
from ..exact_count_learning_v1 import readiness
from ..violin_raw_v1.data import labels_and_states
from .model import ConditioningForecaster

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
GATE=HERE.parent/'path_source_gate_v1';OLD=HERE.parent/'discovery_v1'
LEDGER=HERE.parent/'discovery_v2/LOOP_BUDGET.json'
NAMES=['.gitignore','AGENTS.md','CONTRACT.json','model.py','data.py','control.py','job.py','runner.py','test_runner.py','test_model.py']


def contract():return json.loads((HERE/'CONTRACT.json').read_text())
def versions():return {**readiness.versions(),'sklearn':importlib.metadata.version('scikit-learn'),'threadpoolctl':importlib.metadata.version('threadpoolctl')}
def sources():
    from ..donor_deletion_learning_v1.runner import sources as parent_sources
    result=parent_sources()
    result.update({str((HERE/n).relative_to(REPO)):sha(HERE/n) for n in NAMES})
    return result


def metadata_paths():
    from ..donor_deletion_learning_v1.runner import metadata_paths as previous
    result=previous();donor=HERE.parent/'donor_deletion_learning_v1'
    for n in ['FREEZE.json','RESULTS.json','RESULTS_SEAL.json','OUTPUT_SEAL.json']:
        result['donor_'+n]=donor/n
    for n in ['AGENTS.md','METHOD.md','DECISION.md','PRIOR_REVIEW.md','NEXT_PREPARATION_GATE.md','REVIEW.json']:
        result['reframe_'+n]=HERE.parent/'research_reframe_v1'/n
    return result


def input_paths():
    from ..donor_deletion_learning_v1.runner import input_paths as previous
    result=previous()
    result['retrieval_predictions']=HERE.parent/'donor_deletion_learning_v1/local/all_predictions.npz'
    return result


def metadata_check():
    from ..donor_deletion_learning_v1.runner import metadata_check as previous
    expected,_=previous();donor=HERE.parent/'donor_deletion_learning_v1'
    old=json.loads((donor/'RESULTS.json').read_text())
    if sha(donor/'RESULTS.json')!=contract()['pinned_donor_results_sha256']:raise ValueError('donor results changed')
    if json.loads((donor/'RESULTS_SEAL.json').read_text())['results_sha256']!=sha(donor/'RESULTS.json'):raise ValueError('donor result seal')
    if old['freeze_sha256']!=sha(donor/'FREEZE.json') or old['batch_output_seal_sha256']!=sha(donor/'OUTPUT_SEAL.json'):raise ValueError('donor chain')
    frozen=json.loads((donor/'FREEZE.json').read_text())
    if any(sha(REPO/n)!=h for n,h in frozen['sources'].items()):raise ValueError('older source changed')
    method=HERE.parent/'research_reframe_v1';review=json.loads((method/'REVIEW.json').read_text())
    if review['verdict']!='JUSTIFIED_TO_PREPARE_EXPLORATORY' or review['blockers']:raise ValueError('unapproved reframe')
    if any(sha(method/n)!=h for n,h in review['document_sha256'].items()):raise ValueError('method review changed')
    expected['retrieval_predictions']=old['all_predictions_sha256']
    if set(expected)!=set(input_paths()):raise ValueError('source allowlist changed')
    return expected,{k:sha(v) for k,v in metadata_paths().items()}


def verify_baseline_files(data,paths,expected):
    from ..donor_deletion_learning_v1.runner import verify_baseline_files as previous
    previous(data,paths,expected)
    path=paths['retrieval_predictions']
    if path.is_symlink() or sha(path)!=expected['retrieval_predictions']:raise ValueError('retrieval reference changed')
    methods=['retrieval_erm','retrieval_random_delete','retrieval_adversarial_delete','retrieval_query_group_dro']
    refs=['markov','summary_hgb','joint_aux','mdn_aux']
    with np.load(path,allow_pickle=False) as z, np.load(paths['reference_predictions'],allow_pickle=False) as original:
        if set(z.files)!={'truth',*methods,*refs}:raise ValueError('retrieval reference schema')
        for m in methods:verify_predictions(z[m],np.arange(len(data['X'])),np.arange(len(data['X'])))
        for m in refs:
            if not np.array_equal(z[m],original[m]):raise ValueError('reference alignment changed')
    # Existing source truth is not read or scored in preparation.
    if sha(path)!=expected['retrieval_predictions']:raise ValueError('reference changed during read')


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
    fine_bindings={}
    parent=json.loads((HERE.parent/'exact_count_runner_v1/RESULTS.json').read_text())
    for fold in range(3):
        train,stats,fine=fit_inner(data,fold)
        serialized={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in stats.items()}
        digest=__import__('hashlib').sha256((json.dumps(serialized,indent=2,allow_nan=False)+'\n').encode()).hexdigest()
        if digest!=parent['common_control_hashes'][str(fold)]['preprocessing_sha256']:raise ValueError('preprocessing reference mismatch')
        fine_bindings[str(fold)]={'train_rows':len(train['prefix']),'fine_coarse_consistency':True,
            'fine_statistics_sha256':__import__('hashlib').sha256(json.dumps(fine,sort_keys=True).encode()).hexdigest(),
            'common_preprocessing_sha256':digest}
    summary={'rows':len(data['X']),'people':len(np.unique(data['people'])),'invalid_retained':int((~data['guard']).sum()),
             'fold_rows':[int((data['fold']==k).sum()) for k in range(3)],'fine_training_bindings':fine_bindings,
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


def pin_job_output(directory,job,indices,epochs):
    r=json.loads((directory/'receipt.json').read_text())
    if r['job']!=job['id'] or r['method']!=job['method'] or r['fold']!=job['fold'] or r['epochs_completed']!=epochs:
        raise ValueError('job receipt identity/epoch mismatch')
    expected={'predictions.npz':r['predictions_sha256'],'weights.npz':r['weights_sha256'],
              'trace.json':r['trace_sha256'],'stats.json':r['preprocessing_sha256']}
    for n,h in expected.items():
        if (directory/n).is_symlink() or sha(directory/n)!=h:raise ValueError('job output changed before pin')
    with np.load(directory/'predictions.npz',allow_pickle=False) as z:
        if set(z.files)!={'probability','row_indices'}:raise ValueError('forecast schema')
        verify_predictions(z['probability'],z['row_indices'],indices)
    trace=json.loads((directory/'trace.json').read_text())
    c=contract();steps=int(np.ceil((c['expected_rows']-len(indices))/c['model_config']['batch_size']))
    if (len(trace)!=epochs or r['fine_steps_completed']!=epochs*steps or
        any(e['epoch']!=i+1 or e['fine_steps']!=steps for i,e in enumerate(trace))):raise ValueError('training steps incomplete')
    if r['parameters']!=7583 or r['inference_memory_array_bytes']!=0:raise ValueError('architecture/memory mismatch')
    return {'receipt.json':sha(directory/'receipt.json'),**expected}


def person_mean(values,people):
    return np.array([values[people==u].mean(axis=0) for u in np.unique(people)]).mean(axis=0)


def evaluate_complete(data,state):
    verify_batch_outputs(state)
    c=contract();methods=c['model_config']['methods']
    refs=['markov','summary_hgb','joint_aux','mdn_aux']
    retrieval=['retrieval_erm','retrieval_random_delete','retrieval_adversarial_delete','retrieval_query_group_dro']
    parent=json.loads((HERE.parent/'exact_count_runner_v1/RESULTS.json').read_text())
    donor=json.loads((HERE.parent/'donor_deletion_learning_v1/RESULTS.json').read_text())
    forecasts={m:np.full((len(data['X']),9),np.nan) for m in methods+refs+retrieval}
    with np.load(input_paths()['reference_predictions'],allow_pickle=False) as z:
        for m in refs:forecasts[m][:]=z[m]
        old_truth=z['truth']
    with np.load(input_paths()['retrieval_predictions'],allow_pickle=False) as z:
        for m in retrieval:forecasts[m][:]=z[m]
        if not np.array_equal(old_truth,z['truth']):raise ValueError('historical truth alignment')
    truth=np.zeros_like(old_truth);fold_results={};private={};common={};diagnostics={}
    for fold in range(3):
        train,stats,_=fit_inner(data,fold);_,indices=prediction_inputs(data,fold,stats)
        yh,_=labels_and_states(data['targets'][indices],data['guard'][indices],stats['thresholds'])
        if not np.array_equal(yh,old_truth[indices]):raise ValueError('historical target changed')
        truth[indices]=yh;fold_results[str(fold)]={};diagnostics[str(fold)]={};receipts=[]
        for method in methods:
            d=HERE/'local'/f'job_{fold}_{method}';r=json.loads((d/'receipt.json').read_text());receipts.append(r)
            with np.load(d/'predictions.npz',allow_pickle=False) as z:
                forecasts[method][indices]=verify_predictions(z['probability'],z['row_indices'],indices)
            trace=json.loads((d/'trace.json').read_text())
            diagnostics[str(fold)][method]={
                'maximum_fine_softmax_drift':max(e['fine_step_diagnostics']['softmax_drift']['max'] for e in trace),
                'maximum_fine_query_drift':max(e['fine_step_diagnostics']['query_drift']['max'] for e in trace),
                'maximum_adapter_condition':max(e['fine_step_diagnostics']['adapter_condition']['max'] for e in trace),
                'maximum_adapter_norm':max(e['fine_step_diagnostics']['adapter_norm']['max'] for e in trace),
                'maximum_projector_relative_residual':max(e['fine_step_diagnostics']['projector_relative_residual']['max'] for e in trace),
                'mean_applied_fine_step_norm':float(np.mean([e['fine_step_diagnostics']['applied_norm']['mean'] for e in trace])),
                'resource_costs':{k:r[k] for k in ['parameters','main_step_seconds','fine_step_seconds','held_inference_seconds','inference_memory_array_bytes','weights_artifact_bytes','predictions_artifact_bytes']}}
        keys=['preprocessing_sha256','initial_shared_sha256','initial_model_sha256','minibatch_order_sha256']
        for key in keys:
            if len({r[key] for r in receipts})!=1:raise ValueError('unmatched init/preprocessing/minibatches')
        for key in ['preprocessing_sha256','initial_shared_sha256','minibatch_order_sha256']:
            if receipts[0][key]!=parent['common_control_hashes'][str(fold)][key]:raise ValueError('historical correspondence mismatch')
        common[str(fold)]={k:receipts[0][k] for k in keys}
        for method,p in forecasts.items():
            result,person_losses=score(yh,p[indices],data['people'][indices]);fold_results[str(fold)][method]=result
            private[f'{fold}:{method}:person_loss']=person_losses.tolist()
            reference=parent if method in refs else donor if method in retrieval else None
            if reference and abs(result['log_loss']-reference['fold_results'][str(fold)][method]['log_loss'])>1e-12:
                raise ValueError('reference fold score mismatch')
    aggregate={};promoted=[]
    for method,p in forecasts.items():
        aggregate[method],person_losses=score(truth,p,data['people']);private[method+':person_loss']=person_losses.tolist()
        clipped=np.clip(p,1e-6,1-1e-6);event_loss=-(truth*np.log(clipped)+(1-truth)*np.log1p(-clipped))
        aggregate[method]['eventwise_person_macro_loss']=person_mean(event_loss,data['people']).tolist()
        groups={'invalid':~data['guard'],'valid_all_zero':data['guard']&~truth.any(1),'valid_any_positive':data['guard']&truth.any(1)}
        aggregate[method]['outcome_groups']={}
        for name,mask in groups.items():
            if mask.any():aggregate[method]['outcome_groups'][name]=score(truth[mask],p[mask],data['people'][mask])[0]
        reference=parent if method in refs else donor if method in retrieval else None
        if reference and abs(aggregate[method]['log_loss']-reference['aggregate'][method]['log_loss'])>1e-12:
            raise ValueError('reference overall score mismatch')
    for method in methods:
        gains={b:1-aggregate[method]['log_loss']/aggregate[b]['log_loss'] for b in ['markov','summary_hgb']}
        folds=sum(all(fold_results[str(k)][method]['log_loss']<fold_results[str(k)][b]['log_loss'] for b in gains) for k in range(3))
        aggregate[method].update(relative_gain_vs_baselines=gains,folds_better_than_both=folds)
        if min(gains.values())>=.02 and folds>=2:promoted.append(method)
    special='null_conditioned';matched={}
    for base in ['main_only','ordinary_matched','random_matched']:
        matched[base]={'relative_gain':1-aggregate[special]['log_loss']/aggregate[base]['log_loss'],
            'folds_better':sum(fold_results[str(k)][special]['log_loss']<fold_results[str(k)][base]['log_loss'] for k in range(3))}
    passed=special in promoted and all(x['relative_gain']>0 and x['folds_better']>=2 for x in matched.values())
    dump(HERE/'local/private_scores.json',private);np.savez_compressed(HERE/'local/all_predictions.npz',truth=truth,**forecasts)
    verify_batch_outputs(state)
    return {'status':'COMPLETE_STOP_THIS_BATCH','scope':'historically exposed source exploratory; no independent confirmation',
        'aggregate':aggregate,'fold_results':fold_results,'diagnostics':diagnostics,'promoted_exploratory':promoted,
        'null_operator_screen_pass':passed,'null_vs_matched_controls':matched,'common_control_hashes':common,
        'neural_job_attempts':len(state['completed']),'new_reference_fits':0,'invalid_rows_retained':int((~data['guard']).sum()),
        'eventwise_positive_frequency_person_macro':person_mean(truth,data['people']).tolist(),
        'batch_output_seal_sha256':state['batch_output_seal_sha256'],'actual_worker_seconds':time.monotonic()-state['worker_monotonic_start'],
        'novel_method_established':False,'independent_forecast_improvement_established':False,
        'private_scores_sha256':sha(HERE/'local/private_scores.json'),'all_predictions_sha256':sha(HERE/'local/all_predictions.npz')}


def execution_review():
    review=json.loads((HERE/'FREEZE_REVIEW.json').read_text())
    if (review['verdict']!='APPROVE_FIXED_EXECUTION' or review['blockers']
            or review['freeze_sha256']!=sha(HERE/'FREEZE.json')):
        raise ValueError('full freeze has not passed independent execution review')
    return review


def run():
    execution_review()
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
                execution_review();check();elapsed=time.monotonic()-started;limit=job_limit(c,elapsed,datetime.datetime.now(datetime.timezone.utc))
                job_id=f"job_{j['fold']}_{j['method']}";directory=local/job_id;directory.mkdir()
                active={'id':job_id,'fold':j['fold'],'method':j['method'],'limit_seconds':limit,'nonce':secrets.token_hex(16)}
                register(LEDGER,'JOB_START',job=active);state['active_job']=active;dump(local/'state.json',state)
                job_started=time.monotonic()
                dump(seal_private_output(directory,'request.json'),{**active,'freeze_sha256':sha(HERE/'FREEZE.json'),
                     'parent_pid':os.getpid(),'absolute_monotonic_deadline':job_started+limit,'model_config':c['model_config']})
                try:
                    with seal_private_output(directory,'stdout.log').open('w') as out,seal_private_output(directory,'stderr.log').open('w') as err:
                        spawn=lambda:subprocess.Popen([sys.executable,'-m','agentic_research.null_conditioning_learning_v1.job',job_id],cwd=REPO,stdout=out,stderr=err)
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
