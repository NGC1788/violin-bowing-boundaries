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
NAMES=['.gitignore','AGENTS.md','CONTRACT.json','model.py','data.py','control.py','job.py','runner.py','test_runner.py']


def contract():return json.loads((HERE/'CONTRACT.json').read_text())
def versions():return {**readiness.versions(),'sklearn':importlib.metadata.version('scikit-learn'),'threadpoolctl':importlib.metadata.version('threadpoolctl')}
def sources():
    from ..exact_count_runner_v1.runner import sources as parent_sources
    result=parent_sources()
    result.update({str((HERE/n).relative_to(REPO)):sha(HERE/n) for n in NAMES})
    return result


def metadata_paths():
    from ..exact_count_runner_v1.runner import metadata_paths as parent_metadata
    result=parent_metadata();parent=HERE.parent/'exact_count_runner_v1'
    for name in ['FREEZE.json','RESULTS.json','RESULTS_SEAL.json','OUTPUT_SEAL.json']:
        result['parent_'+name]=parent/name
    for name in ['AGENTS.md','METHOD.md','DECISION.md','PRIOR_REVIEW.md','NEXT_PREPARATION_GATE.md','REVIEW.json']:
        result['method_'+name]=HERE.parent/'method_redesign_v1'/name
    return result


def input_paths():
    from ..exact_count_runner_v1.runner import input_paths as parent_inputs
    result=parent_inputs()
    result['reference_predictions']=HERE.parent/'exact_count_runner_v1/local/all_predictions.npz'
    return result


def metadata_check():
    from ..exact_count_runner_v1.runner import metadata_check as parent_metadata_check
    expected,_=parent_metadata_check();parent=HERE.parent/'exact_count_runner_v1';paths=metadata_paths()
    old=json.loads((parent/'RESULTS.json').read_text())
    if sha(parent/'RESULTS.json')!=contract()['pinned_parent_results_sha256']:raise ValueError('parent results changed')
    if json.loads((parent/'RESULTS_SEAL.json').read_text())['results_sha256']!=sha(parent/'RESULTS.json'):raise ValueError('parent seal changed')
    if old['freeze_sha256']!=sha(parent/'FREEZE.json') or old['batch_output_seal_sha256']!=sha(parent/'OUTPUT_SEAL.json'):raise ValueError('parent chain changed')
    parent_freeze=json.loads((parent/'FREEZE.json').read_text())
    if any(sha(REPO/n)!=h for n,h in parent_freeze['sources'].items()):raise ValueError('older frozen source changed')
    method=HERE.parent/'method_redesign_v1';review=json.loads((method/'REVIEW.json').read_text())
    if review['verdict']!='JUSTIFIED_TO_PREPARE' or review['blockers']:raise ValueError('method unapproved')
    if any(sha(method/n)!=h for n,h in review['document_sha256'].items()):raise ValueError('method review changed')
    expected['reference_predictions']=old['all_predictions_sha256']
    if set(expected)!=set(input_paths()):raise ValueError('source input allowlist changed')
    return expected,{k:sha(v) for k,v in paths.items()}


def verify_baseline_files(data,paths,expected):
    from ..exact_count_runner_v1.runner import verify_baseline_files as parent_verify
    parent_verify(data,paths,expected)
    path=paths['reference_predictions']
    if path.is_symlink() or sha(path)!=expected['reference_predictions']:raise ValueError('reference predictions changed')
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={'truth','joint_aux','mdn_aux','exact_count','moment_count','markov','summary_hgb'}:raise ValueError('reference schema')
        # Do not inspect historical source truth or compute scores in freeze.
        for method in ['markov','summary_hgb','joint_aux','mdn_aux']:
            verify_predictions(z[method],np.arange(len(data['X'])),np.arange(len(data['X'])))
        for k in range(3):
            with np.load(paths[f'hgb_fold{k}'],allow_pickle=False) as old:
                if not np.array_equal(z['summary_hgb'][data['fold']==k],old['probability']):raise ValueError('HGB row correspondence')
    if sha(path)!=expected['reference_predictions']:raise ValueError('reference changed during load')


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
              'trace.json':r['trace_sha256'],'stats.json':r['preprocessing_sha256'],'memory.npz':r['memory_artifact_sha256']}
    for n,h in expected.items():
        if (directory/n).is_symlink() or sha(directory/n)!=h:raise ValueError('job output changed before parent pin')
    with np.load(directory/'memory.npz',allow_pickle=False) as memory:
        if set(memory.files)!={'prefix','labels','groups','source_rows'}:raise ValueError('memory artifact schema')
        donors=len(np.unique(memory['groups']))
        if len(np.intersect1d(memory['source_rows'],indices)):raise ValueError('held row in training memory')
    with np.load(directory/'predictions.npz',allow_pickle=False) as z:
        verify_retrieval_predictions(z,indices,donors)
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


def verify_retrieval_predictions(z,indices,donors):
    expected={'probability','counterfactual_probability','row_indices','donor_codes','group_mass','positive_support','negative_support','dro_logweights'}
    if set(z.files if hasattr(z,'files') else z)!=expected:raise ValueError('prediction schema')
    verify_predictions(z['probability'],z['row_indices'],indices)
    cf=z['counterfactual_probability'];mass=z['group_mass']
    if cf.shape!=(len(indices),donors+1,9) or not np.isfinite(cf).all() or (cf<.001-1e-12).any() or (cf>.999+1e-12).any():raise ValueError('counterfactual forecast shape/range')
    if not np.array_equal(cf[:,0],z['probability']) or not np.array_equal(z['donor_codes'],np.arange(-1,donors)):raise ValueError('counterfactual donor order')
    if mass.shape!=(len(indices),donors) or not np.isfinite(mass).all() or (mass<0).any() or not np.allclose(mass.sum(1),1,atol=1e-12,rtol=0):raise ValueError('memory mass')
    for name in ['positive_support','negative_support']:
        a=z[name]
        if a.shape!=cf.shape or not np.isfinite(a).all() or (a<0).any() or not np.equal(a,np.floor(a)).all():raise ValueError('support schema')
    if z['dro_logweights'].shape!=(donors,) or not np.isfinite(z['dro_logweights']).all():raise ValueError('DRO state')


def person_mean(values,people):
    return np.array([values[people==u].mean(axis=0) for u in np.unique(people)]).mean(axis=0)


def evaluate_complete(data,state):
    verify_batch_outputs(state)
    c=contract();methods=c['model_config']['methods'];references=['markov','summary_hgb','joint_aux','mdn_aux']
    parent=json.loads((HERE.parent/'exact_count_runner_v1/RESULTS.json').read_text())
    forecasts={m:np.full((len(data['X']),9),np.nan) for m in methods+references}
    truth=np.zeros((len(data['X']),9));fold_results={};private={};common={};diagnostics={};worst={m:np.zeros(len(truth)) for m in methods}
    with np.load(input_paths()['reference_predictions'],allow_pickle=False) as reference:
        for m in references:forecasts[m][:]=reference[m]
        old_truth=reference['truth']
    for k in range(3):
        train,stats,_=fit_inner(data,k);_,indices=prediction_inputs(data,k,stats)
        yh,_=labels_and_states(data['targets'][indices],data['guard'][indices],stats['thresholds']);truth[indices]=yh
        if not np.array_equal(yh,old_truth[indices]):raise ValueError('historical fold truth differs')
        receipts=[];fold_results[str(k)]={};diagnostics[str(k)]={}
        for method in methods:
            directory=HERE/'local'/f'job_{k}_{method}';r=json.loads((directory/'receipt.json').read_text());receipts.append(r)
            with np.load(directory/'predictions.npz',allow_pickle=False) as z:
                donors=c['expected_people']-c['fold_people'][k];verify_retrieval_predictions(z,indices,donors)
                forecasts[method][indices]=z['probability'];prob=z['counterfactual_probability']
                clipped=np.clip(prob,1e-6,1-1e-6)
                losses=-(yh[:,None]*np.log(clipped)+(1-yh[:,None])*np.log1p(-clipped)).mean(-1)
                worst[method][indices]=losses.max(1)
                mass=z['group_mass'];entropy=-(mass*np.log(np.maximum(mass,1e-300))).sum(1)
                diagnostics[str(k)][method]={'deletion_risk_person_macro':float(person_mean(losses.max(1),data['people'][indices])),
                    'mean_maximum_donor_mass':float(person_mean(mass.max(1),data['people'][indices])),
                    'mean_donor_entropy_nats':float(person_mean(entropy,data['people'][indices])),
                    'mean_effective_donor_count':float(person_mean(np.exp(entropy),data['people'][indices])),
                    'zero_positive_support_fraction_all_query_donor_event_cells':float((z['positive_support']==0).mean()),
                    'zero_negative_support_fraction_all_query_donor_event_cells':float((z['negative_support']==0).mean()),
                    'train_support':r['training_support_summary'],
                    'resource_costs':{name:r[name] for name in ['inference_memory_array_bytes','memory_artifact_bytes','weights_artifact_bytes','predictions_artifact_bytes','held_full_plus_all_counterfactual_inference_seconds']}}
        for key in ['preprocessing_sha256','initial_shared_sha256','minibatch_order_sha256','training_memory_sha256','memory_artifact_sha256']:
            if len({r[key] for r in receipts})!=1:raise ValueError('unmatched shared preparation/initialization/order/memory')
        for key in ['preprocessing_sha256','initial_shared_sha256','minibatch_order_sha256']:
            if receipts[0][key]!=parent['common_control_hashes'][str(k)][key]:raise ValueError('historical control correspondence')
        common[str(k)]={key:receipts[0][key] for key in ['preprocessing_sha256','initial_shared_sha256','minibatch_order_sha256','training_memory_sha256']}
        for method,prediction in forecasts.items():
            result,loss=score(yh,prediction[indices],data['people'][indices]);fold_results[str(k)][method]=result
            private[f'{k}:{method}:person_loss']=loss.tolist()
            if method in references and abs(result['log_loss']-parent['fold_results'][str(k)][method]['log_loss'])>1e-12:raise ValueError('reference fold score changed')
    aggregate={};promoted=[]
    for method,prediction in forecasts.items():
        aggregate[method],per=score(truth,prediction,data['people']);private[method+':person_loss']=per.tolist()
        p=np.clip(prediction,1e-6,1-1e-6);event_loss=-(truth*np.log(p)+(1-truth)*np.log1p(-p))
        aggregate[method]['eventwise_person_macro_loss']=person_mean(event_loss,data['people']).tolist()
        groups={'invalid':~data['guard'],'valid_all_zero':data['guard']&~truth.any(1),'valid_any_positive':data['guard']&truth.any(1)}
        aggregate[method]['outcome_groups']={}
        for name,mask in groups.items():
            if mask.any():
                detail,_=score(truth[mask],prediction[mask],data['people'][mask]);aggregate[method]['outcome_groups'][name]=detail
        if method in references and abs(aggregate[method]['log_loss']-parent['aggregate'][method]['log_loss'])>1e-12:raise ValueError('reference aggregate changed')
        if method in methods:
            aggregate[method]['deletion_risk_person_macro']=float(person_mean(worst[method],data['people']))
            gains={base:1-aggregate[method]['log_loss']/score(truth,forecasts[base],data['people'])[0]['log_loss'] for base in ['markov','summary_hgb']}
            better=sum(all(fold_results[str(k)][method]['log_loss']<fold_results[str(k)][base]['log_loss'] for base in gains) for k in range(3))
            aggregate[method].update(relative_gain_vs_baselines=gains,folds_better_than_both=better)
            if min(gains.values())>=.02 and better>=2:promoted.append(method)
    special='retrieval_adversarial_delete';matched={}
    for base in ['retrieval_erm','retrieval_random_delete','retrieval_query_group_dro']:
        matched[base]={'relative_gain':1-aggregate[special]['log_loss']/aggregate[base]['log_loss'],
                       'folds_better':sum(fold_results[str(k)][special]['log_loss']<fold_results[str(k)][base]['log_loss'] for k in range(3))}
    operator_pass=special in promoted and all(matched[b]['relative_gain']>0 and matched[b]['folds_better']>=2 for b in ['retrieval_erm','retrieval_random_delete'])
    dump(HERE/'local/private_scores.json',private);np.savez_compressed(HERE/'local/all_predictions.npz',truth=truth,**forecasts)
    verify_batch_outputs(state)
    return {'status':'COMPLETE_STOP_THIS_BATCH','scope':'historically exposed source exploratory, not independent confirmation',
        'aggregate':aggregate,'fold_results':fold_results,'diagnostics':diagnostics,'promoted_exploratory':promoted,
        'adversarial_operator_screen_pass':operator_pass,'adversarial_vs_matched_controls':matched,'common_control_hashes':common,
        'eventwise_positive_frequency_record_macro':truth.mean(0).tolist(),
        'eventwise_positive_frequency_person_macro':person_mean(truth,data['people']).tolist(),
        'neural_job_attempts':len(state['completed']),'new_reference_fits':0,'invalid_rows_retained':int((~data['guard']).sum()),
        'batch_output_seal_sha256':state['batch_output_seal_sha256'],'actual_worker_seconds':time.monotonic()-state['worker_monotonic_start'],
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
                        spawn=lambda:subprocess.Popen([sys.executable,'-m','agentic_research.donor_deletion_learning_v1.job',job_id],cwd=REPO,stdout=out,stderr=err)
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
