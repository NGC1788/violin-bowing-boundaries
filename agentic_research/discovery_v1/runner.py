"""Reviewed, finite source-only exploratory batch; old campaigns stay closed."""
from pathlib import Path
import datetime,fcntl,json,time,argparse
import numpy as np
from .data import HERE,OLD,sha,verify_old_sources,fit_fold,shuffle_tokens
from .models import predict
from ..violin_raw_v1.models import patterns
from ..violin_raw_v1.data import labels_and_states
from ..violin_raw_v1.metrics import row_scores,coherence_rate


def sources():
    return {n:sha(HERE/n) for n in ['AGENTS.md','PLAN.json','data.py','models.py','runner.py','test_discovery.py']}


def frozen_inputs():
    value={n:sha(OLD/'local/prepared'/n) for n in ['features.npz','train_labels.npz']}
    value.update({'meta:'+n:sha(OLD/n) for n in ['data_freeze_manifest.json','execution_freeze_manifest.json','data_partition_receipt.json','RESULTS.json']})
    return value


def freeze():
    if (HERE/'FREEZE.json').exists():raise ValueError('discovery freeze exists')
    verify_old_sources();review=json.loads((HERE/'REVIEW.json').read_text())
    if review['verdict']!='APPROVE' or review['source_sha256']!=sources():raise ValueError('unreviewed source')
    f={'sources':sources(),'source_inputs':frozen_inputs(),'review_sha256':sha(HERE/'REVIEW.json'),
       'scope':'one12job exploratory source-person batch; no olddev/test truth access'}
    (HERE/'FREEZE.json').write_text(json.dumps(f,indent=2)+'\n')
    print(json.dumps({'freeze_sha256':sha(HERE/'FREEZE.json')}))


def check():
    f=json.loads((HERE/'FREEZE.json').read_text())
    if f['sources']!=sources() or f['source_inputs']!=frozen_inputs() or f['review_sha256']!=sha(HERE/'REVIEW.json'):
        raise ValueError('frozen discovery contract changed')
    verify_old_sources();return f


def summarize(y,p,people):
    loss,brier=row_scores(y,p);ids=np.unique(people)
    per=np.asarray([loss[people==pid].mean() for pid in ids]);bs=np.asarray([brier[people==pid].mean() for pid in ids])
    return {'log_loss':float(per.mean()),'brier':float(bs.mean()),'maximum_person_loss':float(per.max()),
            'coherence_rate':coherence_rate(p),'people':len(ids)},per


def job_budget(plan,state,now):
    remaining=(datetime.datetime.fromisoformat(plan['budget']['end_utc'])-now).total_seconds()
    limit=min(plan['budget']['single_job_seconds'],plan['budget']['total_training_seconds']-state['seconds'],remaining)
    if limit<=0 or state['attempts']>=plan['budget']['training_jobs']:raise TimeoutError('discovery budget/deadline')
    return limit


def timed_fit(call,state,persist,clock=time.monotonic):
    started=clock()
    try:
        output=call(started)
    finally:
        elapsed=clock()-started;state['seconds']+=elapsed
        state['active_job']['elapsed_seconds']=elapsed;persist()
    return output,elapsed


def run():
    check();plan=json.loads((HERE/'PLAN.json').read_text());local=HERE/'local';local.mkdir(exist_ok=True)
    with (local/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (local/'state.json').exists():raise ValueError('batch exists; preserve state and do not rerun')
        prepared=json.loads((HERE/'PREPARED.json').read_text())
        receipt_sha=sha(HERE/'PREPARED.json')
        if (prepared['prepared_sha256']!=sha(local/'prepared.npz') or prepared['freeze_sha256']!=sha(HERE/'FREEZE.json') or
                prepared['source_inputs']!=frozen_inputs() or prepared['source_hashes']!=sources()):
            raise ValueError('prepared data/provenance changed')
        with np.load(local/'prepared.npz',allow_pickle=False) as z:
            data={n:z[n] for n in z.files}
        x,seq,t,g,ids,fold=(data[n] for n in ['X','sequence','targets','guard','people','fold'])
        methods=plan['trained_methods'];state={'status':'RUNNING','attempts':0,'seconds':0.,'completed':[],'active_job':None,'freeze_sha256':sha(HERE/'FREEZE.json')}
        state_path=local/'state.json';state_path.write_text(json.dumps(state,indent=2)+'\n')
        outputs={m:np.full((len(x),9),np.nan) for m in methods+['markov','frequency']}
        truth=np.zeros((len(x),9));fold_results={};private={};jobs=[]
        try:
            for k in range(3):
                train=fold!=k;held=~train
                params=fit_fold(x,seq,t,g,ids,train)
                # Refit labels using only inner training thresholds; no historical Y9.
                yt,_=labels_and_states(t[train],g[train],params['thresholds'])
                _,st=labels_and_states(x[held,57:59],x[held,59].astype(bool),params['thresholds'],prefix=True)
                sx=np.clip((x-params['x_mean'])/params['x_scale'],-8,8)
                ss=np.clip((seq-params['seq_mean'])/params['seq_scale'],-8,8)
                shuffled=shuffle_tokens(ss,data['source_ids'])
                outputs['markov'][held]=params['prior'][st]@patterns()
                outputs['frequency'][held]=params['frequency']@patterns()
                fold_results[str(k)]={};private[str(k)]={'thresholds':params['thresholds'].tolist()}
                for method in methods:
                    now=datetime.datetime.now(datetime.timezone.utc)
                    deadline=datetime.datetime.fromisoformat(plan['budget']['end_utc'])
                    limit=job_budget(plan,state,now)
                    check();state['attempts']+=1
                    state['active_job']={'method':method,'fold':k,'start_utc':now.isoformat(),'limit_seconds':limit}
                    state_path.write_text(json.dumps(state,indent=2)+'\n')
                    s=shuffled if method=='shuffled_tcn_joint' else ss
                    call=lambda started:predict(method,s[train],sx[train],yt,t[train],g[train],ids[train],s[held],sx[held],params,plan['parameters'],started,limit)
                    p,seconds=timed_fit(call,state,lambda:state_path.write_text(json.dumps(state,indent=2)+'\n'))
                    if seconds>limit or datetime.datetime.now(datetime.timezone.utc)>=deadline:raise TimeoutError('job budget/deadline')
                    # Predictions are persisted before corresponding held target labels are formed.
                    job=local/f'{method}_fold{k}.npz'
                    np.savez_compressed(job,probability=p,row_indices=np.flatnonzero(held))
                    outputs[method][held]=p
                    state['completed'].append(f'{method}_fold{k}');state['active_job']=None
                    jobs.append({'method':method,'fold':k,'seconds':seconds,'predictions_sha256':sha(job)})
                    state_path.write_text(json.dumps(state,indent=2)+'\n')
                    if state['seconds']>plan['budget']['total_training_seconds']:raise TimeoutError('total job budget')
                    print(json.dumps({'completed':len(state['completed']),'method':method,'fold':k,'seconds':round(seconds,3)}),flush=True)
                yh,_=labels_and_states(t[held],g[held],params['thresholds']);truth[held]=yh
                for method in outputs:
                    fold_results[str(k)][method],per=summarize(yh,outputs[method][held],ids[held])
                    private[str(k)][method+'_person_loss']=per.tolist()
            aggregate={}
            for method in outputs:
                if not np.isfinite(outputs[method]).all():raise ValueError('missing prediction')
                aggregate[method],per=summarize(truth,outputs[method],ids)
                private[method+'_person_loss']=per.tolist()
            promoted=[]
            for method in methods:
                relative={b:float((aggregate[b]['log_loss']-aggregate[method]['log_loss'])/aggregate[b]['log_loss']) for b in ['markov','summary_hgb']}
                directions=sum(all(fold_results[str(k)][method]['log_loss']<fold_results[str(k)][b]['log_loss'] for b in ['markov','summary_hgb']) for k in range(3))
                aggregate[method]['relative_gain_vs_controls']=relative
                aggregate[method]['folds_better_than_both_controls']=directions
                if method!='summary_hgb' and min(relative.values())>=.02 and directions>=2:promoted.append(method)
            check()
            if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat(plan['budget']['end_utc']):raise TimeoutError('discovery deadline before completion')
            if prepared['prepared_sha256']!=sha(local/'prepared.npz') or sha(HERE/'PREPARED.json')!=receipt_sha:
                raise ValueError('prepared input/receipt changed')
            state['status']='COMPLETE';state['stopped']=True;state_path.write_text(json.dumps(state,indent=2)+'\n')
            (local/'details.json').write_text(json.dumps(private,indent=2)+'\n')
            result={'scope':'exploratory original-source-person3fold comparison, not independent confirmation',
                'training_jobs':state['attempts'],'training_seconds':state['seconds'],'dev_test_truth_accessed':False,
                'aggregate':aggregate,'fold_results':fold_results,'jobs':jobs,'promising_exploratory_candidates':promoted,
                'prepared_sha256':prepared['prepared_sha256'],'freeze_sha256':sha(HERE/'FREEZE.json'),
                'prepared_receipt_sha256':receipt_sha,
                'private_details_sha256':sha(local/'details.json'),'novel_method_established':False,
                'old_results_preserved':True,'status':'COMPLETE_STOP_THIS_BATCH',
                'limitations':['source participants historically exposed','single seed, small fixed architecture and30epochs',
                    'fold targets use fold-train quantiles, so pooled score concerns this cross-fitting procedure, not one universal fixed threshold',
                    'summary versus sequence comparisons change representation/model; temporal order contrast uses matched TCN',
                    'MDN versus joint also changes loss andhead, so component ablations needed before a mechanism claim']}
            (HERE/'RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
            (HERE/'RESULTS_SEAL.json').write_text(json.dumps({'results_sha256':sha(HERE/'RESULTS.json')},indent=2)+'\n')
            print(json.dumps({'status':result['status'],'aggregate':aggregate,'promising':promoted},indent=2))
        except Exception as error:
            state.update({'status':'FAILED_STOP','stopped':True,'error_type':type(error).__name__,'reason':str(error)})
            state_path.write_text(json.dumps(state,indent=2)+'\n');raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args()
    freeze() if a.action=='freeze' else run()
