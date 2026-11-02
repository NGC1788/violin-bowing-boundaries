from pathlib import Path
import argparse,json,fcntl,time,datetime
import numpy as np
from .models import train_predict
from ..discovery_v1.data import sha,fit_fold
from ..discovery_v1.runner import check as check_v1,summarize,timed_fit,job_budget
from ..violin_raw_v1.data import labels_and_states
from ..violin_raw_v1.models import patterns

HERE=Path(__file__).resolve().parent;V1=HERE.parent/'discovery_v1'


def sources():return {n:sha(HERE/n) for n in ['AGENTS.md','PLAN.json','models.py','runner.py','test_discovery.py']}


def inputs():
    frozen=check_v1()
    receipt=json.loads((V1/'PREPARED.json').read_text());results=json.loads((V1/'RESULTS.json').read_text())
    seal=json.loads((V1/'RESULTS_SEAL.json').read_text())
    if (receipt['prepared_sha256']!=sha(V1/'local/prepared.npz') or
        receipt['freeze_sha256']!=sha(V1/'FREEZE.json') or receipt['source_inputs']!=frozen['source_inputs'] or
        receipt['source_hashes']!=frozen['sources'] or seal['results_sha256']!=sha(V1/'RESULTS.json') or
        results['prepared_sha256']!=receipt['prepared_sha256'] or results['freeze_sha256']!=sha(V1/'FREEZE.json') or
        results['prepared_receipt_sha256']!=sha(V1/'PREPARED.json')):
        raise ValueError('v1 prepared/result provenance mismatch')
    return {n:sha(V1/n) for n in ['FREEZE.json','PREPARED.json','local/prepared.npz','RESULTS.json','RESULTS_SEAL.json']}


def freeze():
    if (HERE/'FREEZE.json').exists():raise ValueError('freeze exists')
    r=json.loads((HERE/'REVIEW.json').read_text())
    if r['verdict']!='APPROVE' or r['source_sha256']!=sources():raise ValueError('unreviewed source')
    f={'sources':sources(),'inputs':inputs(),'review_sha256':sha(HERE/'REVIEW.json'),'scope':'second exploratory12job batch,sourceonly'}
    (HERE/'FREEZE.json').write_text(json.dumps(f,indent=2)+'\n')
    print(json.dumps({'freeze_sha256':sha(HERE/'FREEZE.json')}))


def check():
    f=json.loads((HERE/'FREEZE.json').read_text())
    if f['sources']!=sources() or f['inputs']!=inputs() or f['review_sha256']!=sha(HERE/'REVIEW.json'):raise ValueError('frozen discoveryv2 changed')
    return f


def build_inputs(x,seq,params):
    _,state=labels_and_states(x[:,57:59],x[:,59].astype(bool),params['thresholds'],prefix=True)
    tokens=np.clip((seq-params['seq_mean'])/params['seq_scale'],-8,8)
    anchor=(np.log1p(x[:,57:59])-params['continuous_mean'])/params['continuous_scale']
    query=np.column_stack([np.clip((x-params['x_mean'])/params['x_scale'],-8,8),np.eye(11)[state],anchor])
    return query,tokens,anchor,state


def run():
    check();plan=json.loads((HERE/'PLAN.json').read_text());local=HERE/'local';local.mkdir(exist_ok=True)
    with (local/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=local/'state.json'
        if path.exists():raise ValueError('v2 batch already started; preserve state')
        with np.load(V1/'local/prepared.npz',allow_pickle=False) as z:data={n:z[n] for n in z.files}
        x,seq,t,g,people,fold=(data[n] for n in ['X','sequence','targets','guard','people','fold'])
        methods=plan['factorial_methods'];out={m:np.full((len(x),9),np.nan) for m in methods+['markov']}
        truth=np.zeros((len(x),9));fr={};private={};jobs=[]
        old=json.loads((V1/'RESULTS.json').read_text())
        state={'status':'RUNNING','attempts':0,'seconds':0.,'active_job':None,'completed':[]}
        def persist():path.write_text(json.dumps(state,indent=2)+'\n')
        persist()
        try:
            for k in range(3):
                train=fold!=k;held=~train;params=fit_fold(x,seq,t,g,people,train)
                query,tokens,anchor,st=build_inputs(x,seq,params)
                yt,_=labels_and_states(t[train],g[train],params['thresholds'])
                out['markov'][held]=params['prior'][st[held]]@patterns();fr[str(k)]={};private[str(k)]={}
                for method in methods:
                    check();now=datetime.datetime.now(datetime.timezone.utc);limit=job_budget(plan,state,now)
                    deadline=datetime.datetime.fromisoformat(plan['budget']['end_utc'])
                    state['attempts']+=1;state['active_job']={'method':method,'fold':k,'limit_seconds':limit,'start_utc':now.isoformat()};persist()
                    call=lambda start:train_predict(method,query[train],tokens[train],anchor[train],yt,t[train],g[train],people[train],query[held],tokens[held],anchor[held],params,plan['parameters'],start,limit)
                    (p,metadata,weights),seconds=timed_fit(call,state,persist)
                    if seconds>limit or datetime.datetime.now(datetime.timezone.utc)>=deadline:raise TimeoutError('job budget/deadline')
                    pred=local/f'{method}_fold{k}_predictions.npz';weight=local/f'{method}_fold{k}_weights.npz'
                    np.savez_compressed(pred,probability=p,row_indices=np.flatnonzero(held));np.savez_compressed(weight,**weights)
                    (local/f'{method}_fold{k}_receipt.json').write_text(json.dumps(metadata,indent=2)+'\n')
                    out[method][held]=p;state['completed'].append(f'{method}_fold{k}');state['active_job']=None;persist()
                    jobs.append({'method':method,'fold':k,'seconds':seconds,'parameter_count':metadata['parameter_count'],
                        'train_trace':metadata['trace'],'prediction_sha256':sha(pred),'weight_sha256':sha(weight)})
                    print(json.dumps({'completed':len(state['completed']),'method':method,'fold':k,'seconds':round(seconds,3)}),flush=True)
                yh,_=labels_and_states(t[held],g[held],params['thresholds']);truth[held]=yh
                for method in out:
                    fr[str(k)][method],per=summarize(yh,out[method][held],people[held]);private[str(k)][method]=per.tolist()
                fr[str(k)]['summary_hgb']=old['fold_results'][str(k)]['summary_hgb']
                if abs(fr[str(k)]['markov']['log_loss']-old['fold_results'][str(k)]['markov']['log_loss'])>1e-12:raise ValueError('v1 fold reference mismatch')
            aggregate={}
            for method in out:
                if not np.isfinite(out[method]).all():raise ValueError('missing prediction')
                aggregate[method],per=summarize(truth,out[method],people);private[method]=per.tolist()
            aggregate['summary_hgb']=old['aggregate']['summary_hgb'];promising=[]
            for method in methods:
                rel={b:(aggregate[b]['log_loss']-aggregate[method]['log_loss'])/aggregate[b]['log_loss'] for b in ['markov','summary_hgb']}
                direction=sum(all(fr[str(k)][method]['log_loss']<fr[str(k)][b]['log_loss'] for b in ['markov','summary_hgb']) for k in range(3))
                aggregate[method]['relative_gain_vs_controls']=rel;aggregate[method]['folds_better_than_both']=direction
                if min(rel.values())>=.02 and direction>=2:promising.append(method)
            def interaction(scores):
                mlp=scores['moment_mlp_joint']['log_loss']-scores['moment_mlp_delta']['log_loss']
                attention=scores['moment_attention_joint']['log_loss']-scores['moment_attention_delta']['log_loss']
                return {'delta_loss_reduction_mlp':mlp,'delta_loss_reduction_attention':attention,
                        'additional_delta_reduction_with_attention':attention-mlp}
            check()
            if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat(plan['budget']['end_utc']):raise TimeoutError('deadline')
            state.update({'status':'COMPLETE_STOP_THIS_BATCH','stopped':True});persist()
            (local/'details.json').write_text(json.dumps(private,indent=2)+'\n')
            result={'scope':'adaptive source-only exploratory secondbatch,notindependentconfirmation','training_jobs':state['attempts'],'seconds':state['seconds'],
                'dev_test_truth_accessed':False,'aggregate':aggregate,'fold_results':fr,'factorial_interaction':interaction(aggregate),
                'fold_interactions':{str(k):interaction(fr[str(k)]) for k in range(3)},'jobs':jobs,
                'promising_exploratory_candidates':promising,'novel_method_established':False,'status':state['status'],
                'freeze_sha256':sha(HERE/'FREEZE.json'),'private_details_sha256':sha(local/'details.json'),
                'limitations':['adaptive reusedsourcefolds andone seed','attentionparametercount/context differs','delta changeshead,anchoringandauxiliaryloss together',
                    'v1HGBreference reusedexactly,notretrained','no independentconfirmation ornewmethodclaim']}
            (HERE/'RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
            (HERE/'RESULTS_SEAL.json').write_text(json.dumps({'results_sha256':sha(HERE/'RESULTS.json')},indent=2)+'\n')
            print(json.dumps({'aggregate':aggregate,'interaction':result['factorial_interaction'],'promising':promising},indent=2))
        except Exception as e:
            state.update({'status':'FAILED_STOP','stopped':True,'error_type':type(e).__name__,'reason':str(e)});persist();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args();freeze() if a.action=='freeze' else run()
