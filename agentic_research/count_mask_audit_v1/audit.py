"""Single frozen synthetic inference audit; never loads actual data."""
from pathlib import Path
import argparse,datetime,fcntl,hashlib,json,os,platform,time
import numpy as np
import scipy
import torch
from threadpoolctl import threadpool_limits
from .kernel import tensor,dp,guard_dp,numpy_counts,tables,direct_path_query,inference,logits_for,cells,correlation_counterexample

HERE=Path(__file__).resolve().parent
LEDGER=HERE.parent/'discovery_v2/LOOP_BUDGET.json'
PARENT=HERE.parent/'path_mask_feasibility_v1'
NAMES=['AGENTS.md','PLAN.json','kernel.py','audit.py','test_kernel.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def sources():return {n:sha(HERE/n) for n in NAMES}
def versions():return {'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__}
def parent_hashes():return {n:sha(PARENT/n) for n in ['METHOD.md','PRIOR_REVIEW.md','REVIEW.json','DECISION.md']}
def limits():
    d=json.loads(LEDGER.read_text());return {k:d[k] for k in ['deadline_kst','max_math_audits','max_seconds_per_math_audit']}


def freeze():
    if (HERE/'FREEZE.json').exists() or (HERE/'local/started.json').exists():raise ValueError('freeze/start already exists')
    r=json.loads((HERE/'REVIEW.json').read_text());u=json.loads((HERE/'UNIT_TESTS.json').read_text())
    if r['verdict']!='APPROVE' or r['blockers'] or r['source_sha256']!=sources():raise ValueError('source not approved')
    if u['source_sha256']!=sources() or u['failed']!=0:raise ValueError('latest source not tested')
    p=json.loads((HERE/'PLAN.json').read_text());lim=limits()
    if p['deadline_kst']!=lim['deadline_kst'] or p['max_seconds']>lim['max_seconds_per_math_audit']:raise ValueError('budget mismatch')
    dump(HERE/'FREEZE.json',{'sources':sources(),'versions':versions(),'parent_hashes':parent_hashes(),
         'review_sha256':sha(HERE/'REVIEW.json'),'tests_sha256':sha(HERE/'UNIT_TESTS.json'),'budget_limits':lim})
    print(json.dumps({'freeze_sha256':sha(HERE/'FREEZE.json')}))


def check():
    f=json.loads((HERE/'FREEZE.json').read_text())
    if (f['sources']!=sources() or f['versions']!=versions() or f['parent_hashes']!=parent_hashes()
       or f['review_sha256']!=sha(HERE/'REVIEW.json') or f['tests_sha256']!=sha(HERE/'UNIT_TESTS.json')
       or f['budget_limits']!=limits()):raise ValueError('frozen numerical contract changed')
    return f


def register(status,*,ledger=LEDGER,freeze_sha=None,seconds=None):
    ledger=Path(ledger);lock=ledger.parent/'local/loop_budget.lock';lock.parent.mkdir(exist_ok=True)
    with lock.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        d=json.loads(ledger.read_text());entries=d['math_audits_started'];name='count_mask_audit_v1'
        found=[e for e in entries if e['audit']==name]
        if status=='RUNNING':
            if found or len(entries)>=d['max_math_audits']:raise ValueError('duplicate attempt or math budget exhausted')
            entries.append({'audit':name,'status':status,'freeze_sha256':freeze_sha,
                            'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
        else:
            if len(found)!=1:raise ValueError('missing attempt')
            found[0].update(status=status,seconds=seconds)
        temp=ledger.with_suffix('.tmp');dump(temp,d);os.replace(temp,ledger)


def evaluate(plan,started,deadline):
    torch.set_num_threads(1);crit=plan['criteria'];small=logits_for(plan);blocks=plan['small_blocks'];guard=plan['small_guard']
    def budget():
        if time.monotonic()-started>plan['max_seconds'] or datetime.datetime.now(datetime.timezone.utc)>=deadline:
            raise TimeoutError('numerical audit time/deadline')
    case_results=[];mass_error=0.;cdf_error=0.;gradient_violation=0.;shared_violation=0.;fd_error=0.;cell_min=1.;coarse_violation=0.
    for conditioned in [False,True]:
        budget();table,_,_=tables(plan,blocks,guard,count_conditioned=conditioned)
        reference,count_ref=direct_path_query(plan,small,blocks,guard,count_conditioned=conditioned)
        theta=tensor(small).requires_grad_();full=inference(theta,blocks,guard,table,plan['mixture'])
        q=full['q'].detach().numpy();cdf_error=max(cdf_error,float(np.abs(q-reference).max()))
        mass_error=max(mass_error,max(float(np.abs(w.detach().numpy()-r).max()) for w,r in zip(full['weights'],count_ref)))
        grads=np.array([torch.autograd.grad(full['q'][i,j],theta,retain_graph=True)[0].numpy() for i in range(3) for j in range(3)])
        exact_guard=float(full['guard'].detach());weights=[w.detach().numpy().tolist() for w in full['weights']]
        a,b=np.indices(count_ref.shape[1:]);enum_guard=float((np.asarray(plan['mixture'])[:,None,None]*count_ref*(a+b>=guard)[None]).sum())
        mass_error=max(mass_error,abs(exact_guard-enum_guard),abs(sum(pi*float(w.sum().detach()) for pi,w in zip(plan['mixture'],full['weights']))-1.))
        rows=[]
        for beam in plan['small_beams']:
            budget();th=tensor(small).requires_grad_();out=inference(th,blocks,guard,table,plan['mixture'],beam)
            approx=out['q'].detach().numpy();residual=float(out['residual'].detach());mass_error=max(mass_error,max(0.,-residual))
            bg=np.array([torch.autograd.grad(out['q'][i,j],th,retain_graph=True)[0].numpy() for i in range(3) for j in range(3)])
            coordinate=float(np.max(np.abs(grads-bg)));shared=float(np.max(np.abs((grads-bg).sum((1,2,3)))))
            gradient_violation=max(gradient_violation,coordinate-residual);shared_violation=max(shared_violation,shared-len(small[0])*residual)
            cdf_error=max(cdf_error,float(np.maximum(approx-q,0).max()),float(np.maximum(q-approx-residual,0).max()))
            cell_min=min(cell_min,float(cells(approx).min()),float(1-approx[-1,-1]))
            step=crit['finite_difference_step'];up=small.copy();down=small.copy();up[0,1,1]+=step;down[0,1,1]-=step
            fu=inference(tensor(up),blocks,guard,table,plan['mixture'],beam,out['traces'])['q'].mean()
            fd=inference(tensor(down),blocks,guard,table,plan['mixture'],beam,out['traces'])['q'].mean()
            finite=float((fu-fd)/(2*step));automatic=float(bg[:,0,1,1].mean());fd_error=max(fd_error,abs(finite-automatic))
            rows.append({'beam':beam,'q':approx.tolist(),'retained_valid':float(out['retained_valid'].detach()),
                         'residual':residual,'gradient':bg.tolist(),'max_coordinate_gradient_error':coordinate,
                         'max_shared_gradient_error':shared,'finite_difference':finite,'autograd_mean_coordinate':automatic,
                         'traces':out['traces']})
        case_results.append({'count_conditioned':conditioned,'enumerated_q':reference.tolist(),'full_q':q.tolist(),
                             'guard':exact_guard,'enumerated_guard':enum_guard,'full_count_weights':weights,
                             'enumerated_count_weights':count_ref.tolist(),
                             'full_gradients':grads.tolist(),'beam_results':rows})
    # Coarsening holds the original count-independent path law fixed.
    _,lo,hi=tables(plan,blocks,guard,coarsening=True);perturbed,_=direct_path_query(plan,small,blocks,guard,perturbed=True)
    coarse=[]
    for beam in [None]+plan['small_beams']:
        lower=inference(tensor(small),blocks,guard,lo,plan['mixture'],beam)
        upper=inference(tensor(small),blocks,guard,hi,plan['mixture'],beam)
        l=lower['q'].numpy();u=upper['q'].numpy()+float(upper['residual'])
        coarse_violation=max(coarse_violation,float(np.maximum(l-perturbed,0).max()),float(np.maximum(perturbed-u,0).max()))
        coarse.append({'beam':beam,'lower':l.tolist(),'upper':u.tolist(),'reference':perturbed.tolist(),'max_width':float((u-l).max())})
    large_rows=[];large_error=0.;concentrated_residual=None
    for case in plan['large_cases']:
        budget();ll=logits_for(plan,case);bl=plan['large_blocks'];minimum=plan['large_guard'];table,_,_=tables(plan,bl,minimum)
        counts=np.array([numpy_counts(ll[h],bl) for h in range(2)])
        reference=(np.asarray(plan['mixture'])[:,None,None,None,None]*counts[:,:,:,None,None]*table).sum((0,1,2))
        dense=inference(tensor(ll),bl,minimum,table,plan['mixture']);large_error=max(large_error,float(np.abs(dense['q'].numpy()-reference).max()))
        row={'name':case['name'],'reference_q':reference.tolist(),'dense_q':dense['q'].tolist(),'guard':float(dense['guard']),
             'numpy_count_weights':counts.tolist(),'dense_count_weights':[w.tolist() for w in dense['weights']],'beams':[]}
        for beam in plan['large_beams']:
            out=inference(tensor(ll),bl,minimum,table,plan['mixture'],beam);q=out['q'].numpy();r=float(out['residual'])
            large_error=max(large_error,float(np.maximum(q-reference,0).max()),float(np.maximum(reference-q-r,0).max()))
            row['beams'].append({'beam':beam,'q':q.tolist(),'residual':r,'retained_valid':float(out['retained_valid']),
                                 'max_abs_probability_error':float(np.abs(q-reference).max())})
            if case['name']=='concentrated' and beam==crit['required_concentrated_beam']:concentrated_residual=r
        large_rows.append(row)
    benchmark={};case=plan['large_cases'][0];ll=logits_for(plan,case)
    for name,beam in [('beam',plan['benchmark']['beam']),('dense',None)]:
        timings=[]
        for _ in range(plan['benchmark']['repetitions']):
            budget();clock=time.monotonic();table,_,_=tables(plan,plan['large_blocks'],plan['large_guard'])
            theta=tensor(ll).requires_grad_();q=inference(theta,plan['large_blocks'],plan['large_guard'],table,plan['mixture'],beam)
            torch.autograd.grad(q['q'].mean(),theta);timings.append(time.monotonic()-clock)
        benchmark[name]={'seconds':timings,'median_seconds':float(np.median(timings))}
    ratio=benchmark['beam']['median_seconds']/benchmark['dense']['median_seconds'];benchmark['beam_dense_ratio']=ratio
    benchmark['acceleration_screen_pass']=ratio<=plan['benchmark']['acceleration_claim_max_beam_dense_ratio']
    negative=correlation_counterexample(plan)
    checks={'count_law_and_dp':mass_error<=crit['mass_abs'],'cdf_enumeration_and_brackets':cdf_error<=crit['probability_abs'],
            'local_mask_gradient_bounds':gradient_violation<=crit['gradient_abs'] and shared_violation<=crit['gradient_abs'],
            'fixed_trace_finite_difference':fd_error<=crit['gradient_fd_abs'],'coherence':cell_min>=-crit['probability_abs'],
            'fixed_law_coarsening':coarse_violation<=crit['coarsening_abs'],'large_independent_dense_and_brackets':large_error<=crit['probability_abs'],
            'concentrated_mass_width':concentrated_residual is not None and concentrated_residual<=crit['required_concentrated_beam_probability_width'],
            'single_example_mask_backward_cost':benchmark['beam']['median_seconds']<=crit['benchmark_forward_backward_median_seconds'],
            'correlated_innovations_negative_control':negative['abs_gap']>=crit['negative_control_min_probability_gap']}
    return {'verdict':'NUMERICAL_PASS_RESTRICTED_COUNT_KERNEL' if all(checks.values()) else 'FAILED_PREDEFINED_GATE_STOP',
            'checks':checks,'maxima':{'mass_error':mass_error,'small_cdf_error_or_bracket_violation':cdf_error,
              'local_gradient_bound_violation':gradient_violation,'shared_gradient_bound_violation':shared_violation,'fixed_trace_fd_error':fd_error,
              'minimum_cell_or_outside':cell_min,'coarsening_violation':coarse_violation,'large_cdf_or_bracket_error':large_error},
            'small_cases':case_results,'coarsening':coarse,'large_cases':large_rows,'benchmark':benchmark,'negative_control':negative,
            'actual_data_accessed':False,'new_models_trained':0,'novel_method_established':False,'real_forecast_improvement_established':False,
            'implementation_limits':['beam uses dense grid with pruning masks; not an O(T L) sparse implementation',
              'CDF values recomputed but Gaussian-head/network gradients not benchmarked','algebraic bounds plus empirical numeric tests; no validated floating-point intervals',
              'two-block conditional-iid Gaussian law only; correlated innovation counterexample retained']}


def run():
    check();plan=json.loads((HERE/'PLAN.json').read_text());local=HERE/'local';local.mkdir(exist_ok=True)
    with (local/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (local/'started.json').exists() or (HERE/'RESULTS.json').exists():raise ValueError('attempt already exists; no retry')
        deadline=datetime.datetime.fromisoformat(plan['deadline_kst'])
        if datetime.datetime.now(datetime.timezone.utc)>=deadline:raise TimeoutError('deadline')
        register('RUNNING',freeze_sha=sha(HERE/'FREEZE.json'));started=time.monotonic()
        dump(local/'started.json',{'freeze_sha256':sha(HERE/'FREEZE.json'),'max_attempts':1})
        try:
            with threadpool_limits(limits=1):result=evaluate(plan,started,deadline)
            check();seconds=time.monotonic()-started
            if seconds>plan['max_seconds'] or datetime.datetime.now(datetime.timezone.utc)>=deadline:raise TimeoutError('budget/deadline')
            result.update(seconds=seconds,freeze_sha256=sha(HERE/'FREEZE.json'));dump(HERE/'RESULTS.json',result)
            dump(HERE/'RESULTS_SEAL.json',{'results_sha256':sha(HERE/'RESULTS.json')})
            register(result['verdict'],seconds=seconds)
            print(json.dumps({k:result[k] for k in ['verdict','checks','maxima','benchmark','seconds']},indent=2))
        except Exception as error:
            seconds=time.monotonic()-started;dump(local/'failure.json',{'type':type(error).__name__,'reason':str(error),'seconds':seconds})
            register('FAILED_STOP',seconds=seconds);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','check','run']);a=p.parse_args()
    if a.action=='freeze':freeze()
    elif a.action=='check':check();print('contract intact')
    else:run()
