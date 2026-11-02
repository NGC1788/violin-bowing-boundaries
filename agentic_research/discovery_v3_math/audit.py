"""One frozen synthetic audit, independent path references and gradient checks."""
from pathlib import Path
import argparse,datetime,fcntl,hashlib,json,os,platform,time
import numpy as np
import scipy
import torch
from threadpoolctl import threadpool_limits
from .decoder import fixture,tensor_case,decode,independent_analytic,path_reference,quadrature,cell_masses,nc_cdf,tensor,gradient_objective

HERE=Path(__file__).resolve().parent
LEDGER=HERE.parent/'discovery_v2/LOOP_BUDGET.json'
LIMIT_KEYS=['deadline_kst','max_math_audits','max_seconds_per_math_audit']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sources():return {n:sha(HERE/n) for n in ['AGENTS.md','PLAN.json','decoder.py','audit.py','test_decoder.py']}
def versions():return {'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__}
def limits():
    d=json.loads(LEDGER.read_text());return {k:d[k] for k in LIMIT_KEYS}


def freeze():
    if (HERE/'FREEZE.json').exists():raise ValueError('freeze already exists')
    review=json.loads((HERE/'REVIEW.json').read_text())
    if review['verdict']!='APPROVE' or review['source_sha256']!=sources():raise ValueError('unreviewed source')
    f={'sources':sources(),'versions':versions(),'budget_limits':limits(),'review_sha256':sha(HERE/'REVIEW.json'),
       'literature_sha256':sha(HERE.parent/'discovery_v2/PATH_DISTRIBUTION_REVIEW.md'),
       'scope':'one synthetic numerical audit; no model fitting or actual data access'}
    (HERE/'FREEZE.json').write_text(json.dumps(f,indent=2)+'\n');print(json.dumps({'freeze_sha256':sha(HERE/'FREEZE.json')}))


def check():
    f=json.loads((HERE/'FREEZE.json').read_text())
    if (f['sources']!=sources() or f['versions']!=versions() or f['budget_limits']!=limits() or
        f['review_sha256']!=sha(HERE/'REVIEW.json') or
        f['literature_sha256']!=sha(HERE.parent/'discovery_v2/PATH_DISTRIBUTION_REVIEW.md')):
        raise ValueError('audit contract changed')
    return f


def ledger_update(status,seconds=None):
    lock=LEDGER.parent/'local/loop_budget.lock';lock.parent.mkdir(exist_ok=True)
    with lock.open('a') as file:
        fcntl.flock(file,fcntl.LOCK_EX)
        d=json.loads(LEDGER.read_text());entries=d['math_audits_started']
        found=[x for x in entries if x['audit']=='discovery_v3_math']
        if status=='RUNNING':
            if found or len(entries)>=d['max_math_audits']:raise ValueError('math budget or duplicate attempt')
            entries.append({'audit':'discovery_v3_math','status':status,'freeze_sha256':sha(HERE/'FREEZE.json'),
                'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
        else:
            if len(found)!=1:raise ValueError('missing math audit registration')
            found[0].update({'status':status,'seconds':seconds})
        temp=LEDGER.with_suffix('.tmp');temp.write_text(json.dumps(d,indent=2)+'\n');os.replace(temp,LEDGER)


def scalar_gradients(plan):
    rows=[];step=plan['criteria']['finite_difference_step']
    for df in [22,45]:
        for nc in [0.,.5,8.,80.]:
            for x in [.1,float(df),df+nc,df+nc+6*np.sqrt(2*(df+2*nc))]:
                xx=tensor(x).requires_grad_();nn=tensor(nc).requires_grad_();out=nc_cdf(xx,nn,df)
                gx,gn=torch.autograd.grad(out,(xx,nn))
                fd_x=float((nc_cdf(tensor(x+step),tensor(nc),df)-nc_cdf(tensor(x-step),tensor(nc),df))/(2*step))
                if nc==0:
                    fd_n=float((nc_cdf(tensor(x),tensor(step),df)-out.detach())/step)
                else:fd_n=float((nc_cdf(tensor(x),tensor(nc+step),df)-nc_cdf(tensor(x),tensor(nc-step),df))/(2*step))
                for variable,a,b in [('x',float(gx),fd_x),('nc',float(gn),fd_n)]:
                    scale=plan['criteria']['gradient_abs']+plan['criteria']['gradient_relative']*abs(b)
                    rows.append({'df':df,'nc':nc,'x':x,'variable':variable,'autograd':a,'finite_difference':b,
                                 'abs_error':abs(a-b),'scaled_error':abs(a-b)/scale})
    return rows


def path_gradients(plan):
    rows=[];order_sensitivity=[];step=plan['criteria']['finite_difference_step']
    for values in [[.5,np.log(.4),0.],[1.,np.log(.6),.7],[0.,np.log(.5),-.3]]:
        theta=tensor(values).requires_grad_();out=gradient_objective(theta,256)
        grad=torch.autograd.grad(out,theta)[0].detach().numpy()
        other=tensor(values).requires_grad_();g512=torch.autograd.grad(gradient_objective(other,512),other)[0].detach().numpy()
        order_sensitivity.append(float(np.max(np.abs(grad-g512))))
        for j in range(3):
            up=np.array(values);down=np.array(values);up[j]+=step;down[j]-=step
            fd=float((gradient_objective(tensor(up),256)-gradient_objective(tensor(down),256))/(2*step))
            scale=plan['criteria']['gradient_abs']+plan['criteria']['gradient_relative']*abs(fd)
            rows.append({'theta':values,'parameter':j,'autograd':float(grad[j]),'finite_difference':fd,
                'abs_error':abs(float(grad[j])-fd),'scaled_error':abs(float(grad[j])-fd)/scale})
    return {'checks':rows,'max_gradient_order256_vs512_abs':max(order_sensitivity),
            'scope':'derivative of truncated quadrature; no full-law derivative-tail certificate'}


def run():
    check();plan=json.loads((HERE/'PLAN.json').read_text());local=HERE/'local';local.mkdir(exist_ok=True)
    with (local/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (local/'started.json').exists():raise ValueError('audit already started; preserve results/failure')
        deadline=datetime.datetime.fromisoformat(plan['budget']['deadline_kst'])
        if datetime.datetime.now(datetime.timezone.utc)>=deadline:raise TimeoutError('audit deadline')
        ledger_update('RUNNING');started=time.monotonic()
        (local/'started.json').write_text(json.dumps({'freeze_sha256':sha(HERE/'FREEZE.json'),'scope':'one synthetic math audit'})+'\n')
        try:
            with threadpool_limits(limits=1):result=evaluate(plan,started,deadline)
            check();result['seconds']=time.monotonic()-started
            if result['seconds']>plan['budget']['max_seconds'] or datetime.datetime.now(datetime.timezone.utc)>=deadline:
                raise TimeoutError('audit budget/deadline')
            result['freeze_sha256']=sha(HERE/'FREEZE.json')
            (HERE/'RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
            (HERE/'RESULTS_SEAL.json').write_text(json.dumps({'results_sha256':sha(HERE/'RESULTS.json')},indent=2)+'\n')
            ledger_update(result['verdict'],result['seconds'])
            print(json.dumps({'verdict':result['verdict'],'checks':result['checks'],'seconds':result['seconds']},indent=2))
        except Exception as error:
            elapsed=time.monotonic()-started
            (local/'failure.json').write_text(json.dumps({'type':type(error).__name__,'reason':str(error),'seconds':elapsed})+'\n')
            ledger_update('FAILED_STOP',elapsed);raise


def evaluate(plan,started,deadline):
    torch.set_num_threads(1);crit=plan['criteria'];cases={};analytic_errors=[];reference_errors=[];quadrature_errors=[];minimum_cell=1.
    def budget():
        if time.monotonic()-started>plan['budget']['max_seconds'] or datetime.datetime.now(datetime.timezone.utc)>=deadline:
            raise TimeoutError('audit budget/deadline')
    for name in ['central','drift','shared','mask']:
        c=fixture(name);orders={};qs={}
        for order in plan['integration']['orders']:
            out=decode(tensor_case(c),order);q=out['joint'].detach().numpy();qs[order]=q
            masses=cell_masses(q)
            if not np.isfinite(q).all() or not np.isfinite(masses).all():raise ArithmeticError('nonfinite probability')
            minimum_cell=min(minimum_cell,float(masses.min()))
            orders[str(order)]={'joint':q.tolist(),'marginals_with_guard':out['marginals_with_guard'].detach().numpy().tolist(),
                'valid_probability':float(out['valid_probability']),'min_pattern_mass':float(masses.min()),'quadrature':quadrature(order)[2]}
            budget()
        discrepancy=float(np.max(np.abs(qs[512]-qs[256])));quadrature_errors.append(discrepancy)
        references=[]
        for power in [14,16]:
            for seed in [0,1]:
                r=path_reference(c,power,seed);error=float(np.max(np.abs(qs[512]-r['joint'])));reference_errors.append(error)
                references.append({**r,'joint':r['joint'].tolist(),'max_abs_error':error});budget()
        case={'orders':orders,'quadrature256_vs512_max_abs':discrepancy,'references':references}
        if name in ['central','drift']:
            reference=independent_analytic(c);error=float(np.max(np.abs(qs[512]-reference)))
            analytic_errors.append(error);case.update({'analytic_joint':reference.tolist(),'analytic_max_abs_error':error})
        cases[name]=case
        (HERE/'local/progress.json').write_text(json.dumps({'completed_fixtures':list(cases)},indent=2)+'\n')
    shared=cases['shared']['orders']['512'];marg=np.array(shared['marginals_with_guard'])
    product_error=float(np.max(np.abs(np.array(shared['joint'])-np.outer(*marg))))
    wrong=fixture('mask');wrong.update(masks=np.ones((1,46),bool),logits=np.zeros(1),slopes=np.zeros(1))
    wrong_q=decode(tensor_case(wrong),512)['joint'].detach().numpy()
    mask_error=float(np.max(np.abs(np.array(cases['mask']['orders']['512']['joint'])-wrong_q)))
    same=decode(tensor_case(fixture('invariance')),512)['joint'].detach().numpy()
    invariant_error=float(np.max(np.abs(same-np.array(shared['joint']))))
    guard22=decode(tensor_case(fixture('guard22')),256);guard23=decode(tensor_case(fixture('guard23')),256)
    guard_ok=bool(np.all(guard22['joint'].numpy()==0) and float(guard22['valid_probability'])==0 and float(guard23['valid_probability'])>.999999)
    budget();scalar=scalar_gradients(plan);path=path_gradients(plan);budget()
    scalar_max=max(r['scaled_error'] for r in scalar);path_max=max(r['scaled_error'] for r in path['checks'])
    durations=[]
    for _ in range(5):
        theta=tensor([.5,np.log(.4),0.]).requires_grad_();start=time.perf_counter()
        torch.autograd.grad(gradient_objective(theta,256),theta);durations.append(time.perf_counter()-start)
    checks={'analytic_accuracy':max(analytic_errors)<=crit['analytic_max_abs_error'],
        'quadrature_order_agreement':max(quadrature_errors)<=crit['quadrature_256_vs_512_max_abs'],
        'all_fixed_qmc_references_agree':max(reference_errors)<=crit['quadrature_vs_max_QMC_reference_abs'],
        'coherence':minimum_cell>=-crit['coherence_tolerance'],
        'scalar_cdf_gradients':scalar_max<=1,'path_parameter_gradients':path_max<=1,
        'constant_level_nonidentifiability':invariant_error<=crit['nonidentifiability_abs'],
        'guard22_vs23':guard_ok,'marginal_product_counterexample':product_error>crit['shared_marginal_product_error_should_exceed'],
        'ignoring_mask_counterexample':mask_error>crit['mask_neglect_error_should_exceed']}
    return {'scope':'restricted synthetic one-factor path/mask numerical audit; no real data or fitting',
        'verdict':'NUMERICAL_PASS_RESTRICTED_MODEL' if all(checks.values()) else 'NUMERICAL_FAIL_NO_RESCUE',
        'checks':checks,'cases':cases,'scalar_gradient_checks':scalar,'path_gradient_checks':path,
        'maxima':{'analytic_abs':max(analytic_errors),'quadrature256_vs512_abs':max(quadrature_errors),
            'qmc_reference_abs':max(reference_errors),'min_pattern_mass':minimum_cell,'scalar_gradient_scaled_error':scalar_max,
            'path_gradient_scaled_error':path_max,'constant_shift_abs':invariant_error},
        'counterexamples':{'unconditional_marginal_product_max_abs_error':product_error,'fullmask_plugin_max_abs_error':mask_error,'wrong_fullmask_joint':wrong_q.tolist()},
        'forward_backward_median_seconds':float(np.median(durations)),
        'actual_data_accessed':False,'new_models_trained':0,'novel_method_established':False,
        'limitations':['one factor, finite mask law, conditional iid homoscedastic Gaussian innovations',
            'mask independent of innovations given latentfactor; not arbitraryMNAR/realobserver model',
            'quadrature/QMC agreement is not certified integration-error bound',
            'normal tail bound is forwardonly; derivative tests validate truncatedimplementation',
            'CPUfloat64first-order gradients, no GPU/higherderivative/scalingclaim',
            'synthetic agreement doesnotestablish realforecast performance ormethodnovelty']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args();freeze() if a.action=='freeze' else run()
