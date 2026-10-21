"""Frozen-plan exact certificate deletion controls. Never modifies source artifacts."""
from pathlib import Path
import argparse,concurrent.futures,gzip,hashlib,itertools,json,shutil,time
from collections import defaultdict
import numpy as np
import model as M
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
PS=(-2.,-1.2)

def write(path,x):path.write_text(json.dumps(x,indent=2,allow_nan=True)+'\n')
def prepare():
    src=HERE/'input_joint.json.gz'
    cols=json.load(gzip.open(src,'rt'))['columns'];rows=sorted(M.extract_rows(cols),key=lambda r:r['id']);write(HERE/'rows.json',rows)
    ref=json.load(open(HERE/'reference_baselines.json'))['sessions']
    cases=json.load(open(HERE/'cases.json'))
    by=defaultdict(list)
    for r in rows:by[r['session']].append(r)
    out={'sessions':{},'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest()}
    for s,rr in sorted(by.items()):
        free=M.fit(rr);fixed={str(p):M.fit(rr,p,free['params']) for p in PS};saved=ref[s]['baseline'];delta=np.array(free['params'])-saved['params']
        assert free['success'] and all(f['success'] for f in fixed.values()),(s,free,fixed)
        assert max(abs(delta))<2e-4 and abs(free['nll']-saved['nll'])<1e-6,(s,delta)
        lo,hi=np.asarray([r['lower'] for r in rr]).T
        out['sessions'][s]={'free':free,'fixed':fixed,'row_ids':[r['id'] for r in rr],'counts':{'n':len(rr),'finite':int((np.isfinite(lo)&np.isfinite(hi)).sum()),'left':int(np.isneginf(lo).sum()),'right':int(np.isposinf(hi).sum())},'reference_param_delta':delta.tolist(),'reference_nll_delta':free['nll']-saved['nll']}
    out['status']='PASS';write(HERE/'baselines.json',out)
    print('BASELINE PASS',len(by),'max param delta',max(max(abs(np.array(z['reference_param_delta']))) for z in out['sessions'].values()),flush=True)
    selected=[]
    for c in cases:
        rr=by[c['session']];base=out['sessions'][c['session']];deleted=[i for i,r in enumerate(rr) if r['id'] in c['ids']];remain=[r for i,r in enumerate(rr) if i not in deleted]
        f=M.fit(remain,init=base['free']['params']);n=M.fit(remain,c['p'],base['fixed'][str(c['p'])]['params'])
        beforeT=2*(base['fixed'][str(c['p'])]['nll']-base['free']['nll']);afterT=2*(n['nll']-f['nll'])
        ellfree=M.row_nll(base['free']['params'],rr);ellfixed=M.row_nll(base['fixed'][str(c['p'])]['params'],rr)
        removed=float(2*np.sum((ellfixed-ellfree)[deleted]));gainfree=float(ellfree.sum()-ellfree[deleted].sum()-f['nll']);gainfixed=float(ellfixed.sum()-ellfixed[deleted].sum()-n['nll']);decomp=removed+2*(gainfixed-gainfree)
        betaidx={rr[i]['column'] for i in deleted};blockids=[r['id'] for r in rr if r['column'] in betaidx];blockrr=[r for r in rr if r['id'] not in blockids];bf=M.fit(blockrr,init=base['free']['params']);bn=M.fit(blockrr,c['p'],base['fixed'][str(c['p'])]['params'])
        selected.append(c|{'deleted_row_indices':deleted,'before':base,'after':{'free':f,'fixed':n},'delta_beta':f['params'][2]-base['free']['params'][2],'delta_speed':f['params'][1]-base['free']['params'][1],'toward_target':abs(base['free']['params'][2]-c['p'])-abs(f['params'][2]-c['p']),'T_before':beforeT,'T_after':afterT,'T_reduction':beforeT-afterT,'decomposition':{'removed_fullfit_contrast':removed,'free_nuisance_gain':gainfree,'fixed_nuisance_gain':gainfixed,'differential_gain':2*(gainfixed-gainfree),'identity_error':beforeT-afterT-decomp},'block':{'ids':blockids,'free':bf,'fixed':bn,'delta_beta':bf['params'][2]-base['free']['params'][2],'T_after':2*(bn['nll']-bf['nll'])}})
    write(HERE/'selected.json',{'status':'selected fitted; awaiting complete controls','cases':selected})
    print('SELECTED',len(selected),'maxdecomperror',max(abs(c['decomposition']['identity_error']) for c in selected),flush=True)

def jobs():
    rows=json.load(open(HERE/'rows.json'));cases=json.load(open(HERE/'cases.json'));base=json.load(open(HERE/'baselines.json'))['sessions'];j={}
    for c in cases:
        key=(c['group'],c['k']);j.setdefault(key,{'group':c['group'],'session':c['session'],'k':c['k'],'p':[]})['p'].append(c['p'])
    for num,(key,z) in enumerate(sorted(j.items())):
        z['number']=num;z['p']=sorted(set(z['p']));z['rows']=[r for r in rows if r['session']==z['session']];z['base']=base[z['session']]
        z['eligible_indices']=[i for i,r in enumerate(z['rows']) if r['group']==z['group'] and r['eligible']]
    return list(j.values())

def task(job,limit=None,save=True):
    start=time.monotonic();rr=job['rows'];X,lo,hi=M.arrays(rr);combos=list(itertools.combinations(job['eligible_indices'],job['k']))
    if limit:combos=combos[:limit]
    fields={};N=len(combos)
    for label in ['free']+[str(p) for p in job['p']]:
        for key,shape,dtype in [('params',(N,4),float),('nll',(N,),float),('grad_max',(N,),float),('success',(N,),bool),('scale_hit',(N,),bool),('iterations',(N,),int),('attempts',(N,),int)]:fields[label+'_'+key]=np.empty(shape,dtype=dtype)
    failures=[]
    for k,delete in enumerate(combos):
        keep=np.ones(len(rr),bool);keep[list(delete)]=False
        for p in [None]+job['p']:
            label='free' if p is None else str(p);warm=job['base']['free' if p is None else 'fixed'];warm=warm if p is None else warm[str(p)]
            f=M.fit_arrays(X[keep],lo[keep],hi[keep],p,warm['params'])
            for key in ('params','nll','grad_max','success','scale_hit','iterations','attempts'):fields[label+'_'+key][k]=f[key]
            if not f['success']:failures.append({'control':k,'delete':list(delete),'p':p,'fit':f})
        for p in job['p']:
            if fields[str(p)+'_nll'][k]<fields['free_nll'][k]-1e-7:failures.append({'control':k,'reason':'fixed_better_than_free','p':p})
    fields['combinations']=np.asarray(combos,int);fields['log_beta_span']=np.array([np.ptp([rr[i]['lb'] for i in c]) for c in combos])
    record={k:v for k,v in job.items() if k not in ('rows','base')};record.update(n_controls=N,fit_count=N*(1+len(job['p'])),seconds=time.monotonic()-start,failures=failures)
    if save:
        out=HERE/'controls';out.mkdir(exist_ok=True);np.savez_compressed(out/f"{job['number']:02}.npz",**fields);write(out/f"{job['number']:02}.json",record)
    return record

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--benchmark',type=int);ap.add_argument('--workers',type=int,default=2);args=ap.parse_args()
    if args.prepare:prepare();return
    jj=jobs()
    if args.benchmark:
        z=task(jj[0],args.benchmark,False);total=sum(len(list(itertools.combinations(j['eligible_indices'],j['k'])))*(1+len(j['p'])) for j in jj);z['total_projected_fits']=total;z['projected_seconds_serial']=z['seconds']*total/z['fit_count'];write(HERE/'benchmark.json',z);print(z);return
    todo=[j for j in jj if not (HERE/'controls'/f"{j['number']:02}.json").exists()];start=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(args.workers,2)) as ex:
        fut={ex.submit(task,j):j for j in todo}
        for f in concurrent.futures.as_completed(fut):
            z=f.result();print('DONE',z['number'],z['session'],z['group'].split('/')[-1],z['k'],z['n_controls'],'fits',z['fit_count'],'sec',round(z['seconds'],1),'fails',len(z['failures']),'elapsed',round(time.monotonic()-start),flush=True)
if __name__=='__main__':main()
