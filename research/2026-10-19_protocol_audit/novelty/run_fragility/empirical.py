"""Fixed-protocol exact decision changes on already exposed derived data.

Reproducible from input_columns.json.gz with NumPy and the two local solvers.
No classifier/likelihood retraining; no original score or label is changed.
"""
from pathlib import Path
import json,gzip,math,hashlib,time
from collections import defaultdict,Counter
import numpy as np
from core import first_run_start,first_run_costs,witness_first_run
from global_profile import optimize_intercept,optimize_incompatibility
HERE=Path(__file__).resolve().parent


def prepare(columns,k):
    kept=[];excluded=Counter()
    for col in columns:
        f=np.array(col['force']);h=np.array(col['labels'])
        if not np.isfinite(f).all():excluded['nonfinite_force']+=1;continue
        assert np.all(np.diff(f)>=0)
        s=first_run_start(h,k)
        if s is None:excluded['no_original_run']+=1;continue
        if s==0:excluded['original_left_censored']+=1;continue
        if not 0<f[s-1]<f[s]:excluded['original_nonpositive_or_zero_width']+=1;continue
        valid=[j for j in range(1,len(h)-k+1) if 0<f[j-1]<f[j]]
        assert s in valid
        kept.append(dict(col=col,original_start=s,valid_starts=valid,
                         intervals={j:(math.log(f[j-1]),math.log(f[j])) for j in valid}))
    return kept,dict(excluded)


def options(kept,k,mode,p,eps):
    result=[]
    for c in kept:
        costs=c['costs'][mode];x=math.log(c['col']['beta']);ls=[]
        for s in c['valid_starts']:
            lo,hi=c['intervals'][s]
            ls.append({'lo':lo-p*x-eps,'hi':hi-p*x+eps,'cost':float(costs[s]),'state':s})
        result.append(ls)
    return result


def witness(kept,opts,solution,k,mode,should_compatible,detail):
    if not solution['feasible']:return None
    total=add=delete=changed_cols=0;lower=[];upper=[];changes=[];moves=[]
    for c,o,choice in zip(kept,opts,solution['choices']):
        target=o[choice];s=target['state'];col=c['col']
        w=witness_first_run(col['labels'],s,k,mode)
        assert w['feasible'] and w['cost']==target['cost']
        assert first_run_start(w['labels'],k)==s
        assert w['cost']==np.sum(w['labels']!=col['labels'])
        total+=w['cost'];changed_cols+=int(w['cost']>0)
        for j in w['flipped_indices']:
            before=col['labels'][j];after=int(w['labels'][j]);add+=int(after==1);delete+=int(after==0)
            if detail:changes.append({'column':col['column'],'trial':col['trial'][j], 'force_rank':int(j),
                'beta':col['beta'],'force':col['force'][j],'before':before,'after':after})
        lower.append(target['lo']);upper.append(target['hi'])
        if detail and s!=c['original_start']:
            moves.append({'column':col['column'],'beta':col['beta'],'old_start':c['original_start'],'new_start':s,
                          'old_interval':c['intervals'][c['original_start']],'new_interval':c['intervals'][s]})
    compatible=max(lower)<=min(upper)
    assert compatible==should_compatible and total==solution['cost']
    if should_compatible:assert max(lower)<=solution['intercept']<=min(upper)
    result={'n_flips':total,'additions':add,'deletions':delete,'changed_columns':changed_cols,
            'intersection':[max(lower),min(upper)],'compatible':compatible}
    if detail:result.update(changes=changes,boundary_moves=moves)
    return result


def summarize(records):
    out={}
    for p in [-2.,-1.2]:
        r=[r for r in records if r.get('k')==2 and r.get('epsilon')==.2 and r.get('mode')=='both' and r.get('p')==p and r.get('status')=='scored']
        successful=[x for x in r if x['flip_cost'] is not None]
        restore=[x for x in r if not x['baseline_compatible'] and x['restore']['feasible']]
        reject=[x for x in r if x['baseline_compatible'] and x['exclude']['feasible']]
        out[str(p)]={'n_folders':len(r),'baseline_compatible':sum(x['baseline_compatible'] for x in r),
           'n_finite_restorations':len(restore),'n_finite_exclusions':len(reject),
           'grid_infeasible':sum(not x['restore']['feasible'] for x in r),
           'flip_count_median':float(np.median([x['flip_cost'] for x in successful])) if successful else None,
           'flip_count_range':[min(x['flip_cost'] for x in successful),max(x['flip_cost'] for x in successful)] if successful else None,
           'median_flip_fraction_eligible':float(np.median([x['flip_fraction_eligible'] for x in successful])) if successful else None,
           'restore_cost_median':float(np.median([x['restore']['cost'] for x in restore])) if restore else None,
           'exclude_cost_median':float(np.median([x['exclude']['cost'] for x in reject])) if reject else None}
    return out


def main():
    started=time.monotonic()
    with gzip.open(HERE/'input_columns.json.gz','rt') as f:inputs=json.load(f)
    groups=defaultdict(list)
    for c in inputs['columns']:groups[c['session'],c['folder']].append(c)
    out={'post_hoc':True,'protocol':'PLAN.md, local commit37c1a46','records':[],'eligibility':[],'witnesses':{},
         'interpretation':'Exact operational closed-interval compatibility, not old MLE confidence-test reversal or physical groundtruth.',
         'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'input_columns.json.gz',HERE/'core.py',HERE/'global_profile.py',Path(__file__),HERE/'PLAN.md']}}
    for (session,folder),cols in groups.items():
        for k in [2,1,3]:
            kept,excluded=prepare(cols,k)
            out['eligibility'].append({'session':session,'folder':folder,'k':k,'all_columns':len(cols),'kept_columns':len(kept),'excluded':excluded,
              'all_beta_range':[min(c['beta'] for c in cols),max(c['beta'] for c in cols)],
              'kept_beta_range':[min(c['col']['beta'] for c in kept),max(c['col']['beta'] for c in kept)] if kept else None,
              'n_eligible_windows':sum(len(c['col']['labels']) for c in kept)})
            if len(kept)<2:continue
            modes=['both','delete','add'] if k==2 else ['both']
            for c in kept:c['costs']={m:first_run_costs(c['col']['labels'],k,m)['costs'] for m in modes}
            geometry={}
            for p in [-2.,-1.2]:
                for eps in ([.2,0.,.1,.4] if k==2 else [.2]):
                    for mode in modes:
                        opts=options(kept,k,mode,p,eps)
                        orig=[next(o for o in ls if o['state']==c['original_start']) for ls,c in zip(opts,kept)]
                        base_compatible=max(o['lo'] for o in orig)<=min(o['hi'] for o in orig)
                        restore=optimize_intercept(opts);exclude=optimize_incompatibility(opts)
                        if mode=='both':geometry[p,eps]=(restore['feasible'],exclude['feasible'])
                        detail=k==2 and eps==.2
                        wr=witness(kept,opts,restore,k,mode,True,detail)
                        we=witness(kept,opts,exclude,k,mode,False,detail)
                        assert (restore['feasible'] and restore['cost']==0)==base_compatible
                        assert (exclude['feasible'] and exclude['cost']==0)==(not base_compatible)
                        target=exclude if base_compatible else restore;w=we if base_compatible else wr
                        nwin=sum(len(c['col']['labels']) for c in kept)
                        record={'status':'scored','session':session,'folder':folder,'speed':cols[0]['speed'],'k':k,'p':p,'epsilon':eps,'mode':mode,
                            'n_columns':len(kept),'n_windows':nwin,'baseline_compatible':base_compatible,
                            'restore':{key:value for key,value in restore.items() if key not in ('choices',)},
                            'exclude':{key:value for key,value in exclude.items() if key not in ('choices',)},
                            'restore_witness_summary':{key:value for key,value in (wr or {}).items() if key not in ('changes','boundary_moves')},
                            'exclude_witness_summary':{key:value for key,value in (we or {}).items() if key not in ('changes','boundary_moves')},
                            'flip_target':'exclude' if base_compatible else 'restore','flip_cost':target['cost'] if target['feasible'] else None,
                            'flip_fraction_eligible':target['cost']/nwin if target['feasible'] else None,
                            'unchanged_verdict_radius':target['cost']-1 if target['feasible'] else None,
                            'unrestricted_grid_can_restore':geometry[p,eps][0]}
                        key=f'{session}|{folder}|K{k}|p{p}|eps{eps}|{mode}'
                        if detail:out['witnesses'][key]={'restore':wr,'exclude':we}
                        out['records'].append(record)
        print(session,folder.split('/')[-1],flush=True)
    out['summary']=summarize(out['records']);out['duration_seconds']=time.monotonic()-started
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [clean(v) for v in x]
        if isinstance(x,float) and not math.isfinite(x):return None
        return x
    (HERE/'results.json').write_text(json.dumps(clean(out),indent=2,allow_nan=False)+'\n')
    print(json.dumps(out['summary'],indent=2),flush=True)
    print('seconds',round(out['duration_seconds'],2),flush=True)

if __name__=='__main__':main()
