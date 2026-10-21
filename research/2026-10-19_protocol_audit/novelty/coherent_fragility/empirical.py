"""Exact coherent-rule profiles and declared summaries on exposed derived data."""
from pathlib import Path
import gzip,json,math,time,hashlib
import numpy as np
from threshold_profile import analyze_thresholds,first_run_start
HERE=Path(__file__).resolve().parent
SLOPES=(-2.,-1.2);EPS=(0.,.1,.2,.4)
RULES=[('periodicity_primary','periodicity','ge',(.7,.9),.8),
       ('ratio_primary','second_flyback_ratio','lt',(.3,.7),.5),
       ('amplitude_stress','amplitude','ge',(0.,math.inf),0.),
       ('height_stress','flyback_height','ge',(0.,math.inf),0.),
       ('periodicity_extended','periodicity','ge',(0.,1.),.8),
       ('ratio_extended','second_flyback_ratio','lt',(0.,1.),.5)]

def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not math.isfinite(x):return 'Infinity' if x>0 else '-Infinity' if x<0 else None
    return x

def load_columns(feature):
    with gzip.open(HERE/'input_features.json.gz','rt') as f:raw=json.load(f)
    out=[]
    for c in raw['columns']:
        s=first_run_start(c['labels'],2);f=c['force']
        eligible=s is not None and s>0 and 0<f[s-1]<f[s]
        out.append(dict(id=c['id'],group=c['group'],beta=c['beta'],force=f,baseline=c['labels'],eligible=bool(eligible),**c['features'][feature]))
    return raw,out

def atom(p,i,base):
    a=p['atoms'];lo=float(a['lo'][i]);hi=float(a['hi'][i]);lc=bool(a['lo_closed'][i])
    if (lo<base or (lc and lo==base)) and base<=hi:t=base
    elif base>hi:t=hi
    else:t=lo if lc else float(np.nextafter(lo,math.inf))
    if not math.isfinite(t):
        finite=max([0.]+[float(v) for v in a['lo'] if math.isfinite(v)])
        t=float(np.nextafter(finite,math.inf))
    assert (lo<t or (lc and lo==t)) and t<=hi
    return dict(index=int(i),lo=lo,hi=hi,lo_closed=lc,hi_closed=True,threshold=t)

def nearest(p,base):
    a=p['atoms'];return np.maximum(a['lo']-base,0)+np.maximum(base-a['hi'],0)

def choose(indices,cost,dist,secondary=None):
    if not len(indices):return None
    secondary=np.zeros(len(cost),dtype=int) if secondary is None else secondary
    return int(min(indices,key=lambda i:(int(cost[i]),int(secondary[i]),float(dist[i]),int(i))))

def regions(p,mask):
    ii=np.flatnonzero(mask)
    if not len(ii):return []
    breaks=np.flatnonzero(np.diff(ii)>1);starts=np.r_[0,breaks+1];ends=np.r_[breaks,len(ii)-1]
    return [dict(lo=float(p['atoms']['lo'][ii[a]]),hi=float(p['atoms']['hi'][ii[b]]),lo_closed=bool(p['atoms']['lo_closed'][ii[a]]),hi_closed=True)
            for a,b in zip(starts,ends)]

def global_choice(p,i,base,si,ei):
    ok=p['compatible'][i,:,si,ei];bad=p['invalid_eligible_columns'][i]>0
    return dict(atom=atom(p,i,base),n_compatible=int(ok.sum()),compatible_groups=[g for g,v in zip(p['groups'],ok) if v],
       n_invalid_groups=int(bad.sum()),invalid_groups=[g for g,v in zip(p['groups'],bad) if v],
       all_edits=int(p['n_all_edits'][i].sum()),eligible_edits=int(p['n_eligible_edits'][i].sum()),
       excluded_column_edits=int((p['n_all_edits'][i]-p['n_eligible_edits'][i]).sum()),
       changed_eligible_columns=int(p['n_changed_columns'][i].sum()),invalid_eligible_columns=int(p['invalid_eligible_columns'][i].sum()))

def main():
    started=time.monotonic();output={'post_hoc':True,'protocol_commit':'e94d8e4','addendum_commit':'61b5304','rules':[],
       'interpretation':'Same scalar rule applied to all 78000 windows. Local optima are oracles. All-column operational compatibility, not original MLE, true-label correction, or generalization.'}
    refs=json.loads((HERE/'arbitrary_reference.json').read_text())['records'];ref={(r['session']+'|'+r['folder'],r['p'],r['epsilon'],r['mode']):r for r in refs}
    for name,feature,relation,domain,base in RULES:
        raw,cols=load_columns(feature);p=analyze_thresholds(cols,relation,domain,SLOPES,EPS,2)
        assert int(p['group_metadata']['n_all_windows'].sum())==78000 and int(p['group_metadata']['n_eligible_windows'].sum())==56600
        base_row=np.flatnonzero((p['atoms']['lo']<=base)&(base<=p['atoms']['hi'])&((p['atoms']['lo']<base)|p['atoms']['lo_closed']))
        assert len(base_row)==1 and p['n_all_edits'][base_row[0]].sum()==0
        dist=nearest(p,base);total=p['n_all_edits'].sum(axis=1);nwin=int(p['group_metadata']['n_all_windows'].sum())
        rule=dict(name=name,feature=feature,relation=relation,domain=domain,base_threshold=base,n_patterns=len(total),
                  n_transition_windows=p['n_transition_windows'],analyses=[],baseline_row=int(base_row[0]))
        for si,slope in enumerate(SLOPES):
            for ei,epsilon in enumerate(EPS):
                ok=p['compatible'][:,:,si,ei];cov=ok.sum(axis=1);m=int(cov.max())
                best=choose(np.flatnonzero(cov==m),total,dist)
                global_best=global_choice(p,best,base,si,ei)
                budgets=[]
                for frac in (.01,.05,.1,.15,.3,1.):
                    ii=np.flatnonzero(total<=math.floor(nwin*frac));mc=int(cov[ii].max());i=choose(ii[cov[ii]==mc],total,dist)
                    budgets.append(dict(budget_fraction=frac,**global_choice(p,i,base,si,ei)))
                local=[]
                for g,group in enumerate(p['groups']):
                    ii=np.flatnonzero(ok[:,g]);i=choose(ii,p['n_eligible_edits'][:,g],dist,total)
                    r=ref[group,slope,epsilon,'both'];bound=r['restore']['cost'] if r['restore']['feasible'] else None
                    row={'group':group,'baseline_compatible':bool(ok[base_row[0],g]),'n_eligible_windows':int(p['group_metadata']['n_eligible_windows'][g]),
                         'n_eligible_columns':int(p['group_metadata']['n_eligible_columns'][g]),'feasible':i is not None,'arbitrary_minimum':bound,
                         'feasible_threshold_regions':regions(p,ok[:,g])}
                    if i is not None:
                        cost=int(p['n_eligible_edits'][i,g]);assert bound is not None and cost>=bound,(name,group,cost,bound)
                        row.update(atom=atom(p,i,base),eligible_edits=cost,folder_all_edits=int(p['n_all_edits'][i,g]),
                            global_all_edits=int(total[i]),changed_eligible_columns=int(p['n_changed_columns'][i,g]),excess_over_arbitrary=cost-bound)
                        if feature in ('amplitude','flyback_height'):
                            dr=ref[group,slope,epsilon,'delete'];assert dr['restore']['feasible'] and cost>=dr['restore']['cost']
                            row['arbitrary_delete_minimum']=dr['restore']['cost']
                    local.append(row)
                rule['analyses'].append(dict(p=slope,epsilon=epsilon,baseline_compatible=int(ok[base_row[0]].sum()),
                    local_feasible=sum(x['feasible'] for x in local),global_best=global_best,budget_frontier=budgets,local=local,
                    all_folders_jointly_feasible=bool(m==len(p['groups']))))
        # Complete profiles, not just winning cases. Numerical arrays only.
        arrays={k:v for k,v in p.items() if isinstance(v,np.ndarray)}
        arrays.update({'atom_'+k:v for k,v in p['atoms'].items()})
        np.savez_compressed(HERE/'profiles'/f'{name}.npz',**arrays)
        rule['groups']=p['groups'];rule['profile_file']=f'profiles/{name}.npz';output['rules'].append(rule)
        print(name,'patterns',len(total),[(a['p'],a['local_feasible'],a['global_best']['n_compatible'],a['global_best']['all_edits']) for a in rule['analyses'] if a['epsilon']==.2],flush=True)
        del p
    output['duration_seconds']=time.monotonic()-started
    output['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'PLAN.md',HERE/'ADDENDUM_ATTAINABILITY.md',HERE/'threshold_profile.py',Path(__file__),HERE/'input_features.json.gz',HERE/'arbitrary_reference.json']}
    (HERE/'results.json').write_text(json.dumps(clean(output),indent=2,allow_nan=False)+'\n');print('seconds',output['duration_seconds'],flush=True)
if __name__=='__main__':
    (HERE/'profiles').mkdir(exist_ok=True);main()
