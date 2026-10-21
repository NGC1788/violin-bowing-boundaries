"""Independent exhaustive controls audit and deterministic canonical refits."""
from pathlib import Path
import argparse, gzip, hashlib, itertools, json, time
import numpy as np
import oracle as O
from independent_refits import fit_rows

HERE=Path(__file__).resolve().parent;BRIDGE=HERE.parent;PROD=BRIDGE/'production'


def rank_summary(values,target):
    # Literal three-way counting, independently of the producer's vector masks.
    below=tied=above=0
    for x in values:
        if abs(float(x)-target)<=1e-9:tied+=1
        elif x<target:below+=1
        else:above+=1
    return {'n':len(values),'below':below,'tied':tied,'above':above,
            'midrank_percentile':(below+.5*tied)/len(values)}


def run():
    start=time.monotonic()
    columns=json.load(gzip.open(PROD/'input_joint.json.gz','rt'))['columns']
    rows=sorted(O.original_rows_from_columns(columns),key=lambda r:r['id'])
    selected=json.load((PROD/'selected.json').open())['cases']
    jobs={};source_hash={};nfits=0;nsubsets=0;objective_errors=[];refit_errors=[];samples=[]
    for path in sorted((PROD/'controls').glob('*.json')):
        meta=json.load(path.open());zpath=path.with_suffix('.npz');z=np.load(zpath)
        assert not meta['failures']
        rr=[r for r in rows if r['session']==meta['session']]
        eligible=[i for i,r in enumerate(rr) if r['eligible'] and r['group']==meta['group']]
        assert eligible==meta['eligible_indices']
        combinations=np.array(list(itertools.combinations(eligible,meta['k'])))
        assert np.array_equal(combinations,z['combinations'])
        assert len(combinations)==meta['n_controls']
        X=np.array([r['x'] for r in rr]);lo=np.array([r['lower'] for r in rr]);hi=np.array([r['upper'] for r in rr])
        spans=np.array([max(X[c,2])-min(X[c,2]) for c in combinations])
        assert np.array_equal(spans,z['log_beta_span'])
        labels=['free']+[str(p) for p in meta['p']]
        for label in labels:
            params=z[label+'_params'];recorded=z[label+'_nll']
            assert len(params)==len(combinations) and z[label+'_success'].all()
            assert not z[label+'_scale_hit'].any()
            assert ((params[:,-1]>-9+1e-6)&(params[:,-1]<3.5-1e-6)).all()
            if label!='free':assert np.all(params[:,2]==float(label))
            # Every saved objective is independently recalculated, in batches.
            for first in range(0,len(params),256):
                pp=params[first:first+256];cc=combinations[first:first+256]
                mean=pp[:,:3]@X.T;sd=np.exp(pp[:,3,None])
                loss=-O.log_mass((lo[None,:]-mean)/sd,(hi[None,:]-mean)/sd)
                for j,c in enumerate(cc):loss[j,c]=0
                got=loss.sum(axis=1);err=float(max(abs(got-recorded[first:first+256])))
                assert err<1e-7,(meta['number'],label,err)
                objective_errors.append(err)
            for i in sorted(set([0,len(combinations)//2,len(combinations)-1])):
                keep=[r for j,r in enumerate(rr) if j not in set(combinations[i])]
                fit=fit_rows(keep,None if label=='free' else float(label))
                assert fit['success'],(meta['number'],label,i,fit)
                e=float(max(abs(np.array(fit['params'])-params[i])))
                f=float(abs(fit['nll']-recorded[i]))
                assert e<2e-5 and f<1e-7,(meta['number'],label,i,e,f)
                refit_errors.append([e,f]);samples.append({'job':meta['number'],'model':label,'index':i,'fit':fit})
        for p in meta['p']:assert np.all(z[str(p)+'_nll']>=z['free_nll']-1e-7)
        nfits+=len(combinations)*len(labels);nsubsets+=len(combinations)
        jobs[meta['number']]={'meta':meta,'arrays':{key:z[key] for key in z.files}}
        source_hash[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        source_hash[str(zpath)]=hashlib.sha256(zpath.read_bytes()).hexdigest()
    n_ranks=0
    for c in selected:
        ctrl=c['controls'];job=jobs[ctrl['job_number']];z=job['arrays'];meta=job['meta']
        assert meta['group']==c['group'] and meta['k']==c['k']
        index=ctrl['selected_index']; assert set(z['combinations'][index])==set(c['deleted_row_indices'])
        base=c['before']['free']['params'][2];betas=z['free_params'][:,2]
        vals={'abs_delta_beta':abs(betas-base),'toward_target':abs(base-c['p'])-abs(betas-c['p']),
              'T_reduction':c['T_before']-2*(z[str(c['p'])+'_nll']-z['free_nll'])}
        target={'abs_delta_beta':abs(c['delta_beta']),'toward_target':c['toward_target'],
                'T_reduction':c['T_reduction']}
        span=z['log_beta_span'][index];match=np.array([.8*span<=x<=1.2*span for x in z['log_beta_span']])
        assert ctrl['n_span_matched']==int(match.sum())
        for label,mask in [('all',np.ones(len(betas),bool)),('span_matched',match)]:
            if not mask.any():assert ctrl[label] is None;continue
            for key,values in vals.items():
                independently=rank_summary(values[mask],target[key]);given=ctrl[label][key]
                for k,v in independently.items():assert v==given[k],(c['group'],c['p'],label,key,k,v,given[k])
                assert given['control_median']==float(np.median(values[mask]))
                assert given['control_min']==float(min(values[mask])) and given['control_max']==float(max(values[mask]))
                n_ranks+=1
    return {'passed':True,'exhaustive_control_jobs':len(jobs),'exhaustive_control_subsets':nsubsets,
            'independent_objective_recalculations':nfits,'max_objective_error':max(objective_errors),
            'independent_canonical_refits':len(samples),'max_refit_parameter_error':max(e[0] for e in refit_errors),
            'max_refit_nll_error':max(e[1] for e in refit_errors),'exact_rank_comparisons':n_ranks,
            'refit_sample_rule':'first, floor(N/2), last subset of every job, every model; deterministic, not selected by influence rank',
            'refit_samples':samples,'source_sha256':source_hash,'seconds':time.monotonic()-start}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=HERE/'control_audit.json');args=parser.parse_args()
    out=run();args.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ['refit_samples','source_sha256']},indent=2))
