"""Declared record-envelope diagnostic plus explicitly posthoc zero-loss coverage."""
from pathlib import Path
import json,sys,math,hashlib
import numpy as np
from empirical import RULES,SLOPES,EPS,load_columns,clean
sys.path.insert(0,str(Path(__file__).resolve().parent/'literature'))
from attainability import first_run_threshold_intervals
HERE=Path(__file__).resolve().parent

def main():
    result=json.loads((HERE/'results.json').read_text())
    out={'attainability_plan':'ADDENDUM_ATTAINABILITY.md, before empirical costs',
         'zero_invalid_diagnostic':'Post hoc after main results: require all 1132 originally eligible boundaries across all39folders to remain finitepositive. No parameter-family/range change.',
         'rules':[]}
    for spec,saved in zip(RULES,result['rules']):
        name,feat,relation,domain,base=spec;raw,cols=load_columns(feat)
        stats=[]
        for c in cols:
            if not c['eligible']:continue
            f=c['force'];valid={s for s in range(1,len(f)-1) if 0<f[s-1]<f[s]}
            values=np.asarray(c['values'],float)
            states=first_run_threshold_intervals(values,c['guard'],c['fixed'],2,relation,domain)
            reachable={x['start'] for x in states}&valid
            assert reachable and reachable<=valid # baseline must be in domain
            stats.append({'id':c['id'],'group':c['group'],'arbitrary_valid_starts':len(valid),'reachable_starts':len(reachable),
                          'fraction':len(reachable)/len(valid)})
        with np.load(HERE/saved['profile_file'],allow_pickle=False) as z:
            no_loss=(z['invalid_eligible_columns'].sum(axis=1)==0)
            assessments=[]
            for si,p in enumerate(SLOPES):
                for ei,eps in enumerate(EPS):
                    cov=z['compatible'][:,:,si,ei].sum(axis=1);eligible=np.flatnonzero(no_loss)
                    maxcov=int(cov[eligible].max());ii=eligible[cov[eligible]==maxcov]
                    total=z['n_all_edits'].sum(axis=1);i=int(ii[np.argmin(total[ii])])
                    assessments.append(dict(p=p,epsilon=eps,max_compatible_folders_without_any_invalid=maxcov,
                        minimum_all_edits_at_that_coverage=int(total[i]),n_patterns_without_any_invalid=int(no_loss.sum())))
        out['rules'].append(dict(name=name,n_columns=len(stats),total_arbitrary_starts=sum(x['arbitrary_valid_starts'] for x in stats),
            total_reachable_starts=sum(x['reachable_starts'] for x in stats),median_fraction=float(np.median([x['fraction'] for x in stats])),
            fraction_range=[min(x['fraction'] for x in stats),max(x['fraction'] for x in stats)],
            exactly_one_reachable=sum(x['reachable_starts']==1 for x in stats),columns=stats,zero_invalid=assessments))
        print(name,'median reachable fraction',out['rules'][-1]['median_fraction'],'no-loss',[(x['p'],x['max_compatible_folders_without_any_invalid']) for x in assessments if x['epsilon']==.2],flush=True)
    out['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'results.json',HERE/'literature/attainability.py']}
    (HERE/'diagnostics.json').write_text(json.dumps(clean(out),indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
