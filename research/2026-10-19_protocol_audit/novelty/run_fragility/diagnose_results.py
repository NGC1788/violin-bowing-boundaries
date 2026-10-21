"""Descriptive integration and finite-support certificates after primary run.

No additional label optimizer or parameter selection. Support certificates
explain already reported infeasibility. The input columns remain fixed.
"""
import json,gzip,math
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from empirical import prepare
HERE=Path(__file__).resolve().parent


def stats(values):
    x=np.asarray(values,float)
    return {'n':len(x),'median':float(np.median(x)) if len(x) else None,
            'range':[float(x.min()),float(x.max())] if len(x) else None}


def main():
    result=json.loads((HERE/'results.json').read_text())
    with gzip.open(HERE/'input_columns.json.gz','rt') as f:inputs=json.load(f)
    groups=defaultdict(list)
    for c in inputs['columns']:groups[c['session'],c['folder']].append(c)
    out={'post_hoc_diagnostic':True,'support_formula':'eps_min=max(0,(max_j min_s lo_js-p*logbeta_j - min_j max_s hi_js-p*logbeta_j)/2); valid because each column candidate union is contiguous.',
         'support_certificates':[],'sensitivity':{},'primary':{},'warnings':[]}
    for (sess,folder),cols in groups.items():
        kept,_=prepare(cols,2)
        if len(kept)<2:continue
        for p in [-2.,-1.2]:
            intervals=[]
            for c in kept:
                values=sorted(c['intervals'].values())
                for first,second in zip(values,values[1:]):assert first[1]>=second[0]
                shift=p*math.log(c['col']['beta'])
                intervals.append((values[0][0]-shift,values[-1][1]-shift,c['col']['column'],c['col']['beta']))
            lo=max(intervals,key=lambda x:x[0]);hi=min(intervals,key=lambda x:x[1])
            eps=max(0.,(lo[0]-hi[1])/2)
            observed=next(r for r in result['records'] if r['session']==sess and r['folder']==folder and r['k']==2 and r['p']==p and r['epsilon']==.2 and r['mode']=='both')
            assert (eps<=.2)==observed['restore']['feasible']
            out['support_certificates'].append({'session':sess,'folder':folder,'speed':cols[0]['speed'],'p':p,
                'minimum_epsilon_from_grid':eps,'obstructing_lower_column':lo[2],'lower_beta':lo[3],'support_lower':lo[0],
                'obstructing_upper_column':hi[2],'upper_beta':hi[3],'support_upper':hi[1]})
    for k in [1,2,3]:
        for eps in ([0.,.1,.2,.4] if k==2 else [.2]):
            for mode in (['both','delete','add'] if k==2 else ['both']):
                for p in [-2.,-1.2]:
                    rows=[r for r in result['records'] if r['k']==k and r['epsilon']==eps and r['mode']==mode and r['p']==p]
                    good=[r for r in rows if r['restore']['feasible']]
                    out['sensitivity'][f'K{k}|eps{eps}|{mode}|p{p}']={'folders':len(rows),
                        'baseline_compatible':sum(r['baseline_compatible'] for r in rows),
                        'restore_feasible':len(good),'restore_cost':stats([r['restore']['cost'] for r in good]),
                        'restore_fraction':stats([r['restore']['cost']/r['n_windows'] for r in good])}
    rows=[r for r in result['records'] if r['k']==2 and r['epsilon']==.2 and r['mode']=='both']
    for p in [-2.,-1.2]:
        rr=[r for r in rows if r['p']==p and r['restore']['feasible']]
        out['primary'][str(p)]={'changed_columns':stats([r['restore_witness_summary']['changed_columns'] for r in rr]),
          'changed_column_fraction':stats([r['restore_witness_summary']['changed_columns']/r['n_columns'] for r in rr]),
          'additions':stats([r['restore_witness_summary']['additions'] for r in rr]),
          'deletions':stats([r['restore_witness_summary']['deletions'] for r in rr]),
          'endpoint_only_witnesses':sum(r['restore_witness_summary']['intersection'][1]-r['restore_witness_summary']['intersection'][0]<=1e-12 for r in rr),
          'by_speed':{str(v):stats([r['restore']['cost'] for r in rr if abs(r['speed']-v)<.01]) for v in [.05,.1,.2]}}
    pairs=[]
    for r in rows:
        if r['p']==-2 and r['restore']['feasible']:
            other=next(q for q in rows if q['p']==-1.2 and q['session']==r['session'] and q['folder']==r['folder'])
            pairs.append({'session':r['session'],'speed':r['speed'],'p2_cost':r['restore']['cost'],'p12_cost':other['restore']['cost'],
                'difference_p2_minus_p12':r['restore']['cost']-other['restore']['cost']})
    out['paired_supported_folders']={'n':len(pairs),'p2_cheaper':sum(r['difference_p2_minus_p12']<0 for r in pairs),
       'p12_cheaper':sum(r['difference_p2_minus_p12']>0 for r in pairs),'equal':sum(r['difference_p2_minus_p12']==0 for r in pairs),
       'difference':stats([r['difference_p2_minus_p12'] for r in pairs]),'rows':pairs}
    eligible=[e for e in result['eligibility'] if e['k']==2];excluded=Counter()
    for e in eligible:excluded.update(e['excluded'])
    out['eligibility']={'total_columns':sum(e['all_columns'] for e in eligible),'kept_columns':sum(e['kept_columns'] for e in eligible),
       'kept_windows':sum(e['n_eligible_windows'] for e in eligible),'excluded':dict(excluded)}
    (HERE/'diagnostics.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ['eligibility','primary','paired_supported_folders']},indent=2))

if __name__=='__main__':main()
