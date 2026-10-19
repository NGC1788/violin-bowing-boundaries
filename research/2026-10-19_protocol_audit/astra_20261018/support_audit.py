"""Post-result covariate-support diagnostic, not a causal matching estimator."""
import json
from pathlib import Path
import numpy as np
from history_prediction_audit import load,HERE,audit

def compare(source,target):
    matched=[];errors=[];total=len(target['y'])
    for speed in range(3):
        for level in range(40):
            for t in (0.,.1,.2,.3):
                a=np.flatnonzero((source['speed']==speed)&(source['level']==level)&np.isclose(source['t'],t))
                b=np.flatnonzero((target['speed']==speed)&(target['level']==level)&np.isclose(target['t'],t))
                distances=np.abs(target['force'][b,None]-source['force'][a][None,:])
                nearest=np.argmin(distances,axis=1)
                for j,i in enumerate(b):
                    k=a[nearest[j]];error=distances[j,nearest[j]]
                    if error<=max(.01,.05*abs(target['force'][i])):
                        matched.append((i,k));errors.append(error)
    i,k=np.array(matched,dtype=int).T
    logacc=np.log(target['accel'][i]/source['accel'][k])
    onset=target['onset'][i]-source['onset'][k];travel=target['travel'][i]-source['travel'][k]
    return {'total_target_windows':total,'matched_windows':len(i),'matched_fraction':len(i)/total,
       'rule':'same sorted beta level, speed and time; nearest current force within max(0.01N,5% absolute target force)',
       'accel_mismatch_factor_over2_fraction':float(np.mean(np.abs(logacc)>np.log(2))),
       'median_accel_ratio_target_over_source':float(np.exp(np.median(logacc))),
       'onset_mismatch_over0p05N_fraction':float(np.mean(np.abs(onset)>.05)),
       'travel_mismatch_over0p02m_fraction':float(np.mean(np.abs(travel)>.02)),
       'median_absolute_force_matching_error_N':float(np.median(errors)),
       'scope':'Descriptive support; reused nearest neighbors, approximate matching, no outcome comparison or causal estimate.'}

if __name__=='__main__':
    ds={k:load(k)[0] for k in ('A1','A2')}
    out={f'{a}_to_{b}':compare(ds[a],ds[b]) for a,b in [('A1','A2'),('A2','A1')]}
    out['source_sha256']=audit.sha(__file__)
    (HERE/'results/support_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
