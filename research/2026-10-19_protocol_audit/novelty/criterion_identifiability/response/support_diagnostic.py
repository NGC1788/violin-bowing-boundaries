"""Analytic support violations of already fitted Gaussian feature surrogates."""
import json,hashlib
import numpy as np
from scipy.special import ndtr
import replay as R

o=json.loads((R.HERE/'results.json').read_text())
res={'post_hoc':True,'no_new_model_fitted':True,'sessions':{},'summary':{}}
for s,e in o['sessions'].items():
    st=R.LS.STEM[s];stem,rep=st if isinstance(st,tuple) else (st,None)
    raw,_=R.T.read2(stem,rep);d=R.LS.build(raw,R.T.BASE)
    res['sessions'][s]={}
    for feature,idx in [('periodicity',4),('second_flyback_ratio',7)]:
        x=np.array([r[idx] for r in raw[0]])
        ef={'observed_positive_range':[float(x[d['H'] & (x>0)].min()),float(x[d['H'] & (x>0)].max())]}
        for m in ('M0','M1'):
            rr=e['features'][feature]['models'][m]['regression'];co=rr['coef']
            valid=d['H'] & np.isfinite(x) & (x>0)
            mu=np.array([co[f'folder_{g}'] for g in d['folder']])+co['log_beta']*np.log(d['beta'])
            if m=='M1':
                valid &= np.isfinite(d['force']) & (d['force']>0)
                mu+=co['log_window_force']*np.log(np.maximum(d['force'],np.finfo(float).tiny))
            mu=mu[valid];sig=rr['sigma']
            if feature=='periodicity':
                low=ndtr((np.log(.8)-mu)/sig);high=ndtr((mu-np.log(1.))/sig)
            else:
                low=np.zeros(len(mu));high=ndtr((mu-np.log(.5))/sig)
            ef[m]={'n':len(mu),'p_below_support_mean':float(low.mean()),'p_above_support_mean':float(high.mean()),
                   'p_outside_mean':float((low+high).mean())}
        res['sessions'][s][feature]=ef
for feature in ('periodicity','second_flyback_ratio'):
    res['summary'][feature]={}
    for m in ('M0','M1'):
        vv=[e[feature][m] for e in res['sessions'].values()]
        res['summary'][feature][m]={'diagram_mean_p_outside':float(np.mean([v['p_outside_mean'] for v in vv])),
            'window_weighted_p_outside':float(sum(v['n']*v['p_outside_mean'] for v in vv)/sum(v['n'] for v in vv)),
            'diagram_range':[min(v['p_outside_mean'] for v in vv),max(v['p_outside_mean'] for v in vv)]}
res['sha256']={str(p.relative_to(R.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R.Path(__file__),R.HERE/'results.json']}
(R.HERE/'support_diagnostic.json').write_text(json.dumps(res,indent=2))
print(json.dumps(res['summary'],indent=2))
