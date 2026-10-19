"""Exploratory, specimen-separated audit of instantaneous vs protocol-history inputs."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import warnings
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import log_loss, brier_score_loss, jaccard_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures,StandardScaler

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(BASE/'timeres/audit_20261017'))
import window_audit as audit
STEMS={'A1':'2024-03-25_TypeA_sample1','A2':'2024-04-19_TypeA_sample2'}
FEATURES=['current_force_asinh','log_beta','log_speed','plateau_end_s',
          'onset_force_asinh','log_accel','travel_center_m']
CONFIGS={'instant':[0,1,2],'clock':[0,1,2,3],'history':[0,1,2,3,4,5,6],
         'no_onset':[0,1,2,3,5,6],'no_accel':[0,1,2,3,4,6],'no_travel':[0,1,2,3,4,5]}


def unique_map(keys):
    mapping={k:i for i,k in enumerate(keys)}
    if len(mapping)!=len(keys):raise ValueError('duplicate join keys')
    return mapping


def load(tag):
    feature_path=BASE/f'timeres/features/{STEMS[tag]}.csv.gz'
    ramp_path=BASE/('claude_extra/ramp_meta_A2.csv' if tag=='A2' else f'claude_extra/ramp/ramp_meta_{STEMS[tag]}.csv')
    block_path=BASE/f'claude_extra/blocks/blocks_{tag}.npz'
    data=audit.load_features(feature_path);levels,grid=audit.recover_grid(data)
    with ramp_path.open() as h:meta=list(csv.DictReader(h))
    if any(r['error'] for r in meta):raise ValueError('ramp metadata errors')
    lookup=unique_map([(r['folder'],int(r['trial'])) for r in meta])
    blocks=np.load(block_path,allow_pickle=False)
    leaf_map=unique_map([str(f).split('/')[-1] for f in data['folder_names']])
    blockkeys=[(str(data['folder_names'][leaf_map[str(f)]]),int(t)) for f,t in zip(blocks['folder'],blocks['trial'])]
    blookup=unique_map(blockkeys)
    full_keys=[(str(data['folder'][i]),int(data['trial'][i])) for i in data['first']]
    if set(full_keys)!=set(lookup) or set(full_keys)!=set(blookup):raise ValueError('stroke coverage differs between inputs')
    for i,key in zip(data['first'],full_keys):
        if abs(data['beta'][i]-float(meta[lookup[key]]['beta']))>1e-12 or abs(data['beta'][i]-blocks['beta'][blookup[key]])>1e-12:
            raise ValueError('beta mismatch in independent join')
    selected=np.flatnonzero(np.isclose(data['window_s'],.2) & np.isin(np.round(data['t'],3),[0.,.1,.2,.3]))
    counts=np.bincount(data['stroke_index'][selected],minlength=len(full_keys))
    if not (counts==4).all():raise ValueError('not every stroke has four common-horizon windows')
    subset={k:data[k][selected] for k in ('amplitude_std','periodicity','f0_hz','flybacks_per_period','folder')}
    labels=audit.vector_labels(subset,.4063327834010124,{str(f):98.08 for f in data['folder_names']})
    floor,refs=audit.pipeline_nuisance(data)
    native=audit.vector_labels(subset,floor,refs)
    onset=np.empty(len(selected));acc=np.empty_like(onset);travel=np.empty_like(onset)
    fields=list(blocks['fields']);pos=blocks['data'][:,fields.index('position'),:]
    block_ends=(np.arange(pos.shape[1])+1)*float(blocks['block_s'])
    for j,i in enumerate(selected):
        key=(str(data['folder'][i]),int(data['trial'][i]));r=meta[lookup[key]];row=pos[blookup[key]]
        onset[j]=float(r['force_at_onset']);acc[j]=float(r['accel'])
        when=float(r['plateau_on_s'])+float(data['t'][i])+.1
        origin=float(r['onset_s']);valid=np.isfinite(row)
        if when>block_ends[valid][-1] or origin<0:raise ValueError('block timeline does not cover feature center')
        # np.interp uses neighboring block endpoints; at most .05 s after center, before window end.
        travel[j]=np.interp(when,np.r_[0,block_ends[valid]],np.r_[0,row[valid]])-np.interp(origin,np.r_[0,block_ends[valid]],np.r_[0,row[valid]])
    X=np.column_stack((np.arcsinh(data['force'][selected]/.1),np.log(data['beta'][selected]),
        np.log(data['speed'][selected]),data['t'][selected]+.2,np.arcsinh(onset/.1),np.log(acc),travel))
    if not np.isfinite(X).all():raise ValueError('nonfinite model inputs')
    d={'X':X,'y':labels=='helmholtz','level':levels[selected],
       'speed':data['folder_codes'][selected], 't':data['t'][selected],
       'trial':data['trial'][selected], 'stroke':data['stroke_index'][selected],
       'force':data['force'][selected], 'onset':onset,'accel':acc,'travel':travel}
    report={'stroke_count':len(full_keys),'rows':len(selected),'grid':grid,
            'H_count':int(d['y'].sum()),'H_changes_global_vs_archive_reference':int(np.count_nonzero((native=='helmholtz')!=d['y'])),
            'files_sha256':{str(p.relative_to(BASE)):audit.sha(p) for p in (feature_path,ramp_path,block_path)}}
    # Rank from measured onset force within each position and speed: do not assume A1 trial order.
    onset_rank=np.zeros(len(data['first']),int)
    for code in range(3):
        for level in range(40):
            idx=data['first'][(data['folder_codes'][data['first']]==code) & (levels[data['first']]==level)]
            idx=idx[np.argsort([float(meta[lookup[(str(data['folder'][i]),int(data['trial'][i]))]]['force_at_onset']) for i in idx])]
            onset_rank[data['stroke_index'][idx]]=np.arange(1,len(idx)+1)
    report['force_change_by_speed']=[]
    for code in range(3):
        si=data['first'][data['folder_codes'][data['first']]==code]
        first_mask=np.isclose(data['window_s'],.2)&np.isclose(data['t'],0)&(data['folder_codes']==code)
        wi=np.flatnonzero(np.isclose(data['window_s'],.2)&(data['folder_codes']==code))
        # Sort by stroke then time and take last; excludes no rows by force or label.
        ordered=wi[np.lexsort((data['t'][wi],data['stroke_index'][wi]))]
        last=ordered[np.r_[data['stroke_index'][ordered][1:]!=data['stroke_index'][ordered][:-1],True]]
        start_by_stroke={int(data['stroke_index'][i]):float(data['force'][i]) for i in np.flatnonzero(first_mask)}
        for rlo,rhi in ((1,20),(21,35),(36,50)):
            take=[i for i in last if rlo<=onset_rank[data['stroke_index'][i]]<=rhi]
            fon=np.array([float(meta[lookup[(str(data['folder'][i]),int(data['trial'][i]))]]['force_at_onset']) for i in take])
            ff=np.array([start_by_stroke[int(data['stroke_index'][i])] for i in take]);fl=data['force'][take]
            report['force_change_by_speed'].append({'speed_index':code,'onset_rank':[rlo,rhi],'n':len(take),
              'onset_median_N':float(np.median(fon)),'first_window_median_N':float(np.median(ff)),
              'last_window_median_N':float(np.median(fl)),'fraction_last_below_half_onset':float(np.mean(fl<.5*fon))})
    report['acceleration_support']=[]
    for code in range(3):
        m=(d['speed']==code)&(d['t']==0)&(onset_rank[d['stroke']]<=20)
        report['acceleration_support'].append({'speed_index':code,'low_onset_rank_n':int(m.sum()),
            'accel_quantiles_5_50_95':np.percentile(acc[m],[5,50,95]).tolist()})
    # A predictive design audit, not a physical model. Source-conditioned collinearity.
    m=d['t']==0
    design=make_pipeline(PolynomialFeatures(2,include_bias=False),StandardScaler(),LinearRegression())
    controls=np.column_stack([X[m,1],X[m,2],X[m,4]])
    design.fit(controls,np.log(acc[m]))
    report['in_sample_R2_accel_from_beta_speed_onset_poly2']=float(design.score(controls,np.log(acc[m])))
    return d,report


def metrics(y,p):
    return {'n':len(y),'log_loss':float(log_loss(y,p,labels=[False,True])),
        'brier':float(brier_score_loss(y,p)),'IoU':float(jaccard_score(y,p>=.5,zero_division=0)),
        'balanced_accuracy':float(balanced_accuracy_score(y,p>=.5))}


def paired_cluster(y,p0,p1,level):
    y=np.asarray(y,dtype=float)
    eps=1e-12
    def loss(p):
        p=np.clip(p,eps,1-eps);return -y*np.log(p)-(1-y)*np.log1p(-p)
    delta=loss(p1)-loss(p0)
    groups=[delta[level==k] for k in np.unique(level)]
    sums=np.array([g.sum() for g in groups]);counts=np.array([len(g) for g in groups])
    rng=np.random.default_rng(20261018);draws=rng.integers(0,len(groups),(2000,len(groups)))
    boots=sums[draws].sum(1)/counts[draws].sum(1)
    return {'history_minus_clock_log_loss':float(delta.mean()),'position_cluster_percentile95':np.percentile(boots,[2.5,97.5]).tolist(),
            'scope':'Conditional position-cluster sensitivity; only two exposed specimens; not population confidence.'}


def model(learner):
    if learner=='logistic_poly2':
        return make_pipeline(PolynomialFeatures(2,include_bias=False),StandardScaler(),
                             LogisticRegression(C=1,max_iter=3000,solver='lbfgs'))
    return HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=15,learning_rate=.07,
        min_samples_leaf=30,l2_regularization=1,early_stopping=False,random_state=20261018)


def main():
    output=HERE/'results';output.mkdir(exist_ok=True)
    ds={};report={'scope':'Exploratory predictive audit, not a novel method or causal test','inputs':{},'fits':[]}
    for tag in ('A1','A2'):
        ds[tag],report['inputs'][tag]=load(tag);print(f'{tag}: joins and support validated',flush=True)
    predictions={}
    for train,test in (('A1','A2'),('A2','A1')):
        a,b=ds[train],ds[test]
        for learner in ('logistic_poly2','hist_gradient_boosting'):
            for name,cols in CONFIGS.items():
                estimator=model(learner);estimator.fit(a['X'][:,cols],a['y'])
                p=estimator.predict_proba(b['X'][:,cols])[:,1]
                predictions[f'{train}_to_{test}_{learner}_{name}']=p
                item={'train':train,'test':test,'learner':learner,'inputs':name,'metrics':metrics(b['y'],p),'by_speed_time':[]}
                for speed in range(3):
                    for t in (0.,.1,.2,.3):
                        m=(b['speed']==speed)&np.isclose(b['t'],t)
                        item['by_speed_time'].append({'speed_index':speed,'t':t,**metrics(b['y'][m],p[m])})
                report['fits'].append(item);print(train,test,learner,name,item['metrics'],flush=True)
            p0=predictions[f'{train}_to_{test}_{learner}_clock'];p1=predictions[f'{train}_to_{test}_{learner}_history']
            report.setdefault('paired_history_comparisons',[]).append({'train':train,'test':test,'learner':learner,**paired_cluster(b['y'],p0,p1,b['level'])})
    report['source_sha256']=audit.sha(__file__);report['analysis_plan_sha256']=audit.sha(HERE/'ANALYSIS_PLAN.md')
    for tag,d in ds.items():
        for k in ('y','level','speed','t','trial','stroke'):predictions[f'{tag}_{k}']=d[k]
    np.savez_compressed(output/'history_predictions.npz',**predictions)
    (output/'history_prediction_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
