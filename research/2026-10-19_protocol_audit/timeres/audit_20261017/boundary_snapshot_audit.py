"""Descriptive finite-horizon boundary audit; not an equilibrium-law estimator.

Each observation brackets the first K consecutive H-labelled force samples in
one experimental position column. Treating an absent run as right censoring is
an explicit extrapolation assumption, not proof that a physical threshold lies
above the force grid. No windows are treated as independent replicates.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr

import window_audit as audit


def log_interval_probability(lo, hi, mu, sigma):
    """Stable normal interval masses, including left/right censoring."""
    a, b = (lo-mu)/sigma, (hi-mu)/sigma
    # In the positive tail subtract survival probabilities instead of CDFs.
    left = log_ndtr(np.where(a > 0, -a, b))
    right = log_ndtr(np.where(a > 0, -b, a))
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        out = left + np.log(-np.expm1(np.minimum(right-left, 0)))
    return out


def fit(intervals, *, multistart=True):
    if len(intervals) < 8:
        return {'success': False, 'reason': 'fewer than eight intervals'}
    speed = np.log([r['speed'] for r in intervals])
    beta = np.log([r['beta'] for r in intervals])
    lo = np.array([r['lo'] for r in intervals]); hi = np.array([r['hi'] for r in intervals])
    Z = np.column_stack((speed, beta)); center = Z.mean(axis=0)
    X = np.column_stack((np.ones(len(Z)), Z-center))
    if np.linalg.matrix_rank(X) < 3:
        return {'success': False, 'reason': 'rank deficient three-parameter design'}
    proxy = np.where(np.isfinite(lo) & np.isfinite(hi), (np.where(np.isfinite(lo),lo,0)+np.where(np.isfinite(hi),hi,0))/2,
                     np.where(np.isfinite(lo), lo+.2, hi-.2))
    init = np.linalg.lstsq(X, proxy, rcond=None)[0]
    def objective(x):
        value = log_interval_probability(lo, hi, X @ x[:3], np.exp(x[3]))
        return -float(value.sum()) if np.isfinite(value).all() else 1e100
    results = [minimize(objective, np.r_[init, np.log(s)], method='L-BFGS-B',
                bounds=[(None,None)]*3+[(-6.,4.)],
                options={'maxiter': 1500, 'ftol': 1e-11, 'gtol': 1e-6})
               for s in ((.15,.5,1.5) if multistart else (.5,))]
    valid = [r for r in results if r.success and np.isfinite(r.fun) and r.fun < 1e90]
    if not valid:
        return {'success': False, 'reason': 'no optimizer start converged',
                'messages': [str(r.message) for r in results]}
    best = min(valid, key=lambda r:r.fun)
    c = best.x[:3].copy(); c[0] -= center @ c[1:]
    return {'success': True, 'intercept': float(c[0]), 'speed_exp': float(c[1]),
            'beta_exp': float(c[2]), 'sigma': float(np.exp(best.x[3])), 'nll': float(best.fun),
            'sigma_bound_hit': bool(abs(best.x[3]+6)<1e-4 or abs(best.x[3]-4)<1e-4),
            'converged_starts': len(valid), 'starts': len(results),
            'gradient_max': float(np.max(np.abs(best.jac))),
            'objective_spread_converged': float(max(r.fun for r in valid)-best.fun)}


def legacy_levels(data):
    levels = np.empty(len(data['trial']), dtype=int)
    for code in range(len(data['folder_names'])):
        mask = data['folder_codes'] == code
        rounded = np.round(np.log(data['beta'][mask]),4)
        unique, inverse = np.unique(rounded, return_inverse=True)
        gaps = np.diff(unique)
        cut = np.median(gaps[gaps>0])/3
        groups = np.r_[0,np.cumsum(gaps>cut)]
        levels[mask] = groups[inverse]
    return levels


def extract(data, levels, labels, selected, *, run=2):
    index = np.flatnonzero(selected)
    groups = {}
    for i in index:
        groups.setdefault((int(data['folder_codes'][i]),int(levels[i])),[]).append(i)
    out=[]
    for (folder,level), idx in groups.items():
        n_strokes = len(idx)
        idx = [i for i in idx if data['force'][i] > 0]
        if len(idx)<6:
            continue
        idx = np.array(idx)[np.argsort(data['force'][idx],kind='stable')]
        H = labels[idx]=='helmholtz'
        starts = np.flatnonzero(np.convolve(H.astype(int),np.ones(run,dtype=int),'valid') == run)
        first = int(starts[0]) if len(starts) else None
        lf = np.log(data['force'][idx])
        lo = lf[-1] if first is None else -np.inf if first==0 else lf[first-1]
        hi = np.inf if first is None else lf[first]
        out.append({'folder':folder,'level':level,'speed':float(np.median(data['speed'][idx])),
                    'beta':float(np.median(data['beta'][idx])), 'lo':float(lo),'hi':float(hi),
                    'n_forces':len(idx), 'n_strokes':n_strokes, 'H_points':int(H.sum()),
                    'H_blocks':int(np.count_nonzero(H & ~np.r_[False,H[:-1]])),
                    'singletons_before_run':int(H[:first].sum()) if first is not None else int(H.sum()),
                    'first_run':first})
    return out


def summarize(intervals):
    return {'columns':len(intervals), 'by_speed':{
        str(k):{'columns':len(s), 'left':int(sum(np.isneginf(r['lo']) for r in s)),
                'right':int(sum(np.isposinf(r['hi']) for r in s)),
                'finite':int(sum(np.isfinite([r['lo'],r['hi']]).all() for r in s)),
                'forces_min':min(r['n_forces'] for r in s),'forces_max':max(r['n_forces'] for r in s)}
        for k in sorted({r['folder'] for r in intervals})
        for s in [[r for r in intervals if r['folder']==k]]},
        'columns_with_multiple_H_blocks':sum(r['H_blocks']>1 for r in intervals),
        'columns_with_H_before_accepted_run':sum(r['singletons_before_run']>0 for r in intervals)}


def common_complete(intervals, reference, n_folders):
    expected={(r['folder'],r['level']):r['n_strokes'] for r in reference}
    complete=[r for r in intervals if r['n_strokes']==expected.get((r['folder'],r['level']))]
    sets=[{r['level'] for r in complete if r['folder']==f} for f in range(n_folders)]
    common=set.intersection(*sets)
    return [r for r in complete if r['level'] in common]


def bootstrap_columns(intervals, reps, seed=20261017):
    """Same position indices resampled across all speeds; conditional sensitivity only."""
    groups={level:[r for r in intervals if r['level']==level] for level in {r['level'] for r in intervals}}
    keys=sorted(groups); rng=np.random.default_rng(seed); values=[]; failures=0
    for _ in range(reps):
        sampled=[r for key in rng.choice(keys,len(keys),replace=True) for r in groups[int(key)]]
        result=fit(sampled,multistart=False)
        if result['success'] and not result['sigma_bound_hit']:
            values.append([result['speed_exp'], result['beta_exp']])
        else:
            failures+=1
    return {'reps':reps, 'successful':len(values),'failed':failures,
            'speed_percentile_95':np.percentile(np.array(values)[:,0],[2.5,97.5]).tolist() if values else None,
            'beta_percentile_95':np.percentile(np.array(values)[:,1],[2.5,97.5]).tolist() if values else None,
            'scope':'Position-cluster resampling conditional on A2, fixed classifier and grid; not across-specimen confidence.'}


def run(path, output, reps):
    data=audit.load_features(path); levels,grid_info=audit.recover_grid(data)
    old_levels=legacy_levels(data)
    floor,refs=audit.pipeline_nuisance(data)
    settings=[(.1,.8,2),(.2,.8,2),(.4,.8,2),(.2,.7,2),(.2,.9,2),(.2,.8,1),(.2,.8,3)]
    reports=[]; exported=[]
    for w,per,k in settings:
        labels=audit.vector_labels(data,floor,refs,per)
        wm=np.isclose(data['window_s'],w)
        reference=extract(data,levels,labels,wm & np.isclose(data['t'],0),run=k)
        for t in (0.,.1,.2,.3,.4,.6,.8,1.):
            selection=wm & np.isclose(data['t'],t)
            intervals=extract(data,levels,labels,selection,run=k)
            common=common_complete(intervals,reference,len(data['folder_names']))
            item={'window_s':w,'decision_end_s':round(t+w,3),'window_start_s':t,
                  'min_periodicity':per,'force_run':k, 'all_available':summarize(intervals),
                  'complete_common':summarize(common)}
            if len({r['folder'] for r in common})==3 and len(common)>=24:
                item['fit_common']=fit(common)
                if w==.2 and per==.8 and k==2 and t<=.3:
                    item['cluster_sensitivity']=bootstrap_columns(common,reps)
                # Sensitivity removes assumed right-censored absent runs, with selection bias explicitly retained.
                item['fit_detected_only']=fit([r for r in common if np.isfinite(r['hi'])])
            else:
                item['fit_common']={'success':False,'reason':'three speeds with >=8 complete shared positions unavailable'}
            if w==.2 and per==.8 and k==2 and t<=.3:
                old=extract(data,old_levels,labels,selection,run=k)
                item['legacy_fragmented']=summarize(old)
                item['legacy_fragmented_fit']=fit(old)
            for r in intervals:
                exported.append({'window_s':w,'t':t,'min_periodicity':per,'force_run':k,**r})
            reports.append(item)
            if item['fit_common']['success']:
                print(f'w={w}, p={per}, k={k}, t={t}: n={len(common)} speed={item["fit_common"]["speed_exp"]:.4f} beta={item["fit_common"]["beta_exp"]:.4f}',flush=True)
    # Original acquisition windows: independent frozen table with the same operational boundary rule.
    original_path=audit.REPO/'research/2026-10-14/lockbox_A2/regimes.csv.gz'
    with gzip.open(original_path,'rt') as h: original=list(csv.DictReader(h))
    first= data['first']; folder_map={str(x).split('/')[-1]:i for i,x in enumerate(data['folder_names'])}
    od={'force':np.array([float(r['c1_window_mean']) for r in original]),
        'speed':np.array([float(r['c2_window_mean']) for r in original]),
        'beta':np.array([float(r['beta']) for r in original]),
        'folder_codes':np.array([folder_map[r['condition']] for r in original])}
    oi=extract(od,np.array([int(r['beta_level']) for r in original]),np.array([r['label'] for r in original]),np.ones(len(original),bool))
    original_report={'summary':summarize(oi),'fit':fit(oi),'cluster_sensitivity':bootstrap_columns(oi,reps)}
    output.mkdir(exist_ok=True,parents=True)
    result={'status':'exploratory correction of A2; no held-out confirmation or steady-state identification',
      'feature_sha256':audit.sha(path),'source_sha256':audit.sha(__file__),
      'audit_source_sha256':audit.sha(audit.__file__), 'grid':grid_info,'original_acquisition_windows':original_report,
      'assumptions':['First K consecutive H labels defines an operational force-grid boundary.',
        'No detected H run uses a right-censoring convention requiring an unverified physical single-band assumption.',
        'Detected-only fits are selection-biased sensitivity analyses, not a replacement primary estimate.',
        'Measured grid levels are paired by sorted position index across speeds; counts and beta ranges must remain comparable.',
        'All three speeds and complete force support at each shared position are required.',
        'Window start, center and end differ by a constant within each setting; decision availability is at window end.',
        'Overlapping windows are separate descriptive snapshots, never independent replicates.',
        'Window-length comparisons change the finite-window periodicity detector as well as observation horizon.'],
      'snapshots':reports}
    (output/'boundary_snapshots.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with gzip.open(output/'boundary_intervals.csv.gz','wt',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=list(exported[0]));writer.writeheader();writer.writerows(exported)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,default=audit.TIMERES/'features/2024-04-19_TypeA_sample2.csv.gz')
    p.add_argument('--output',type=Path,default=audit.HERE/'results')
    p.add_argument('--bootstrap',type=int,default=200)
    a=p.parse_args();run(a.features,a.output,a.bootstrap)
