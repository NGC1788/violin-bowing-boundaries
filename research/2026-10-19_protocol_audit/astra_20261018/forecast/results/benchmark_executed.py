"""Nonoverlapping future-window prediction on two exposed development specimens."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, jaccard_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
sys.path.insert(0, str(BASE / 'timeres/audit_20261017'))
import window_audit as audit

STEMS = {'A1': '2024-03-25_TypeA_sample1', 'A2': '2024-04-19_TypeA_sample2'}
FLOOR = .4063327834010124
ORIGIN = .2
PAST = ((.2, 0.), (.1, 0.), (.1, .05), (.1, .1))
TARGETS = (.25, .30)
SEED = 20261018


def window_indices(data, width, start):
    idx = np.flatnonzero(np.isclose(data['window_s'], width, atol=1e-9, rtol=0)
                        & np.isclose(data['t'], start, atol=1e-9, rtol=0))
    order = np.argsort(data['stroke_index'][idx])
    idx = idx[order]
    if not np.array_equal(data['stroke_index'][idx], np.arange(len(data['first']))):
        raise ValueError(f'Incomplete or duplicate window support: {width}, {start}')
    return idx


def labels(data, idx):
    subset = {k: data[k][idx] for k in ('amplitude_std', 'periodicity', 'f0_hz',
                                       'flybacks_per_period', 'folder')}
    return audit.vector_labels(subset, FLOOR, {str(f): 98.08 for f in data['folder_names']}) == 'helmholtz'


def features(data, idx):
    """Only the explicitly supplied past rows can reach model inputs."""
    force, beta, speed = (data[k][idx] for k in ('force', 'beta', 'speed'))
    if not np.isfinite(np.column_stack((force, beta, speed))).all() or np.any(beta <= 0) or np.any(speed <= 0):
        raise ValueError('Invalid past controls')
    values = [np.arcsinh(force/.1), np.log(beta), np.log(speed), labels(data, idx).astype(float)]
    names = ['force_asinh', 'log_beta', 'log_speed', 'H']
    for field, transform in (
        ('periodicity', lambda x: x),
        ('f0_hz', lambda x: np.arcsinh(x/98.08)),
        ('amplitude_std', lambda x: np.arcsinh(x/.1)),
        ('flybacks_per_period', lambda x: x),
        ('second_flyback_ratio', lambda x: x),
        ('flyback_height', lambda x: np.arcsinh(x/.1)),
    ):
        raw = data[field][idx]
        good = np.isfinite(raw)
        values.extend((transform(np.where(good, raw, 0.)), (~good).astype(float)))
        names.extend((field, field+'_missing'))
    x = np.column_stack(values)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite transformed feature')
    return x, names


def make_dataset(data):
    if any(start+width > ORIGIN+1e-12 for width, start in PAST):
        raise ValueError('Input crosses origin')
    if any(start < ORIGIN for start in TARGETS):
        raise ValueError('Target overlaps input')
    # Build X solely from past; target rows are looked up only after X is fixed.
    past_indices = [window_indices(data, w, t) for w, t in PAST]
    parts = [features(data, idx) for idx in past_indices]
    x = np.column_stack([a for a, _ in parts])
    names = [f'w{w:.2f}_t{t:.2f}_{name}' for (w,t), (_,ns) in zip(PAST,parts) for name in ns]
    snapshot = parts[0][0].shape[1]
    origin_H = labels(data, past_indices[0])
    levels, grid = audit.recover_grid(data)
    nominal = np.array([.05, .10, .20])
    past_speed = data['speed'][past_indices[0]]
    speed = np.abs(past_speed[:,None]-nominal).argmin(1)
    if np.any(np.abs(past_speed-nominal[speed]) > .005):
        raise ValueError('Unknown commanded speed; do not silently map it')
    y = {start: labels(data, window_indices(data, .2, start)) for start in TARGETS}
    dataset = dict(X=x, y=y, H=origin_H, level=levels[past_indices[0]], speed=speed,
                   snapshot=snapshot, stroke=np.arange(len(origin_H)), feature_names=names)
    info = {'strokes': len(origin_H), 'grid': grid, 'input_dimensions': len(names),
            'snapshot_dimensions': snapshot, 'past_windows': PAST,
            'origin_s': ORIGIN, 'target_windows': [[t,t+.2] for t in TARGETS],
            'minimum_input_target_gap_s': min(TARGETS)-ORIGIN,
            'past_H_count': int(origin_H.sum()), 'coverage': []}
    for start, yy in y.items():
        for code in range(3):
            m = speed == code
            info['coverage'].append({'start': start, 'speed': float(nominal[code]),
                'n': int(m.sum()), 'future_H': int(yy[m].sum()),
                'nonH_to_H': int((m & ~origin_H & yy).sum()),
                'H_to_nonH': int((m & origin_H & ~yy).sum())})
    return dataset, info


def markov_prob(train_H, train_y, test_H, train_speed=None, test_speed=None):
    p = np.empty(len(test_H))
    ts = np.zeros(len(train_H),int) if train_speed is None else train_speed
    vs = np.zeros(len(test_H),int) if test_speed is None else test_speed
    if not set(vs).issubset(set(ts)):
        raise ValueError('No source speed support')
    for code in np.unique(vs):
        for h in (False, True):
            m = (ts==code) & (train_H==h)
            p[(vs==code) & (test_H==h)] = (train_y[m].sum()+.5)/(m.sum()+1.)
    return p


def estimator(name):
    if name.startswith('logistic'):
        return make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(),
                             LogisticRegression(C=1,max_iter=3000,solver='lbfgs'))
    return HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=15,learning_rate=.07,
            min_samples_leaf=30,l2_regularization=1,early_stopping=False,random_state=SEED)


def fitted_probability(name, x, y, target, artifact=None):
    if len(np.unique(y)) < 2:
        probability = (y.sum()+.5)/(len(y)+1.)
        if artifact is not None: joblib.dump({'constant_probability': probability},artifact)
        return np.full(len(target),probability)
    model = estimator(name).fit(x,y)
    if name.startswith('logistic') and int(model[-1].n_iter_.max()) >= 3000:
        raise RuntimeError('Logistic failed to converge; do not hide it')
    if artifact is not None: joblib.dump(model,artifact)
    return model.predict_proba(target)[:,1]


def predict(name, a, b, start, artifact_dir=None):
    yy = a['y'][start]
    if name == 'fixed_persistence': return np.where(b['H'],.98,.02)
    if name == 'markov': return markov_prob(a['H'],yy,b['H'])
    if name == 'markov_speed': return markov_prob(a['H'],yy,b['H'],a['speed'],b['speed'])
    cols = slice(0,3) if name=='hgb_controls' else (slice(0,a['snapshot']) if name.endswith('snapshot') else slice(None))
    ax, bx = a['X'][:,cols], b['X'][:,cols]
    if name == 'hgb_history_two_heads':
        p = np.empty(len(bx))
        for h in (False,True):
            m,n = a['H']==h,b['H']==h
            if not n.any(): continue
            artifact = None if artifact_dir is None else artifact_dir/f'{name}_{int(h)}.joblib'
            p[n] = fitted_probability('hgb', ax[m], yy[m], bx[n],artifact)
        return p
    artifact = None if artifact_dir is None else artifact_dir/f'{name}.joblib'
    return fitted_probability(name,ax,yy,bx,artifact)


def metrics(y,p):
    if not len(y): return {'n':0}
    return {'n':len(y),'H':int(y.sum()),'mean_probability':float(p.mean()),
            'log_loss':float(log_loss(y,p,labels=[False,True])),
            'brier':float(brier_score_loss(y,p)),
            'IoU':float(jaccard_score(y,p>=.5,zero_division=0)),
            'balanced_accuracy':float(balanced_accuracy_score(y,p>=.5)) if len(np.unique(y))==2 else None}


def loss_vector(y,p):
    y = np.asarray(y,dtype=float)
    p = np.clip(p,1e-15,1-1e-15)
    return -y*np.log(p)-(1-y)*np.log1p(-p)


def paired(y,pbase,pcandidate,level):
    delta = loss_vector(y,pcandidate)-loss_vector(y,pbase)
    groups = np.unique(level)
    sums = np.array([delta[level==g].sum() for g in groups])
    counts = np.array([(level==g).sum() for g in groups])
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0,len(groups),(2000,len(groups)))
    boot = sums[draws].sum(1)/counts[draws].sum(1)
    return {'candidate_minus_baseline_log_loss':float(delta.mean()),
            'position_cluster_percentile95':np.percentile(boot,[2.5,97.5]).tolist(),
            'scope':'Conditional on these measured specimens/sessions; not population inference.'}


MODELS = ('fixed_persistence','markov','markov_speed','hgb_controls','logistic_snapshot',
          'logistic_history','hgb_snapshot','hgb_history','hgb_history_two_heads')


def main():
    out = HERE/'results'
    out.mkdir(exist_ok=True)
    if (out/'results.json').exists():
        raise FileExistsError('Preserve completed benchmark; use a versioned directory for changes')
    report = {'scope':'Exposed A1/A2 development; no superiority or novelty claim',
              'protocol_sha256':audit.sha(HERE/'PROTOCOL.md'),
              'code_sha256':audit.sha(__file__), 'inputs':{}, 'fits':[], 'paired':[]}
    ds = {}
    for tag,stem in STEMS.items():
        path = BASE/f'timeres/features/{stem}.csv.gz'
        data = audit.load_features(path)
        if len(data['first']) != 6000 or len(data['folder_names']) != 3:
            raise ValueError('Acquisition inventory requires 6000 strokes in 3 folders')
        for folder in data['folder_names']:
            trial_ids = np.unique(data['trial'][data['folder']==folder])
            if not np.array_equal(trial_ids,np.arange(1,2001)):
                raise ValueError('Acquisition inventory requires trial IDs 1..2000 per folder')
        ds[tag], report['inputs'][tag] = make_dataset(data)
        if not np.array_equal(np.bincount(ds[tag]['speed'],minlength=3),[2000,2000,2000]):
            raise ValueError('Expected 2000 strokes per commanded speed')
        report['inputs'][tag]['sha256'] = audit.sha(path)
        print(f'{tag} verified: {len(ds[tag]["H"])} complete strokes',flush=True)
    (out/'prefit_manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    arrays = {}
    # All fitted predictions are saved before evaluating either direction.
    for train,test in (('A1','A2'),('A2','A1')):
        a,b = ds[train],ds[test]
        for start in TARGETS:
            artifact_dir = out/'models'/f'{train}_to_{test}_{start:.2f}'
            artifact_dir.mkdir(parents=True,exist_ok=True)
            for name in MODELS:
                p = predict(name,a,b,start,artifact_dir)
                if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
                    raise ValueError('Invalid predictions')
                arrays[f'{train}_to_{test}_{start:.2f}_{name}'] = p
                print('fitted',train,test,start,name,flush=True)
    for tag,d in ds.items():
        for key in ('H','level','speed','stroke'): arrays[tag+'_'+key] = d[key]
        for start in TARGETS: arrays[f'{tag}_y_{start:.2f}'] = d['y'][start]
    np.savez_compressed(out/'predictions.npz',**arrays)
    report['predictions_sha256_before_evaluation'] = audit.sha(out/'predictions.npz')
    report['model_sha256'] = {str(p.relative_to(out)):audit.sha(p) for p in sorted((out/'models').rglob('*.joblib'))}
    (out/'prescore_manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for train,test in (('A1','A2'),('A2','A1')):
        b=ds[test]
        for start in TARGETS:
            predictions={name:arrays[f'{train}_to_{test}_{start:.2f}_{name}'] for name in MODELS}
            for name,p in predictions.items():
                r = {'train':train,'test':test,'target_start':start,'model':name,
                     'metrics':metrics(b['y'][start],p),'by_speed':[],'by_past_H':[]}
                for speed in range(3):
                    m = b['speed']==speed
                    r['by_speed'].append({'speed_index':speed,**metrics(b['y'][start][m],p[m])})
                for h in (False,True):
                    m = b['H']==h
                    r['by_past_H'].append({'past_H':h,**metrics(b['y'][start][m],p[m])})
                report['fits'].append(r)
                print(train,test,start,name,round(r['metrics']['log_loss'],6),flush=True)
            for baseline,candidate in (('hgb_snapshot','hgb_history'),('logistic_snapshot','logistic_history'),
                 ('markov_speed','hgb_snapshot'),('markov_speed','hgb_history'),('hgb_history','hgb_history_two_heads')):
                report['paired'].append({'train':train,'test':test,'target_start':start,
                     'baseline':baseline,'candidate':candidate,
                     **paired(b['y'][start],predictions[baseline],predictions[candidate],b['level'])})
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__': main()
