#!/usr/bin/env python3
"""Post-hoc test of position-dependent width transport; no novelty claim.

Changes only k(b)=k0+k1*b from the previously fixed global k. All source
selection precedes target scoring. Prospective A2 is never loaded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, exprel
from threadpoolctl import threadpool_limits

import affine_transport as af
import benchmark_data as bd
from benchmark_sets import evaluate_panels, metrics
import multispeed_scaling as ms
from set_envelope import label_family

BOUNDS = af.BOUNDS + [(-2., 2.)]


def transport(b, u, t, a0, a1, k0, k1, *, gradient=False):
    b, u, t = np.broadcast_arrays(np.asarray(b, float), np.asarray(u, float), np.asarray(t, float))
    k = k0 + k1 * b
    q = k * t
    e = np.exp(-q)
    j = t * exprel(-q)
    a = a0 + a1 * b
    z = e * u - a * j
    if not gradient:
        return z
    small = abs(q) < 1e-4
    dj = np.empty_like(t)
    ks, ts = k[small], t[small]
    dj[small] = -.5 * ts**2 + ks * ts**3 / 3 - ks**2 * ts**4 / 8 + ks**3 * ts**5 / 30
    dj[~small] = (q[~small] * e[~small] + np.expm1(-q[~small])) / k[~small]**2
    dk = -t * e * u - a * dj
    return z, np.stack((-j, -b * j, dk, b * dk), axis=-1)


def design(theta, coordinates, norm, *, derivatives=False):
    b, u, t = coordinates
    z, dz = transport(b, u, t, *theta[6:], gradient=True)
    B = (b - norm['mean'][0]) / norm['scale'][0]
    Z = (z - norm['mean'][1]) / norm['scale'][1]
    basis = np.column_stack((np.ones_like(B), B, Z, B*B, B*Z, Z*Z))
    score = basis @ theta[:6]
    if derivatives:
        warp = (theta[2] + theta[4]*B + 2*theta[5]*Z)[:, None] * dz / norm['scale'][1]
        return score, np.column_stack((basis, warp))
    return score


def objective(theta, coordinates, y, norm, lam, warp_lam):
    score, jac = design(theta, coordinates, norm, derivatives=True)
    delta = theta[6:] - np.array([1., 0., 0., 0.])
    value = np.mean(np.logaddexp(0., score) - y*score)
    value += .5*lam*np.dot(theta[1:6], theta[1:6]) + .5*warp_lam*np.dot(delta, delta)
    grad = jac.T @ (expit(score)-y) / len(y)
    grad[1:6] += lam*theta[1:6]
    grad[6:] += warp_lam*delta
    return float(value), grad


def fit_source(X, y, lam, warp_lam, *, starts=af.START_K):
    coords, norm = af.raw_coordinates(X), af.normalization(X)
    y = np.asarray(y, float)
    if np.unique(y).size != 2:
        raise ValueError('source must contain both classes')
    fits = []
    for k in starts:
        theta = np.zeros(10)
        theta[0] = np.log(np.mean(y)/(1-np.mean(y)))
        theta[5], theta[6], theta[8] = -.1, 1., k
        r = minimize(objective, theta, args=(coords, y, norm, lam, warp_lam), method='L-BFGS-B',
                     jac=True, bounds=BOUNDS,
                     options={'maxiter': 500, 'ftol': 1e-11, 'gtol': 2e-7, 'maxls': 40})
        fits.append({'initial_k0': k, 'initial_k1': 0., 'success': bool(r.success),
                     'message': str(r.message), 'objective': float(r.fun), 'iterations': int(r.nit),
                     'theta': r.x.tolist()})
    eligible = [i for i, r in enumerate(fits) if r['success'] and np.isfinite(r['objective'])]
    if not eligible:
        raise RuntimeError('all source optimization starts failed')
    chosen = min(eligible, key=lambda i: fits[i]['objective'])
    return {'theta': fits[chosen]['theta'], 'normalization': norm, 'lambda': lam, 'warp_lambda': warp_lam,
            'selected_start': chosen, 'starts': fits, 'reference_speed': .1}


def predict_frozen(model, X):
    return expit(design(np.asarray(model['theta']), af.raw_coordinates(X), model['normalization']))


def select_source(X, y, family, levels, speeds, ids, log_path, *, lambdas=af.LAMBDAS, warp_lambdas=af.WARP_LAMBDAS):
    candidates = []
    folds = ms.two_speed_folds(levels, speeds)
    for lam, warp_lam in itertools.product(lambdas, warp_lambdas):
        p = np.full(len(y), np.nan)
        for fold, (train, valid) in enumerate(folds):
            fit = fit_source(X[train], y[train], lam, warp_lam)
            p[valid] = predict_frozen(fit, X[valid])
            ms.append_jsonl(log_path, {'lambda': lam, 'warp_lambda': warp_lam, 'fold': fold,
                'train_ids_sha256': bd.id_hash(ids[train]), 'valid_ids_sha256': bd.id_hash(ids[valid]),
                'heldout_beta_levels': sorted(np.unique(levels[valid]).tolist()), 'fit': fit,
                'metrics': metrics(p[valid], y[valid], family[valid])})
        if not np.isfinite(p).all():
            raise AssertionError('missing OOF probabilities')
        candidates.append({'lambda': lam, 'warp_lambda': warp_lam, 'metrics': metrics(p, y, family)})
    chosen = max(range(len(candidates)), key=lambda i: candidates[i]['metrics']['family_min_iou'])
    best = candidates[chosen]
    return {'model': fit_source(X, y, best['lambda'], best['warp_lambda']),
            'source_candidates': candidates, 'selected_index': chosen}


def run(output, full_run):
    output, full_run = Path(output), Path(full_run)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('preserve historical outputs')
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'POST-HOC development on already inspected corpora; not confirmatory',
        'hypothesis': 'A beta-dependent width rate addresses the global-k source incompatibility diagnostic.',
        'changed': 'k=k0+k1*log(beta/.1); one extra parameter; k1 bounded[-2,2], penalized toward0',
        'same': ['source folds', 'normalization', 'base concavity', 'loss', 'regularization grids', 'three initial k0 starts'],
        'source_selection': 'pooled source OOF minimum IoU over fixed observer family',
        'decision_before_run': 'Accept as broad predictive improvement only if BOTH corpora combined macro family minimum improve over full flow, learned alpha and raw3D. Report all18 tasks and separate transfer scopes regardless.',
        'novelty': 'Conditional linear flows and varying coefficients are established; predictive test only.',
        'prospective_A2': 'excluded; no new candidate added to its frozen evaluation',
        'published_headline': 'D held-out speeds only; E conditional boundary-focused diagnostic excluded',
        'code_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        (Path(__file__), Path(af.__file__), Path(ms.__file__), Path(bd.__file__))},
        'lambda_grid': af.LAMBDAS, 'warp_lambda_grid': af.WARP_LAMBDAS, 'bounds': BOUNDS}
    ms.write_json(output/'protocol.json', protocol, exclusive=True)
    cello = bd.load_dataset(include_ambiguous=True)
    cello['family'] = label_family(cello['output_features'], cello['reference_f0_hz'])
    datasets = {'cello': cello, 'published': af.load_published()}
    ms.write_json(output/'input_manifests.json', {k: v['manifest'] for k, v in datasets.items()}, exclusive=True)
    tasks = [('cello', s, pair) for s, pair in itertools.product(('A','B','C','D'), ms.SPEED_PAIRS)]
    tasks += [('published', 'D4', pair) for pair in itertools.combinations((.05,.1,.15,.2), 2)]
    frozen = {}
    for domain, source, pair in tasks:
        data = datasets[domain]
        train, test = ms.build_masks(data, source, pair)
        if domain == 'published':
            test['combined'] = test['source_heldout_speed'].copy()
        key = f'{domain}__{source}__{pair[0]:g}_{pair[1]:g}'
        prior_path = full_run/key/'frozen_models.json'
        prior = json.loads(prior_path.read_text())
        training_hash = bd.id_hash(data['ids'][train])
        if training_hash != prior['training_ids_sha256']:
            raise ValueError('source ID mismatch')
        target_hash = {scope: bd.id_hash(data['ids'][mask]) for scope, mask in test.items()}
        if target_hash != prior['target_ids_sha256']:
            raise ValueError('target ID mismatch')
        directory = output/key
        directory.mkdir()
        selection = select_source(data['X'][train], data['y_fixed'][train] == 1, data['family'][train],
            data['beta_level'][train], data['speed'][train], data['ids'][train], directory/'source_cv.jsonl')
        payload = {'domain': domain, 'source': source, 'training_speeds': pair,
            'training_ids': data['ids'][train].tolist(), 'training_ids_sha256': training_hash,
            'target_ids_sha256': target_hash, 'conditional_flow': selection,
            'full_flow_model': prior['flow']['model'], 'baseline_models': prior['baseline_models'],
            'prior_frozen_sha256': hashlib.sha256(prior_path.read_bytes()).hexdigest()}
        path = directory/'frozen_models.json'
        ms.write_json(path, payload, exclusive=True)
        frozen[key] = (path, domain, test)
        print('FROZEN', key, 'warp', selection['model']['theta'][6:], flush=True)
    ms.write_json(output/'all_models_frozen.json', {k: {'path': str(p.relative_to(output)),
        'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for k, (p, _, _) in frozen.items()}, exclusive=True)
    results = {}
    for key, (path, domain, test) in frozen.items():
        data, payload = datasets[domain], json.loads(path.read_text())
        probabilities = {'conditional_flow': predict_frozen(payload['conditional_flow']['model'], data['X']),
                         'full_flow': af.predict_frozen(payload['full_flow_model'], data['X'])}
        probabilities.update({m: ms.predict_frozen(model, data['X']) for m, model in payload['baseline_models'].items()})
        results[key] = {'domain': domain, 'source': payload['source'], 'training_speeds': payload['training_speeds'],
            'warp': payload['conditional_flow']['model']['theta'][6:], 'methods': {m: {
                scope: evaluate_panels(p, data['y_fixed'] == 1, data['family'], data['group'], mask)
                for scope, mask in test.items()} for m, p in probabilities.items()}}
    ms.write_json(output/'results.json', results, exclusive=True)
    aggregates = {}
    for domain in datasets:
        rows = [v for v in results.values() if v['domain'] == domain]
        methods = ('conditional_flow', 'full_flow', 'learned_alpha', 'raw_3d')
        aggregates[domain] = {'tasks': len(rows), 'methods': {m: {scope: {agg: {metric: float(np.mean([
            r['methods'][m][scope][agg][metric] for r in rows])) for metric in ('family_min_iou', 'nominal_iou')}
            for agg in ('macro_panels','global_same_rule')} for scope in ('combined','source_heldout_speed','other_specimens')}
            for m in methods}, 'paired': {}}
        for m in methods[1:]:
            d = np.array([r['methods']['conditional_flow']['combined']['macro_panels']['family_min_iou'] -
                          r['methods'][m]['combined']['macro_panels']['family_min_iou'] for r in rows])
            aggregates[domain]['paired'][m] = {'mean': float(d.mean()), 'wins': int(sum(d>1e-12)), 'losses': int(sum(d< -1e-12))}
    summary = {'status': protocol['status'], 'aggregate': aggregates,
               'all_tasks': {k: {m: v['combined']['macro_panels']['family_min_iou'] for m,v in row['methods'].items()}
                             for k,row in results.items()}, 'seconds': time.monotonic()-started,
               'broad_improvement_gate': all(d['paired'][m]['mean'] > 0 for d in aggregates.values() for m in ('full_flow','learned_alpha','raw_3d')),
               'caveat': 'Overlapping tasks are not independent replicates; gate passing is not novelty or statistical significance.'}
    ms.write_json(output/'summary.json', summary, exclusive=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--full-run', type=Path, default=af.ROOT/'reports/affine_transport/2026-10-13-v2')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, args.full_run)
