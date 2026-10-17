#!/usr/bin/env python3
"""Independently audit and publish the complete conditional-transport experiment.

Without --run, reads the public gzip bundle (one host-path metadata redaction).
No reports/ or A2
inputs are needed. Numerical scoring below does not call the experiment's
predictor, metric, fold, or source-selection implementations.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import affine_transport as af
import benchmark_data as bd
import conditional_transport as cf
from set_envelope import label_family

METHODS = ('conditional_flow', 'full_flow', 'learned_alpha', 'raw_3d')
SCALARS = ('nominal_iou', 'family_min_iou', 'family_mean_iou', 'family_max_iou',
           'box_lower_iou', 'box_upper_iou', 'brier_nominal')
BUNDLE_NAME = 'conditional_transport_artifacts.json.gz'


def flow_score(model, X):
    b, u, t = np.log(X / [.1, 1., .1]).T
    theta = np.asarray(model['theta'])
    a0, a1, k0 = theta[6:9]
    k = k0 + (theta[9] * b if len(theta) == 10 else 0.)
    k = np.broadcast_to(k, b.shape)
    factor = np.empty_like(k)
    np.divide(-np.expm1(-k*t), k, out=factor, where=k != 0.)
    factor[k == 0.] = t[k == 0.]
    z = np.exp(-k*t)*u - (a0 + a1*b)*factor
    B = (b-model['normalization']['mean'][0])/model['normalization']['scale'][0]
    Z = (z-model['normalization']['mean'][1])/model['normalization']['scale'][1]
    return theta[0]+theta[1]*B+theta[2]*Z+theta[3]*B*B+theta[4]*B*Z+theta[5]*Z*Z


def flow_probability(model, X):
    return expit(flow_score(model, X))


def baseline_probability(model, X):
    coordinates = np.log(X/[.1, 1., .1])
    if model['alpha'] is not None:
        coordinates = coordinates[:, :2].copy()
        coordinates[:, 1] -= model['alpha'] * np.log(X[:, 2]/.1)
    features = np.column_stack([np.prod(coordinates**power, axis=1)
                                for power in model['polynomial_powers']])
    features = (features-model['polynomial_feature_mean'])/model['polynomial_feature_scale']
    return expit(features @ np.asarray(model['logistic_coefficient']) + model['logistic_intercept'])


def iou(a, b):
    denominator = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b)/denominator) if denominator else 1.


def score(p, nominal, family):
    pred = p >= .5
    values = np.asarray([iou(pred, labels) for labels in family.T])
    lower, upper = family.all(axis=1), family.any(axis=1)
    lo_truth = np.where(pred, lower, upper)
    hi_truth = np.where(pred, upper, lower)
    return dict(n=len(p), nominal_iou=iou(pred, nominal), family_min_iou=float(values.min()),
                family_mean_iou=float(values.mean()), family_max_iou=float(values.max()),
                box_lower_iou=iou(pred, lo_truth), box_upper_iou=iou(pred, hi_truth),
                brier_nominal=float(np.mean((p-nominal)**2)), predicted_positive=int(pred.sum()),
                family_ious=values.tolist())


def evaluate(p, data, mask):
    groups, y, family = data['group'], data['y_fixed'] == 1, data['family']
    panels = {str(g): score(p[mask & (groups == g)], y[mask & (groups == g)], family[mask & (groups == g)])
              for g in sorted(set(groups[mask]))}
    macro = {k: float(np.mean([r[k] for r in panels.values()])) for k in SCALARS}
    macro['worst_panel_family_min_iou'] = min(r['family_min_iou'] for r in panels.values())
    specimens = {s: {k: float(np.mean([r[k] for g, r in panels.items() if g.split('/')[0] == s]))
                       for k in SCALARS} for s in sorted({g.split('/')[0] for g in panels})}
    return dict(panels=panels, macro_panels=macro, specimen_panel_means=specimens,
                global_same_rule=score(p[mask], y[mask], family[mask]))


class Comparison:
    def __init__(self):
        self.count = 0
        self.maximum_error = 0.

    def compare(self, actual, expected, path=''):
        if isinstance(expected, dict):
            if set(actual) != set(expected):
                raise AssertionError(f'keys differ: {path}')
            for key in expected:
                self.compare(actual[key], expected[key], f'{path}/{key}')
        elif isinstance(expected, list):
            if len(actual) != len(expected):
                raise AssertionError(f'length differs: {path}')
            for i, (a, e) in enumerate(zip(actual, expected)):
                self.compare(a, e, f'{path}/{i}')
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            error = abs(float(actual)-expected)
            if not np.isfinite(error) or error > 1e-10:
                raise AssertionError(f'numeric mismatch {path}: {actual} != {expected}')
            self.maximum_error = max(self.maximum_error, error)
            self.count += 1
        elif actual != expected:
            raise AssertionError(f'value mismatch: {path}')


def audit(files):
    read = lambda name: json.loads(files[name])
    protocol, results, summary = (read(f'{name}.json') for name in ('protocol', 'results', 'summary'))
    for name, expected in protocol['code_sha256'].items():
        assert hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest() == expected, name
    manifest = read('all_models_frozen.json')
    assert len(manifest) == len(results) == 18
    comparison, starts, ids_checks = Comparison(), Counter(), 0
    cello = bd.load_dataset(include_ambiguous=True)
    cello['family'] = label_family(cello['output_features'], cello['reference_f0_hz'])
    assert np.array_equal(cello['family'][:, 7], cello['y_fixed'] == 1)
    data_by_domain = {'cello': cello, 'published': af.load_published()}
    input_manifests = read('input_manifests.json')
    comparison.compare(cello['manifest'], input_manifests['cello'])
    for key in ('sha256', 'strict_only', 'scope'):
        comparison.compare(data_by_domain['published']['manifest'][key], input_manifests['published'][key])
    observed_tasks = set()
    for task, entry in manifest.items():
        assert hashlib.sha256(files[entry['path']].encode()).hexdigest() == entry['sha256']
        model_file = read(entry['path'])
        domain, source, pair = model_file['domain'], model_file['source'], model_file['training_speeds']
        observed_tasks.add((domain, source, tuple(pair)))
        data = data_by_domain[domain]
        train = (data['string'] == source) & np.isin(data['speed'], pair)
        targets = {'combined': ~train, 'source_heldout_speed': (data['string'] == source) & ~train,
                   'other_specimens': data['string'] != source}
        if domain == 'published':
            targets['combined'] = targets['source_heldout_speed'].copy()
        assert model_file['training_ids'] == data['ids'][train].tolist()
        assert model_file['training_ids_sha256'] == bd.id_hash(data['ids'][train])
        assert model_file['target_ids_sha256'] == {scope: bd.id_hash(data['ids'][mask]) for scope, mask in targets.items()}
        assert not any((train & mask).any() for mask in targets.values())
        ids_checks += 4
        selection = model_file['conditional_flow']
        X, y, family = data['X'][train], data['y_fixed'][train] == 1, data['family'][train]
        levels, ids = data['beta_level'][train], data['ids'][train]
        fold_columns = np.array_split(np.unique(levels), 5)
        oof = {}
        logs = [json.loads(line) for line in files[f'{task}/source_cv.jsonl'].splitlines()]
        assert len(logs) == 45
        for row in logs:
            valid = np.isin(levels, fold_columns[row['fold']])
            tr = ~valid
            assert row['train_ids_sha256'] == bd.id_hash(ids[tr])
            assert row['valid_ids_sha256'] == bd.id_hash(ids[valid])
            assert row['heldout_beta_levels'] == sorted(fold_columns[row['fold']].tolist())
            key = (row['lambda'], row['warp_lambda'])
            oof.setdefault(key, np.full(len(y), np.nan))[valid] = flow_probability(row['fit'], X[valid])
            comparison.compare(score(flow_probability(row['fit'], X[valid]), y[valid], family[valid]), row['metrics'])
            check_fit(row['fit'], X[tr], y[tr], comparison, starts)
            ids_checks += 2
        for candidate in selection['source_candidates']:
            key = candidate['lambda'], candidate['warp_lambda']
            assert np.isfinite(oof[key]).all()
            comparison.compare(score(oof[key], y, family), candidate['metrics'])
        best = max(range(9), key=lambda j: selection['source_candidates'][j]['metrics']['family_min_iou'])
        assert best == selection['selected_index']
        chosen = selection['source_candidates'][best]
        assert all(selection['model'][key] == chosen[key] for key in ('lambda', 'warp_lambda'))
        check_fit(selection['model'], X, y, comparison, starts)
        probabilities = {'conditional_flow': flow_probability(selection['model'], data['X']),
                         'full_flow': flow_probability(model_file['full_flow_model'], data['X'])}
        probabilities.update({name: baseline_probability(model, data['X'])
                              for name, model in model_file['baseline_models'].items()})
        for name, p in probabilities.items():
            for scope, mask in targets.items():
                comparison.compare(evaluate(p, data, mask), results[task]['methods'][name][scope], f'{task}/{name}/{scope}')
            comparison.compare(results[task]['methods'][name]['combined']['macro_panels']['family_min_iou'],
                               summary['all_tasks'][task][name])
    expected_tasks = {('cello', s, pair) for s in 'ABCD' for pair in ((.05,.1),(.05,.2),(.1,.2))}
    expected_tasks |= {('published', 'D4', pair) for pair in itertools.combinations((.05,.1,.15,.2), 2)}
    assert observed_tasks == expected_tasks
    for domain, aggregate in summary['aggregate'].items():
        rows = [r for r in results.values() if r['domain'] == domain]
        assert aggregate['tasks'] == len(rows)
        for method in METHODS:
            for scope in ('combined', 'source_heldout_speed', 'other_specimens'):
                for agg in ('macro_panels', 'global_same_rule'):
                    for metric in ('family_min_iou', 'nominal_iou'):
                        comparison.compare(np.mean([r['methods'][method][scope][agg][metric] for r in rows]),
                                           aggregate['methods'][method][scope][agg][metric])
        for method in METHODS[1:]:
            delta = np.array([r['methods']['conditional_flow']['combined']['macro_panels']['family_min_iou'] -
                              r['methods'][method]['combined']['macro_panels']['family_min_iou'] for r in rows])
            comparison.compare(dict(mean=float(delta.mean()), wins=int(sum(delta>1e-12)), losses=int(sum(delta < -1e-12))),
                               aggregate['paired'][method])
    gate = all(d['paired'][method]['mean'] > 0 for d in summary['aggregate'].values() for method in METHODS[1:])
    assert gate == summary['broad_improvement_gate']
    gradient_maximum = gradient_check()
    return dict(status='PASS independent numerical and source-isolation audit; predictive improvement gate FAILED',
                tasks=18, source_cv_fits=810, final_source_fits=18, optimizer_starts=dict(starts),
                independently_compared_numeric_values=comparison.count, maximum_numeric_error=comparison.maximum_error,
                source_target_or_fold_id_checks=ids_checks, gradient_checks=48, maximum_gradient_absolute_error=gradient_maximum,
                broad_improvement_gate=gate, A2_accessed=False,
                scope='Post-hoc sample1 and published D/E only. E is conditional diagnostic, excluded from headline.',
                bundle_files={name: hashlib.sha256(text.encode()).hexdigest() for name, text in sorted(files.items())})


def check_fit(model, X, y, comparison, starts):
    b, u, t = np.log(X/[.1,1.,.1]).T
    coords = np.column_stack((b, u-t))
    scale = coords.std(axis=0)
    scale[scale < 1e-8] = 1.
    comparison.compare(coords.mean(axis=0).tolist(), model['normalization']['mean'])
    comparison.compare(scale.tolist(), model['normalization']['scale'])
    for start in model['starts']:
        starts['successful' if start['success'] else 'failed'] += 1
        theta = np.asarray(start['theta'])
        p = flow_probability({**model, 'theta': theta}, X)
        score_value = flow_score({**model, 'theta': theta}, X)
        comparison.compare(cf.predict_frozen({**model, 'theta': theta}, X).tolist(), p.tolist())
        value = np.mean(np.logaddexp(0.,score_value)-y*score_value)
        value += .5*model['lambda']*np.sum(theta[1:6]**2)
        value += .5*model['warp_lambda']*np.sum((theta[6:]-[1.,0.,0.,0.])**2)
        comparison.compare(float(value), start['objective'])
    eligible = [i for i, r in enumerate(model['starts']) if r['success'] and np.isfinite(r['objective'])]
    chosen = min(eligible, key=lambda i: model['starts'][i]['objective'])
    assert chosen == model['selected_start']
    assert model['theta'] == model['starts'][chosen]['theta']


def gradient_check():
    rng = np.random.default_rng(7318)
    X = np.exp(rng.uniform([-3., -5., -3.], [-1., 2., -1.], (67, 3)))
    norm = af.normalization(X)
    coords = af.raw_coordinates(X)
    y = rng.integers(0,2,len(X))
    largest = 0.
    for i in range(48):
        theta = rng.uniform(-1., 1., 10)
        theta[5] = -abs(theta[5])
        if i % 4 == 0:
            theta[8:] = 0.
        if i % 4 == 1:
            theta[8:] *= 1e-7
        args = coords, y, norm, .01, .03
        analytic = cf.objective(theta, *args)[1]
        numeric = approx_derivative(lambda q: cf.objective(q, *args)[0], theta, method='3-point', abs_step=1e-5).ravel()
        largest = max(largest, float(np.max(abs(analytic-numeric))))
        np.testing.assert_allclose(analytic, numeric, atol=2e-6, rtol=2e-5)
    return largest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, help='Optional original run to bundle; default audits public bundle')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.run:
        files = {str(p.relative_to(args.run)): p.read_bytes().decode('utf-8')
                 for p in sorted(args.run.rglob('*')) if p.is_file()}
        original_manifest = files['input_manifests.json']
        manifest = json.loads(original_manifest)
        published_source = Path(manifest['published']['source'])
        if published_source.is_absolute():
            manifest['published']['source'] = str(published_source.relative_to(ROOT))
        files['input_manifests.json'] = json.dumps(manifest, indent=2) + '\n'
        transformations = [{
            'file': 'input_manifests.json',
            'field': 'published.source',
            'change': 'Repository-relative source path; JSON indentation normalized. All other files retain original bytes.',
            'original_file_sha256': hashlib.sha256(original_manifest.encode()).hexdigest(),
        }]
        payload = json.dumps(dict(format='utf8-file-map-v1', files=files,
                                  publication_transformations=transformations), ensure_ascii=False,
                             separators=(',',':')).encode()
    else:
        payload = gzip.decompress((Path(__file__).with_name(BUNDLE_NAME)).read_bytes())
        bundle = json.loads(payload)
        assert bundle['format'] == 'utf8-file-map-v1'
        files = bundle['files']
        transformations = bundle.get('publication_transformations', [])
    report = audit(files)
    report['bundle_uncompressed_sha256'] = hashlib.sha256(payload).hexdigest()
    report['publication_transformations'] = transformations
    if args.run:
        (args.output/BUNDLE_NAME).write_bytes(gzip.compress(payload, mtime=0))
    for name in ('protocol', 'summary'):
        (args.output/f'conditional_transport_{name}.json').write_text(files[f'{name}.json'])
    (args.output/'conditional_transport_independent_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    summary = json.loads(files['summary.json'])
    fig, axes = plt.subplots(2, 1, figsize=(10,7.4), constrained_layout=True)
    for ax, domain in zip(axes, ('cello', 'published')):
        rows = {key: value for key, value in summary['all_tasks'].items() if key.startswith(domain+'__')}
        for method, label, color in [('full_flow','vs global width flow','#287c8e'),
                                     ('learned_alpha','vs learned uniform exponent','#b26a16'),
                                     ('raw_3d','vs raw 3D quadratic','#696969')]:
            delta = [r['conditional_flow']-r[method] for r in rows.values()]
            ax.plot(range(len(rows)), delta, 'o-', label=label, color=color, lw=1, ms=4)
        ax.axhline(0, color='black', lw=.8)
        ax.set_xticks(range(len(rows)), [key.split('__')[1]+'\n'+key.split('__')[2].replace('_',', ') for key in rows], fontsize=8)
        ax.set_ylabel('Conditional flow minus comparator IoU')
        ax.set_title('Cello: all 12 source/pair tasks (15-rule panel minima)' if domain=='cello'
                     else 'Published D-string: all 6 pairs (nominal IoU; E excluded)', fontsize=11)
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_xlabel('Source and training speeds (m/s); overlapping tasks are not independent replicates')
    fig.suptitle('Position-dependent width: broad improvement gate failed', fontsize=13)
    fig.savefig(args.output/'conditional_transport_comparison.png', dpi=170)
    print(json.dumps({k:v for k,v in report.items() if k!='bundle_files'}, indent=2))


if __name__ == '__main__':
    main()
