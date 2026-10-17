#!/usr/bin/env python3
"""Publish and independently check the complete nominal-control width diagnostic.

The witness checker does not import width_law_certificate or reuse its band
enumerator, optimizer, or witness implementation. Its check establishes that the
reported upper bound is attained. Global optimality additionally relies on the
finite-envelope argument and exhaustive small-instance tests, not on witnesses.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import affine_transport as af
import benchmark_data as bd
import set_envelope as se

HERE = Path(__file__).resolve().parent
BUNDLE = 'width_law_artifacts.json.gz'


def portable_paths(value):
    if isinstance(value, dict):
        return {key: portable_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_paths(item) for item in value]
    if isinstance(value, str) and value.startswith(str(ROOT)+'/'):
        return str(Path(value).relative_to(ROOT))
    return value


def read_run(directory):
    directory = Path(directory)
    summary = json.loads((directory / 'summary.json').read_text())
    manifest_bytes = (directory / 'input_manifests.json').read_bytes()
    return {'summary': summary,
            'publication_transform': {'description': 'Repository-local absolute source path replaced with repo-relative path; numerical data and task results unchanged. Original ignored run preserved.',
                'original_input_manifest_file_sha256': hashlib.sha256(manifest_bytes).hexdigest()},
            'input_manifests': portable_paths(json.loads(manifest_bytes)),
            'tasks': {r['task']: json.loads(gzip.decompress((directory / (r['task']+'.json.gz')).read_bytes()))
                      for r in summary['all_tasks']}}


def minimum_band_errors(force, labels):
    """Independent linear-time maximum-subarray baseline with indivisible force ties."""
    force, labels = np.asarray(force), np.asarray(labels, bool)
    _, inverse = np.unique(force, return_inverse=True)
    gains = np.bincount(inverse, weights=2*labels.astype(int)-1)
    best = running = 0.
    for gain in gains:
        running = max(0., running+gain)
        best = max(best, running)
    return int(labels.sum()-best)


def verify_task(result, data, labels):
    source, speeds = result['source'], result['source_speeds']
    ids, independent, total, rows = [], 0, 0, 0
    if len(result['columns']) != len(result['witnesses']):
        raise AssertionError('column/witness count mismatch')
    expected_levels = sorted(np.unique(data['beta_level'][data['string'] == source]).tolist())
    if [column['beta_level'] for column in result['columns']] != expected_levels:
        raise AssertionError('incomplete nominal beta grid')
    for column, witness in zip(result['columns'], result['witnesses']):
        widths, pair_cost = [], 0
        for side, speed in zip(('first', 'second'), speeds):
            mask = ((data['string'] == source) & (data['speed'] == speed)
                    & (data['beta_level'] == column['beta_level']))
            force, y = data['X'][mask, 1], labels[mask]
            ids.extend(data['ids'][mask].tolist())
            w = witness[side]
            lo, hi = w['lower_log_force'], w['upper_log_force']
            if not np.isfinite([lo, hi]).all() or not lo < hi:
                raise AssertionError('non-finite or empty band')
            pred = (np.log(force) >= lo) & (np.log(force) <= hi)
            errors = np.flatnonzero(pred != y)
            if len(errors) != w['cost'] or sorted(w['disagreement_original_indices']) != errors.tolist():
                raise AssertionError('reclassification disagrees with witness')
            if len(y) != column[side+'_n']:
                raise AssertionError('wrong column size')
            if not np.isclose(hi-lo, w['width'], rtol=2e-11, atol=2e-14):
                raise AssertionError('stored width differs from endpoints')
            widths.append(hi-lo)
            independent += minimum_band_errors(force, y)
            total += len(errors)
            pair_cost += len(errors)
            rows += len(y)
        ratio = np.exp(result['witness_k']*np.log(speeds[1]/speeds[0]))
        if not np.isclose(widths[1], ratio*widths[0], rtol=2e-10, atol=2e-13):
            raise AssertionError('shared width law violated')
        if pair_cost != witness['cost']:
            raise AssertionError('pair cost mismatch')
    if rows != result['source_rows'] or bd.id_hash(np.asarray(ids)) != result['source_ids_sha256']:
        raise AssertionError('source membership mismatch')
    if independent != result['independent_band_cost'] or total != result['shared_width_cost']:
        raise AssertionError('objective mismatch')
    if total-independent != result['extra_disagreements']:
        raise AssertionError('repair gap mismatch')
    if total != min(c['cost'] for c in result['cost_cells']):
        raise AssertionError('witness is not a minimizer of stored envelope')
    return {'rows_reclassified': rows, 'columns': len(result['columns']),
            'independent_band_cost': independent, 'attained_shared_width_cost': total}


def aggregate(tasks):
    grouped = defaultdict(list)
    for r in tasks:
        grouped[(r['domain'], r['source'], tuple(r['source_speeds']))].append(r)
    pairs = []
    for (domain, source, speeds), rows in grouped.items():
        nominal = next(r for r in rows if r['rule_index'] == (7 if domain == 'cello' else 0))
        pairs.append({'domain': domain, 'source': source, 'speeds': list(speeds),
            'source_rows': nominal['source_rows'], 'nominal_band_errors': nominal['independent_band_cost'],
            'nominal_extra_width_errors': nominal['extra_disagreements'],
            'nominal_extra_constant_width_errors': nominal['constant_width_extra_disagreements'],
            'rule_count': len(rows), 'rules_with_positive_gap': sum(r['extra_disagreements'] > 0 for r in rows),
            'extra_width_range': [min(r['extra_disagreements'] for r in rows), max(r['extra_disagreements'] for r in rows)]})
    result = {'pairs': pairs, 'domains': {}}
    for domain in ('cello', 'published'):
        rows = [r for r in tasks if r['domain'] == domain]
        nominal = [r for r in rows if r['rule_index'] == (7 if domain == 'cello' else 0)]
        result['domains'][domain] = {'tasks': len(rows), 'positive_gap_tasks': sum(r['extra_disagreements'] > 0 for r in rows),
            'nominal_tasks': len(nominal), 'nominal_positive_gap_tasks': sum(r['extra_disagreements'] > 0 for r in nominal),
            'max_extra_errors': max(r['extra_disagreements'] for r in rows),
            'max_extra_error_fraction': max(r['extra_disagreement_fraction'] for r in rows),
            'nominal_max_extra_error_fraction': max(r['extra_disagreement_fraction'] for r in nominal),
            'nominal_independent_band_fraction_range': [min(r['independent_band_fraction'] for r in nominal), max(r['independent_band_fraction'] for r in nominal)]}
    return result


def verify_bundle(bundle):
    datasets = {'cello': bd.load_dataset(include_ambiguous=True), 'published': af.load_published()}
    for name, data in datasets.items():
        if portable_paths(data['manifest']) != bundle['input_manifests'][name]:
            raise AssertionError('input manifest mismatch: '+name)
    for name, digest in bundle['summary']['protocol']['code_sha256'].items():
        if hashlib.sha256((ROOT / 'scripts' / name).read_bytes()).hexdigest() != digest:
            raise AssertionError('experiment code mismatch: '+name)
    identities = [(r['domain'], r['source'], tuple(r['source_speeds']), r['rule_index'])
                  for r in bundle['summary']['all_tasks']]
    expected = {('cello', s, pair, rule) for s, pair, rule in itertools.product(
        ('A', 'B', 'C', 'D'), itertools.combinations((.05, .1, .2), 2), range(15))}
    expected |= {('published', 'D4', pair, 0) for pair in itertools.combinations((.05, .1, .15, .2), 2)}
    if set(identities) != expected or len(identities) != len(expected):
        raise AssertionError('complete all-observer task set required')
    if set(bundle['tasks']) != {r['task'] for r in bundle['summary']['all_tasks']}:
        raise AssertionError('summary and full task set differ')
    data = datasets['cello']
    family = se.label_family(data['output_features'], data['reference_f0_hz'])
    results = {}
    for item in bundle['summary']['all_tasks']:
        result = bundle['tasks'][item['task']]
        labels = family[:, result['rule_index']] if result['domain'] == 'cello' else datasets['published']['y_fixed'] == 1
        results[item['task']] = verify_task(result, datasets[result['domain']], labels)
        for key in ('source_rows', 'source_ids_sha256', 'independent_band_cost', 'shared_width_cost',
                    'extra_disagreements', 'constant_width_extra_disagreements', 'witness_k'):
            if item[key] != result[key]:
                raise AssertionError('summary/task mismatch: '+key)
    variations = []
    for source in sorted(set(data['string'])):
        for speed in sorted(set(data['speed'])):
            for level in sorted(set(data['beta_level'])):
                mask = (data['string'] == source) & (data['speed'] == speed) & (data['beta_level'] == level)
                if mask.any():
                    x = data['X'][mask]
                    variations.append([np.ptp(x[:, 0])/np.median(x[:, 0]), np.ptp(x[:, 2])/speed])
    return {'scope': 'Independent reclassification and dynamic-programming band baseline; not an independent global-optimality proof or physical validation',
        'tasks_verified': len(results), 'rows_reclassified_with_reuse': sum(r['rows_reclassified'] for r in results.values()),
        'columns_verified_with_reuse': sum(r['columns'] for r in results.values()),
        'data_controls': {'nominal_cello_columns': len(variations), 'quantiles': [0, .5, .95, 1],
            'relative_beta_range_quantiles': np.quantile(variations, [0, .5, .95, 1], axis=0)[:, 0].tolist(),
            'relative_speed_range_quantiles': np.quantile(variations, [0, .5, .95, 1], axis=0)[:, 1].tolist()},
        'task_checks': results}


def figure(tasks, out):
    cello = [r for r in tasks if r['domain'] == 'cello']
    keys = sorted(set((r['source'], tuple(r['source_speeds'])) for r in cello))
    matrix = np.array([[next(r['extra_disagreements'] for r in cello if (r['source'], tuple(r['source_speeds'])) == key and r['rule_index'] == rule)
                        for rule in range(15)] for key in keys])
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.8), gridspec_kw={'width_ratios': [1.7, 1]}, constrained_layout=True)
    ax = axes[0]
    image = ax.imshow(matrix, cmap='YlOrRd', vmin=0, vmax=12, aspect='auto')
    ax.set_yticks(range(len(keys)), [f'{s}: {v[0]:g}–{v[1]:g}' for s, v in keys])
    ax.set_xticks(range(15), [f'{f:g}\n{p:g}' for f in (.3, .4, .5, .6, .7) for p in (.75, .8, .85)])
    ax.set_xlabel('Observer: fraction threshold / periodicity threshold')
    ax.set_ylabel('Cello source specimen and two speeds (m/s)')
    for row in range(len(keys)):
        for col in range(15):
            ax.text(col, row, str(matrix[row, col]), ha='center', va='center', fontsize=8,
                    color='white' if matrix[row, col] >= 8 else '#242424')
    ax.set_title('All 15 observer rules: extra disagreements G')
    fig.colorbar(image, ax=ax, shrink=.75, label='Extra rows beyond independent bands')
    pub = [r for r in tasks if r['domain'] == 'published']
    y = np.arange(len(pub))
    axes[1].barh(y-.17, [r['extra_disagreements'] for r in pub], height=.32, label='Shared k, optimized')
    axes[1].barh(y+.17, [r['constant_width_extra_disagreements'] for r in pub], height=.32, label='Constant width, k = 0')
    axes[1].set_yticks(y, [f'{r["source_speeds"][0]:g}–{r["source_speeds"][1]:g}' for r in pub])
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Extra disagreements beyond independent bands')
    axes[1].set_ylabel('Published violin D-string source speeds (m/s)')
    axes[1].set_title('Author classifications on the nominal grid')
    axes[1].legend(loc='lower right', fontsize=9)
    fig.suptitle('Nominal-grid width relaxation: few extra source disagreements; not prediction accuracy', fontsize=12)
    fig.savefig(out / 'width_law_sensitivity.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, help='Package this completed run before verification')
    parser.add_argument('--out', type=Path, default=HERE)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.run:
        bundle = read_run(args.run)
        payload = json.dumps(bundle, separators=(',', ':'), allow_nan=False).encode()
        (args.out / BUNDLE).write_bytes(gzip.compress(payload, mtime=0))
    else:
        bundle = json.loads(gzip.decompress((args.out / BUNDLE).read_bytes()))
    audit = verify_bundle(bundle)
    audit['bundle_sha256'] = hashlib.sha256((args.out / BUNDLE).read_bytes()).hexdigest()
    summary = aggregate(bundle['summary']['all_tasks'])
    summary['protocol'] = bundle['summary']['protocol']
    summary['runtime_seconds'] = bundle['summary']['seconds']
    for name, value in (('width_law_summary.json', summary), ('width_law_witness_audit.json', audit)):
        (args.out / name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    figure(bundle['summary']['all_tasks'], args.out)
    print(json.dumps({key: value for key, value in audit.items() if key != 'task_checks'}, indent=2))


if __name__ == '__main__':
    main()
