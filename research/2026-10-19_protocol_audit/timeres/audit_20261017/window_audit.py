"""Audit time-resolved A2 features against the preserved acquisition-window cache.

No original feature, result, archive, or frozen repository file is modified.
Window-cache replay verifies implementation agreement, not physical truth.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.csv as arrow_csv

REPO = Path.home() / 'violin-bowing-boundaries'
HERE = Path(__file__).resolve().parent
TIMERES = HERE.parent
sys.path.insert(0, str(REPO / 'scripts'))
import grid_summary as grid
import regime_map as rm


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_features(path):
    table = arrow_csv.read_csv(path)
    columns = {name: table[name].to_numpy(zero_copy_only=False) for name in table.column_names}
    errors = columns['error']
    if any(e not in (None, '') for e in errors):
        raise ValueError('feature errors must be resolved, not silently discarded')
    required = ('trial', 'beta', 'window_s', 't', 'plateau_s', 'force', 'speed')
    if any(not np.isfinite(columns[name]).all() for name in required):
        raise ValueError('nonfinite controls or time')
    if np.any(columns['beta'] <= 0) or np.any(columns['speed'] <= 0):
        raise ValueError('nonpositive beta or speed')
    folders, codes = np.unique(columns['folder'], return_inverse=True)
    keys, first, inverse = np.unique(np.column_stack((codes, columns['trial'])), axis=0,
                                    return_index=True, return_inverse=True)
    record_keys = np.column_stack((codes, columns['trial'], np.rint(columns['window_s']*1000),
                                  np.rint(columns['t']*1000)))
    if np.unique(record_keys, axis=0).shape[0] != len(codes):
        raise ValueError('duplicate feature keys; compare contents before deduplication')
    if not np.array_equal(columns['beta'], columns['beta'][first][inverse]):
        raise ValueError('beta changes between windows of one stroke')
    columns.update(folder_names=folders, folder_codes=codes, trial_keys=keys,
                   first=first, stroke_index=inverse)
    return columns


def recover_grid(data):
    """Unrounded per-stroke controls and the repository's established gap criterion."""
    first = data['first']
    stroke_levels = np.full(len(first), -1, int)
    info = {}
    for code, folder in enumerate(data['folder_names']):
        mask = data['trial_keys'][:, 0] == code
        beta = data['beta'][first[mask]]
        threshold, ratio = grid.natural_break(beta)
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError('no separated position grid; do not invent columns')
        levels = grid.assign_levels(beta, threshold)
        stroke_levels[mask] = levels
        rounded = np.unique(np.round(np.log(beta), 4))
        gaps = np.diff(rounded)
        old_cut = np.median(gaps[gaps > 0])/3
        old_count = 1 + np.count_nonzero(gaps > old_cut)
        counts = np.bincount(levels)
        info[str(folder)] = {'trials': int(mask.sum()), 'recovered_columns': len(counts),
            'trials_per_column': counts.tolist(), 'gap_threshold': threshold,
            'gap_ratio': ratio, 'legacy_prepare_columns': int(old_count)}
    return stroke_levels[data['stroke_index']], info


def vector_labels(data, floor, references, min_periodicity=.8):
    amp, per, f0, fb = (data[k] for k in ('amplitude_std','periodicity','f0_hz','flybacks_per_period'))
    ref = np.array([references[str(folder)] for folder in data['folder']])
    labels = np.full(len(amp), 'ambiguous', dtype='<U16')
    decided = (~(amp >= floor)) & (~(per >= min_periodicity))
    labels[decided] = 'no_oscillation'
    aperiodic = ~decided & (~np.isfinite(per) | (per < .5))
    labels[aperiodic] = 'aperiodic'
    valid = ~decided & ~aperiodic & (per >= min_periodicity)
    sub = valid & np.isfinite(ref) & np.isfinite(f0) & (f0 < .75*ref)
    labels[sub] = 'subharmonic'
    valid &= ~sub
    labels[valid & (fb == 1)] = 'helmholtz'
    labels[valid & (fb >= 2)] = 'multiple_slip'
    return labels


def pipeline_nuisance(data):
    controls = (data['force'] <= 0) & np.isfinite(data['amplitude_std'])
    floor = max(.02, 3*float(np.median(data['amplitude_std'][controls]))) if controls.sum() >= 10 else .02
    refs = {}
    for folder in data['folder_names']:
        mask = ((data['folder'] == folder) & (data['periodicity'] >= .8)
                & (data['amplitude_std'] >= floor) & np.isfinite(data['f0_hz']))
        refs[str(folder)] = float(np.median(data['f0_hz'][mask]))
    return floor, refs


def cached_features(task):
    condition, trial = task
    window = np.load(REPO/'data/cache/2024-04-19_TypeA_sample2'/condition/f'window_{trial}.npy',
                     allow_pickle=False)[:, 2]
    per, f0 = rm.dominant_period(window)
    shape = rm.flyback_structure(window, rm.RATE_HZ/f0 if np.isfinite(f0) else np.nan, .5, 3.)
    return {'condition': condition, 'trial': trial, 'amplitude_std': float(np.std(window)),
            'periodicity': per, 'f0_hz': f0, 'flybacks_per_period': shape['flybacks_per_period'],
            'second_flyback_ratio': shape['second_flyback_ratio']}


def replay_cached_windows(rows, summary):
    expected = {(r['condition'], int(r['trial'])): r for r in rows}
    tasks = sorted(expected)
    numeric_error = {name: 0. for name in ('amplitude_std','periodicity','f0_hz',
                                         'flybacks_per_period','second_flyback_ratio')}
    mismatch, binary_mismatch, returned = [], [], []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in pool.map(cached_features, tasks, chunksize=16):
            key = (f['condition'], f['trial'])
            target = expected[key]
            for name in numeric_error:
                a, b = f[name], float(target[name])
                if np.isnan(a) and np.isnan(b):
                    continue
                if not np.isfinite([a,b]).all():
                    raise ValueError(f'nonmatching nonfinite feature: {key} {name}')
                numeric_error[name] = max(numeric_error[name], abs(a-b))
            ref = summary['conditions'][key[0]]['reference_f0_hz']
            f['label'] = rm.classify(f['amplitude_std'], f['periodicity'], f['f0_hz'],
                                     f['flybacks_per_period'], ref, summary['thresholds_applied'])
            if f['label'] != target['label']:
                mismatch.append(list(key))
            if (f['label']=='helmholtz') != (target['label']=='helmholtz'):
                binary_mismatch.append(list(key))
            returned.append(f)
    return returned, {'trials': len(tasks), 'maximum_feature_errors': numeric_error,
        'multiclass_mismatches': mismatch, 'binary_H_mismatches': binary_mismatch,
        'scope': 'Exact original cache-window replay; not nearest sliding-window comparison or raw extraction replay.'}


def run(path, output, *, cache_replay=True):
    output.mkdir(parents=True, exist_ok=True)
    data = load_features(path)
    levels, grid_info = recover_grid(data)
    source = REPO/'research/2026-10-14/lockbox_A2'
    with gzip.open(source/'regimes.csv.gz', 'rt') as f:
        original = list(csv.DictReader(f))
    expected = {(r['condition'], int(r['trial'])): r for r in original}
    matched, beta_error = 0, 0.
    for i in data['first']:
        key = (str(data['folder'][i]).split('/')[-1], int(data['trial'][i]))
        r = expected[key]
        matched += levels[i] == int(r['beta_level'])
        beta_error = max(beta_error, abs(data['beta'][i]-float(r['beta'])))
    summary = json.loads((source/'feature_summary.json').read_text())
    ref = {str(f): summary['conditions'][str(f).split('/')[-1]]['reference_f0_hz']
           for f in data['folder_names']}
    floor = summary['thresholds_applied']['min_amplitude']
    new_floor, new_ref = pipeline_nuisance(data)
    frozen_labels = vector_labels(data, floor, ref)
    new_labels = vector_labels(data, new_floor, new_ref)
    checks = {'rows': len(data['trial']), 'unique_strokes': len(data['first']),
        'error_rows': 0, 'duplicate_keys': 0, 'grid': grid_info,
        'strokes_matching_original_beta_level': int(matched), 'maximum_beta_error': beta_error,
        'frozen_floor': floor, 'pipeline_floor': new_floor, 'frozen_refs': ref, 'pipeline_refs': new_ref,
        'sliding_full_label_changes_from_nuisance': int(np.count_nonzero(frozen_labels != new_labels)),
        'sliding_H_changes_from_nuisance': int(np.count_nonzero((frozen_labels=='helmholtz') != (new_labels=='helmholtz')))}
    if matched != len(data['first']):
        raise AssertionError('recovered control grid differs from the original independent grid')
    if cache_replay:
        cached, checks['cache_replay'] = replay_cached_windows(original, summary)
        cd = {key: np.array([r[key] for r in cached]) for key in
              ('amplitude_std','periodicity','f0_hz','flybacks_per_period')}
        cd['folder'] = np.array(['2024-04-19_TypeA_sample2/'+r['condition'] for r in cached])
        relabel = vector_labels(cd, new_floor, new_ref)
        oldlabels = np.array([r['label'] for r in cached])
        checks['cache_replay']['pipeline_nuisance_full_label_changes'] = int(np.count_nonzero(oldlabels != relabel))
        checks['cache_replay']['pipeline_nuisance_H_changes'] = int(np.count_nonzero((oldlabels=='helmholtz') != (relabel=='helmholtz')))
        with gzip.open(output/'original_window_features_replayed.json.gz','wt') as handle:
            json.dump(cached, handle)
    controls = []
    for window in (.1,.2,.4):
        signal = np.sin(2*np.pi*98*np.arange(round(window*rm.RATE_HZ))/rm.RATE_HZ)
        p, f = rm.dominant_period(signal)
        controls.append({'window_s': window, 'signal': 'noise-free 98 Hz sine',
                         'biased_periodicity': p, 'estimated_f0': f,
                         'scope': 'illustrates window-length bias, not a Helmholtz positive control'})
    checks['periodicity_window_length_controls'] = controls
    checks.update(created_utc=datetime.now(timezone.utc).isoformat(),
        feature_file_sha256=sha(path), source_sha256={p.name:sha(p) for p in
            (Path(__file__), Path(rm.__file__), Path(grid.__file__), TIMERES/'analysis_multi.py')})
    (output/'window_audit.json').write_text(json.dumps(checks, indent=2, allow_nan=False)+'\n')
    print(json.dumps(checks, indent=2, allow_nan=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', type=Path, default=TIMERES/'features/2024-04-19_TypeA_sample2.csv.gz')
    p.add_argument('--output', type=Path, default=HERE/'results')
    p.add_argument('--no-cache-replay', action='store_true')
    a=p.parse_args()
    run(a.features, a.output, cache_replay=not a.no_cache_replay)
