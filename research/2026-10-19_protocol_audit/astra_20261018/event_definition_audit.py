"""Descriptive audit of retrospective suffix events versus causal H-run detection.

Each label is available only at the end of its analysis window. First-K detection
is an observed sequential-rule event, not a physical establishment/steady-lock
transition. Force can change throughout a stroke; no steady boundary, relaxation
law, critical slowing down, or bifurcation type is inferred here.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
RNE = HERE.parent
REPO = Path.home() / 'violin-bowing-boundaries'
sys.path.insert(0, str(RNE / 'timeres/audit_20261017'))
import window_audit as wa

STEP = .05
WINDOWS = (.1, .2, .4)
RUNS = (1, 2, 3)
FRACTIONS = (.25, .5, .75)
HORIZONS = (.3, .4, .5, .8, 1., 2.)


def sequence_arrays(start, h):
    t = np.asarray(start, float)
    H = np.asarray(h)
    if t.ndim != 1 or H.shape != t.shape or not len(t):
        raise ValueError('a nonempty one-dimensional sequence is required')
    if not np.isfinite(t).all() or np.any(t < 0) or np.any(np.diff(t) <= 0):
        raise ValueError('window start times must be finite, nonnegative and strictly increasing')
    if not np.isin(H, (False, True)).all():
        raise ValueError('H must contain binary decisions')
    return t, H.astype(bool)


def causal_index(start, h, k, *, step=STEP):
    """First completion of K adjacent H window decisions; index is of completion."""
    t, H = sequence_arrays(start, h)
    if not isinstance(k, (int, np.integer)) or k < 1 or step <= 0:
        raise ValueError('positive integer K and positive step are required')
    if len(H) < k:
        return None
    # Running counts over fixed K windows avoid Python loops over long strokes.
    counts = np.convolve(H.astype(np.int64), np.ones(k, dtype=np.int64), mode='valid')
    valid = counts == k
    if k > 1:
        gaps = ~np.isclose(np.diff(t), step, atol=1e-8, rtol=0)
        valid &= np.convolve(gaps.astype(np.int64), np.ones(k-1, dtype=np.int64), mode='valid') == 0
    found = np.flatnonzero(valid)
    return int(found[0]+k-1) if len(found) else None


def suffix_index(h, threshold=.8):
    """Historical first H whose remaining *observed* suffix has >=threshold H."""
    H = np.asarray(h, bool)
    if H.ndim != 1 or not len(H) or not 0 < threshold <= 1:
        raise ValueError('nonempty binary sequence and threshold in (0,1] required')
    count = np.cumsum(H[::-1], dtype=np.int64)[::-1]
    candidates = np.flatnonzero(H & (count >= threshold*np.arange(len(H), 0, -1)))
    return int(candidates[0]) if len(candidates) else None


def episodes(start, h, *, step=STEP):
    """Observed H runs and endpoint spans; these are NOT physical dwell times."""
    t, H = sequence_arrays(start, h)
    adjacent = np.r_[False, np.isclose(np.diff(t), step, atol=1e-8, rtol=0)]
    starts = np.flatnonzero(H & ~(np.r_[False, H[:-1]] & adjacent))
    ends = np.flatnonzero(H & ~(np.r_[H[1:], False] & np.r_[adjacent[1:], False]))
    assert len(starts) == len(ends)
    return [{'start_index': int(a), 'end_index': int(b), 'decisions': int(b-a+1),
             'decision_span_s': float(t[b]-t[a]),
             'left_observation_edge': bool(a == 0), 'right_observation_edge': bool(b == len(H)-1),
             'left_gap_edge': bool(a > 0 and not adjacent[a]),
             'right_gap_edge': bool(b < len(H)-1 and not adjacent[b+1])}
            for a, b in zip(starts, ends)]


def compare_prefix(start, h, n, k=None):
    """Compare only an event located in the observed prefix to a prefix re-analysis."""
    t, H = sequence_arrays(start, h)
    if not 1 <= n < len(H):
        raise ValueError('prefix must be nonempty and strictly shorter than full followup')
    full = suffix_index(H) if k is None else causal_index(t, H, k)
    partial = suffix_index(H[:n]) if k is None else causal_index(t[:n], H[:n], k)
    available = full if full is not None and full < n else None
    return {'changed': available != partial,
            'prefix_event_retracted_or_later': partial is not None and available is None,
            'future_qualifies_earlier_event': partial is None and available is not None,
            'event_index_changed': partial is not None and available is not None and partial != available,
            'full_event_index': full, 'prefix_event_index': partial,
            'available_full_event_index': available}


def quantiles(values):
    v = np.asarray(values, float)
    if not len(v):
        return None
    return {name: float(x) for name, x in zip(('min', 'p25', 'median', 'p75', 'max'),
                                             np.quantile(v, [0, .25, .5, .75, 1]))}


def fixed_a2_nuisance(data):
    path = REPO/'research/2026-10-14/lockbox_A2/feature_summary.json'
    spec = json.loads(path.read_text())
    return spec['thresholds_applied']['min_amplitude'], {
        str(f): spec['conditions'][str(f).split('/')[-1]]['reference_f0_hz']
        for f in data['folder_names']}, wa.sha(path)


def group_sequences(data, window):
    idx = np.flatnonzero(np.isclose(data['window_s'], window))
    order = np.lexsort((data['t'][idx], data['stroke_index'][idx]))
    idx = idx[order]
    cuts = np.r_[0, np.flatnonzero(np.diff(data['stroke_index'][idx]))+1, len(idx)]
    return [idx[a:b] for a, b in zip(cuts[:-1], cuts[1:])]


def summarize_group(rows):
    n = len(rows)
    result = {'strokes': n, 'window_decisions': sum(r['windows'] for r in rows),
              'strokes_with_time_gaps': sum(r['gaps'] > 0 for r in rows),
              'strokes_any_nonpositive_force_window': sum(r['nonpositive_force_windows'] > 0 for r in rows),
              'strokes_multiple_H_episodes': sum(r['H_episodes'] > 1 for r in rows),
              'strokes_any_H': sum(r['H_windows'] > 0 for r in rows),
              'strokes_any_H_to_nonH': sum(r['H_to_nonH'] > 0 for r in rows),
              'H_to_nonH_transitions': sum(r['H_to_nonH'] for r in rows),
              'nonH_to_H_transitions': sum(r['nonH_to_H'] for r in rows),
              'censor_last_decision_end_s': quantiles([r['last_decision_end_s'] for r in rows]),
              'methods': {}}
    for method in ('suffix80', 'causal_K1', 'causal_K2', 'causal_K3'):
        detected = [r for r in rows if r[method+'_end_s'] is not None]
        result['methods'][method] = {
            'detected': len(detected), 'not_detected_by_last_observed_decision': n-len(detected),
            'detection_end_s': quantiles([r[method+'_end_s'] for r in detected]),
            'later_nonH_observed': sum(r[method+'_later_nonH'] for r in detected),
            'later_nonH_fraction_among_detected': sum(r[method+'_later_nonH'] for r in detected)/len(detected) if detected else None}
    result['legacy_force_filtered'] = {
        'eligible_strokes_at_least3_positive_force_windows': sum(r['legacy_eligible'] for r in rows),
        'filtered_sequences_with_time_gaps': sum(r['legacy_positive_force_gaps'] > 0 and r['legacy_eligible'] for r in rows),
        'legacy_suffix_detection_changes_vs_unfiltered_among_eligible': sum(r['legacy_suffix_changed'] for r in rows if r['legacy_eligible']),
        'legacy_suffix_not_detected': sum(r['legacy_suffix_end_s'] is None for r in rows if r['legacy_eligible']),
        'legacy_suffix_first_window_mapped_to_0p025_s': sum(r['legacy_suffix_zero'] for r in rows if r['legacy_eligible'])}
    return result


def audit_archive(feature, output):
    started = time.monotonic()
    initial_sha = wa.sha(feature)
    data = wa.load_features(feature)
    floor, refs = wa.pipeline_nuisance(data)
    modes = {'pipeline_fixed': (floor, refs)}
    calibration = {'pipeline_fixed': {'floor': floor, 'references': refs,
        'scope': 'estimated once from the complete archive; held fixed during prefix audit; not online recalibration'}}
    if feature.stem.replace('.csv', '') == '2024-04-19_TypeA_sample2':
        f, r, digest = fixed_a2_nuisance(data)
        modes = {'frozen_acquisition_fixed': (f, r), **modes}
        calibration['frozen_acquisition_fixed'] = {'floor': f, 'references': r,
            'source_sha256': digest, 'scope': 'original A2 acquisition-window nuisance held fixed'}
    labels = {name: wa.vector_labels(data, f, r) == 'helmholtz' for name, (f, r) in modes.items()}
    mode_names = list(modes)
    nuisance_changes = int(np.count_nonzero(labels[mode_names[0]] != labels[mode_names[-1]]))
    rows, episode_rows, comparisons = [], [], defaultdict(lambda: defaultdict(int))
    for window in WINDOWS:
        groups = group_sequences(data, window)
        if len(groups) != len(data['first']):
            raise ValueError('each requested window length must represent every stroke')
        for mode, H_all in labels.items():
            for idx in groups:
                t, H = sequence_arrays(data['t'][idx], H_all[idx])
                end = t+window
                if np.any(end > data['plateau_s'][idx]+1e-7):
                    raise ValueError('window ends outside recorded plateau')
                adjacency = np.isclose(np.diff(t), STEP, atol=1e-8, rtol=0)
                ep = episodes(t, H)
                folder = str(data['folder'][idx[0]])
                row = {'archive': feature.stem.replace('.csv', ''), 'nuisance': mode,
                    'window_s': window, 'folder': folder, 'trial': int(data['trial'][idx[0]]),
                    'beta': float(data['beta'][idx[0]]), 'speed': float(np.median(data['speed'][idx])),
                    'windows': len(idx), 'gaps': int(np.count_nonzero(~adjacency)),
                    'first_force_N': float(data['force'][idx[0]]), 'last_force_N': float(data['force'][idx[-1]]),
                    'nonpositive_force_windows': int(np.count_nonzero(data['force'][idx] <= 0)),
                    'H_windows': int(H.sum()), 'H_episodes': len(ep),
                    'H_to_nonH': int(np.count_nonzero(H[:-1] & ~H[1:] & adjacency)),
                    'nonH_to_H': int(np.count_nonzero(~H[:-1] & H[1:] & adjacency)),
                    'first_decision_end_s': float(end[0]), 'last_decision_end_s': float(end[-1]),
                    'longest_H_decision_span_s': max((e['decision_span_s'] for e in ep), default=0.)}
                for name, k in [('suffix80', None)]+[(f'causal_K{k}', k) for k in RUNS]:
                    event = suffix_index(H) if k is None else causal_index(t, H, k)
                    row[name+'_index'] = event
                    row[name+'_end_s'] = float(end[event]) if event is not None else None
                    row[name+'_later_nonH'] = bool(np.any(~H[event+1:])) if event is not None else False
                positive = data['force'][idx] > 0
                tp, hp = t[positive], H[positive]
                row['legacy_eligible'] = len(tp) >= 3
                row['legacy_positive_force_gaps'] = int(np.count_nonzero(~np.isclose(np.diff(tp), STEP, atol=1e-8, rtol=0)))
                old_event = suffix_index(hp) if len(tp) >= 3 else None
                old_end = float(tp[old_event]+window) if old_event is not None else None
                row['legacy_suffix_end_s'] = old_end
                row['legacy_suffix_zero'] = bool(old_event == 0) if old_event is not None else False
                row['legacy_suffix_changed'] = bool(old_end != row['suffix80_end_s']) if row['legacy_eligible'] else False
                rows.append(row)
                for e in ep:
                    episode_rows.append({k: row[k] for k in ('archive','nuisance','window_s','folder','trial')} | e)
                prefixes = [(f'fraction_{fraction:g}', max(1, int(np.floor(fraction*len(t))))) for fraction in FRACTIONS]
                prefixes += [(f'end_horizon_{horizon:g}', int(np.count_nonzero(end <= horizon+1e-8)))
                             for horizon in HORIZONS if end[-1] > horizon+1e-8]
                for prefix_name, n in prefixes:
                    if n < 1 or n >= len(t):
                        continue
                    for method, k in [('suffix80', None)]+[(f'causal_K{k}', k) for k in RUNS]:
                        c = compare_prefix(t, H, n, k)
                        key = (mode, window, folder, prefix_name, method)
                        comparisons[key]['comparisons'] += 1
                        for field in ('changed', 'prefix_event_retracted_or_later', 'future_qualifies_earlier_event', 'event_index_changed'):
                            comparisons[key][field] += int(c[field])
                        if k is not None and c['changed']:
                            raise AssertionError('causal first-K event changed when future was removed')
            print(f'{feature.stem}: window={window}, nuisance={mode} complete', flush=True)
    if wa.sha(feature) != initial_sha:
        raise ValueError('feature archive changed during audit; wait for pipeline completion')
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['nuisance'], row['window_s'], row['folder'])].append(row)
    summaries = [{'nuisance': key[0], 'window_s': key[1], 'folder': key[2], **summarize_group(rs)}
                 for key, rs in grouped.items()]
    comparison_rows = [{'nuisance': k[0], 'window_s': k[1], 'folder': k[2], 'prefix': k[3], 'method': k[4], **v}
                       for k, v in comparisons.items()]
    report = {'status': 'descriptive event-definition audit; no physical lock, CSD, equilibrium-law or bifurcation inference',
        'archive': feature.stem.replace('.csv', ''), 'created_utc': datetime.now(timezone.utc).isoformat(),
        'input_sha256': initial_sha, 'feature_rows': len(data['trial']), 'strokes': len(data['first']),
        'calibration': calibration, 'binary_H_nuisance_changes': nuisance_changes,
        'definitions': {'causal_K': 'first completion of K consecutive H decisions, available at window END; reset across time gaps',
            'suffix80': 'first H index for which >=80% of all remaining observed decisions are H; retrospective and horizon dependent',
            'not_detected': 'no detector event by the final observed window END; no cure fraction or permanent non-H assertion',
            'prefix_test': 'full-series event counts only if its index is in the prefix; compare with event recomputed from that same prefix',
            'H_episode': 'contiguous labelled H decision run; decision-time span, not latent physical dwell duration',
            'force_filtering': 'NONE in primary audit; legacy per-window force>0 filtering audited separately',
            'causality_scope': 'conditional on fixed archive-level classifier constants; current nuisance estimation is not an online algorithm'},
        'limitations': ['Window overlap causes strong dependence; decision counts and prefix comparisons are not independent trials.',
            'Changing force and design-confounded acceleration/travel prevent attributing detection times to relaxation at constant force.',
            'The H classifier is a waveform proxy, not independent contact-state truth.',
            'Later non-H decisions may reflect switching, changing controls or detector noise; this audit does not distinguish them.',
            'Different window lengths change detector bias and latency; they are not interchangeable physical clocks.',
            'Prefix stability establishes a stopping rule under fixed labels; it does not establish independent censoring for a survival model.'],
        'groups': summaries, 'prefix_comparisons': comparison_rows,
        'code_sha256': {Path(p).name: wa.sha(p) for p in (__file__, wa.__file__)},
        'seconds': time.monotonic()-started}
    stem = report['archive']
    output.mkdir(parents=True, exist_ok=True)
    target = output/(stem+'_event_audit.json')
    if target.exists():
        raise FileExistsError('preserve prior audit; choose another output directory')
    target.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for suffix, records in [('stroke_events', rows), ('H_episodes', episode_rows)]:
        with gzip.open(output/(stem+'_'+suffix+'.csv.gz'), 'wt', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, nargs='+', default=[RNE/'timeres/features'/f'{s}.csv.gz' for s in
        ('2024-03-25_TypeA_sample1', '2024-04-19_TypeA_sample2')])
    parser.add_argument('--output', type=Path, default=HERE/'event_audit_results')
    args = parser.parse_args()
    for feature in args.features:
        report = audit_archive(feature, args.output)
        print(json.dumps({'archive': report['archive'], 'strokes': report['strokes'], 'seconds': report['seconds'],
            'causal_prefix_changes': sum(r['changed'] for r in report['prefix_comparisons'] if r['method'].startswith('causal')),
            'suffix_prefix_changes': sum(r['changed'] for r in report['prefix_comparisons'] if r['method']=='suffix80')}, indent=2))


if __name__ == '__main__':
    main()
