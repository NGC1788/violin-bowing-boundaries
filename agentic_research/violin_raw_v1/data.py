"""Pure data-contract functions; no WAV access, model fit, or automatic export.

Actual use requires the separate metadata audit, independent review and frozen
protocol. Functions here are exercised on synthetic examples first.
"""
from __future__ import annotations
import hashlib
import json
import numpy as np

SPLITS = ('train', 'dev', 'test')
SEED = 20261030
PROPORTIONS = (.68, .14, .18)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def participant_split(participants, seed=SEED):
    """Cohort-stratified hash ordering; counts depend on metadata only.

    Largest-remainder allocation of 68/14/18 percent, ties train/dev/test.
    No musical score, target, feature, or duration enters this function.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('integer seed required')
    participants = list(participants)
    ids = [p['participant_id'] for p in participants]
    if len(set(ids)) != len(ids) or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError('duplicate or malformed participant identity')
    if not participants or any(p['cohort'] not in ('CG', 'FG', 'EG') for p in participants):
        raise ValueError('only declared CG/FG/EG cohorts are eligible')
    manifest, counts = {}, {}
    for cohort in sorted({p['cohort'] for p in participants}):
        group = [p['participant_id'] for p in participants if p['cohort'] == cohort]
        n = len(group)
        exact = n * np.asarray(PROPORTIONS)
        sizes = np.floor(exact).astype(int)
        for i in sorted(range(3), key=lambda i: (-(exact[i] - sizes[i]), i))[:n - int(sizes.sum())]:
            sizes[i] += 1
        if np.any(sizes == 0):
            raise ValueError('cohort too small for the declared three-way split')
        ranked = sorted(group, key=lambda pid: (digest({'seed': seed, 'cohort': cohort, 'participant': pid}), pid))
        cursor = 0
        for name, size in zip(SPLITS, sizes):
            for pid in ranked[cursor:cursor + size]:
                manifest[pid] = name
            cursor += int(size)
        counts[cohort] = dict(zip(SPLITS, map(int, sizes)))
    return manifest, {'seed': seed, 'proportions': list(PROPORTIONS), 'cohort_counts': counts,
                      'manifest_sha256': digest(manifest), 'selection_inputs': ['participant_id', 'cohort']}


def deduplicate_manifest(records):
    """Exact PCM aliases within a person; cross-person full/prefix copies fail.

    The whole-signal checksum is an identity audit, never a predictor feature.
    Constant/silent shared prefixes are also flagged; no automatic exception.
    """
    records = list(records)
    if len({r['source_id'] for r in records}) != len(records):
        raise ValueError('repeated source identity')
    full, prefix, kept, aliases = {}, {}, [], {}
    for r in sorted(records, key=lambda row: row['source_id']):
        for key in ('participant_id', 'source_id', 'pcm_sha256', 'prefix_pcm_sha256'):
            if not isinstance(r.get(key), str) or not r[key]:
                raise ValueError('missing identity field: ' + key)
        pid, content, short = r['participant_id'], r['pcm_sha256'], r['prefix_pcm_sha256']
        if content in full and full[content]['participant_id'] != pid:
            raise ValueError('identical PCM crosses participant identities')
        if short in prefix and prefix[short] != pid:
            raise ValueError('identical prefix PCM crosses participant identities')
        prefix[short] = pid
        if content in full:
            if short != full[content]['prefix_pcm_sha256']:
                raise ValueError('identical full content has inconsistent prefix checksum')
            aliases[full[content]['source_id']].append(r['source_id'])
        else:
            full[content] = r
            aliases[r['source_id']] = [r['source_id']]
            kept.append(r)
    return kept, aliases


def _partition(participant_ids, splits):
    p, s = np.asarray(participant_ids), np.asarray(splits)
    if p.ndim != 1 or s.shape != p.shape or not np.isin(s, SPLITS).all():
        raise ValueError('invalid partition arrays')
    if not np.isin(SPLITS, s).all():
        raise ValueError('all three partitions must contain recordings')
    for pid in np.unique(p):
        if len(np.unique(s[p == pid])) != 1:
            raise ValueError('participant crosses partitions')
    return p, s


def _guard(guard, n):
    g = np.asarray(guard)
    if g.shape != (n,) or g.dtype != np.bool_:
        raise ValueError('guard must be a boolean vector')
    return g


def fit_thresholds(targets, guard, splits, participant_ids):
    """Only train AND valid rows are converted, inspected, or fit to quantiles."""
    participants, split = _partition(participant_ids, splits)
    y = np.asarray(targets)
    if y.shape != (len(split), 2):
        raise ValueError('targets must have shape Nx2')
    g = _guard(guard, len(split))
    selected = (split == 'train') & g
    train_y = y[selected].astype(np.float64)
    if len(train_y) < 4 or not np.isfinite(train_y).all() or (train_y < 0).any():
        raise ValueError('invalid or insufficient valid training targets')
    thresholds = np.quantile(train_y, [.25, .5, .75], axis=0, method='linear').T
    if not (np.diff(thresholds, axis=1) > 0).all():
        raise ValueError('degenerate training quantiles; do not add jitter or tune on held-out data')
    fit_ids = sorted(set(participants[selected].tolist()))
    return thresholds, {'valid_train_rows': int(selected.sum()),
                        'train_rows': int((split == 'train').sum()),
                        'fit_participant_ids_sha256': digest(fit_ids),
                        'fit_target_bytes_sha256': hashlib.sha256(train_y.astype('<f8').tobytes()).hexdigest(),
                        'quantiles': [.25, .5, .75], 'quantile_method': 'linear',
                        'quantile_weighting': 'recording-weighted valid training targets; evaluation is participant-macro'}


def patterns():
    # Independent construction; must match the model's sorted bank exactly.
    bank = [np.zeros((3, 3), dtype=np.uint8)]
    for i in range(3):
        for j in range(3):
            grid = np.zeros((3, 3), dtype=np.uint8)
            grid[i:, j:] = 1
            bank.append(grid)
    return np.unique(np.asarray(bank).reshape(10, 9), axis=0)


def labels_and_states(targets, guard, thresholds, *, prefix=False):
    y, t = np.asarray(targets, float), np.asarray(thresholds, float)
    if y.ndim != 2 or y.shape[1] != 2 or t.shape != (2, 3):
        raise ValueError('invalid target/threshold shapes')
    g = _guard(guard, len(y))
    if not np.isfinite(t).all() or (t < 0).any() or not (np.diff(t, axis=1) > 0).all():
        raise ValueError('thresholds must be finite, nonnegative and strictly ordered')
    if not np.isfinite(y[g]).all() or (y[g] < 0).any():
        raise ValueError('valid targets must be finite and nonnegative')
    if not np.isfinite(y[:, 1]).all() or (y[:, 1] < 0).any():
        raise ValueError('power must be valid even when pitch guard is false')
    labels = np.zeros((len(y), 9), dtype=np.uint8)
    a = y[g, 0, None] <= t[0]
    b = y[g, 1, None] <= t[1]
    labels[g] = (a[:, :, None] & b[:, None, :]).reshape(-1, 9)
    matches = (labels[:, None, :] == patterns()[None, :, :]).all(2)
    if not matches.any(1).all():
        raise AssertionError('unexpected rule pattern')
    states = matches.argmax(1).astype(np.int64)
    if prefix:
        states[~g] = 10  # Separate unavailable-prefix state; future invalid remains all-false.
    return labels, states


def assemble_rows(records, split_manifest):
    """Assemble already-extracted rows. This function never opens actual audio."""
    unique, aliases = deduplicate_manifest(records)
    participants = np.asarray([r['participant_id'] for r in unique])
    if set(participants) != set(split_manifest):
        raise ValueError('partition manifest must match all eligible participants')
    split = np.asarray([split_manifest[p] for p in participants])
    _partition(participants, split)
    x = np.asarray([r['X'] for r in unique], dtype=np.float64)
    if x.shape != (len(unique), 60) or not np.isfinite(x).all():
        raise ValueError('expected finite 60-dimensional prefix features')
    target = np.asarray([r['targets'] for r in unique], float)
    prefix_target = np.asarray([r['prefix_targets'] for r in unique], float)
    guard = _guard(np.asarray([r['target_guard'] for r in unique]), len(unique))
    prefix_guard = _guard(np.asarray([r['prefix_guard'] for r in unique]), len(unique))
    thresholds, receipt = fit_thresholds(target, guard, split, participants)
    labels, states = labels_and_states(target, guard, thresholds)
    prefix_labels, prefix_states = labels_and_states(prefix_target, prefix_guard, thresholds, prefix=True)
    return {'X': x, 'targets': target, 'target_guard': guard, 'prefix_targets': prefix_target,
            'prefix_guard': prefix_guard, 'Y9': labels, 'pattern_ids': states,
            'prefix_Y9': prefix_labels, 'prefix_states': prefix_states,
            'participant_ids': participants, 'split': split, 'thresholds': thresholds,
            'source_ids': np.asarray([r['source_id'] for r in unique]), 'alias_manifest': aliases,
            'metadata': {'rows': len(unique), 'aliases_removed': sum(len(a) - 1 for a in aliases.values()),
                         'manifest_sha256': digest(split_manifest), 'threshold_fit': receipt,
                         'participant_counts': {s: len(set(participants[split == s])) for s in SPLITS},
                         'row_counts': {s: int((split == s).sum()) for s in SPLITS},
                         'invalid_target_rows': {s: int(((split == s) & ~guard).sum()) for s in SPLITS},
                         'feature_inputs_exclude': ['participant', 'cohort', 'stage', 'file name',
                                                   'duration', 'PCM hashes', 'future guard', 'future targets']}}
