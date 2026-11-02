"""Participant-macro proper scores and paired uncertainty for a fixed split."""
from __future__ import annotations
import numpy as np
from .models import patterns

EPSILON = 1e-6
BASELINE_ORDER = ('frequency', 'markov', 'hgb')


def row_scores(y, probability):
    y, p = np.asarray(y, float), np.asarray(probability, float)
    if y.ndim != 2 or y.shape[1] != 9 or p.shape != y.shape or not np.isin(y, [0, 1]).all():
        raise ValueError('expected matched Nx9 binary labels and probabilities')
    if len(y) == 0 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('empty labels or invalid probabilities')
    clipped = np.clip(p, EPSILON, 1 - EPSILON)
    loss = -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)).mean(1)
    brier = np.square(p - y).mean(1)
    return loss, brier


def participant_means(values, participant_ids, mask=None):
    values, ids = np.asarray(values, float), np.asarray(participant_ids)
    if values.shape != ids.shape or values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('unaligned participant scores')
    selected = np.ones(len(ids), bool) if mask is None else np.asarray(mask)
    if selected.shape != ids.shape or selected.dtype != np.bool_:
        raise ValueError('invalid subset mask')
    people = np.unique(ids)
    people = np.asarray([pid for pid in people if (selected & (ids == pid)).any()])
    means = np.asarray([values[selected & (ids == pid)].mean() for pid in people])
    return people, means


def coherence_rate(probability, tolerance=1e-6):
    p = np.asarray(probability, float)
    if p.ndim != 2 or p.shape[1] != 9 or not len(p) or not np.isfinite(p).all():
        raise ValueError('invalid probability shape')
    matrix = np.vstack((np.ones(10), patterns().T))
    mass = np.linalg.solve(matrix, np.column_stack((np.ones(len(p)), p)).T).T
    return float((mass.min(1) >= -tolerance).mean())


def summarize(y, probability, participant_ids, target_guard):
    guard = np.asarray(target_guard)
    if guard.shape != (len(y),) or guard.dtype != np.bool_:
        raise ValueError('target guard must be a boolean vector')
    loss, brier = row_scores(y, probability)
    people, means = participant_means(loss, participant_ids)
    _, brier_means = participant_means(brier, participant_ids)
    result = {'log_loss': float(means.mean()), 'brier': float(brier_means.mean()),
              'participants': len(people), 'rows': len(y), 'coherence_rate': coherence_rate(probability),
              'participant_loss': means.tolist()}
    for name, selected in [('valid', guard), ('invalid', ~guard)]:
        subset_people, subset_loss = participant_means(loss, participant_ids, selected)
        result[name] = {'rows': int(selected.sum()), 'participants': len(subset_people),
                        'log_loss': float(subset_loss.mean()) if len(subset_loss) else None}
    return result


def select_baseline(dev_scores):
    if set(dev_scores) != set(BASELINE_ORDER) or not all(np.isfinite(v) for v in dev_scores.values()):
        raise ValueError('all three finite dev scores required')
    return min(BASELINE_ORDER, key=lambda name: (dev_scores[name], BASELINE_ORDER.index(name)))


def paired_bootstrap(candidate, comparator, repeats=5000, seed=20261102):
    a, b = np.asarray(candidate, float), np.asarray(comparator, float)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('paired participant loss vectors required')
    difference = a - b
    indices = np.random.default_rng(seed).integers(0, len(a), size=(repeats, len(a)))
    bootstrap = difference[indices].mean(1)
    return {'mean_difference': float(difference.mean()),
            'percentile_95_interval': np.quantile(bootstrap, [.025, .975]).tolist(),
            'participants': len(a), 'resamples': repeats, 'seed': seed,
            'scope': 'Exploratory paired participant uncertainty; no trial/seed pseudo-replication or population guarantee.'}
