"""Trusted compact-input loader: policies receive public metadata and query labels only."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import gzip
import hashlib
import json
import numpy as np

SPLIT_SEED = 20261030
TRAIN_SESSIONS = ('A1', 'A2', 'A3', 'A4', 'A_T2', 'B1', 'B_T2s1', 'B_T2r1', 'B_T2r2')
DEV_SESSIONS = ('C1', 'C_T2', 'D1', 'D2')


@dataclass(frozen=True)
class Column:
    id: str
    session: str
    group: str
    public: np.ndarray  # [50,3]: log measured force, log beta, log measured speed
    labels: np.ndarray  # private to trusted harness/oracle, [rules,50]


def rule_bank(name: str) -> np.ndarray:
    if name == 'baseline':
        return np.asarray([[.8, .5]], dtype=float)
    if name == 'nine':
        return np.asarray([(p, r) for p in (.7, .8, .9) for r in (.3, .5, .7)], dtype=float)
    raise ValueError('Unknown rule_bank')


def _numeric(values):
    return np.asarray([np.nan if v is None else v for v in values], dtype=float)


def select_columns(columns, limit, sessions):
    """Fixed paired group-stratified subset; NEVER use model seed for split selection."""
    by_group = {s: sorted((c for c in columns if c.session == s), key=lambda c: c.id) for s in sessions}
    if any(not x for x in by_group.values()):
        raise ValueError('Missing declared train/development family')
    rng = np.random.default_rng(SPLIT_SEED)
    for s in sessions:
        ix = rng.permutation(len(by_group[s]))
        by_group[s] = [by_group[s][i] for i in ix]
    total = sum(map(len, by_group.values()))
    if limit is None:
        limit = total
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < len(sessions) or limit > total:
        raise ValueError(f'Column limit must be {len(sessions)}..{total} or null')
    result = []
    for i in range(max(map(len, by_group.values()))):
        for s in sessions:
            if i < len(by_group[s]) and len(result) < limit:
                result.append(by_group[s][i])
    return result


def load_dataset(path: Path, bank='nine', max_train_columns=None, max_eval_columns=16):
    blob = Path(path).read_bytes()
    raw = json.loads(gzip.decompress(blob))
    thresholds = rule_bank(bank)
    columns = []
    floor_count = 0
    seen = set()
    for c in raw['columns']:
        if c['id'] in seen:
            raise ValueError('Duplicate column ID')
        seen.add(c['id'])
        force = np.asarray(c['force'], dtype=float)
        if len(force) != 50 or np.any(np.diff(force) < 0) or not np.isfinite(force).all():
            raise ValueError('Expected 50 finite force-ordered observations')
        if c['beta'] <= 0 or c['speed'] <= 0:
            raise ValueError('Nonpositive beta or speed')
        per, ratio = _numeric(c['periodicity']), _numeric(c['ratio'])
        guard, fixed = np.asarray(c['guard'], dtype=bool), np.asarray(c['fixed'], dtype=bool)
        labels = fixed[None, :] | (guard[None, :] & (per[None, :] >= thresholds[:, 0, None])
                                  & (ratio[None, :] < thresholds[:, 1, None]))
        base = fixed | (guard & (per >= .8) & (ratio < .5))
        if not np.array_equal(base, np.asarray(c['baseline'], dtype=bool)):
            raise ValueError('Pinned observer reconstruction differs from original baseline')
        floor_count += int((force <= 1e-4).sum())
        public = np.column_stack((np.log(np.maximum(force, 1e-4)),
                                  np.full(50, np.log(c['beta'])), np.full(50, np.log(c['speed']))))
        columns.append(Column(c['id'], c['session'], c['group'], public, labels))
    sessions = {c.session for c in columns}
    if sessions != set(TRAIN_SESSIONS + DEV_SESSIONS):
        raise ValueError(f'Unexpected session manifest: {sessions}')
    train = select_columns(columns, max_train_columns, TRAIN_SESSIONS)
    dev = select_columns(columns, max_eval_columns, DEV_SESSIONS)
    if {c.id for c in train} & {c.id for c in dev}:
        raise AssertionError('Train/development overlap')
    receipt = {'input_sha256': hashlib.sha256(blob).hexdigest(), 'split_seed': SPLIT_SEED,
               'train_ids': [c.id for c in train], 'dev_ids': [c.id for c in dev],
               'rules': thresholds.tolist(), 'force_log_floor': 1e-4, 'floored_observations_all_data': floor_count,
               'scope': 'Retrospective exposed-data development; measured force is public recorded metadata; no independent final test.'}
    return train, dev, thresholds, receipt


def conditional_features(public, thresholds):
    """[columns,50,3] + known rule parameters -> [columns,rules,50,5]."""
    public = np.asarray(public, dtype=float)
    base = np.broadcast_to(public[:, None, :, :], (len(public), len(thresholds), public.shape[1], 3))
    rule = np.broadcast_to(thresholds[None, :, None, :], base.shape[:-1] + (2,))
    return np.concatenate((base, rule), axis=-1)
