"""Pure path extraction and train-only mask-catalogue summaries. No fitting."""
import hashlib
import numpy as np
from ..violin_raw_v1.features import interval_frames, PREFIX, TARGET, extract_prefix


def prefix_inputs(samples, sample_rate, subtype):
    # Values, length, finite checks, and normalization see only [0, 1).
    cut = samples[:round(sample_rate)].copy()
    frames = interval_frames(cut, sample_rate, subtype, PREFIX)
    sequence = np.column_stack([frames[k] for k in
                                ('pitch_cents', 'power_dbfs', 'periodicity', 'voiced')]).astype(np.float32)
    return extract_prefix(cut, sample_rate, subtype)['X'], sequence


def future_labels(samples, sample_rate, subtype):
    # interval_frames slices TARGET before conversion, checking or descriptors.
    frames = interval_frames(samples, sample_rate, subtype, TARGET)
    paths = {k: frames[k].copy() for k in ('pitch_cents', 'power_dbfs', 'periodicity', 'voiced')}
    if any(v.shape != (46,) for v in paths.values()):
        raise ValueError('expected exactly 46 future frames')
    targets, guard = reconstruct(paths['pitch_cents'][None], paths['power_dbfs'][None],
                                 paths['voiced'][None])
    return {**paths, 'targets': targets[0], 'guard': guard[0],
            'support': np.column_stack((frames['starts'], frames['ends']))}


def reconstruct(pitch, power, voiced):
    pitch, power, voiced = np.asarray(pitch), np.asarray(power), np.asarray(voiced)
    if pitch.ndim != 2 or pitch.shape != power.shape or pitch.shape != voiced.shape or pitch.shape[1] != 46:
        raise ValueError('paths must be matching [rows, 46] arrays')
    if voiced.dtype != np.bool_ or not np.isfinite(power).all() or not np.isfinite(pitch[voiced]).all():
        raise ValueError('invalid finite values or boolean mask')
    guard = voiced.sum(axis=1) >= 23
    pitch_std = np.full(len(pitch), np.nan)
    for i in np.flatnonzero(guard):
        pitch_std[i] = np.std(pitch[i, voiced[i]], ddof=1)
    return np.column_stack((pitch_std, np.std(power, axis=1, ddof=1))), guard


def mask_keys(voiced):
    voiced = np.asarray(voiced)
    if voiced.ndim != 2 or voiced.shape[1] != 46 or voiced.dtype != np.bool_:
        raise ValueError('mask must be boolean [rows, 46]')
    return [r.tobytes() for r in np.packbits(voiced, axis=1, bitorder='little')]


def build_catalogue(train_masks):
    """This API receives INNER TRAIN masks only, never a full-array selector."""
    return frozenset(mask_keys(train_masks))


def catalogue_counts(catalogue, held_masks):
    keys = mask_keys(held_masks)
    unknown = np.array([k not in catalogue for k in keys], bool)
    valid = np.asarray(held_masks).sum(axis=1) >= 23
    return {'held_rows': len(keys), 'held_valid_rows': int(valid.sum()),
            'held_unknown_rows': int(unknown.sum()),
            'held_valid_unknown_rows': int((unknown & valid).sum()),
            'held_unknown_fraction': float(unknown.mean()) if len(keys) else None,
            'held_valid_unknown_fraction': float(unknown[valid].mean()) if valid.any() else None}


def mutation_ids(source_ids, count):
    return set(sorted(map(str, source_ids), key=lambda s: hashlib.sha256(s.encode()).hexdigest())[:count])


def budget_check(started, maximum, *, clock, now, deadline):
    if clock() - started > maximum or now() >= deadline:
        raise TimeoutError('source preparation budget/deadline reached')
