"""Exact finite-cell profiling of a shared two-feature waveform rule.

H = fixed | (guard & (periodicity >= u) & (ratio < v)).

This module performs no I/O or preferred-threshold selection.  It enumerates a
single group's complete reachable classifier arrangement on a closed, finite
rectangle.  Global aggregation must retain the SAME (u,v) across all groups;
independent group optima are not a globally attainable classifier.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


def _domain(value: Any, name: str) -> tuple[float, float]:
    try:
        a = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{name} must be two ordered finite endpoints') from exc
    if a.shape != (2,) or not np.isfinite(a).all() or a[0] > a[1]:
        raise ValueError(f'{name} must be two ordered finite endpoints')
    return float(a[0]), float(a[1])


def threshold_atoms(scores: Iterable[float], domain: tuple[float, float]) -> dict:
    """Cells for a predicate whose labels change immediately AFTER each score.

    Every atom is right closed.  The first is left closed, later atoms left
    open.  Scores equal to domain_lo produce its singleton as the first atom;
    scores equal to domain_hi do not split the domain.  Finite scores outside
    the domain are ignored.  NaN/inf scores are rejected.
    """
    lo, hi = _domain(domain, 'domain')
    a = np.asarray(list(scores), dtype=float)
    if a.ndim != 1 or not np.isfinite(a).all():
        raise ValueError('scores must be a finite one-dimensional iterable')
    levels = np.unique(a[(a >= lo) & (a < hi)])
    left = np.r_[lo, levels]
    right = np.r_[levels, hi]
    left_closed = np.zeros(len(right), dtype=bool)
    left_closed[0] = True
    return dict(lo=left, hi=right, lo_closed=left_closed,
                hi_closed=np.ones(len(right), dtype=bool),
                representative=right.copy())


def map_atoms(atoms: dict, representatives: Iterable[float]) -> np.ndarray:
    """Map finer global representatives onto local right-closed atoms."""
    r = np.asarray(representatives, dtype=float)
    if r.ndim != 1 or not np.isfinite(r).all():
        raise ValueError('representatives must be a finite one-dimensional array')
    hi = np.asarray(atoms['hi'], dtype=float)
    lo = np.asarray(atoms['lo'], dtype=float)
    if (not len(hi) or np.any(r < lo[0]) or np.any(r > hi[-1])):
        raise ValueError('representatives must lie in the local atom domain')
    return np.searchsorted(hi, r, side='left')


def _binary(value: Any, name: str, n: int) -> np.ndarray:
    a = np.asarray(value)
    if a.shape != (n,):
        raise ValueError(f'{name} must be a one-dimensional array of matching length')
    try:
        good = np.all((a == 0) | (a == 1))
    except (TypeError, ValueError):
        good = False
    if not good:
        raise ValueError(f'{name} must contain only binary values')
    return a.astype(bool, copy=True)


def _first_run(labels: np.ndarray, k: int) -> int | None:
    run = 0
    for i, label in enumerate(labels):
        run = run + 1 if label else 0
        if run == k:
            return i-k+1
    return None


def _vector(value: Any, name: str, nonnegative: bool = False) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError(f'{name} must be a nonempty finite one-dimensional array')
    if nonnegative and np.any(a < 0):
        raise ValueError(f'{name} must be nonnegative')
    return a.copy()


def analyze_group(columns: Iterable[dict],
                  u_domain: tuple[float, float] = (.7, .9),
                  v_domain: tuple[float, float] = (.3, .7),
                  slopes: Iterable[float] = (-2., -1.2),
                  epsilons: Iterable[float] = (0., .1, .2, .4),
                  k: int = 2) -> dict:
    """Profile every reachable joint rule in ONE group, including tied events.

    Required column keys: id,group,beta,force,baseline,eligible,guard,fixed,
    periodicity,ratio.  Force must be ascending finite.  Fixed and guard are
    disjoint; scores must be finite on guarded windows and may otherwise be
    NaN.  Original eligibility remains fixed; every eligible baseline must
    have a positive, nonzero-width finite first-K-run force bracket.

    Counts have shape [U,V], lower/upper/gap and endpoint witness indices have
    [U,V,S], and compatible has [U,V,S,E].  Any eligible invalid boundary makes
    all law comparisons in that cell false.  Endpoint and invalid witness
    indices refer to metadata['eligible_column_ids']; -1 means unavailable.
    In valid cells, the two endpoint witnesses prove the entire interval
    intersection: L=max(log(Flo)-p*log(beta)), U=min(log(Fhi)-p*log(beta)).
    Compatibility is evaluated explicitly as L-epsilon <= U+epsilon.

    Both threshold axes are CLOSED finite intervals.  Cartesian atoms retain
    exact endpoint behavior; their upper-right representative is sufficient.
    Axis events unable to change any label anywhere in the rectangle are
    safely omitted.  Numeric exactness refers to the finite combinatorial
    enumeration, with force logs evaluated in IEEE float64.
    """
    ud = _domain(u_domain, 'u_domain')
    vd = _domain(v_domain, 'v_domain')
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError('k must be a positive integer')
    k = int(k)
    p = _vector(slopes, 'slopes')
    eps = _vector(epsilons, 'epsilons', nonnegative=True)
    cols, ids, group = [], set(), None
    for raw in columns:
        name, g = raw['id'], raw['group']
        if not isinstance(name, str) or not name or name in ids:
            raise ValueError('column id must be a unique nonempty string')
        if not isinstance(g, str) or not g:
            raise ValueError('column group must be a nonempty string')
        if group is not None and group != g:
            raise ValueError('analyze_group requires exactly one group')
        group = g
        ids.add(name)
        beta = float(raw['beta'])
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError(f'{name}: beta must be positive and finite')
        force = np.asarray(raw['force'], dtype=float)
        if (force.ndim != 1 or not len(force) or not np.isfinite(force).all()
                or np.any(np.diff(force) < 0)):
            raise ValueError(f'{name}: force must be nonempty, sorted and finite')
        force = force.copy()
        n = len(force)
        baseline = _binary(raw['baseline'], f'{name}: baseline', n)
        guard = _binary(raw['guard'], f'{name}: guard', n)
        fixed = _binary(raw['fixed'], f'{name}: fixed', n)
        if np.any(fixed & guard):
            raise ValueError(f'{name}: fixed and guard must be disjoint')
        periodicity = np.asarray(raw['periodicity'], dtype=float)
        ratio = np.asarray(raw['ratio'], dtype=float)
        for key, score in [('periodicity', periodicity), ('ratio', ratio)]:
            if score.shape != (n,) or np.isinf(score).any():
                raise ValueError(f'{name}: {key} must be finite-or-NaN and match force')
            if np.any(guard & np.isnan(score)):
                raise ValueError(f'{name}: guarded {key} must be finite')
        eligible = raw['eligible']
        if not isinstance(eligible, (bool, np.bool_)):
            raise ValueError(f'{name}: eligible must be boolean')
        s = _first_run(baseline, int(k))
        if eligible and (s is None or s == 0 or not (0 < force[s-1] < force[s])):
            raise ValueError(f'{name}: eligible baseline lacks a valid finite force bracket')
        cols.append(dict(id=name, beta=beta, force=force, baseline=baseline,
                         guard=guard, fixed=fixed, periodicity=periodicity.copy(),
                         ratio=ratio.copy(), eligible=bool(eligible)))
    if not cols:
        raise ValueError('columns must be nonempty')

    # Pruning needs the other axis's least restrictive ENDPOINT, not baseline.
    # A ratio equal to v_hi can never pass strict <; periodicity equal to u_lo
    # CAN pass >= at the closed left endpoint and therefore must be retained.
    u_events = np.concatenate([c['periodicity'][c['guard'] & (c['ratio'] < vd[1])]
                               for c in cols])
    v_events = np.concatenate([c['ratio'][c['guard'] & (c['periodicity'] >= ud[0])]
                               for c in cols])
    ua, va = threshold_atoms(u_events, ud), threshold_atoms(v_events, vd)
    U, V, C, S, E = len(ua['hi']), len(va['hi']), len(cols), len(p), len(eps)
    N = max(len(c['force']) for c in cols)
    shape = (C, N)
    baseline = np.zeros(shape, dtype=bool)
    guard = np.zeros(shape, dtype=bool)
    fixed = np.zeros(shape, dtype=bool)
    periodicity = np.full(shape, np.nan)
    ratio = np.full(shape, np.nan)
    force = np.full(shape, np.nan)
    logforce = np.full(shape, np.nan)
    beta_log = np.asarray([math.log(c['beta']) for c in cols])
    eligible_indices = np.asarray([i for i,c in enumerate(cols) if c['eligible']], dtype=int)
    for ci, c in enumerate(cols):
        n = len(c['force'])
        for arr,key in [(baseline,'baseline'), (guard,'guard'), (fixed,'fixed'),
                        (periodicity,'periodicity'), (ratio,'ratio'), (force,'force')]:
            arr[ci,:n] = c[key]
        logforce[ci,:n] = [math.log(f) if f > 0 else math.nan for f in c['force']]

    counts = {key: np.empty((U,V), dtype=np.int64) for key in (
        'n_all_edits', 'n_eligible_edits', 'n_changed_columns', 'n_all_changed_columns',
        'invalid_eligible_columns', 'no_run_eligible_columns',
        'left_censored_eligible_columns', 'nonpositive_or_tied_eligible_columns')}
    lower = np.full((U,V,S), np.nan)
    upper = np.full((U,V,S), np.nan)
    witness_dtype = np.int16 if len(eligible_indices) <= np.iinfo(np.int16).max else np.int64
    lower_witness = np.full((U,V,S), -1, dtype=witness_dtype)
    upper_witness = np.full((U,V,S), -1, dtype=witness_dtype)
    invalid_witness = np.full((U,V), -1, dtype=witness_dtype)
    ratio_gate = ratio[None,:,:] < va['representative'][:,None,None]
    column_index = np.arange(C)[None,:]
    for ui, u in enumerate(ua['representative']):
        labels = fixed[None,:,:] | (ratio_gate & (guard & (periodicity >= u))[None,:,:])
        differences = labels != baseline[None,:,:]
        cost = np.count_nonzero(differences, axis=2)
        counts['n_all_edits'][ui] = cost.sum(axis=1)
        counts['n_eligible_edits'][ui] = cost[:,eligible_indices].sum(axis=1)
        counts['n_changed_columns'][ui] = np.count_nonzero(cost[:,eligible_indices], axis=1)
        counts['n_all_changed_columns'][ui] = np.count_nonzero(cost, axis=1)
        if k > N:
            has_run = np.zeros((V,C), dtype=bool)
            starts = np.zeros((V,C), dtype=int)
        else:
            runs = labels[:,:,:N-k+1].copy()
            for offset in range(1,k):
                runs &= labels[:,:,offset:offset+N-k+1]
            has_run = np.any(runs, axis=2)
            starts = np.argmax(runs, axis=2)
        prev = np.maximum(starts-1, 0)
        flo, fhi = force[column_index,prev], force[column_index,starts]
        finite = has_run & (starts > 0) & (flo > 0) & (fhi > flo)
        bad = ~finite[:,eligible_indices]
        invalid_counts = np.count_nonzero(bad, axis=1)
        counts['invalid_eligible_columns'][ui] = invalid_counts
        counts['no_run_eligible_columns'][ui] = np.count_nonzero(~has_run[:,eligible_indices], axis=1)
        counts['left_censored_eligible_columns'][ui] = np.count_nonzero(
            (has_run & (starts == 0))[:,eligible_indices], axis=1)
        counts['nonpositive_or_tied_eligible_columns'][ui] = np.count_nonzero(
            (has_run & (starts > 0) & ~finite)[:,eligible_indices], axis=1)
        if len(eligible_indices):
            invalid_witness[ui] = np.where(invalid_counts > 0, np.argmax(bad,axis=1), -1)
            lo = (logforce[column_index,prev][:,eligible_indices,None]
                  - beta_log[eligible_indices][None,:,None]*p)
            hi = (logforce[column_index,starts][:,eligible_indices,None]
                  - beta_log[eligible_indices][None,:,None]*p)
            valid_rows = invalid_counts == 0
            lower[ui,valid_rows] = np.max(lo[valid_rows], axis=1)
            upper[ui,valid_rows] = np.min(hi[valid_rows], axis=1)
            lower_witness[ui,valid_rows] = np.argmax(lo[valid_rows], axis=1)
            upper_witness[ui,valid_rows] = np.argmin(hi[valid_rows], axis=1)

    compatible = lower[:,:,:,None]-eps <= upper[:,:,:,None]+eps
    metadata = dict(n_all_windows=sum(len(c['force']) for c in cols),
                    n_eligible_windows=sum(len(cols[i]['force']) for i in eligible_indices),
                    n_all_columns=C, n_eligible_columns=len(eligible_indices),
                    column_ids=[c['id'] for c in cols],
                    eligible_column_ids=[cols[i]['id'] for i in eligible_indices],
                    eligible_indices=eligible_indices,
                    n_guarded_windows=sum(int(c['guard'].sum()) for c in cols))
    return dict(schema_version=1, group=group, u_domain=ud, v_domain=vd,
                k=int(k), slopes=p, epsilons=eps, u_atoms=ua, v_atoms=va,
                **counts, lower=lower, upper=upper, gap=lower-upper,
                lower_witness=lower_witness, upper_witness=upper_witness,
                invalid_witness=invalid_witness, compatible=compatible,
                metadata=metadata,
                interpretation='Exact joint-rule operational compatibility; not physical truth or an MLE test.')
