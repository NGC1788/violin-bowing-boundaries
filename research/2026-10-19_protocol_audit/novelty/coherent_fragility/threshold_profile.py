"""Exact threshold-arrangement profiling for coherent first-K-run decisions.

One scalar threshold is shared by EVERY supplied column and group.  No label is
optimized independently.  A group is a predeclared folder; its fixed eligible
columns constrain a common-intercept, fixed-slope log-force law.  Groups have
separate intercepts but share the classifier threshold.

Only NumPy is needed.  This module neither loads empirical files nor selects a
preferred threshold.  See THEORY.md for the exactness argument and limitations.
"""
from __future__ import annotations

import math
from typing import Any, Iterable
import numpy as np


def first_run_start(labels: Iterable[bool], k: int) -> int | None:
    """Index of the first K consecutive positive labels, or None."""
    run = 0
    for i, positive in enumerate(labels):
        run = run + 1 if positive else 0
        if run == k:
            return i - k + 1
    return None


def _binary(value: Any, name: str, n: int | None = None) -> np.ndarray:
    a = np.asarray(value)
    if a.ndim != 1 or (n is not None and len(a) != n):
        raise ValueError(f'{name} must be a one-dimensional array of matching length')
    try:
        ok = np.all((a == 0) | (a == 1))
    except (TypeError, ValueError):
        ok = False
    if not ok:
        raise ValueError(f'{name} must contain only binary values')
    return np.asarray(a, dtype=bool).copy()


def _finite_vector(value: Any, name: str, nonnegative: bool = False) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError(f'{name} must be a nonempty finite one-dimensional array')
    if nonnegative and np.any(a < 0):
        raise ValueError(f'{name} must be nonnegative')
    return a.copy()


def _valid_bracket(force: np.ndarray, labels: np.ndarray, k: int):
    s = first_run_start(labels, k)
    if s is None:
        return None, 'no_run'
    if s == 0:
        return None, 'left_censored'
    if not (0 < force[s - 1] < force[s]):
        return None, 'nonpositive_or_tied'
    return (math.log(force[s - 1]), math.log(force[s])), None


def analyze_thresholds(columns: Iterable[dict], relation: str,
                       domain: tuple[float, float],
                       slopes: Iterable[float] = (-2., -1.2),
                       epsilons: Iterable[float] = (.2,), k: int = 2) -> dict:
    """Enumerate every reachable classifier pattern exactly, including ties.

    Required column keys are ``id, group, beta, force, baseline, guard, values,
    fixed, eligible``. Arrays must have the same length; force is ascending,
    finite, and already in force order. ``eligible`` is fixed by the caller
    from the ORIGINAL classifier, and every True column must have an original
    finite positive nonzero-width first-run bracket. Setting it False retains
    its labels in the all-window edit count but excludes it from law constraints.

    For relation='ge': H = fixed | (guard & (values >= threshold)).
    For relation='lt': H = fixed | (guard & (values < threshold)).
    Guard and fixed must be disjoint. Values can be finite or NaN, but a NaN
    must have guard=False. The domain is CLOSED and may have infinite endpoints.

    Returns NumPy arrays (not JSON). ``groups`` gives deterministic sorted group
    order. ``atoms`` describes threshold cells: every cell is right closed,
    the first is left closed, all subsequent cells left open. Its representative
    is its right endpoint, which may be infinite and has exactly the cell's
    labels. The singleton [domain_lo, domain_lo] is retained when needed.

    Main arrays have shape [n_patterns, n_groups]: n_all_edits,
    n_eligible_edits, n_changed_columns (eligible columns with >=1 changed
    label), n_all_changed_columns, invalid_eligible_columns. Per-cause invalid
    counts are also supplied. ``lower, upper, gap`` have an additional slope
    axis, BEFORE epsilon expansion. ``compatible`` has further slope, epsilon
    axes. It is false if ANY eligible boundary becomes invalid or if the group
    has no eligible columns; the latter is reported separately via metadata.
    This deliberately prevents disappearance/censoring from becoming support.

    The routine returns profiles, not a physical model-selection p-value.
    Exactness is combinatorial; endpoint calculations use IEEE float64 logs.
    """
    if relation not in ('ge', 'lt'):
        raise ValueError("relation must be 'ge' or 'lt'")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError('k must be a positive integer')
    p = _finite_vector(slopes, 'slopes')
    eps = _finite_vector(epsilons, 'epsilons', nonnegative=True)
    try:
        d = np.asarray(domain, dtype=float)
    except (TypeError, ValueError) as e:
        raise ValueError('domain must contain two ordered non-NaN endpoints') from e
    if d.shape != (2,) or np.isnan(d).any() or d[0] > d[1]:
        raise ValueError('domain must contain two ordered non-NaN endpoints')
    dlo, dhi = float(d[0]), float(d[1])

    cols = []
    ids = set()
    for raw in columns:
        name, group = raw['id'], raw['group']
        if not isinstance(name, str) or not name or name in ids:
            raise ValueError('column id must be a unique nonempty string')
        if not isinstance(group, str) or not group:
            raise ValueError('column group must be a nonempty string')
        ids.add(name)
        beta = float(raw['beta'])
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError(f'{name}: beta must be positive and finite')
        force = np.asarray(raw['force'], dtype=float)
        if force.ndim != 1 or not np.isfinite(force).all() or np.any(np.diff(force) < 0):
            raise ValueError(f'{name}: force must be a sorted finite one-dimensional array')
        force = force.copy()
        n = len(force)
        baseline = _binary(raw['baseline'], f'{name}: baseline', n)
        guard = _binary(raw['guard'], f'{name}: guard', n)
        fixed = _binary(raw['fixed'], f'{name}: fixed', n)
        values = np.asarray(raw['values'], dtype=float)
        if values.shape != (n,) or np.isinf(values).any():
            raise ValueError(f'{name}: values must be finite-or-NaN and match force length')
        values = values.copy()
        if np.any(guard & np.isnan(values)):
            raise ValueError(f'{name}: NaN values require guard=False')
        if np.any(fixed & guard):
            raise ValueError(f'{name}: fixed and guard must be disjoint')
        eligible = raw['eligible']
        if not isinstance(eligible, (bool, np.bool_)):
            raise ValueError(f'{name}: eligible must be boolean')
        bracket, reason = _valid_bracket(force, baseline, k)
        if eligible and reason is not None:
            raise ValueError(f'{name}: eligible baseline lacks a finite positive first-run bracket')
        labels = fixed | (guard & ((values >= dlo) if relation == 'ge' else (values < dlo)))
        cols.append(dict(id=name, group=group, beta=beta, force=force,
                         baseline=baseline, guard=guard, fixed=fixed, values=values,
                         eligible=bool(eligible), labels=labels,
                         log_beta=math.log(beta)))

    groups = sorted({c['group'] for c in cols})
    gi = {g: i for i, g in enumerate(groups)}
    G, C, S, E = len(groups), len(cols), len(p), len(eps)
    members = [[] for _ in groups]
    eligible_members = [[] for _ in groups]
    event_values, event_cols, event_positions = [], [], []
    for ci, c in enumerate(cols):
        g = gi[c['group']]
        c['group_index'] = g
        members[g].append(ci)
        if c['eligible']:
            eligible_members[g].append(ci)
        pos = np.flatnonzero(c['guard'] & (c['values'] >= dlo) & (c['values'] < dhi))
        event_values.extend(c['values'][pos].tolist())
        event_cols.extend([ci] * len(pos))
        event_positions.extend(pos.tolist())
    ev = np.asarray(event_values, dtype=float)
    ec = np.asarray(event_cols, dtype=np.int64)
    ep = np.asarray(event_positions, dtype=np.int64)
    order = np.argsort(ev, kind='stable')
    ev, ec, ep = ev[order], ec[order], ep[order]
    if len(ev):
        starts = np.r_[0, 1 + np.flatnonzero(ev[1:] != ev[:-1])]
        event_levels = ev[starts]
        ends = np.r_[starts[1:], len(ev)]
    else:
        starts = ends = np.array([], dtype=int)
        event_levels = np.array([], dtype=float)
    P = len(event_levels) + 1
    atom_lo = np.r_[dlo, event_levels]
    atom_hi = np.r_[event_levels, dhi]
    atom_lc = np.zeros(P, dtype=bool)
    atom_lc[0] = True
    atoms = dict(lo=atom_lo, hi=atom_hi, lo_closed=atom_lc,
                 hi_closed=np.ones(P, dtype=bool), representative=atom_hi.copy())

    counts = {name: np.empty((P, G), dtype=np.int64) for name in (
        'n_all_edits', 'n_eligible_edits', 'n_changed_columns',
        'n_all_changed_columns', 'invalid_eligible_columns',
        'no_run_eligible_columns', 'left_censored_eligible_columns',
        'nonpositive_or_tied_eligible_columns')}
    lower = np.full((P, G, S), np.nan)
    upper = np.full((P, G, S), np.nan)
    cost = np.zeros(C, dtype=np.int64)
    col_lower = np.full((C, S), np.nan)
    col_upper = np.full((C, S), np.nan)
    reason = [None] * C
    valid = np.zeros(C, dtype=bool)
    member_arrays = [np.asarray(x, dtype=int) for x in members]
    eligible_arrays = [np.asarray(x, dtype=int) for x in eligible_members]

    def refresh_column(ci):
        c = cols[ci]
        cost[ci] = np.count_nonzero(c['labels'] != c['baseline'])
        if not c['eligible']:
            return
        bracket, reason[ci] = _valid_bracket(c['force'], c['labels'], k)
        valid[ci] = bracket is not None
        if bracket is not None:
            col_lower[ci] = bracket[0] - p * c['log_beta']
            col_upper[ci] = bracket[1] - p * c['log_beta']
        else:
            col_lower[ci] = col_upper[ci] = np.nan

    def refresh_group(row, g):
        all_i, elig_i = member_arrays[g], eligible_arrays[g]
        counts['n_all_edits'][row, g] = cost[all_i].sum()
        counts['n_eligible_edits'][row, g] = cost[elig_i].sum()
        counts['n_changed_columns'][row, g] = np.count_nonzero(cost[elig_i])
        counts['n_all_changed_columns'][row, g] = np.count_nonzero(cost[all_i])
        bad = int(np.count_nonzero(~valid[elig_i]))
        counts['invalid_eligible_columns'][row, g] = bad
        for reason_name, key in (
            ('no_run', 'no_run_eligible_columns'),
            ('left_censored', 'left_censored_eligible_columns'),
            ('nonpositive_or_tied', 'nonpositive_or_tied_eligible_columns')):
            counts[key][row, g] = sum(reason[i] == reason_name for i in elig_i)
        if len(elig_i) and not bad:
            lower[row, g] = np.max(col_lower[elig_i], axis=0)
            upper[row, g] = np.min(col_upper[elig_i], axis=0)
        else:
            lower[row, g] = upper[row, g] = np.nan

    for ci in range(C):
        refresh_column(ci)
    for g in range(G):
        refresh_group(0, g)
    new_label = relation == 'lt'
    for row, (a, b) in enumerate(zip(starts, ends), start=1):
        # All equal scores flip atomically. Intermediates would be unreachable
        # by a single threshold and MUST NOT be scored as feasible solutions.
        dirty_cols = set()
        for j in range(int(a), int(b)):
            ci = int(ec[j])
            cols[ci]['labels'][ep[j]] = new_label
            dirty_cols.add(ci)
        for arr in counts.values():
            arr[row] = arr[row - 1]
        lower[row] = lower[row - 1]
        upper[row] = upper[row - 1]
        dirty_groups = set()
        for ci in dirty_cols:
            refresh_column(ci)
            dirty_groups.add(cols[ci]['group_index'])
        for g in dirty_groups:
            refresh_group(row, g)

    gap = lower - upper
    # Compute with the SAME explicit shifted-endpoint predicate as the
    # single-threshold oracle, rather than gap <= 2*eps (different rounding).
    compatible = ((lower[..., None] - eps) <= (upper[..., None] + eps))
    metadata = dict(
        n_all_windows=np.asarray([sum(len(cols[i]['force']) for i in x) for x in members], dtype=np.int64),
        n_eligible_windows=np.asarray([sum(len(cols[i]['force']) for i in x) for x in eligible_members], dtype=np.int64),
        n_all_columns=np.asarray([len(x) for x in members], dtype=np.int64),
        n_eligible_columns=np.asarray([len(x) for x in eligible_members], dtype=np.int64),
        eligible_column_ids=[[cols[i]['id'] for i in x] for x in eligible_members],
        column_ids=[[cols[i]['id'] for i in x] for x in members])
    global_counts = {key: arr.sum(axis=1) for key, arr in counts.items()}
    global_counts['n_compatible_groups'] = compatible.sum(axis=1)
    return dict(schema_version=1, relation=relation, domain=(dlo, dhi), k=int(k),
                slopes=p, epsilons=eps, groups=groups, atoms=atoms,
                **counts, lower=lower, upper=upper, gap=gap,
                compatible=compatible, group_metadata=metadata,
                global_counts=global_counts,
                n_guarded_windows=sum(int(c['guard'].sum()) for c in cols),
                n_transition_windows=len(ev),
                interpretation='Exact shared-threshold operational compatibility; not physical truth or an MLE test.')
