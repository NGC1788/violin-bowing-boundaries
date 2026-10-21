"""Exact label-edit costs for a first consecutive-success run.

The labels are ordered from low to high force. A run starts at ``s`` if
``labels[s:s+k]`` are all one, and it is the first run if no earlier start
has that property. These functions describe an operational label summary;
they do not establish physical ground truth or classifier correctness.

Only NumPy is required. All finite costs are exact integer Hamming counts;
the float cost array is used solely to represent infeasibility as infinity.
"""

from __future__ import annotations

import operator
from typing import Any

import numpy as np


_MODES = frozenset(("both", "delete", "add"))


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not a boolean")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _validated(labels: Any, k: Any, mode: str) -> tuple[np.ndarray, int]:
    k = _positive_integer(k, "k")
    if not isinstance(mode, str) or mode not in _MODES:
        raise ValueError("mode must be 'both', 'delete', or 'add'")
    arr = np.asarray(labels)
    if arr.ndim != 1:
        raise ValueError("labels must be a one-dimensional binary sequence")
    if arr.dtype.kind not in "biuf":
        raise ValueError("labels must have a boolean or real numeric dtype")
    if not np.all((arr == 0) | (arr == 1)):
        raise ValueError("labels must contain only zero and one")
    return arr.astype(np.uint8, copy=True), k


def _edit_costs(labels: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-position costs of writing zero and writing one."""
    c0 = labels.astype(float)
    c1 = (1 - labels).astype(float)
    if mode == "delete":
        c1[labels == 0] = np.inf
    elif mode == "add":
        c0[labels == 1] = np.inf
    return c0, c1


def _step(dp: np.ndarray, c0: float, c1: float) -> np.ndarray:
    """Append one bit while forbidding a length-k run of ones."""
    result = np.empty_like(dp)
    result[0] = np.min(dp) + c0
    result[1:] = dp[:-1] + c1
    return result


def first_run_start(labels: Any, k: int = 2) -> int | None:
    """Return the first length-k success-run start, or ``None``.

    This helper applies the same input validation as the optimization API.
    """
    arr, k = _validated(labels, k, "both")
    trailing = 0
    for i, bit in enumerate(arr):
        trailing = trailing + 1 if bit else 0
        if trailing == k:
            return i - k + 1
    return None


def first_run_costs(labels: Any, k: int = 2, mode: str = "both") -> dict:
    """Minimum admissible Hamming edits for every possible first run.

    Parameters
    ----------
    labels : one-dimensional binary sequence
        Original labels, ordered from lower to higher force. The input is
        never changed. Empty sequences and sequences shorter than k work.
    k : positive integer
        Required consecutive successes; k=1 is included.
    mode : {'both', 'delete', 'add'}
        ``delete`` permits only 1 -> 0, ``add`` only 0 -> 1, and ``both``
        permits either direction. Keeping a label always has zero cost.

    Returns
    -------
    dict
        ``costs[s]`` is the minimum cost of making s the *first* run start.
        Its length is max(n-k+1, 0). ``no_run_cost`` is the minimum cost of
        having no length-k run anywhere. Impossible targets have ``inf``.
        ``n``, ``k`` and ``mode`` record the interpretation of the result.

    Complexity is O(n*k) time and O(n+k) memory (including the output).
    The special n<k case returns immediately after input validation.
    """
    arr, k = _validated(labels, k, mode)
    n = len(arr)
    if n < k:
        return {
            "costs": np.empty(0, dtype=float), "no_run_cost": 0,
            "n": n, "k": k, "mode": mode,
        }
    c0, c1 = _edit_costs(arr, mode)
    # Separate forbidden-write counts from finite sums: inf - inf would
    # otherwise corrupt sliding-window sums in the directional modes.
    forbidden = np.concatenate(([0], np.cumsum(~np.isfinite(c1))))
    finite_sum = np.concatenate(([0.0], np.cumsum(np.where(np.isfinite(c1), c1, 0.0))))
    window_cost = finite_sum[k:] - finite_sum[:-k]
    window_cost[(forbidden[k:] - forbidden[:-k]) != 0] = np.inf

    dp = np.full(k, np.inf)
    dp[0] = 0.0
    costs = np.empty(n - k + 1, dtype=float)
    for s in range(n + 1):
        if s <= n - k:
            # No earlier run, and the bit immediately before s is zero.
            # The empty prefix at s=0 also has dp[0]=0 by convention.
            costs[s] = dp[0] + window_cost[s]
        if s < n:
            dp = _step(dp, c0[s], c1[s])
    minimum = np.min(dp)
    return {
        "costs": costs,
        "no_run_cost": int(minimum) if np.isfinite(minimum) else np.inf,
        "n": n, "k": k, "mode": mode,
    }


def _prefix_witness(
    c0: np.ndarray, c1: np.ndarray, length: int, k: int, end_state: int | None,
) -> tuple[np.ndarray | None, float]:
    """Optimal run-free prefix, optionally constrained to its end state.

    Only the predecessor of state zero needs storing. Every positive state
    has its unique predecessor one smaller. Ties choose the smaller state.
    """
    dp = np.full(k, np.inf)
    dp[0] = 0.0
    zero_parent = np.empty(length, dtype=np.int64)
    for t in range(length):
        zero_parent[t] = np.argmin(dp)
        dp = _step(dp, c0[t], c1[t])
    state = int(np.argmin(dp)) if end_state is None else end_state
    cost = float(dp[state])
    if not np.isfinite(cost):
        return None, np.inf
    output = np.empty(length, dtype=np.uint8)
    for t in range(length - 1, -1, -1):
        if state == 0:
            output[t] = 0
            state = int(zero_parent[t])
        else:
            output[t] = 1
            state -= 1
    assert state == 0
    return output, cost


def witness_first_run(
    labels: Any, start: int | None, k: int = 2, mode: str = "both",
) -> dict:
    """Construct one minimum-edit witness for a target first run.

    ``start=None`` requests no run. A non-None start must be an integer in
    [0,n-k], otherwise ValueError is raised (TypeError for nonintegers).
    A finite witness returns edited ``labels``, ``flipped_indices``, an
    integer ``cost``, the requested ``start``, and ``feasible=True``.

    An impossible target returns the unchanged input, empty flip indices,
    ``cost=inf`` and ``feasible=False``. Those labels are NOT a witness for
    the requested target. Callers must check feasibility before using them.

    Witness construction is O(n*k) time and O(n+k) memory. Suffix labels
    after a forced first run stay unchanged, because they cannot affect it.
    """
    arr, k = _validated(labels, k, mode)
    n = len(arr)
    if start is not None:
        if isinstance(start, (bool, np.bool_)):
            raise TypeError("start must be an integer or None, not a boolean")
        try:
            start = operator.index(start)
        except TypeError as exc:
            raise TypeError("start must be an integer or None") from exc
        if not 0 <= start <= n - k:
            raise ValueError("start must be between zero and n-k inclusive")
    result = {
        "labels": arr.copy(), "flipped_indices": np.empty(0, dtype=np.int64),
        "cost": np.inf, "start": start, "feasible": False,
    }
    if n < k:
        # A non-None start was already rejected above.
        result.update(cost=0, feasible=True)
        return result
    c0, c1 = _edit_costs(arr, mode)
    if start is None:
        edited, optimum = _prefix_witness(c0, c1, n, k, None)
    else:
        forced_cost = float(np.sum(c1[start:start + k]))
        if not np.isfinite(forced_cost):
            return result
        prefix, prefix_cost = _prefix_witness(c0, c1, start, k, 0)
        if prefix is None:
            return result
        edited = arr.copy()
        edited[:start] = prefix
        edited[start:start + k] = 1
        optimum = prefix_cost + forced_cost
    if edited is None:
        return result
    flipped = np.flatnonzero(edited != arr)
    assert len(flipped) == optimum
    result.update(
        labels=edited, flipped_indices=flipped, cost=int(optimum), feasible=True,
    )
    return result
