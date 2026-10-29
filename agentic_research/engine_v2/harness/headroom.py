"""Trusted, label-aware feasibility diagnostic before adaptive model search.

K=2 joint certificates reduce to a minimum vertex cover on a path with forced
vertices. This is classical dynamic programming, not a new general algorithm.
The output is private audit evidence: individual bounds/IDs must never enter
model inputs, acquisition policies, or LLM candidate-selection context.
"""
from __future__ import annotations
from fractions import Fraction
from pathlib import Path
from typing import Any, TypedDict
import hashlib
import json
import math
import numpy as np
from .data import load_dataset


class BoundRecord(TypedDict):
    id: str
    exact_joint_minimum: int


class BaselineReference(TypedDict):
    run_id: str
    mean_queries: float
    manifest_sha: str
    metrics_sha256: str
    sealed_metrics_sha256: str | None
    config: dict[str, Any]


class HeadroomReport(TypedDict):
    schema_version: int
    gate_pass: bool
    attainable_10percent: bool
    eval_ids: list[str]
    eval_ids_sha: str
    input_sha256: str
    manifest_sha: str
    protocol_sha: str
    module_sha256: str
    chosen_baseline: BaselineReference
    exact_joint_bounds: list[BoundRecord]
    mean_exact_joint_bound: float
    headroom_fraction: float
    required_relative_improvement: float


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def minimum_joint_certificate(labels):
    """Return exact minimum size and one witness for binary rule×position labels.

    A finite first pair forces its two positive positions. Every earlier pair
    needs a queried actual zero. One available zero is forced; two give the
    adjacent-vertex clause x_i OR x_(i+1). Unioning these clauses across rules
    yields a path vertex-cover problem, solved in O(rules*n+n) time.
    """
    rows = np.atleast_2d(np.asarray(labels))
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] == 0 or not np.isin(rows, [0, 1]).all():
        raise ValueError('Expected a nonempty binary rule-by-position matrix')
    n = rows.shape[1]
    forced = [False]*n
    edge = [False]*max(0, n-1)
    for row in rows:
        start = next((i for i in range(n-1) if row[i] and row[i+1]), n)
        if start < n:
            forced[start] = forced[start+1] = True
        for i in range(min(start, n-1)):
            left_zero, right_zero = not row[i], not row[i+1]
            if left_zero and right_zero:
                edge[i] = True
            elif left_zero:
                forced[i] = True
            elif right_zero:
                forced[i+1] = True
            else:
                raise AssertionError('A pair before the first pair cannot be all positive')
    infinity = n+1
    costs = [infinity if forced[0] else 0, 1]
    back = []
    for i in range(1, n):
        nxt, previous = [infinity, infinity], [-1, -1]
        for bit in (0, 1):
            if forced[i] and bit == 0:
                continue
            for prior in (0, 1):
                if edge[i-1] and prior == bit == 0:
                    continue
                value = costs[prior]+bit
                if value < nxt[bit]:
                    nxt[bit], previous[bit] = value, prior
        costs = nxt
        back.append(previous)
    bit = min((0, 1), key=lambda b: costs[b])
    optimum = costs[bit]
    witness = []
    for i in range(n-1, -1, -1):
        if bit:
            witness.append(i)
        if i:
            bit = back[i-1][bit]
            if bit < 0:
                raise AssertionError('Missing optimal predecessor')
    witness.reverse()
    if len(witness) != optimum:
        raise AssertionError('Certificate reconstruction cost mismatch')
    return int(optimum), tuple(witness)


def _baseline_reference(row, eval_ids, input_sha, benchmark, rules):
    if row.get('status', 'completed') != 'completed':
        raise ValueError('Headroom accepts completed baseline results only')
    metrics, config = row['metrics'], row['config']
    if config.get('prior') not in ('constant', 'uniform'):
        raise ValueError('A learned prior cannot define the nonlearned reference baseline')
    if config.get('policy') not in ('ascending', 'random', 'even_first', 'uncertainty', 'entropy'):
        raise ValueError('Unknown nonlearned baseline policy')
    for name, default in [('rule_bank', 'nine'), ('max_eval_columns', 16), ('max_train_columns', None)]:
        if config.get(name, default) != benchmark.get(name, default):
            raise ValueError(f'Baseline benchmark mismatch: {name}')
    if metrics.get('input_sha256') != input_sha or metrics.get('rules') != rules:
        raise ValueError('Baseline data hash or rule count mismatch')
    if metrics.get('certificate_error_count') != 0 or metrics.get('all_certified') is not True:
        raise ValueError('Baseline is not a verified zero-error certified run')
    columns = metrics['columns']
    if [x['id'] for x in columns] != eval_ids or metrics.get('eval_columns') != len(eval_ids):
        raise ValueError('Baseline must use the identical ordered evaluation selection')
    queries = [x['queries'] for x in columns]
    if any(isinstance(q, bool) or not isinstance(q, int) or not 0 <= q <= 50 for q in queries):
        raise ValueError('Invalid baseline query accounting')
    if any(x.get('certificate_errors') != 0 for x in columns):
        raise ValueError('Per-column certificate error')
    total = sum(queries)
    if total <= 0:
        raise ValueError('Expected positive baseline query cost')
    score = float(metrics['mean_queries'])
    if not math.isfinite(score) or not math.isclose(score, total/len(queries), rel_tol=0, abs_tol=1e-12):
        raise ValueError('Baseline mean is inconsistent with recorded query counts')
    return total, queries, {
        'run_id': str(row['run_id']), 'mean_queries': score,
        'metrics_sha256': _digest(metrics), 'sealed_metrics_sha256': row.get('result_sha', row.get('metrics_sha')),
        'config': config,
    }


def make_report(data_path: Path, protocol: dict, baseline_runs: list[dict], manifest_sha: str) -> HeadroomReport:
    """Read fixed data and verified baseline rows; return a JSON-serializable report.

    Caller must first freeze/check the campaign with control.integrity and obtain
    baseline_runs from the same campaign's integrity-checked control.report.
    Caller writes/seals this report. This function performs no output writes and
    runs no training/policy evaluation. A passing bound is only a necessary
    possibility test, not evidence that an implementable policy attains it.
    """
    if not isinstance(manifest_sha, str) or len(manifest_sha) != 64 or any(c not in '0123456789abcdef' for c in manifest_sha):
        raise ValueError('Expected the current frozen manifest SHA-256')
    benchmark = protocol.get('benchmark', {})
    if benchmark.get('eval_selection_seed', 20261030) != 20261030:
        raise ValueError('Unexpected selection seed')
    _, dev, thresholds, receipt = load_dataset(data_path, benchmark.get('rule_bank', 'nine'),
        benchmark.get('max_train_columns'), benchmark.get('max_eval_columns', 16))
    eval_ids = [c.id for c in dev]
    if not baseline_runs:
        raise ValueError('At least one completed nonlearned reference baseline is required')
    validated = [_baseline_reference(row, eval_ids, receipt['input_sha256'], benchmark, len(thresholds))
                 for row in baseline_runs]
    total_baseline, best_queries, chosen = min(validated, key=lambda value: (value[0], value[2]['run_id']))
    chosen['manifest_sha'] = manifest_sha
    bounds = [{'id': c.id, 'exact_joint_minimum': minimum_joint_certificate(c.labels)[0]} for c in dev]
    if any(b['exact_joint_minimum'] > q for b, q in zip(bounds, best_queries)):
        raise AssertionError('An exact hindsight minimum exceeds an observed certified query cost')
    total_bound = sum(b['exact_joint_minimum'] for b in bounds)
    required = protocol.get('acceptance', {}).get('minimum_mean_query_reduction_vs_best_nonlearned', .10)
    if isinstance(required, bool) or not isinstance(required, (int, float)) or not math.isfinite(required) or not 0 <= required < 1:
        raise ValueError('Required relative improvement must be in [0,1)')
    # Rational comparison avoids a rounding-dependent boundary at exactly 10%.
    possible = Fraction(total_bound) <= (1-Fraction(str(required)))*total_baseline
    ten_percent_possible = 10*total_bound <= 9*total_baseline
    return {
        'schema_version': 1, 'gate_pass': bool(possible), 'attainable_10percent': bool(ten_percent_possible),
        'eval_ids': eval_ids, 'eval_ids_sha': _digest(eval_ids),
        'input_sha256': receipt['input_sha256'], 'manifest_sha': manifest_sha,
        'protocol_sha': _digest(protocol), 'module_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'benchmark_sha': _digest(benchmark), 'benchmark_id': protocol.get('version', 'unspecified'),
        'chosen_baseline': chosen, 'baseline_candidates_checked': len(validated),
        'baseline_selection_scope': 'Best mean among the supplied, completed nonlearned baselines in this frozen campaign.',
        'exact_joint_bounds': bounds, 'mean_exact_joint_bound': total_bound/len(bounds),
        'total_exact_joint_bound': total_bound, 'total_baseline_queries': total_baseline,
        'headroom_fraction': (total_baseline-total_bound)/total_baseline,
        'required_relative_improvement': float(required),
        'reason': ('The hindsight bound does not rule out the declared improvement; policy attainability remains unproved.'
                   if possible else 'Even the label-aware optimal certificate cannot attain the declared improvement against this baseline.'),
        'scope': 'Label-aware hindsight diagnostic for K=2 on exposed archived data. Not a deployable policy, learned result, or unseen confirmation.',
        'privacy': 'Individual IDs/bounds stay in trusted audit storage and must not be sent to models, acquisition policies or the candidate-planning LLM.',
        'algorithm': 'Classical minimum vertex cover on a path with forced vertices; linear dynamic programming.',
        'reference': 'Independent audit joint_certificate_reference.py motivated this implementation; independent brute-force subset/completion tests verify it.',
    }
