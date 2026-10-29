"""Trusted sparse audit benchmark. Hidden labels are revealed only through query()."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from .data import load_dataset
from .models import fit_prior
from .run_distribution import certificate, first_run, run_distribution, information_gain, certificate_lower_bound

BUDGETS = (5, 10, 20, 35, 50)
POLICIES = ('ascending', 'random', 'even_first', 'uncertainty', 'entropy')


class QueryOracle:
    def __init__(self, hidden_labels):
        self.__labels = np.asarray(hidden_labels, dtype=np.int8).copy()
        self.queried = set()

    def query(self, position):
        if isinstance(position, bool) or not isinstance(position, (int, np.integer)):
            raise TypeError('Position must be integer')
        position = int(position)
        if not 0 <= position < self.__labels.shape[1] or position in self.queried:
            raise ValueError('Invalid or repeated signal query')
        self.queried.add(position)
        return self.__labels[:, position].copy()


def even_order(n):
    # For K=2, observing actual zeros at all even positions hits every adjacent
    # pair. This is a stronger structure-aware baseline than uniform spacing.
    return list(range(0, n, 2)) + list(range(1, n, 2))


def simulate(prior, query, policy, rng, budgets=BUDGETS):
    """Adaptive learner sees a prior and query replies, never the hidden label array."""
    if policy not in POLICIES:
        raise ValueError('Unknown acquisition policy')
    p = np.asarray(prior, dtype=float).copy()
    if p.ndim != 2 or p.shape[1] != 50 or not np.isfinite(p).all():
        raise ValueError('Expected rule×50 finite prior')
    rules, n = p.shape
    observed = np.full((rules, n), -1, dtype=np.int8)
    chosen = []
    snapshots = {}
    sequence = list(range(n)) if policy == 'ascending' else list(rng.permutation(n)) if policy == 'random' else even_order(n)
    for count in range(n+1):
        certified, low, high = certificate(observed)
        q = np.where(observed >= 0, observed, p)
        if count in budgets:
            snapshots[count] = run_distribution(q)
        if certified.all():
            terminal = np.zeros((rules, n+1))
            terminal[np.arange(rules), low] = 1.
            for budget in budgets:
                if budget >= count:
                    snapshots[budget] = terminal.copy()
            return {'queries': chosen, 'observed': observed, 'certified_start': low,
                    'snapshots': np.stack([snapshots[b] for b in budgets]), 'certificate_count': count}
        if count == n:
            raise AssertionError('Fully observed labels must certify')
        unknown = observed[0] < 0
        if policy in ('ascending', 'random', 'even_first'):
            pos = next(i for i in sequence if unknown[i])
        elif policy == 'uncertainty':
            score = (q[~certified]*(1-q[~certified])).sum(axis=0)
            score[~unknown] = -np.inf
            pos = int(np.argmax(score))
        else:
            score = information_gain(q[~certified], unknown)
            pos = int(np.argmax(score))
        reply = np.asarray(query(pos))
        if reply.shape != (rules,) or not np.isin(reply, [0, 1]).all():
            raise ValueError('Invalid label-vector oracle response')
        observed[:, pos] = reply
        chosen.append(int(pos))
    raise AssertionError('Unreachable')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_experiment(config: dict, data_path: Path, output_dir: Path) -> dict:
    """Write metrics/model/predictions only in a fresh caller-owned output directory."""
    config = dict(config)
    if config.get('policy', 'ascending') not in POLICIES:
        raise ValueError('Unknown policy')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir/name).exists() for name in ('metrics.json', 'predictions.npz', 'run_config.json')):
        raise FileExistsError('Preserve prior run; choose a fresh output directory')
    wall = time.perf_counter()
    train, dev, rules, receipt = load_dataset(data_path, config.get('rule_bank', 'nine'),
        config.get('max_train_columns'), config.get('max_eval_columns', 16))
    (output_dir/'run_config.json').write_text(json.dumps(config, sort_keys=True, indent=2, allow_nan=False)+'\n')
    (output_dir/'data_receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    cpu = time.process_time()
    with threadpool_limits(limits=2):
        model, training = fit_prior(config, train, rules)
    train_cpu = time.process_time()-cpu
    model.save(output_dir)
    acquisition_wall, acquisition_cpu = time.perf_counter(), time.process_time()
    records, prediction, truth, query_trace, priors = [], [], [], [], []
    seed = int(config.get('seed', 0))
    for column in dev:
        prior = model.predict(column.public)
        oracle = QueryOracle(column.labels)
        id_seed = int(hashlib.sha256(column.id.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed ^ id_seed)
        result = simulate(prior, oracle.query, config.get('policy', 'ascending'), rng)
        # Full labels become visible only here, after adaptive acquisition has terminated.
        target = first_run(column.labels)
        errors = int(np.count_nonzero(result['certified_start'] != target))
        if errors:
            raise AssertionError('Certificate error: invalidate run instead of publishing a successful metric')
        lower, per_rule_lower = certificate_lower_bound(column.labels)
        if len(oracle.queried) != result['certificate_count'] or result['certificate_count'] < lower:
            raise AssertionError('Query accounting/certificate lower bound violation')
        records.append({'id': column.id, 'session': column.session, 'group': column.group,
                        'queries': result['certificate_count'], 'oracle_lower_bound': lower,
                        'per_rule_lower_bound': per_rule_lower, 'certificate_errors': errors})
        prediction.append(result['snapshots']); truth.append(target); priors.append(prior)
        query_trace.append(result['queries']+[-1]*(50-len(result['queries'])))
    acq_seconds = time.perf_counter()-acquisition_wall
    acq_cpu = time.process_time()-acquisition_cpu
    prediction, truth = np.stack(prediction), np.stack(truth)
    rank = np.arange(51)
    means = (prediction*rank).sum(axis=-1)
    absolute = np.abs(means-truth[:, None, :])
    counts = np.array([r['queries'] for r in records])
    bounds = np.array([r['oracle_lower_bound'] for r in records])
    anytime = {str(b): {'mean_absolute_start_rank_error': float(absolute[:, i].mean()),
                        'per_rule_mae': absolute[:, i].mean(axis=0).tolist(),
                        'fully_certified_column_fraction': float((counts <= b).mean())} for i, b in enumerate(BUDGETS)}
    groups = {}
    for s in sorted({c.session for c in dev}):
        mask = np.array([c.session == s for c in dev])
        groups[s] = {'columns': int(mask.sum()), 'mean_queries': float(counts[mask].mean()),
                     'mean_lower_bound': float(bounds[mask].mean()),
                     'anytime_mae': {str(b): float(absolute[mask, i].mean()) for i, b in enumerate(BUDGETS)}}
    errors = sum(r['certificate_errors'] for r in records)
    metrics = {
        'schema_version': 1, 'id': config.get('id', 'unnamed'), 'config': config,
        'config_sha256': hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
        'input_sha256': receipt['input_sha256'], 'harness_sha256': {p.name: _sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        'scope': 'Sparse auditing of already recorded signals against a pinned proxy observer; retrospective exposed development, no physical truth or new unseen confirmation.',
        'train_columns': len(train), 'eval_columns': len(dev), 'rules': len(rules),
        'train_sessions': sorted({c.session for c in train}), 'eval_sessions': sorted({c.session for c in dev}),
        'mean_queries': float(counts.mean()), 'median_queries': float(np.median(counts)),
        'p90_queries': float(np.percentile(counts, 90)), 'max_queries': int(counts.max()),
        'queries_saved_vs_full_grid_fraction': float(1-counts.mean()/50),
        'mean_lower_bound': float(bounds.mean()), 'mean_queries_minus_lower_bound': float((counts-bounds).mean()),
        'lower_bound_scope': 'Maximum of exact per-rule hindsight certificate sizes; a joint lower bound, not joint optimum and not a deployable baseline.',
        'certificate_error_count': errors, 'all_certified': errors == 0,
        'anytime': anytime, 'groups': groups, 'training': training,
        'timing': {'train_wall_seconds': training['fit_seconds'], 'train_cpu_seconds': train_cpu,
                   'acquisition_wall_seconds': acq_seconds, 'acquisition_cpu_seconds': acq_cpu,
                   'total_wall_seconds': time.perf_counter()-wall},
        'certificate_definition': 'First K=2 run unchanged between unknown bits all-0 and all-1, simultaneously for every declared rule.',
        'entropy_policy_scope': 'Sum of per-rule marginal first-run information gains under independent Bernoulli priors; not exact joint entropy.',
        'columns': records,
    }
    np.savez_compressed(output_dir/'predictions.npz', probabilities=prediction, target_start=truth,
        prior=np.stack(priors), query_indices=np.asarray(query_trace), budgets=np.asarray(BUDGETS),
        ids=np.asarray([c.id for c in dev]), thresholds=rules)
    metrics['predictions_sha256'] = _sha(output_dir/'predictions.npz')
    (output_dir/'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False)+'\n')
    return metrics
