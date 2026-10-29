"""Independent brute-force validation and integrity checks; no real-data scoring."""
from functools import lru_cache
from types import SimpleNamespace
import itertools
import json
import numpy as np
import pytest
from harness import headroom


def brute_first(row):
    for i in range(len(row)-1):
        if row[i] == row[i+1] == 1:
            return i
    return len(row)


@lru_cache(None)
def completion_outcomes(n):
    result = {}
    for observed in itertools.product((-1, 0, 1), repeat=n):
        missing = [i for i, x in enumerate(observed) if x == -1]
        possibilities = set()
        for fill in itertools.product((0, 1), repeat=len(missing)):
            full = list(observed)
            for i, x in zip(missing, fill):
                full[i] = x
            possibilities.add(brute_first(full))
        result[observed] = possibilities
    return result


def brute_certificate(rows, outcomes):
    n = len(rows[0])
    for size in range(n+1):
        for chosen in itertools.combinations(range(n), size):
            keep = set(chosen)
            if all(len(outcomes[tuple(int(row[i]) if i in keep else -1 for i in range(n))]) == 1 for row in rows):
                return size
    raise AssertionError('Fully observed matrix must certify')


def test_exact_joint_path_dp_against_6160_independent_exhaustive_cases():
    rng = np.random.default_rng(20261031)
    count = 0
    for n in range(1, 8):
        outcomes = completion_outcomes(n)
        vectors = list(itertools.product((0, 1), repeat=n))
        banks = list(itertools.product(vectors, repeat=2)) if n <= 6 else []
        banks.extend(rng.integers(0, 2, (9, n)).tolist() for _ in range(100))
        for rows in banks:
            size, witness = headroom.minimum_joint_certificate(rows)
            assert size == brute_certificate(rows, outcomes)
            assert len(set(witness)) == len(witness) == size
            assert all(len(outcomes[tuple(int(row[i]) if i in witness else -1 for i in range(n))]) == 1 for row in rows)
            count += 1
    assert count == 6160


def test_joint_bound_can_be_stricter_than_maximum_individual_bound():
    rows = [[0, 0, 1, 1], [1, 1, 0, 0]]
    joint, _ = headroom.minimum_joint_certificate(rows)
    individual = [headroom.minimum_joint_certificate(r)[0] for r in rows]
    assert joint > max(individual)


@pytest.mark.parametrize('bad', [[], [[]], [[0, 2]], [[0, np.nan]], np.zeros((2, 2, 2))])
def test_invalid_label_matrices_rejected(bad):
    with pytest.raises(ValueError):
        headroom.minimum_joint_certificate(bad)


def fake_report_inputs(monkeypatch):
    dev = []
    for i in range(4):
        row = [1, 1]+[0]*48 if i < 3 else [0, 1, 1]+[0]*47
        dev.append(SimpleNamespace(id=f'dev-{i}', labels=np.tile(row, (9, 1))))
    receipt = {'input_sha256': 'a'*64}
    monkeypatch.setattr(headroom, 'load_dataset', lambda *args, **kwargs: ([], dev, np.zeros((9, 2)), receipt))
    protocol = {'version': 'synthetic-only', 'benchmark': {'rule_bank': 'nine', 'max_eval_columns': 4,
        'max_train_columns': None, 'eval_selection_seed': 20261030},
        'acceptance': {'minimum_mean_query_reduction_vs_best_nonlearned': .1}}
    queries = [3, 2, 2, 3]  # exact lower bounds [2,2,2,3] => 10% headroom exactly
    config = {'prior': 'uniform', 'policy': 'ascending', 'rule_bank': 'nine',
              'max_eval_columns': 4, 'max_train_columns': None}
    metrics = {'input_sha256': 'a'*64, 'rules': 9, 'eval_columns': 4,
               'certificate_error_count': 0, 'all_certified': True, 'mean_queries': 2.5,
               'columns': [{'id': c.id, 'queries': q, 'certificate_errors': 0} for c, q in zip(dev, queries)]}
    baseline = {'status': 'completed', 'run_id': 'baseline', 'result_sha': 'b'*64, 'config': config, 'metrics': metrics}
    return protocol, baseline


def test_headroom_report_deterministic_with_exact_gate_boundary(monkeypatch):
    protocol, baseline = fake_report_inputs(monkeypatch)
    a = headroom.make_report('unused', protocol, [baseline], 'c'*64)
    b = headroom.make_report('unused', protocol, [baseline], 'c'*64)
    assert a == b
    assert a['gate_pass'] and a['attainable_10percent']
    assert a['headroom_fraction'] == pytest.approx(.1)
    assert a['mean_exact_joint_bound'] == 2.25
    assert a['total_exact_joint_bound'] == 9
    assert a['chosen_baseline']['manifest_sha'] == 'c'*64
    json.dumps(a, allow_nan=False)
    protocol['acceptance']['minimum_mean_query_reduction_vs_best_nonlearned'] = .100001
    c = headroom.make_report('unused', protocol, [baseline], 'c'*64)
    assert not c['gate_pass'] and c['attainable_10percent']


def test_baseline_result_hash_alias_is_stable(monkeypatch):
    protocol, baseline = fake_report_inputs(monkeypatch)
    a = headroom.make_report('unused', protocol, [baseline], 'c'*64)
    baseline['metrics_sha'] = baseline.pop('result_sha')
    b = headroom.make_report('unused', protocol, [baseline], 'c'*64)
    assert a == b


@pytest.mark.parametrize('tamper', ['ids', 'score', 'rules', 'data', 'error', 'prior', 'status', 'query', 'config'])
def test_report_rejects_mismatched_or_invalid_baseline(monkeypatch, tamper):
    protocol, baseline = fake_report_inputs(monkeypatch)
    m = baseline['metrics']
    if tamper == 'ids': m['columns'].reverse()
    if tamper == 'score': m['mean_queries'] += .01
    if tamper == 'rules': m['rules'] = 1
    if tamper == 'data': m['input_sha256'] = 'd'*64
    if tamper == 'error': m['certificate_error_count'] = 1
    if tamper == 'prior': baseline['config']['prior'] = 'mlp_bce'
    if tamper == 'status': baseline['status'] = 'failed'
    if tamper == 'query': m['columns'][0]['queries'] = 2.5
    if tamper == 'config': baseline['config']['max_eval_columns'] = 16
    with pytest.raises(ValueError): headroom.make_report('unused', protocol, [baseline], 'c'*64)


def test_bound_exceeding_baseline_invalidates_the_audit(monkeypatch):
    protocol, baseline = fake_report_inputs(monkeypatch)
    baseline['metrics']['columns'][-1]['queries'] = 2
    baseline['metrics']['mean_queries'] = 2.25
    with pytest.raises(AssertionError): headroom.make_report('unused', protocol, [baseline], 'c'*64)


def test_best_provided_completed_nonlearned_baseline_is_selected(monkeypatch):
    protocol, baseline = fake_report_inputs(monkeypatch)
    slow = json.loads(json.dumps(baseline))
    slow['run_id'] = 'slow'
    slow['metrics']['columns'][0]['queries'] += 1
    slow['metrics']['mean_queries'] += .25
    result = headroom.make_report('unused', protocol, [slow, baseline], 'c'*64)
    assert result['chosen_baseline']['run_id'] == 'baseline'
    assert result['baseline_candidates_checked'] == 2
