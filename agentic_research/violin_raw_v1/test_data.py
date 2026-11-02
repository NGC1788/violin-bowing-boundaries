import copy
import itertools
import numpy as np
import pytest
from agentic_research.violin_raw_v1 import data, models


def participants():
    return [{'participant_id': f'{c}_{i}', 'cohort': c}
            for c, n in [('CG', 28), ('FG', 29), ('EG', 15)] for i in range(n)]


def fixture():
    rows, split = [], {'a': 'train', 'b': 'train', 'c': 'dev', 'd': 'test'}
    rng = np.random.default_rng(7)
    for pid in split:
        for j in range(5):
            sid = f'{pid}/{j}'
            valid = j != 4
            rows.append({'source_id': sid, 'participant_id': pid,
                         'pcm_sha256': data.digest(['full', sid]), 'prefix_pcm_sha256': data.digest(['prefix', sid]),
                         'X': rng.normal(size=60), 'targets': np.array([j + .1 if valid else np.nan, j + 1.1]),
                         'prefix_targets': np.array([j + .2 if valid else np.nan, j + 1.2]),
                         'target_guard': valid, 'prefix_guard': valid})
    return rows, split


def test_participant_split_metadata_only_and_order_invariant():
    people = participants()
    first, receipt = data.participant_split(people)
    other, _ = data.participant_split(list(reversed(people)))
    assert first == other
    assert {s: list(first.values()).count(s) for s in data.SPLITS} == {'train': 49, 'dev': 10, 'test': 13}
    assert receipt['cohort_counts']['EG'] == {'train': 10, 'dev': 2, 'test': 3}
    with pytest.raises(ValueError): data.participant_split(people + [people[0]])
    with pytest.raises(ValueError): data.participant_split([{'participant_id': 'a', 'cohort': 'AG'}])
    with pytest.raises(ValueError): data.participant_split(people[:3])


def test_thresholds_ignore_all_held_out_values_and_guards():
    rows, manifest = fixture()
    original = data.assemble_rows(rows, manifest)
    y = original['targets'].copy()
    g = original['target_guard'].copy()
    held = original['split'] != 'train'
    y[held] = np.nan
    g[held] = ~g[held]
    changed, receipt = data.fit_thresholds(y, g, original['split'], original['participant_ids'])
    np.testing.assert_array_equal(changed, original['thresholds'])
    assert receipt == original['metadata']['threshold_fit']
    # Even invalid train pitch values cannot set the quantiles.
    y[(original['split'] == 'train') & ~g] = 1e50
    changed2, _ = data.fit_thresholds(y, g, original['split'], original['participant_ids'])
    np.testing.assert_array_equal(changed2, changed)


def test_rule_bank_exhaustive_parity_and_ties():
    t = np.array([[1., 2., 3.], [1., 2., 3.]])
    values = np.array(list(itertools.product([.5, 1.5, 2.5, 3.5], repeat=2)))
    labels, states = data.labels_and_states(values, np.ones(16, bool), t)
    assert len(np.unique(labels, axis=0)) == 10
    np.testing.assert_array_equal(data.patterns(), models.patterns())
    np.testing.assert_array_equal(states, models.pattern_ids(labels))
    permitted = {tuple(x) for x in data.patterns()}
    assert sum(tuple(y) in permitted for y in itertools.product([0, 1], repeat=9)) == 10
    labels, _ = data.labels_and_states(np.array([[1., 2.]]), np.ones(1, bool), t)
    np.testing.assert_array_equal(labels[0].reshape(3, 3), [[0, 1, 1]] * 3)


def test_invalid_future_retained_and_prefix_gets_state_10():
    rows, manifest = fixture()
    ds = data.assemble_rows(rows, manifest)
    assert len(ds['X']) == 20
    assert int((~ds['target_guard']).sum()) == 4
    assert (ds['Y9'][~ds['target_guard']] == 0).all()
    assert (ds['prefix_states'][~ds['prefix_guard']] == 10).all()
    assert ds['pattern_ids'].max() < 10
    assert ds['metadata']['participant_counts'] == {'train': 2, 'dev': 1, 'test': 1}


def test_duplicate_aliases_and_cross_person_collision_failure():
    rows, manifest = fixture()
    duplicate = copy.deepcopy(rows[0]); duplicate['source_id'] = 'z_alias'
    ds = data.assemble_rows(rows + [duplicate], manifest)
    assert len(ds['X']) == 20 and ds['metadata']['aliases_removed'] == 1
    assert ds['alias_manifest']['a/0'] == ['a/0', 'z_alias']
    duplicate['participant_id'] = 'd'
    with pytest.raises(ValueError, match='PCM crosses'): data.assemble_rows(rows + [duplicate], manifest)
    duplicate['pcm_sha256'] = data.digest('different full PCM')
    with pytest.raises(ValueError, match='prefix PCM crosses'): data.assemble_rows(rows + [duplicate], manifest)


def test_participant_mixing_is_rejected_before_quantiles():
    y = np.ones((6, 2)); guard = np.ones(6, bool)
    with pytest.raises(ValueError, match='participant crosses'):
        data.fit_thresholds(y, guard, ['train', 'train', 'train', 'train', 'dev', 'test'], ['a'] * 6)


def test_degenerate_train_quantiles_fail_without_jitter():
    y = np.ones((6, 2)); guard = np.ones(6, bool)
    with pytest.raises(ValueError, match='degenerate'):
        data.fit_thresholds(y, guard, ['train'] * 4 + ['dev', 'test'], ['a'] * 4 + ['b', 'c'])


def test_future_mutation_changes_labels_only_not_prefix_or_train_fit():
    rows, manifest = fixture()
    original = data.assemble_rows(rows, manifest)
    changed = copy.deepcopy(rows)
    for r in changed:
        if manifest[r['participant_id']] == 'test':
            r['targets'] = np.array([1e5, 1e6]); r['target_guard'] = True
    altered = data.assemble_rows(changed, manifest)
    for name in ('X', 'prefix_states', 'prefix_Y9', 'thresholds', 'split', 'participant_ids'):
        np.testing.assert_array_equal(original[name], altered[name])
    assert not np.array_equal(original['Y9'], altered['Y9'])


def test_invalid_shapes_power_or_manifest_fail():
    rows, manifest = fixture()
    with pytest.raises(ValueError): data.assemble_rows(rows, {**manifest, 'absent': 'test'})
    changed = copy.deepcopy(rows); changed[0]['X'] = np.zeros(59)
    with pytest.raises(ValueError): data.assemble_rows(changed, manifest)
    t = np.array([[1, 2, 3], [1, 2, 3]])
    with pytest.raises(ValueError): data.labels_and_states([[np.nan, np.nan]], np.array([False]), t)
    with pytest.raises(ValueError): data.labels_and_states([[1, 2]], np.array([1]), t)
