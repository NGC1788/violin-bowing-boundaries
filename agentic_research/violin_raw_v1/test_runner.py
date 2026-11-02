"""Synthetic lifecycle and adversarial access checks; no real held-out labels."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from agentic_research.violin_raw_v1 import runner
from agentic_research.violin_raw_v1.models import patterns


def fixture(tmp_path, monkeypatch):
    base = tmp_path / 'agentic_research/violin_raw_v1'; base.mkdir(parents=True)
    campaign = runner.Campaign(base)
    campaign.prepared.mkdir(parents=True)
    (base / 'execution_freeze_manifest.json').write_text('synthetic test manifest')
    protocol = json.loads((runner.HERE / 'protocol.json').read_text())
    # Monkeypatch only a temporary campaign's verification in synthetic tests.
    # Production has no bypass flag and actual files are never touched.
    monkeypatch.setattr(campaign, 'verify', lambda **kwargs: ({}, protocol))
    n = 12; rng = np.random.default_rng(12)
    features = {'X': rng.normal(size=(n, 60)), 'prefix_states': np.arange(n) % 10,
                'prefix_Y9': patterns()[np.arange(n) % 10], 'prefix_guard': np.ones(n, bool),
                'participant_ids': np.array(['a_CG'] * 4 + ['b_FG'] * 2 + ['c_EG'] * 2 + ['d_CG'] * 2 + ['e_FG'] * 2),
                'split': np.array(['train'] * 4 + ['dev'] * 4 + ['test'] * 4),
                'source_ids': np.array([f's{i}' for i in range(n)])}
    np.savez_compressed(campaign.prepared / 'features.npz', **features)
    for split, count in [('train', 4), ('dev', 4), ('test', 4)]:
        np.savez_compressed(campaign.prepared / f'{split}_labels.npz', Y9=patterns()[np.arange(count)],
                            target_guard=np.array([True, False, True, True]), targets=np.ones((count, 2)))
    return campaign, features, protocol


def fake_fit(method, seed, x, y, states, people, xe, se, config):
    assert len(x) == len(y) == 4 and len(xe) == 8
    return np.tile(np.linspace(.1, .9, 9), (len(xe), 1)), {'method': method, 'seed': seed, 'seconds': 0.0}


def test_training_never_parses_held_out_labels_even_if_they_are_invalid(tmp_path, monkeypatch):
    campaign, _, _ = fixture(tmp_path, monkeypatch)
    for split in ('dev', 'test'): (campaign.prepared / f'{split}_labels.npz').write_bytes(b'not a numpy archive')
    calls = []; original = np.load
    def guarded(path, *args, **kwargs):
        name = Path(path).name; calls.append(name)
        assert name not in ('dev_labels.npz', 'test_labels.npz')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', guarded)
    state = campaign.train(fake_fit)
    assert state['status'] == 'TRAINED' and state['attempts'] == len(state['completed']) == 12
    assert calls == ['features.npz', 'train_labels.npz']
    assert (campaign.run / 'predictions_seal.json').exists()
    assert not (campaign.run / 'test_opened.json').exists()


def test_full_lifecycle_seals_selection_before_single_test_read(tmp_path, monkeypatch):
    campaign, _, _ = fixture(tmp_path, monkeypatch)
    campaign.train(fake_fit)
    original = np.load; test_reads = []
    def observed(path, *args, **kwargs):
        if Path(path).name == 'test_labels.npz':
            assert (campaign.run / 'predictions_seal.json').exists()
            assert (campaign.run / 'selection.json').exists()
            assert (campaign.run / 'test_opened.json').exists()
            test_reads.append(str(path))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', observed)
    result = campaign.evaluate()
    assert len(test_reads) == 1 and result['selected_baseline'] == 'frequency'
    assert result['training_attempts'] == 12 and result['test']['mlp_ind']['seeds'] == 3
    assert json.loads((campaign.run / 'state.json').read_text())['stopped']
    with pytest.raises(ValueError, match='already evaluated'): campaign.evaluate()
    assert len(test_reads) == 1
    # train() also does not start a thirteenth attempt after completion.
    assert campaign.train(lambda *a: pytest.fail('unexpected retry'))['status'] == 'EVALUATED'


def test_failed_or_interrupted_fit_cannot_be_retried(tmp_path, monkeypatch):
    campaign, _, _ = fixture(tmp_path, monkeypatch)
    def broken(*args): raise RuntimeError('synthetic training failure')
    with pytest.raises(RuntimeError): campaign.train(broken)
    state = json.loads((campaign.run / 'state.json').read_text())
    assert state['status'] == 'FAILED' and state['attempts'] == 1
    with pytest.raises(ValueError, match='no retry'): campaign.train(fake_fit)
    with pytest.raises(ValueError, match='halted'): campaign.evaluate()
    assert not (campaign.run / 'test_opened.json').exists()


def test_timeout_is_recorded_and_stops_without_test_access(tmp_path, monkeypatch):
    campaign, _, protocol = fixture(tmp_path, monkeypatch)
    protocol['budget']['trial_seconds'] = .01
    def too_slow(*args):
        import time
        time.sleep(.2)
        pytest.fail('timer did not stop the synthetic job')
    with pytest.raises(TimeoutError): campaign.train(too_slow)
    assert json.loads((campaign.run / 'state.json').read_text())['status'] == 'FAILED'
    assert not (campaign.run / 'test_opened.json').exists()


@pytest.mark.parametrize('kind', ['prediction', 'incomplete_seal'])
def test_prediction_tampering_or_incomplete_seal_blocks_test(tmp_path, monkeypatch, kind):
    campaign, _, _ = fixture(tmp_path, monkeypatch)
    campaign.train(fake_fit)
    prediction = campaign.run / 'frequency_s0/predictions.npz'
    original = prediction.read_bytes()
    if kind == 'prediction':
        prediction.write_bytes(original + b'changed')
    else:
        seal_path = campaign.run / 'predictions_seal.json'
        seal = json.loads(seal_path.read_text()); seal['completed_jobs'].pop()
        seal_path.write_text(json.dumps(seal))
    with pytest.raises(ValueError): campaign.evaluate()
    assert not (campaign.run / 'test_opened.json').exists()
    assert (campaign.run / 'halt.json').exists()
    assert json.loads((campaign.run / 'state.json').read_text())['status'] == 'SAFETY_HALT'
    prediction.write_bytes(original)
    with pytest.raises(ValueError, match='halted'): campaign.evaluate()
    assert not (campaign.run / 'test_opened.json').exists()


def test_file_hash_deadline_and_predictor_field_guards(tmp_path, monkeypatch):
    p = tmp_path / 'fixed'; p.write_text('original'); hashes = {'fixed': runner.file_sha(p)}
    runner.check_hashes(tmp_path, hashes); p.write_text('changed')
    with pytest.raises(ValueError): runner.check_hashes(tmp_path, hashes)
    campaign, features, _ = fixture(tmp_path, monkeypatch)
    np.savez_compressed(campaign.prepared / 'features.npz', **features, target_guard=np.ones(12, bool))
    with pytest.raises(ValueError, match='predictor fields'): runner.load_training_inputs(campaign.prepared)


def test_seed_loss_mean_is_not_ensemble_probability_loss():
    y = np.zeros((4, 9)); people = np.array(['a_CG', 'a_CG', 'b_EG', 'b_EG']); guard = np.ones(4, bool)
    p = [np.full((4, 9), .1), np.full((4, 9), .9)]
    result = runner.score_family(y, p, people, guard)
    assert abs(result['log_loss'] - (-np.log(.9) - np.log(.1)) / 2) < 1e-12
    assert abs(result['log_loss'] - (-np.log(.5))) > .1


@pytest.mark.parametrize('stop', ['deadline', 'attempt_budget', 'interrupted'])
def test_deadline_attempt_budget_and_interrupted_state_block_new_fits(tmp_path, monkeypatch, stop):
    campaign, _, protocol = fixture(tmp_path, monkeypatch)
    if stop == 'deadline': protocol['budget']['end_utc'] = '2000-01-01T00:00:00+00:00'
    elif stop == 'attempt_budget': protocol['budget']['max_training_attempts'] = 0
    else:
        campaign.run.mkdir(parents=True)
        (campaign.run / 'state.json').write_text(json.dumps({'status': 'RUNNING', 'attempts': 1}))
    with pytest.raises((TimeoutError, ValueError)):
        campaign.train(lambda *a: pytest.fail('fit called despite termination condition'))
    assert not (campaign.run / 'test_opened.json').exists()


def freeze_fixture(tmp_path, monkeypatch):
    base = tmp_path / 'agentic_research/violin_raw_v1'; base.mkdir(parents=True)
    campaign = runner.Campaign(base); campaign.prepared.mkdir(parents=True)
    (base / 'local/reviews').mkdir()
    monkeypatch.setattr(runner, 'verify_data_freeze', lambda *a: {})
    (base / 'data_freeze_manifest.json').write_text('synthetic data freeze')
    files = {}
    for name in ('features.npz', 'train_labels.npz', 'dev_labels.npz', 'test_labels.npz'):
        p = campaign.prepared / name; p.write_bytes(('opaque synthetic ' + name).encode()); files[name] = runner.file_sha(p)
    receipt = {'data_freeze_sha256': runner.file_sha(base / 'data_freeze_manifest.json'), 'files': files}
    runner.write_json(campaign.prepared / 'receipt.json', receipt)
    runner.write_json(base / 'PREPARED_SUMMARY.json', {'data_freeze_sha256': receipt['data_freeze_sha256'],
        'prepared_receipt_sha256': runner.file_sha(campaign.prepared / 'receipt.json'), 'file_sha256': files})
    for name in ('runner.py', 'models.py', 'metrics.py', 'test_runner.py'):
        (base / name).write_text('# synthetic source only\n')
    (base / 'protocol.json').write_text((runner.HERE / 'protocol.json').read_text())
    source = {str((base / name).relative_to(campaign.repository)): runner.file_sha(base / name)
              for name in ('runner.py', 'models.py', 'metrics.py', 'protocol.json', 'test_runner.py', 'PREPARED_SUMMARY.json')}
    snapshot_sha = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    runner.write_json(base / 'local/reviews/execution_snapshot.json', {'snapshot_sha256': snapshot_sha, 'source_hashes': source})
    runner.write_json(base / 'EXECUTION_REVIEW.json', {'snapshot_sha256': snapshot_sha,
        'scope': 'training_evaluation', 'verdict': 'approve', 'blockers': []})
    return campaign


def test_freeze_binds_review_source_map_and_previously_prepared_data(tmp_path, monkeypatch):
    campaign = freeze_fixture(tmp_path, monkeypatch)
    manifest = campaign.freeze()
    assert manifest['prepared_files']['test_labels.npz'] == runner.file_sha(campaign.prepared / 'test_labels.npz')
    with pytest.raises(ValueError, match='existing'): campaign.freeze()


def test_stale_review_cannot_approve_a_changed_source_map(tmp_path, monkeypatch):
    campaign = freeze_fixture(tmp_path, monkeypatch)
    changed = campaign.base / 'runner.py'; changed.write_text('# a different implementation')
    path = campaign.base / 'local/reviews/execution_snapshot.json'; snapshot = json.loads(path.read_text())
    snapshot['source_hashes'][str(changed.relative_to(campaign.repository))] = runner.file_sha(changed)
    runner.write_json(path, snapshot)  # The old review SHA was deliberately retained.
    with pytest.raises(ValueError, match='bind'): campaign.freeze()


def test_substituting_prepared_data_even_with_rehashed_receipt_is_rejected(tmp_path, monkeypatch):
    campaign = freeze_fixture(tmp_path, monkeypatch)
    path = campaign.prepared / 'receipt.json'; receipt = json.loads(path.read_text())
    changed = campaign.prepared / 'test_labels.npz'; changed.write_bytes(b'different synthetic labels')
    receipt['files']['test_labels.npz'] = runner.file_sha(changed)
    runner.write_json(path, receipt)
    with pytest.raises(ValueError, match='reviewed public summary'): campaign.freeze()


def test_lock_contention_does_not_halt_the_lock_owner(tmp_path, monkeypatch):
    import fcntl
    campaign, _, _ = fixture(tmp_path, monkeypatch)
    campaign.run.mkdir(parents=True)
    with (campaign.run / 'campaign.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError): campaign.train(fake_fit)
    assert not (campaign.run / 'halt.json').exists()
    assert not (campaign.run / 'state.json').exists()
