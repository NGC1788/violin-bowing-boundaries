import json
import wave
import numpy as np
import pytest
from agentic_research.violin_raw_v1.prepare import read_supported_samples, write_partitioned, verify_freeze
from agentic_research.violin_raw_v1.inventory import wav_metadata


def test_actual_decoder_matches_fixed_pcm_support_and_checks_identity(tmp_path):
    root = tmp_path; (root / 'violin_wav').mkdir()
    path = root / 'violin_wav/example.wav'
    x = np.arange(88200, dtype=np.int16)
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100); w.writeframes(x.tobytes())
    row = {'local_path': str(path), **wav_metadata(path)}
    np.testing.assert_array_equal(read_supported_samples(row, root), x[:77175])
    with pytest.raises(ValueError): read_supported_samples({**row, 'frames': 12}, root)
    with pytest.raises(ValueError): read_supported_samples(row, root / 'other')


def test_held_out_targets_never_stored_in_predictor_file(tmp_path):
    n = 6
    ds = {'X': np.zeros((n, 60)), 'prefix_states': np.arange(n), 'prefix_Y9': np.zeros((n, 9)),
          'prefix_guard': np.ones(n, bool), 'participant_ids': np.array(['a', 'a', 'b', 'b', 'c', 'c']),
          'split': np.array(['train', 'train', 'dev', 'dev', 'test', 'test']), 'source_ids': np.arange(n),
          'Y9': np.ones((n, 9)), 'targets': np.ones((n, 2)), 'target_guard': np.ones(n, bool)}
    hashes = write_partitioned(ds, tmp_path)
    assert len(hashes) == 4
    with np.load(tmp_path / 'features.npz', allow_pickle=False) as features:
        assert set(features.files) == {'X', 'prefix_states', 'prefix_Y9', 'prefix_guard', 'participant_ids', 'split', 'source_ids'}
        assert not {'Y9', 'targets', 'target_guard'}.intersection(features.files)
    for split in ('train', 'dev', 'test'):
        with np.load(tmp_path / f'{split}_labels.npz', allow_pickle=False) as labels:
            assert labels['Y9'].shape == (2, 9)


def test_missing_approval_or_expired_freeze_cannot_access_audio(tmp_path):
    path = tmp_path / 'freeze.json'
    path.write_text(json.dumps({'approved_scope': 'not approved'}))
    with pytest.raises(ValueError, match='reviewed'): verify_freeze(path)
    path.write_text(json.dumps({'approved_scope': 'data_preparation_only', 'end_utc': '2000-01-01T00:00:00+00:00'}))
    with pytest.raises(ValueError, match='deadline'): verify_freeze(path)
