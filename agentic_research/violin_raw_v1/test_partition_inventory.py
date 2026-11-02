import pytest
from agentic_research.violin_raw_v1.partition_inventory import reconcile, write_once


def fixture():
    rows = []
    for i in range(8):
        name = f'Participant{i}_CG/pre/Note1_Trial1_pitch.wav'
        rows.append({'source_id': name, 'participant_id': f'Participant{i}_CG', 'cohort': 'CG',
                     'eligible_duration': True, 'pcm_sha256': str(i), 'prefix_pcm_sha256': 'prefix' + str(i)})
    names = [r['source_id'] for r in rows] + ['Participant1_AG2/pre/Note1_Trial1_pitch.wav']
    return rows, names, {'archive_entries': len(names), 'exclusions': {'unclassified_wav': 1}}


def test_ag2_resolution_preserves_declared_selection():
    rows, names, base = fixture()
    manifest, receipt = reconcile(rows, names, base)
    assert len(manifest) == 8 and not receipt['selection_changed']
    assert receipt['exclusions']['AG2_outside_declared_cohorts'] == 1
    assert not receipt['future_targets_computed']


def test_unknown_core_recording_or_changed_selected_set_blocks_partition():
    rows, names, base = fixture()
    names[-1] = 'Participant1_CG/pre/surprise.wav'
    with pytest.raises(ValueError, match='unresolved'): reconcile(rows, names, base)
    rows, names, base = fixture()
    with pytest.raises(ValueError, match='change selected'): reconcile(rows[:-1], names, base)


def test_seal_is_idempotent_but_cannot_be_replaced(tmp_path):
    path = tmp_path / 'seal.json'
    write_once(path, {'a': 1}); write_once(path, {'a': 1})
    with pytest.raises(ValueError): write_once(path, {'a': 2})
