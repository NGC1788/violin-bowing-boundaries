import io
import tarfile
import wave
import numpy as np
import pytest
from agentic_research.violin_raw_v1.inventory import classify_member, wav_metadata, scan_archive, summarize


def wav_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(np.arange(32000, dtype=np.int16).tobytes())
    return buffer.getvalue()


def test_member_selection_excludes_voice_reference_and_ag():
    row, reason = classify_member('root/Participant12_CG/pre/Note3_Trial2_pitch.wav')
    assert row['participant_id'] == 'Participant12_CG' and reason == 'selected_violin'
    for tail in ['Note2_GS.wav', 'Note2_MC.wav', 'Note2_PV_pitch.wav']:
        assert classify_member('Participant1_EG/feed/' + tail)[0]
    for path, expected in [('Participant1_CG/pre/Note2_Sing_Trial1_pitch.wav', 'voice'),
                           ('Participant1_CG/pre/Note2_Trial1_ref.wav', 'reference'),
                           ('Participant1_AG/pre/Note2_Trial1_pitch.wav', 'different_AG_cohort'),
                           ('Participant1_CG/pre/surprise.wav', 'unclassified_wav')]:
        assert classify_member(path) == (None, expected)
    for path in ['/a.wav', '../a.wav', 'a/../../b.wav', 'C:\\a.wav']:
        with pytest.raises(ValueError): classify_member(path)


def test_wav_headers_and_identity_are_not_feature_extraction(tmp_path):
    path = tmp_path / 'example.wav'; path.write_bytes(wav_bytes())
    meta = wav_metadata(path)
    assert meta['duration_seconds'] == 2 and meta['sample_rate'] == 16000
    assert meta['eligible_duration'] and meta['subtype'] == 'PCM_16'
    assert not meta['prefix_constant_pcm']
    changed = bytearray(wav_bytes()); changed[-2:] = b'\0\0'
    path.write_bytes(changed); altered = wav_metadata(path)
    assert altered['pcm_sha256'] != meta['pcm_sha256']
    assert altered['prefix_pcm_sha256'] == meta['prefix_pcm_sha256']
    path.write_bytes(changed[:-1])
    with pytest.raises(ValueError): wav_metadata(path)


def test_real_archive_reader_selects_and_extracts_only_violin(tmp_path):
    archive = tmp_path / 'fixture.tar'
    with tarfile.open(archive, 'w') as tar:
        for name in ['Participant1_CG/pre/Note1_Trial1_pitch.wav',
                     'Participant1_CG/pre/Note1_Sing_Trial1_pitch.wav']:
            entry = tarfile.TarInfo(name); entry.size = len(wav_bytes())
            tar.addfile(entry, io.BytesIO(wav_bytes()))
    records, base = scan_archive(archive, tmp_path / 'extracted')
    assert len(records) == 1 and base['exclusions']['voice'] == 1
    assert len(list((tmp_path / 'extracted').glob('*.wav'))) == 1
    summary, collisions = summarize(records, base)
    assert summary['metadata_gate_passed'] and summary['duration_eligible'] == 1
    assert not collisions['pcm_sha256']


def test_cross_participant_prefix_copy_blocks_metadata_gate(tmp_path):
    path = tmp_path / 'x.wav'; path.write_bytes(wav_bytes())
    meta = wav_metadata(path)
    rows = [{**meta, 'participant_id': pid, 'cohort': 'CG', 'stage': 'pre', 'source_id': pid}
            for pid in ['a', 'b']]
    summary, _ = summarize(rows, {'exclusions': {}})
    assert not summary['metadata_gate_passed']
    assert summary['pcm_sha256_cross_participant_groups'] == 1


def test_selected_symlink_is_rejected(tmp_path):
    path = tmp_path / 'bad.tar'
    with tarfile.open(path, 'w') as tar:
        entry = tarfile.TarInfo('Participant1_CG/pre/Note1_Trial1_pitch.wav')
        entry.type = tarfile.SYMTYPE; entry.linkname = '/etc/passwd'; tar.addfile(entry)
    with pytest.raises(ValueError, match='ordinary file'): scan_archive(path, tmp_path / 'out')
