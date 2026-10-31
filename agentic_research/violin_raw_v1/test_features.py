import numpy as np
import pytest
from agentic_research.violin_raw_v1.features import extract_prefix, extract_example, interval_frames, PREFIX, TARGET, feature_names


def tone(sr=16000, hz=440, seconds=2, amplitude=.2):
    return amplitude * np.sin(2 * np.pi * hz * np.arange(round(sr * seconds)) / sr)


@pytest.mark.parametrize('sr', [8000, 16000, 22050, 44100, 48000])
def test_disjoint_physical_support_and_known_tone(sr):
    x = tone(sr)
    row = extract_example(x, sr, 'DOUBLE')
    assert row['X'].shape == (60,)
    assert len(feature_names()) == len(set(feature_names())) == 60
    a, b = row['frame_support'], row['target_support']
    assert len(a) == 96 and len(b) == 46
    assert a.min() >= 0 and a[:, 1].max() <= sr
    assert b[:, 0].min() >= round(1.25 * sr) and b[:, 1].max() <= round(1.75 * sr)
    assert b[:, 0].min() - a[:, 1].max() >= round(.25 * sr)
    assert row['prefix_guard'] and row['target_guard']
    assert row['targets'][0] < 2.0
    frames = interval_frames(x, sr, 'DOUBLE', PREFIX)
    assert abs(np.median(frames['pitch_cents'])) < 3.0
    assert abs(np.median(frames['power_dbfs']) - 20 * np.log10(.2 / np.sqrt(2))) < .05


@pytest.mark.parametrize('hz', [196, 220, 440, 880, 1318.51])
def test_pitch_scale_and_harmonic_rich_tones(hz):
    sr = 16000
    x = tone(sr, hz) + .08 * np.sin(2 * np.pi * 2 * hz * np.arange(sr * 2) / sr)
    f = interval_frames(x, sr, 'DOUBLE', PREFIX)
    assert f['voiced'].all()
    assert abs(np.median(f['pitch_cents']) - 1200 * np.log2(hz / 440)) < 10


def test_future_and_gap_values_cannot_change_input_bitwise():
    sr = 16000
    x = tone(sr)
    before = extract_prefix(x, sr, 'DOUBLE')
    rng = np.random.default_rng(34)
    for future in (rng.normal(0, 1e8, sr), np.full(sr, np.nan), np.full(sr, np.inf)):
        altered = np.concatenate((x[:sr], future))
        after = extract_prefix(altered, sr, 'DOUBLE')
        np.testing.assert_array_equal(before['X'], after['X'])
        np.testing.assert_array_equal(before['prefix_targets'], after['prefix_targets'])
        np.testing.assert_array_equal(before['frame_support'], after['frame_support'])
    np.testing.assert_array_equal(before['X'], extract_prefix(x[:sr], sr, 'DOUBLE')['X'])


def test_prefix_and_beyond_target_changes_cannot_change_target():
    sr = 16000
    x = tone(sr)
    expected = extract_example(x, sr, 'DOUBLE')
    changed = x.copy()
    changed[:sr] = 0
    changed[sr:round(1.25 * sr)] = np.nan
    changed[round(1.75 * sr):] = np.nan
    actual = extract_example(changed, sr, 'DOUBLE')
    np.testing.assert_array_equal(expected['targets'], actual['targets'])
    assert actual['target_guard'] and not actual['prefix_guard']


def test_fixed_pcm_scaling_and_stereo():
    sr = 16000
    pcm = np.rint(tone(sr) * 32767).astype(np.int16)
    direct = pcm.astype(float) / 32768
    reference = extract_example(direct, sr, 'DOUBLE')
    representations = [(pcm, 'PCM_16'), (np.column_stack([pcm, pcm]), 'PCM_16'),
                       (pcm.astype(np.int32) << 16, 'PCM_24'), (pcm.astype(np.int32) << 16, 'PCM_32')]
    for samples, subtype in representations:
        actual = extract_example(samples, sr, subtype)
        np.testing.assert_array_equal(reference['X'], actual['X'])
        np.testing.assert_array_equal(reference['targets'], actual['targets'])
    unsigned = np.rint(tone(sr) * 127 + 128).astype(np.uint8)
    np.testing.assert_array_equal(extract_prefix(unsigned, sr, 'PCM_U8')['X'],
                                  extract_prefix((unsigned.astype(float) - 128) / 128, sr, 'DOUBLE')['X'])


def test_silence_and_noise_are_not_dropped_or_imputed_as_valid():
    sr = 16000
    silence = extract_example(np.zeros(sr * 2), sr, 'DOUBLE')
    assert not silence['prefix_guard'] and not silence['target_guard']
    assert np.isnan(silence['targets'][0]) and silence['targets'][1] == 0
    assert np.isfinite(silence['X']).all()
    x = np.random.default_rng(77).normal(0, .1, sr * 2)
    noise = extract_example(x, sr, 'DOUBLE')
    assert not noise['target_guard']
    assert np.isfinite(noise['targets'][1])
    quiet_future = tone(sr)
    quiet_future[round(1.25 * sr):] = 0
    row = extract_example(quiet_future, sr, 'DOUBLE')
    assert row['prefix_guard'] and not row['target_guard']


def test_variability_changes_target_but_not_input():
    sr = 16000
    original = tone(sr)
    changed = original.copy()
    start, stop = round(1.25 * sr), round(1.75 * sr)
    t = np.arange(stop - start) / sr
    # A prescribed frequency sweep creates more instability than a fixed tone.
    changed[start:stop] = (.1 + .2 * t) * np.sin(2 * np.pi * (440 * t + 30 * t * t))
    a, b = extract_example(original, sr, 'DOUBLE'), extract_example(changed, sr, 'DOUBLE')
    np.testing.assert_array_equal(a['X'], b['X'])
    assert b['targets'][0] > a['targets'][0] + 10
    assert b['targets'][1] > a['targets'][1] + .1


def test_invalid_input_rejected_in_supported_interval_only():
    x = tone()
    with pytest.raises(ValueError): extract_prefix(x, 16000.0, 'DOUBLE')
    with pytest.raises(ValueError): extract_prefix(x[:100], 16000, 'DOUBLE')
    with pytest.raises(ValueError): extract_prefix(x, 16000, 'PCM_16')
    with pytest.raises(ValueError): extract_prefix(x.astype(np.int16), 16000, 'DOUBLE')
    with pytest.raises(ValueError): extract_prefix(np.ones((32000, 3)), 16000, 'DOUBLE')
    with pytest.raises(ValueError): interval_frames(x, 16000, 'DOUBLE', (0, 2))
    x[0] = np.nan
    with pytest.raises(ValueError): extract_prefix(x, 16000, 'DOUBLE')
