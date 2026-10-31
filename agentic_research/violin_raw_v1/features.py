"""Explicit waveform support for a provisional, not-yet-frozen violin study.

No filtering, onset alignment, resampling, or normalization across intervals.
The model predicts these measured descriptors, not human musical quality.
"""
from __future__ import annotations

import numpy as np

PREFIX = (0.0, 1.0)
TARGET = (1.25, 1.75)
FRAME_SECONDS = 0.05
HOP_SECONDS = 0.01
MIN_HZ, MAX_HZ = 100.0, 1600.0
RMS_FLOOR = 1e-12
VOICED_RMS = 1e-4
VOICED_PERIODICITY = 0.5
PEAK_RELATIVE_HEIGHT = 0.95


def _rate(sample_rate: int) -> int:
    if isinstance(sample_rate, (bool, np.bool_)) or not isinstance(sample_rate, (int, np.integer)):
        raise ValueError('sample_rate must be an integer')
    if not 8000 <= sample_rate <= 192000:
        raise ValueError('unsupported sample rate')
    return int(sample_rate)


def _mono_interval(samples: np.ndarray, sample_rate: int, subtype: str,
                   interval: tuple[float, float]) -> tuple[np.ndarray, int]:
    """Slice before conversion and value checks. Never inspect other samples."""
    if not isinstance(samples, np.ndarray) or samples.ndim not in (1, 2):
        raise ValueError('samples must be a mono or stereo numpy array')
    if samples.ndim == 2 and samples.shape[1] not in (1, 2):
        raise ValueError('only mono or stereo audio is supported')
    start, stop = (round(t * sample_rate) for t in interval)
    if samples.shape[0] < stop:
        raise ValueError('insufficient recorded support')
    x = samples[start:stop]
    # scipy.io.wavfile.read conventions: 24-bit PCM is left-aligned int32.
    required = {'PCM_U8': ('u', 1), 'PCM_16': ('i', 2),
                'PCM_24': ('i', 4), 'PCM_32': ('i', 4)}
    if subtype in required:
        if (x.dtype.kind, x.dtype.itemsize) != required[subtype]:
            raise ValueError('PCM subtype and array dtype disagree')
        y = x.astype(np.float64)
        if subtype == 'PCM_U8':
            y = (y - 128.0) / 128.0
        else:
            y /= 2.0 ** (15 if subtype == 'PCM_16' else 31)
    elif subtype in ('FLOAT', 'DOUBLE'):
        if x.dtype.kind != 'f':
            raise ValueError('floating WAV data must use floating arrays')
        y = x.astype(np.float64, copy=True)
    else:
        raise ValueError('unsupported audio subtype')
    if not np.isfinite(y).all():
        raise ValueError('nonfinite sample inside the requested interval')
    if y.ndim == 2:
        y = y.mean(axis=1)
    return y, start


def _frame_descriptor(frame: np.ndarray, sample_rate: int) -> tuple[float, float, float, bool]:
    """Local normalized ACF; first peak within 95% of the strongest peak.

    Rectangular frame and overlap-energy normalization are deliberate. A
    Hann envelope can move ACF peaks; this draft avoids that source of bias.
    ACF is a conventional descriptor, not a proposed new pitch algorithm.
    """
    x = frame - frame.mean()
    energy = np.sum(x * x)
    rms = float(np.sqrt(energy / len(x)))
    dbfs = float(20.0 * np.log10(max(rms, RMS_FLOOR)))
    if energy == 0:
        return 0.0, dbfs, 0.0, False
    lo = max(2, int(np.ceil(sample_rate / MAX_HZ)))
    hi = min(len(x) - 2, int(np.floor(sample_rate / MIN_HZ)))
    nfft = 1 << (2 * len(x) - 1).bit_length()
    spectrum = np.fft.rfft(x, n=nfft)
    acf = np.fft.irfft(spectrum * spectrum.conj(), n=nfft)[:hi + 2]
    energies = np.concatenate(([0.0], np.cumsum(x * x)))
    lag = np.arange(hi + 2)
    norm = np.sqrt(np.maximum(energies[len(x) - lag] * (energies[-1] - energies[lag]), 0))
    corr = np.divide(acf, norm, out=np.zeros_like(acf), where=norm > 0)
    corr = np.clip(corr, -1.0, 1.0)
    candidate = np.arange(lo, hi + 1)
    peaks = candidate[(corr[candidate] > corr[candidate - 1]) &
                      (corr[candidate] >= corr[candidate + 1]) & (corr[candidate] > 0)]
    if len(peaks) == 0:
        return 0.0, dbfs, 0.0, False
    strongest = float(corr[peaks].max())
    k = int(peaks[np.flatnonzero(corr[peaks] >= PEAK_RELATIVE_HEIGHT * strongest)[0]])
    a, b, c = corr[k - 1:k + 2]
    denominator = a - 2 * b + c
    shift = float(np.clip(0.5 * (a - c) / denominator, -0.5, 0.5)) if denominator < 0 else 0.0
    hz = float(np.clip(sample_rate / (k + shift), MIN_HZ, MAX_HZ))
    voiced = rms >= VOICED_RMS and b >= VOICED_PERIODICITY
    cents = float(1200 * np.log2(hz / 440.0)) if voiced else 0.0
    return cents, dbfs, float(b), bool(voiced)


def interval_frames(samples: np.ndarray, sample_rate: int, subtype: str,
                    interval: tuple[float, float]) -> dict:
    sample_rate = _rate(sample_rate)
    if interval not in (PREFIX, TARGET):
        raise ValueError('only the declared prefix and target intervals are accepted')
    mono, offset = _mono_interval(samples, sample_rate, subtype, interval)
    width = round(FRAME_SECONDS * sample_rate)
    # Round physical frame starts independently, rather than accumulating hop drift.
    starts = np.array([round(i * HOP_SECONDS * sample_rate)
                       for i in range(round((interval[1] - interval[0]) / HOP_SECONDS) + 1)], dtype=np.int64)
    starts = starts[starts + width <= len(mono)]
    if len(starts) < 2:
        raise ValueError('not enough complete frames')
    rows = np.asarray([_frame_descriptor(mono[s:s + width], sample_rate) for s in starts])
    return {'pitch_cents': rows[:, 0], 'power_dbfs': rows[:, 1],
            'periodicity': rows[:, 2], 'voiced': rows[:, 3].astype(bool),
            'starts': starts + offset, 'ends': starts + offset + width,
            'sample_rate': sample_rate}


def _summaries(values: np.ndarray, times: np.ndarray) -> list[float]:
    if len(values) == 0:
        return [0.0] * 6
    centered = times - times.mean()
    denominator = np.sum(centered * centered)
    slope = np.sum(centered * (values - values.mean())) / denominator if denominator > 0 else 0.0
    return [float(values.mean()), float(np.std(values, ddof=1)) if len(values) >= 2 else 0.0,
            float(np.quantile(values, .1)), float(np.quantile(values, .9)),
            float(np.abs(np.diff(values)).mean()) if len(values) >= 2 else 0.0, float(slope)]


def _targets(frames: dict) -> tuple[np.ndarray, bool]:
    mask = frames['voiced']
    guard = int(mask.sum()) >= max(2, int(np.ceil(len(mask) / 2)))
    pitch = float(np.std(frames['pitch_cents'][mask], ddof=1)) if guard else float('nan')
    power = float(np.std(frames['power_dbfs'], ddof=1))
    return np.array([pitch, power], dtype=np.float64), guard


def feature_names() -> list[str]:
    stats = ('mean', 'std', 'p10', 'p90', 'mean_abs_diff', 'slope_per_second')
    channels = ('pitch_cents_voiced', 'power_dbfs', 'periodicity')
    windows = ('last4', 'last16', 'all')
    names = [f'{channel}_{window}_{stat}' for channel in channels for window in windows for stat in stats]
    return names + [f'voiced_fraction_{w}' for w in windows] + [
        'prefix_matching_target_pitch_std_or_zero', 'prefix_matching_target_power_std', 'prefix_guard']


def extract_prefix(samples: np.ndarray, sample_rate: int, subtype: str) -> dict:
    frames = interval_frames(samples, sample_rate, subtype, PREFIX)
    count = len(frames['voiced'])
    windows = (4, 16, count)
    values = []
    for channel in ('pitch_cents', 'power_dbfs', 'periodicity'):
        for n in windows:
            y = frames[channel][-n:]
            times = frames['starts'][-n:] / sample_rate
            if channel == 'pitch_cents':
                mask = frames['voiced'][-n:]
                y, times = y[mask], times[mask]
            values.extend(_summaries(y, times))
    values.extend(float(frames['voiced'][-n:].mean()) for n in windows)
    # Exactly the number of complete target frames, without inspecting the target.
    width = round(FRAME_SECONDS * sample_rate)
    target_length = round(TARGET[1] * sample_rate) - round(TARGET[0] * sample_rate)
    target_count = sum(round(i * HOP_SECONDS * sample_rate) + width <= target_length
                       for i in range(round((TARGET[1] - TARGET[0]) / HOP_SECONDS) + 1))
    tail = {k: frames[k][-target_count:] for k in ('pitch_cents', 'power_dbfs', 'voiced')}
    prefix_targets, guard = _targets(tail)
    values.extend([float(prefix_targets[0]) if guard else 0.0, float(prefix_targets[1]), float(guard)])
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (60,) or not np.isfinite(result).all():
        raise ArithmeticError('invalid prefix features')
    return {'X': result, 'prefix_targets': prefix_targets, 'prefix_guard': guard,
            'feature_names': feature_names(),
            'frame_support': np.column_stack((frames['starts'], frames['ends']))}


def extract_example(samples: np.ndarray, sample_rate: int, subtype: str) -> dict:
    prefix = extract_prefix(samples, sample_rate, subtype)
    future = interval_frames(samples, sample_rate, subtype, TARGET)
    target, guard = _targets(future)
    return {**prefix, 'targets': target, 'target_guard': guard,
            'target_support': np.column_stack((future['starts'], future['ends']))}
