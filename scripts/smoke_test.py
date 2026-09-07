#!/usr/bin/env python3
"""Synthetic pipeline checks, not a scientific evaluation of violin signals."""
import datetime as dt
import json
from pathlib import Path
import tempfile
import uuid


def main():
    print("Loading audio/image/table libraries (first import may take longer)...", flush=True)
    import cv2
    import librosa
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy.signal
    import soundfile as sf
    from sklearn.linear_model import LogisticRegression

    checks = {}
    print("Checking synthetic audio IO, resampling, and mel features...", flush=True)
    with tempfile.TemporaryDirectory(prefix="violin_smoke_") as temporary:
        directory = Path(temporary)
        sr = 48000
        t = np.arange(sr, dtype=np.float64) / sr
        audio = 0.2 * np.sin(2 * np.pi * 440 * t)
        wav = directory / "synthetic.wav"
        sf.write(wav, audio, sr, subtype="PCM_24")
        restored, restored_sr = sf.read(wav, dtype="float64")
        err = float(np.max(np.abs(restored - audio)))
        assert restored_sr == sr and err < 2e-7, (restored_sr, err)
        checks["wav_48k_pcm24_roundtrip_max_error"] = err
        resampled = scipy.signal.resample_poly(restored, 1, 3)
        assert len(resampled) == 16000
        spectrum = np.abs(np.fft.rfft(resampled))
        peak = float(np.fft.rfftfreq(len(resampled), 1 / 16000)[np.argmax(spectrum)])
        assert abs(peak - 440) < 1.0, peak
        checks["resampled_sine_peak_hz"] = peak
        mel = librosa.feature.melspectrogram(y=resampled, sr=16000, n_fft=512,
                                             hop_length=160, n_mels=40)
        assert mel.shape[0] == 40 and np.isfinite(mel).all()
        checks["mel_shape"] = list(mel.shape)
        print("Checking image/table IO, a tiny classifier, and a headless plot...", flush=True)
        picture = np.zeros((64, 96, 3), dtype=np.uint8)
        picture[16:48, 20:60, 1] = 200
        png = directory / "synthetic.png"
        assert cv2.imwrite(str(png), picture)
        assert np.array_equal(cv2.imread(str(png)), picture)
        checks["opencv_png_roundtrip"] = "PASS (video tracking not tested)"
        table = pd.DataFrame({"take_id": ["synthetic_1", "synthetic_2"], "force_n": [0.5, 1.0]})
        parquet = directory / "synthetic.parquet"
        table.to_parquet(parquet, index=False)
        pd.testing.assert_frame_equal(pd.read_parquet(parquet), table)
        checks["parquet_roundtrip"] = "PASS"
        x = np.array([[-3.], [-2.], [-1.], [1.], [2.], [3.]])
        y = np.array([0, 0, 0, 1, 1, 1])
        model = LogisticRegression(random_state=20260907).fit(x, y)
        assert np.array_equal(model.predict(x), y)
        checks["sklearn_synthetic_fit"] = "PASS (not a held-out performance estimate)"
        fig, ax = plt.subplots()
        ax.plot(t[:480], audio[:480])
        fig.savefig(directory / "synthetic_plot.png")
        plt.close(fig)
        assert (directory / "synthetic_plot.png").stat().st_size > 0
        checks["headless_plot"] = "PASS"

    now = dt.datetime.now(dt.timezone.utc)
    report_dir = Path("reports/smoke")
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / (now.strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8] + ".json")
    payload = {"time_utc": now.isoformat(), "status": "PASS", "synthetic_only": True, "checks": checks}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print("SMOKE PASS. Report:", path)


if __name__ == "__main__":
    main()
