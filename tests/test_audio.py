"""Audio extraction produces the dataset's exact 518-column schema."""

import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from musicmap.audio import FEATURE_SIZES, STATISTICS, feature_names


def test_feature_schema_matches_fma_layout():
    names = feature_names()
    assert len(names) == 518 == 7 * sum(FEATURE_SIZES.values())
    assert len(set(names)) == len(names)
    assert {name.split("__")[1] for name in names} == set(STATISTICS)
    assert "mfcc__mean__01" in names and "mfcc__max__20" in names
    assert "tonnetz__std__06" in names and "zcr__kurtosis__01" in names


def test_synthetic_signal_extracts_all_descriptors():
    pytest.importorskip("librosa")
    from musicmap.audio import features_from_signal
    sr = 22050
    t = np.arange(sr * 4) / sr
    rng = np.random.default_rng(0)
    x = 0.4 * np.sin(2 * np.pi * 330 * t) + 0.05 * rng.normal(size=t.size)
    features = features_from_signal(x, sr)
    assert list(features.index) == feature_names()
    # Constant frames (e.g. a pure tone's chroma) give undefined skew/kurtosis;
    # the model pipeline imputes those with training medians.
    assert features.notna().mean() > 0.9
    assert features["rmse__mean__01"] > 0


def test_too_short_audio_is_rejected():
    pytest.importorskip("librosa")
    from musicmap.audio import features_from_signal
    with pytest.raises(ValueError, match="at least one second"):
        features_from_signal(np.zeros(100), 22050)


def test_low_sample_rate_cli_reports_actionable_error_without_traceback(tmp_path, monkeypatch, capsys):
    pytest.importorskip("librosa")
    sf = pytest.importorskip("soundfile")
    from musicmap.cli import main
    sr = 8000
    path = tmp_path / "low-rate.wav"
    sf.write(path, np.sin(2 * np.pi * 330 * np.arange(sr * 2) / sr), sr)
    output = tmp_path / "features.csv"
    monkeypatch.setattr(sys, "argv", ["musicmap", "extract", str(path), "--output", str(output)])
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 2
    error = capsys.readouterr().err
    assert "at least 11025 Hz; got 8000 Hz" in error
    assert "resample the file to 22050 Hz" in error
    assert "Traceback" not in error
    assert not output.exists()


def test_lowest_supported_rate_extracts_all_descriptors():
    pytest.importorskip("librosa")
    from musicmap.audio import features_from_signal
    sr = 11025
    t = np.arange(sr * 2) / sr
    rng = np.random.default_rng(13)
    signal = 0.4 * np.sin(2 * np.pi * 660 * t) + 0.05 * rng.normal(size=t.size)
    features = features_from_signal(signal, sr)
    assert list(features.index) == feature_names()
    assert np.isfinite(features).mean() > 0.9
    assert features["rmse__mean__01"] > 0


def test_m4a_fallback_preserves_native_rate_and_middle_excerpt(tmp_path, monkeypatch):
    librosa = pytest.importorskip("librosa")
    sf = pytest.importorskip("soundfile")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe are required for the M4A integration check")
    from musicmap import audio

    sr = 32000
    t = np.arange(sr * 8) / sr
    frequency = np.where(t < 2, 110, np.where(t < 6, 660, 1760))
    mono = np.sin(2 * np.pi * frequency * t)
    wav, m4a = tmp_path / "source.wav", tmp_path / "music with spaces.m4a"
    sf.write(wav, np.column_stack([0.4 * mono, 0.2 * mono]), sr)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(wav), "-c:a", "aac", str(m4a)],
                   check=True, capture_output=True)

    def unsupported(*args, **kwargs):
        raise RuntimeError("The native decoder does not support AAC")

    captured = {}

    def capture(signal, rate):
        captured.update(signal=signal, sr=rate)
        return pd.Series({"rmse__mean__01": float(np.sqrt(np.mean(signal ** 2)))})

    monkeypatch.setattr(librosa, "load", unsupported)
    monkeypatch.setattr(audio, "features_from_signal", capture)
    result = audio.extract_file(m4a, duration=2)
    signal = captured["signal"]
    assert captured["sr"] == sr
    assert signal.ndim == 1 and abs(len(signal) - 2 * sr) <= 1
    spectrum = np.fft.rfft(signal)
    peak = np.fft.rfftfreq(len(signal), 1 / sr)[np.abs(spectrum).argmax()]
    assert peak == pytest.approx(660, abs=1)
    # The channel mean has amplitude .3, not an energy-increasing stereo downmix.
    assert result["rmse__mean__01"] == pytest.approx(0.3 / np.sqrt(2), rel=0.08)
    assert result.name == "music with spaces"


def test_unsupported_audio_without_ffmpeg_gives_conversion_guidance(tmp_path, monkeypatch):
    librosa = pytest.importorskip("librosa")
    from musicmap import audio
    path = tmp_path / "recording.m4a"
    path.write_bytes(b"unsupported audio")

    def unsupported(*args, **kwargs):
        raise RuntimeError("Format not recognised")

    monkeypatch.setattr(librosa, "load", unsupported)
    monkeypatch.setattr(audio.shutil, "which", lambda name: None)
    with pytest.raises(ValueError, match="Install FFmpeg.*convert the file to WAV"):
        audio.extract_file(path, offset=0)


def test_invalid_audio_cli_reports_error_without_traceback(tmp_path, monkeypatch, capsys):
    pytest.importorskip("librosa")
    from musicmap.cli import main
    path = tmp_path / "broken.m4a"
    path.write_bytes(b"not an audio file")
    monkeypatch.setattr(sys, "argv", ["musicmap", "extract", str(path), "--output", str(tmp_path / "out.csv")])
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 2
    error = capsys.readouterr().err
    assert "Could not decode broken.m4a" in error
    assert "Traceback" not in error
