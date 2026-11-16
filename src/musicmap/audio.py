"""Extract FMA-compatible descriptors from an audio file (optional ``audio`` extra).

This follows the official FMA extraction recipe (github.com/mdeff/fma, features.py):
11 librosa features, each summarized over time by seven statistics, giving the same
518 named columns as the prepared dataset. Library versions and decoders differ from
the 2017 extraction, so values are close to, but not bit-identical with, the published
features. Predictions for your own audio are therefore a demonstration, not a
benchmark result.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd

STATISTICS = ("kurtosis", "max", "mean", "median", "min", "skew", "std")
FEATURE_SIZES = {
    "chroma_cens": 12, "chroma_cqt": 12, "chroma_stft": 12, "mfcc": 20, "rmse": 1,
    "spectral_bandwidth": 1, "spectral_centroid": 1, "spectral_contrast": 7,
    "spectral_rolloff": 1, "tonnetz": 6, "zcr": 1,
}


def feature_names() -> list[str]:
    return [
        f"{family}__{stat}__{index:02d}"
        for family, size in sorted(FEATURE_SIZES.items())
        for stat in STATISTICS
        for index in range(1, size + 1)
    ]


def _require_librosa():
    try:
        import librosa  # noqa: F401
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Audio extraction needs the optional audio dependencies: "
            'python -m pip install -e ".[audio]"'
        ) from error
    import librosa
    return librosa


def features_from_signal(x: np.ndarray, sr: int) -> pd.Series:
    """Compute the 518 FMA descriptors for a mono signal."""
    from scipy import stats
    librosa = _require_librosa()
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1 or len(x) < sr:
        raise ValueError("Audio must be mono and at least one second long.")
    values: dict[str, float] = {}

    def summarize(name: str, frames: np.ndarray) -> None:
        frames = np.asarray(frames, dtype=np.float64)
        with np.errstate(all="ignore"):
            summary = {
                "mean": np.mean(frames, axis=1), "std": np.std(frames, axis=1),
                "skew": stats.skew(frames, axis=1), "kurtosis": stats.kurtosis(frames, axis=1),
                "median": np.median(frames, axis=1), "min": np.min(frames, axis=1),
                "max": np.max(frames, axis=1),
            }
        for stat, vector in summary.items():
            for index, value in enumerate(vector, start=1):
                values[f"{name}__{stat}__{index:02d}"] = float(value)

    summarize("zcr", librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512))
    cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12, n_bins=7 * 12, tuning=None))
    summarize("chroma_cqt", librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7))
    cens = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
    summarize("chroma_cens", cens)
    summarize("tonnetz", librosa.feature.tonnetz(chroma=cens))
    del cqt
    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    summarize("chroma_stft", librosa.feature.chroma_stft(S=stft ** 2, n_chroma=12))
    summarize("rmse", librosa.feature.rms(S=stft))
    summarize("spectral_centroid", librosa.feature.spectral_centroid(S=stft))
    summarize("spectral_bandwidth", librosa.feature.spectral_bandwidth(S=stft))
    summarize("spectral_contrast", librosa.feature.spectral_contrast(S=stft, n_bands=6))
    summarize("spectral_rolloff", librosa.feature.spectral_rolloff(S=stft))
    mel = librosa.feature.melspectrogram(sr=sr, S=stft ** 2)
    summarize("mfcc", librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20))
    series = pd.Series(values, dtype=np.float64).reindex(feature_names())
    if series.isna().all():
        raise ValueError("Feature extraction failed for this audio.")
    return series.replace([np.inf, -np.inf], np.nan)


def extract_file(path: Path, offset: float | None = None, duration: float = 30.0) -> pd.Series:
    """Load a 30-second excerpt (from the middle by default, like FMA clips)."""
    librosa = _require_librosa()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Audio excerpt duration must be positive and finite.")
    if offset is not None and (not np.isfinite(offset) or offset < 0):
        raise ValueError("Audio excerpt offset must be nonnegative and finite.")
    requested_offset = offset
    if offset is None:
        try:
            total = float(librosa.get_duration(path=str(path)))
        except Exception:  # noqa: BLE001 - some decoders cannot report duration
            total = duration
        offset = max(0.0, (total - duration) / 2)
    try:
        x, sr = librosa.load(str(path), sr=None, mono=True, offset=offset, duration=duration)
    except (OSError, RuntimeError, ValueError):
        # Some libsndfile builds cannot decode AAC/M4A. Decode only the requested
        # excerpt with FFmpeg, keeping the source sample rate used by FMA.
        x, sr = _load_with_ffmpeg(path, requested_offset, duration)
    return features_from_signal(x, sr).rename(path.stem)


def _load_with_ffmpeg(path: Path, offset: float | None, duration: float) -> tuple[np.ndarray, int]:
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise ValueError(
            f"Could not decode {path.name}. Install FFmpeg (including ffprobe) "
            "for M4A/AAC support, or convert the file to WAV."
        )
    try:
        probe = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries",
             "stream=sample_rate,channels,duration:format=duration", "-of", "json", str(path.resolve())],
            check=True, capture_output=True, timeout=30,
        )
        info = json.loads(probe.stdout)
        stream = info["streams"][0]
        sr = int(stream["sample_rate"])
        channels = int(stream["channels"])
        if sr <= 0 or channels <= 0:
            raise ValueError("Invalid audio stream properties.")
        if offset is None:
            total = duration
            for candidate in (stream.get("duration"), info.get("format", {}).get("duration")):
                try:
                    total = float(candidate)
                    break
                except (TypeError, ValueError):
                    continue
            offset = max(0.0, (total - duration) / 2) if np.isfinite(total) else 0.0
        # Match librosa.to_mono's channel mean, rather than FFmpeg's default
        # surround/stereo mixing matrix, which can change the signal's energy.
        channel_mean = "+".join(f"{1 / channels}*c{i}" for i in range(channels))
        decoded = subprocess.run(
            [ffmpeg, "-nostdin", "-v", "error", "-ss", str(offset), "-i", str(path.resolve()),
             "-t", str(duration), "-map", "0:a:0", "-vn", "-af", f"pan=mono|c0={channel_mean}",
             "-ac", "1", "-ar", str(sr),
             "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"],
            check=True, capture_output=True, timeout=60,
        )
        return np.frombuffer(decoded.stdout, dtype="<f4").copy(), sr
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError) as error:
        raise ValueError(
            f"Could not decode {path.name}. Check that it is a valid audio file, "
            "or convert it to WAV and try again."
        ) from error


def extract_files(paths: list[Path]) -> pd.DataFrame:
    rows = [extract_file(Path(p)) for p in paths]
    frame = pd.DataFrame(rows)
    frame.index.name = "track_id"
    return frame
