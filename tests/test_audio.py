"""Audio extraction produces the dataset's exact 518-column schema."""

import numpy as np
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
