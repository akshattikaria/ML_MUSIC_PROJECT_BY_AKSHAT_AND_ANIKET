"""Regression checks for label handling, split isolation, and data alignment."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from musicmap.data import Dataset, validate_tables
from musicmap.discovery import grouped_permutation
from musicmap.experiment import build_model, run_experiment


@pytest.fixture
def dataset():
    rng = np.random.default_rng(7)
    labels = np.array(["Electronic", "Folk", "Rock"])
    y = np.tile(labels, 60)
    ids = pd.Index(np.arange(1000, 1180), name="track_id")
    centers = np.tile(np.arange(3), 60)[:, None] * 3
    X = pd.DataFrame(
        centers + rng.normal(0, 0.4, (180, 4)), index=ids,
        columns=["mfcc__mean__01", "mfcc__mean__02", "spectral_contrast__mean__01", "zcr__mean__01"],
    )
    metadata = pd.DataFrame({
        "genre": y,
        "split": ["training"] * 120 + ["validation"] * 30 + ["test"] * 30,
        "artist_id": np.repeat(np.arange(60), 3),
        "album_id": np.repeat(np.arange(60), 3),
        "title": [f"Song {i}" for i in range(180)],
        "artist": [f"Artist {i // 3}" for i in range(180)],
    }, index=ids)
    return Dataset(X, metadata, {"genres": labels.tolist(), "processed_sha256": {}})


def test_alignment_and_metadata_leakage_are_rejected(dataset):
    validate_tables(dataset.X, dataset.metadata)
    with pytest.raises(ValueError, match="identical track IDs"):
        validate_tables(dataset.X.iloc[::-1], dataset.metadata)
    leaked = dataset.X.assign(artist_id=dataset.metadata["artist_id"])
    with pytest.raises(ValueError, match="documented FMA audio feature"):
        validate_tables(leaked, dataset.metadata)
    with pytest.raises(ValueError, match="unique"):
        validate_tables(pd.concat([dataset.X, dataset.X.iloc[[0]]]), dataset.metadata)


def test_ann_accepts_text_labels_and_survives_serialization(dataset, tmp_path):
    model = build_model("ANN (MLP)", {"hidden_layer_sizes": (16,), "alpha": 0.01, "batch_size": 32}, seed=42)
    training = dataset.ids("training")
    with threadpool_limits(limits=1):
        model.fit(dataset.X.loc[training], dataset.y.loc[training])
        before = model.predict(dataset.X.iloc[-12:])
        probabilities = model.predict_proba(dataset.X.iloc[-12:])
    assert set(before).issubset(set(dataset.y))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    path = tmp_path / "ann.joblib"
    joblib.dump(model, path)
    with threadpool_limits(limits=1):
        np.testing.assert_array_equal(before, joblib.load(path).predict(dataset.X.iloc[-12:]))


def test_test_partition_is_requested_after_selection_and_never_fits_preprocessing(dataset, tmp_path):
    output = tmp_path / "reports"

    class GuardedDataset(Dataset):
        def ids(self, split):
            if split == "test":
                assert (output / "selection.json").exists(), "Test partition requested before selection lock"
            return super().ids(split)

    # Make contamination obvious if test features accidentally enter a fitting step.
    dataset.X.loc[dataset.metadata["split"].eq("test")] += 1_000_000
    guarded = GuardedDataset(dataset.X, dataset.metadata, dataset.manifest)
    results = run_experiment(guarded, output, profile="quick", threads=1)
    selection = json.loads((output / "selection.json").read_text())
    assert selection["test_used_for_selection"] is False
    assert len(results) == 5
    assert results["selected_by_validation"].sum() == 1
    assert results.loc[results["selected_by_validation"], "model"].iloc[0] == selection["champion"]
    training = dataset.ids("training")
    development = dataset.metadata.index[dataset.metadata["split"].ne("test")]
    tuned = joblib.load(tmp_path / "models/selected_training_model.joblib")
    final = joblib.load(tmp_path / "models/best_model.joblib")
    np.testing.assert_allclose(tuned.named_steps["imputer"].statistics_, dataset.X.loc[training].median())
    np.testing.assert_allclose(final.named_steps["imputer"].statistics_, dataset.X.loc[development].median())
    predictions = pd.read_csv(output / "test_predictions.csv")
    assert len(predictions) == 30
    assert set(predictions["track_id"]) == set(dataset.metadata.index[dataset.metadata["split"].eq("test")])


def test_group_permutation_detects_signal_without_mutating_inputs():
    X = pd.DataFrame({
        "mfcc__mean__01": np.tile([-1.0, 1.0], 60),
        "mfcc__std__01": np.tile([-2.0, 2.0], 60),
        "zcr__mean__01": np.linspace(0, 1, 120),
    })
    y = pd.Series(np.where(X["mfcc__mean__01"] > 0, "Rock", "Folk"))

    class KnownRule:
        def predict(self, frame):
            return np.where(frame["mfcc__mean__01"] > 0, "Rock", "Folk")

    original = X.copy(deep=True)
    result = grouped_permutation(KnownRule(), X, y, repeats=3).set_index("family")
    assert result.at["mfcc", "mean_f1_drop"] > 0.3
    assert result.at["zcr", "mean_f1_drop"] == 0
    assert result.at["mfcc", "n_features"] == 2
    pd.testing.assert_frame_equal(X, original)
