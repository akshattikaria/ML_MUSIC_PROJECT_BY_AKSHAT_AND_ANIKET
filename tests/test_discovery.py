"""The stored similarity catalog and live queries must share a coordinate rule."""

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from musicmap.data import Dataset
from musicmap.discovery import _fit_pca_space, nearest_to_features


def test_randomized_pca_catalog_matches_reloaded_query_projection(tmp_path):
    # More features than retained PCs exposes randomized fit_transform's
    # approximation difference; a tiny full-rank matrix would miss the bug.
    rng = np.random.default_rng(6)
    training = rng.normal(size=(240, 100)).astype(np.float32)
    validation = rng.normal(size=(8, 100)).astype(np.float32)
    pca, catalog, held_out = _fit_pca_space(training, validation, seed=42)
    path = tmp_path / "pca.joblib"
    joblib.dump(pca, path)
    loaded = joblib.load(path)
    np.testing.assert_allclose(catalog, loaded.transform(training), rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(held_out, loaded.transform(validation), rtol=1e-7, atol=1e-7)


def test_training_query_has_zero_distance_to_its_stored_catalog_vector(tmp_path):
    rng = np.random.default_rng(12)
    ids = pd.Index(np.arange(1000, 1240), name="track_id")
    features = pd.DataFrame(rng.normal(size=(len(ids), 100)).astype(np.float32), index=ids)
    validation = features.iloc[:4] + 2
    pca, catalog, _ = _fit_pca_space(features, validation, seed=42)
    joblib.dump(Pipeline([("pca", pca)]), tmp_path / "discovery_transform.joblib")
    np.savez_compressed(tmp_path / "training_embeddings.npz", values=catalog, track_ids=ids)
    metadata = pd.DataFrame({
        "title": [f"Track {i}" for i in ids], "artist": "Artist", "genre": "Rock",
        "split": "training",
    }, index=ids)
    dataset = Dataset(features, metadata, {})
    query = features.iloc[[11]]
    result = nearest_to_features(dataset, tmp_path, query, k=1)
    assert result.index.tolist() == query.index.tolist()
    assert result.distance.iloc[0] < 1e-5
