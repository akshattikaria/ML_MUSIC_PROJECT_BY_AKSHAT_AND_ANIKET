"""Check grouped tuning, leakage isolation, and fixed-seed ANN averaging."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from musicmap.data import Dataset
from musicmap.improvement import (
    SeedAveragedMLP, build_candidate, make_group_folds, run_improvement,
)


@pytest.fixture
def grouped_dataset():
    rng = np.random.default_rng(9)
    index = pd.Index(np.arange(1000, 1180), name="track_id")
    labels = np.tile(["Electronic", "Folk", "Rock"], 60)
    centers = np.tile(np.arange(3), 60)[:, None] * 3
    X = pd.DataFrame(
        centers + rng.normal(0, 0.35, size=(180, 6)),
        index=index,
        columns=[
            "mfcc__mean__01", "mfcc__mean__02", "mfcc__mean__03",
            "spectral_centroid__mean__01", "zcr__mean__01", "rmse__mean__01",
        ],
    )
    # Each artist contributes one example of every class.
    metadata = pd.DataFrame({
        "genre": labels,
        "split": ["training"] * 120 + ["validation"] * 30 + ["test"] * 30,
        "artist_id": np.repeat(np.arange(60), 3),
        "album_id": np.repeat(np.arange(60), 3),
        "title": [f"Track {i}" for i in range(180)],
        "artist": [f"Artist {i // 3}" for i in range(180)],
    }, index=index)
    return Dataset(X, metadata, {
        "genres": ["Electronic", "Folk", "Rock"], "processed_sha256": {},
    })


def test_group_folds_isolate_artists_and_cover_training_once(grouped_dataset):
    dataset = grouped_dataset
    folds, audits = make_group_folds(dataset, n_splits=3, seed=42)
    training_ids = dataset.ids("training")
    training_meta = dataset.metadata.loc[training_ids]
    assert len(folds) == len(audits) == 3
    held_out_positions = []
    for (fit, validation), audit in zip(folds, audits):
        assert set(fit).isdisjoint(validation)
        assert set(fit) | set(validation) == set(range(len(training_ids)))
        fit_meta, val_meta = training_meta.iloc[fit], training_meta.iloc[validation]
        assert set(fit_meta.artist_id).isdisjoint(val_meta.artist_id)
        assert set(fit_meta.genre) == set(val_meta.genre) == set(dataset.manifest["genres"])
        assert audit["artist_overlap"] == 0
        assert set(audit["fit_track_ids"]) == set(fit_meta.index)
        assert set(audit["validation_track_ids"]) == set(val_meta.index)
        held_out_positions.extend(validation)
    assert sorted(held_out_positions) == list(range(len(training_ids)))


@pytest.mark.parametrize("reduction", ["all", "select", "pca"])
def test_reduction_fits_only_fold_training_and_caps_tiny_rank(grouped_dataset, reduction):
    dataset = grouped_dataset
    folds, _ = make_group_folds(dataset, n_splits=3, seed=42)
    fit, validation = folds[0]
    X = dataset.X.loc[dataset.ids("training")].copy()
    y = dataset.y.loc[X.index]
    # Contamination would make the learned statistics obviously incorrect.
    X.iloc[validation] += 1_000_000
    X.iloc[fit[0], 0] = np.nan
    candidate = build_candidate("SVM (RBF)", {
        "scaling": "standard", "reduction": reduction, "n_components": 256,
        "classifier": {"C": 1.0, "gamma": "scale"},
    }, seed=42)
    with threadpool_limits(limits=1):
        candidate.fit(X.iloc[fit], y.iloc[fit])
        predictions = candidate.predict(X.iloc[validation])
    imputers = [
        value for value in candidate.get_params(deep=True).values()
        if isinstance(value, SimpleImputer) and hasattr(value, "statistics_")
    ]
    assert imputers, "Candidate pipeline must contain a fitted imputer"
    np.testing.assert_allclose(imputers[0].statistics_, X.iloc[fit].median())
    assert len(predictions) == len(validation)
    assert set(predictions).issubset(set(y))


def test_ann_probabilities_average_fixed_seeds_and_survive_serialization(grouped_dataset, tmp_path):
    dataset = grouped_dataset
    train = dataset.ids("training")
    model = SeedAveragedMLP(
        parameters={"hidden_layer_sizes": (8,), "max_iter": 15, "batch_size": 32},
        seeds=(42, 43, 44),
    )
    with threadpool_limits(limits=1):
        # This serialization check intentionally uses only 15 epochs.
        with pytest.warns(ConvergenceWarning):
            model.fit(dataset.X.loc[train].to_numpy(), dataset.y.loc[train])
        examples = dataset.X.iloc[-9:].to_numpy()
        actual = model.predict_proba(examples)
        expected = np.mean(
            [estimator.predict_proba(examples) for estimator in model.estimators_],
            axis=0,
        )
        predictions = model.predict(examples)
    assert len(model.estimators_) == 3
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(actual.sum(axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(predictions, model.classes_[np.argmax(actual, axis=1)])
    path = tmp_path / "seed_average.joblib"
    joblib.dump(model, path)
    with threadpool_limits(limits=1):
        restored = joblib.load(path)
        np.testing.assert_allclose(restored.predict_proba(examples), actual)
        np.testing.assert_array_equal(restored.predict(examples), predictions)


def test_quantile_ranks_exclude_shifted_validation_values(grouped_dataset):
    dataset = grouped_dataset
    training = dataset.ids("training")
    held_out = dataset.ids("validation")
    dataset.X.loc[held_out] += 1_000_000
    candidate = build_candidate("SVM (RBF)", {
        "scaling": "quantile", "reduction": "all",
        "classifier": {"C": 1.0, "gamma": "scale"},
    }, seed=42)
    with threadpool_limits(limits=1):
        # Tiny synthetic data have fewer rows than the production quantile count.
        with pytest.warns(UserWarning, match="n_quantiles"):
            candidate.fit(dataset.X.loc[training], dataset.y.loc[training])
        transformer = candidate.named_steps["scaler"]
        np.testing.assert_allclose(
            transformer.quantiles_[-1], dataset.X.loc[training].max(),
        )
        predicted = candidate.predict(dataset.X.loc[held_out])
    assert len(predicted) == len(held_out)
    assert set(predicted).issubset(set(dataset.y))


def test_improved_selection_locks_before_test_and_refit_excludes_test(grouped_dataset, tmp_path):
    output = tmp_path / "reports"

    class GuardedDataset(Dataset):
        def ids(self, split):
            if split == "test":
                assert (output / "selection.json").exists(), "Test requested before model selection was locked"
            return super().ids(split)

    dataset = grouped_dataset
    dataset.X.loc[dataset.metadata["split"].eq("test")] += 1_000_000
    guarded = GuardedDataset(dataset.X, dataset.metadata, dataset.manifest)
    spaces = {
        "SVM (RBF)": [{
            "scaling": "standard", "reduction": "select", "n_components": 4,
            "classifier": {"C": 1.0, "gamma": "scale"},
        }],
        "ANN (MLP)": [{
            "scaling": "standard", "reduction": "all",
            "classifier": {"hidden_layer_sizes": (8,), "max_iter": 15, "batch_size": 32},
        }],
        "Dummy baseline": [{
            "scaling": "standard", "reduction": "all", "classifier": {},
        }],
    }
    results = run_improvement(guarded, output, seed=42, threads=1, cv_folds=3, spaces=spaces)
    assert len(results) == 3
    assert results["selected_by_validation"].sum() == 1
    selection = json.loads((output / "selection.json").read_text())
    assert selection["test_used_for_selection"] is False
    assert selection["champion"] in {"SVM (RBF)", "ANN (MLP)"}
    final = joblib.load(tmp_path / "models/best_model.joblib")
    tuned = joblib.load(tmp_path / "models/selected_training_model.joblib")
    training = dataset.ids("training")
    development = dataset.metadata.index[dataset.metadata["split"].ne("test")]
    for model, ids in ((tuned, training), (final, development)):
        imputer = next(
            value for value in model.get_params(deep=True).values()
            if isinstance(value, SimpleImputer) and hasattr(value, "statistics_")
        )
        np.testing.assert_allclose(imputer.statistics_, dataset.X.loc[ids].median())
    predictions = pd.read_csv(output / "test_predictions.csv")
    assert len(predictions) == 30
    assert set(predictions.track_id) == set(dataset.metadata.index[dataset.metadata["split"].eq("test")])
    cv_results = pd.read_csv(output / "cv_results.csv")
    assert cv_results["cv_mean_f1"].notna().all()
    assert (output / "cv_splits.json").exists()
