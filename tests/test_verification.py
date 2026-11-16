"""Regression checks for stale artifact and stale verification evidence detection."""

import importlib.util
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


spec = importlib.util.spec_from_file_location(
    "verify_artifacts", Path(__file__).resolve().parents[1] / "scripts/verify_artifacts.py",
)
verification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verification)


class SavedPredictor:
    classes_ = np.array(["Folk", "Rock"])
    feature_names_in_ = np.array(["mfcc__mean__01"])

    def __init__(self, predictions, probabilities):
        self.predictions = np.asarray(predictions)
        self.probabilities = np.asarray(probabilities)

    def predict(self, X):
        return self.predictions

    def predict_proba(self, X):
        return self.probabilities


def test_stale_model_is_rejected_even_when_accuracy_is_identical():
    X = pd.DataFrame({"mfcc__mean__01": [0, 1, 2, 3]})
    expected = np.array(["Folk", "Folk", "Rock", "Rock"])
    actual = np.array(["Rock", "Rock", "Folk", "Folk"])
    truth = np.array(["Folk", "Rock", "Folk", "Rock"])
    assert np.mean(expected == truth) == np.mean(actual == truth)
    probabilities = np.array([[0.8, 0.2], [0.8, 0.2], [0.2, 0.8], [0.2, 0.8]])
    stale = SavedPredictor(actual, probabilities[:, ::-1])
    with pytest.raises(ValueError, match="predictions differ"):
        verification.verify_model_predictions(stale, X, expected, probabilities, stale.classes_)


def test_same_labels_with_stale_probability_values_are_rejected():
    X = pd.DataFrame({"mfcc__mean__01": [0, 1]})
    expected = np.array(["Folk", "Rock"])
    probabilities = np.array([[0.8, 0.2], [0.2, 0.8]])
    stale = SavedPredictor(expected, [[0.6, 0.4], [0.4, 0.6]])
    with pytest.raises(ValueError, match="probabilities differ"):
        verification.verify_model_predictions(stale, X, expected, probabilities, stale.classes_)


def test_ann_check_does_not_treat_a_forests_trees_as_named_ensemble_members():
    X = pd.DataFrame({"mfcc__mean__01": [0, 1, 2, 3]})
    forest = Pipeline([("model", ExtraTreesClassifier(n_estimators=2, random_state=42))])
    forest.fit(X, ["Folk", "Folk", "Rock", "Rock"])
    assert verification.verify_ann_average(forest, X, [42, 43, 44]) == 0


def test_artist_overlap_is_checked_from_actual_ids_instead_of_saved_flag():
    metadata = pd.DataFrame({"artist_id": [10, 10, 20, 20]}, index=[1, 2, 3, 4])
    folds = [
        {"fit_track_ids": [1, 3], "validation_track_ids": [2, 4], "artist_overlap": 0},
        {"fit_track_ids": [2, 4], "validation_track_ids": [1, 3], "artist_overlap": 0},
    ]
    with pytest.raises(ValueError, match="Artists overlap"):
        verification.verify_folds(metadata, metadata.index, folds)


def test_failed_artifact_check_preserves_previous_verification_file(tmp_path, monkeypatch):
    junit = tmp_path / "pytest.xml"
    junit.write_text('<testsuites><testsuite><testcase name="passed"/></testsuite></testsuites>')
    previous = tmp_path / "verification.json"
    previous.write_text('{"previous_evidence": true}\n')

    def stale_artifacts(root):
        raise ValueError("Saved model predictions differ from the report CSV.")

    monkeypatch.setattr(verification, "verify_artifacts", stale_artifacts)
    with pytest.raises(ValueError, match="predictions differ"):
        verification.verify_and_write(tmp_path, junit, previous)
    assert previous.read_text() == '{"previous_evidence": true}\n'


def test_junit_counts_real_outcomes_and_rejects_failed_runs(tmp_path):
    junit = tmp_path / "pytest.xml"
    junit.write_text('<testsuites><testsuite tests="999"><testcase name="one"/><testcase name="two"/><testcase name="skip"><skipped/></testcase></testsuite></testsuites>')
    assert verification.test_summary(junit) == {"passed": 2, "failed": 0, "errors": 0, "skipped": 1}
    junit.write_text('<testsuites><testsuite><testcase name="one"/><testcase name="broken"><failure/></testcase></testsuite></testsuites>')
    with pytest.raises(ValueError, match="failed tests"):
        verification.test_summary(junit)


@pytest.fixture
def discovery_files(tmp_path):
    X = pd.DataFrame(
        np.random.default_rng(42).normal(size=(12, 3)),
        index=pd.Index(range(100, 112), name="track_id"),
        columns=["mfcc__mean__01", "spectral_centroid__mean__01", "zcr__mean__01"],
    )
    query = X.iloc[:2] + 0.15
    transform = Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=2))]).fit(X)
    models = tmp_path / "models"
    models.mkdir()
    joblib.dump(transform, models / "discovery_transform.joblib")
    np.savez_compressed(models / "training_embeddings.npz", values=transform.transform(X), track_ids=X.index)
    return models, X, query


def test_discovery_verification_projects_queries_and_reconciles_catalog(discovery_files):
    models, X, query = discovery_files
    evidence = verification.verify_discovery_artifacts(models, X, query)
    assert evidence == {
        "discovery_transform_matches_catalog": True,
        "discovery_catalog_rows_checked": 12,
        "discovery_query_rows_checked": 2,
        "discovery_components": 2,
    }


@pytest.mark.parametrize("wrong_shape", [False, True])
def test_discovery_verification_rejects_stale_or_incompatible_coordinates(discovery_files, wrong_shape):
    models, X, query = discovery_files
    projection = joblib.load(models / "discovery_transform.joblib").transform(X)
    stale = projection[:, :1] if wrong_shape else projection + 0.1
    np.savez_compressed(models / "training_embeddings.npz", values=stale, track_ids=X.index)
    with pytest.raises(ValueError, match="Similarity catalog (dimensions|coordinates) differ"):
        verification.verify_discovery_artifacts(models, X, query)


def test_corrupt_discovery_transform_fails_cli_without_replacing_evidence(
    discovery_files, tmp_path, monkeypatch, capsys,
):
    models, X, query = discovery_files
    (models / "discovery_transform.joblib").write_bytes(b"invalid serialized transform")
    junit = tmp_path / "tests.xml"
    junit.write_text('<testsuites><testsuite><testcase name="passed"/></testsuite></testsuites>')
    previous = tmp_path / "verification.json"
    previous.write_text('{"previous_evidence": true}\n')
    monkeypatch.setattr(
        verification, "verify_artifacts", lambda root: verification.verify_discovery_artifacts(models, X, query),
    )
    monkeypatch.setattr(sys, "argv", [
        "verify_artifacts", "--root", str(tmp_path), "--test-report", str(junit), "--output", str(previous),
    ])
    assert verification.main() == 1
    assert previous.read_text() == '{"previous_evidence": true}\n'
    error = capsys.readouterr().err
    assert "Could not load the saved discovery transform" in error
    assert "Traceback" not in error
