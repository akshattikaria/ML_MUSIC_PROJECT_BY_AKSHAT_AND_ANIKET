"""Verify a saved MusicMap run and publish evidence only after every check passes.

Run pytest with --junitxml first, then invoke this script with --test-report.
This verifies existing artifacts; it never fits models or changes their results.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, log_loss, top_k_accuracy_score
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from musicmap.data import load_dataset, split_audit
from musicmap.experiment import metrics
from musicmap.improvement import IMPROVED_SLUGS, SeedAveragedMLP


def require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_summary(path: Path) -> dict:
    """Count actual JUnit testcases rather than hard-coding a suite size."""
    cases = list(ET.parse(path).getroot().iter("testcase"))
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    for case in cases:
        outcome = next((name for name in ("error", "failure", "skipped") if case.find(name) is not None), None)
        key = {"error": "errors", "failure": "failed", "skipped": "skipped"}.get(outcome, "passed")
        counts[key] += 1
    require(counts["passed"] > 0, "JUnit report contains no passed tests.")
    require(not counts["failed"] and not counts["errors"], "JUnit report contains failed tests or collection errors.")
    return counts


def verify_folds(metadata: pd.DataFrame, training_ids, folds: list) -> int:
    training = set(training_ids)
    held_out = Counter()
    require(len(folds) >= 2, "At least two saved CV folds are required.")
    for fold in folds:
        fit, hold = fold["fit_track_ids"], fold["validation_track_ids"]
        require(len(fit) == len(set(fit)) and len(hold) == len(set(hold)), "Duplicate track IDs inside a CV fold.")
        require(not set(fit) & set(hold), "CV fitting and holdout tracks overlap.")
        require(set(fit) | set(hold) == training, "CV fold does not partition exactly the training tracks.")
        fit_artists = set(metadata.loc[fit, "artist_id"])
        hold_artists = set(metadata.loc[hold, "artist_id"])
        require(not fit_artists & hold_artists, "Artists overlap across a CV fold.")
        held_out.update(hold)
    require(set(held_out) == training and set(held_out.values()) == {1}, "CV must hold out each training track exactly once.")
    return len(held_out)


def verify_model_predictions(model, X, expected_labels, expected_probabilities, labels):
    """Detect stale models even if their aggregate accuracy happens to agree."""
    require(list(model.classes_) == list(labels), "Saved model class order differs from the report.")
    require(list(model.feature_names_in_) == list(X.columns), "Saved model feature schema differs from prepared data.")
    actual = np.asarray(model.predict(X))
    probability = np.asarray(model.predict_proba(X))
    require(np.array_equal(actual, np.asarray(expected_labels)), "Saved model predictions differ from the report CSV.")
    require(probability.shape == np.asarray(expected_probabilities).shape, "Saved model probability shape differs from the report.")
    require(np.all(np.isfinite(probability)) and np.all(probability >= 0) and np.all(probability <= 1), "Invalid saved-model probabilities.")
    require(np.allclose(probability.sum(axis=1), 1), "Saved-model probabilities are not normalized.")
    require(np.allclose(probability, expected_probabilities, atol=1e-7, rtol=1e-7), "Saved model probabilities differ from the report CSV.")
    return actual, probability


def verify_ann_average(model, X, expected_seeds) -> int:
    """Check ANN members through their own upstream pipeline transformations."""
    if isinstance(model, Pipeline):
        transformed = model[:-1].transform(X) if len(model.steps) > 1 else X
        return verify_ann_average(model.steps[-1][1], transformed, expected_seeds)
    if isinstance(model, SeedAveragedMLP):
        require(tuple(model.seeds) == tuple(expected_seeds), "Saved ANN seeds differ from selection metadata.")
        require(len(model.estimators_) == len(model.seeds), "Saved ANN seed member count differs.")
        expected = np.mean([member.predict_proba(X) for member in model.estimators_], axis=0)
        require(np.allclose(model.predict_proba(X), expected), "ANN probabilities do not equal the fixed-seed average.")
        return 1
    members = getattr(model, "estimators_", [])
    # Stacks/voters contain named pipelines; a tree forest contains unnamed
    # decision trees and is a leaf for this ANN-specific check.
    return sum(
        verify_ann_average(member[1], X, expected_seeds)
        for member in members if isinstance(member, tuple) and len(member) == 2
    )


def verify_discovery_artifacts(models: Path, X_train: pd.DataFrame, X_query: pd.DataFrame) -> dict:
    """Exercise the saved query transform and reconcile every catalog coordinate."""
    try:
        transform = joblib.load(models / "discovery_transform.joblib")
    except Exception as error:
        raise ValueError("Could not load the saved discovery transform.") from error
    require(
        list(getattr(transform, "feature_names_in_", [])) == list(X_train.columns),
        "Discovery transform feature schema differs from prepared data.",
    )
    require(list(X_query.columns) == list(X_train.columns), "Discovery query feature schema differs.")
    try:
        with threadpool_limits(limits=2):
            projected_train = np.asarray(transform.transform(X_train))
            projected_query = np.asarray(transform.transform(X_query))
    except Exception as error:
        raise ValueError("The saved discovery transform could not project training/query features.") from error
    require(
        projected_train.ndim == 2 and projected_train.shape[0] == len(X_train)
        and projected_train.shape[1] > 0 and np.isfinite(projected_train).all(),
        "Invalid training projection from the saved discovery transform.",
    )
    require(
        projected_query.shape == (len(X_query), projected_train.shape[1])
        and np.isfinite(projected_query).all(),
        "Invalid query projection from the saved discovery transform.",
    )
    try:
        with np.load(models / "training_embeddings.npz", allow_pickle=False) as catalog:
            ids, coordinates = catalog["track_ids"], catalog["values"]
    except Exception as error:
        raise ValueError("Could not load the saved similarity catalog.") from error
    require(np.array_equal(ids, X_train.index), "Similarity catalog IDs differ from training IDs.")
    require(
        coordinates.shape == projected_train.shape and np.isfinite(coordinates).all(),
        "Similarity catalog dimensions differ from the saved discovery transform.",
    )
    require(
        np.allclose(coordinates, projected_train, atol=1e-6, rtol=1e-6),
        "Similarity catalog coordinates differ from the saved discovery transform.",
    )
    return {
        "discovery_transform_matches_catalog": True,
        "discovery_catalog_rows_checked": len(X_train),
        "discovery_query_rows_checked": len(X_query),
        "discovery_components": projected_train.shape[1],
    }


def verify_artifacts(root: Path) -> dict:
    reports, models = root / "reports", root / "models"
    dataset = load_dataset(root)  # Includes processed-file checksums and schema validation.
    audit = split_audit(dataset.metadata)
    require(not any(audit["artist_overlap_counts"].values()), "Artists overlap across official partitions.")
    require(audit["missing_artist_ids"] == 0, "Missing artist IDs prevent grouping checks.")
    train_ids, validation_ids, test_ids = (dataset.ids(name) for name in ("training", "validation", "test"))
    selection, run = read_json(reports / "selection.json"), read_json(reports / "run_info.json")
    folds = read_json(reports / "cv_splits.json")
    fold_rows = verify_folds(dataset.metadata, train_ids, folds)
    require(set(selection["training_track_ids"]) == set(train_ids), "Selection metadata training IDs differ.")
    require(set(selection["validation_track_ids"]) == set(validation_ids), "Selection metadata validation IDs differ.")
    require(selection["test_used_for_selection"] is False, "Selection metadata reports test use.")
    comparison = pd.read_csv(reports / "model_comparison.csv").set_index("model")
    predictions = pd.read_csv(reports / "test_predictions.csv").set_index("track_id")
    require(predictions.index.equals(test_ids), "Prediction CSV IDs/order differ from the official test split.")
    require(np.array_equal(predictions["actual"], dataset.y.loc[test_ids]), "Prediction CSV actual labels differ from prepared data.")
    champion = selection["champion"]
    require(champion == run["champion"], "Run and selection champion names differ.")
    marked = comparison.index[comparison["selected_by_validation"]].tolist()
    require(marked == [champion], "Comparison table marks a different champion.")
    matrices = read_json(reports / "confusion_matrices.json")
    labels = matrices["labels"]
    require(labels == list(dataset.manifest["genres"]), "Report class order differs from the dataset.")
    class_reports = read_json(reports / "classification_reports.json")
    for name, row in comparison.iterrows():
        predicted = predictions[IMPROVED_SLUGS[name]]
        for metric, value in metrics(predictions.actual, predicted, labels).items():
            require(np.isclose(value, row["test_" + metric]), f"{name}: test {metric} does not reconcile.")
        require(np.array_equal(confusion_matrix(predictions.actual, predicted, labels=labels), matrices["matrices"][name]), f"{name}: confusion matrix does not reconcile.")
        actual_report = classification_report(predictions.actual, predicted, labels=labels, output_dict=True, zero_division=0)
        for label in labels:
            for metric, value in actual_report[label].items():
                require(np.isclose(value, class_reports[name][label][metric]), f"{name}: per-genre {label} {metric} differs.")
    probability_columns = ["p_" + label for label in labels]
    expected_probability = predictions[probability_columns].to_numpy()
    final = joblib.load(models / "best_model.joblib")
    training = joblib.load(models / "selected_training_model.joblib")
    with threadpool_limits(limits=2):
        actual, probability = verify_model_predictions(final, dataset.X.loc[test_ids], predictions[IMPROVED_SLUGS[champion]], expected_probability, labels)
        require(np.array_equal(np.asarray(labels)[probability.argmax(axis=1)], actual), "Champion probabilities and predicted labels differ.")
        require(np.allclose(probability.max(axis=1), predictions.champion_confidence), "Champion confidence differs from probabilities.")
        chosen = comparison.loc[champion]
        require(np.isclose(top_k_accuracy_score(predictions.actual, probability, k=3, labels=labels), chosen.test_top3_accuracy), "Champion top-3 accuracy differs.")
        require(np.isclose(log_loss(predictions.actual, probability, labels=labels), chosen.test_log_loss), "Champion log-loss differs.")
        require(list(training.feature_names_in_) == list(dataset.X.columns) and list(training.classes_) == labels, "Training-only model schema differs.")
        validation_prediction = training.predict(dataset.X.loc[validation_ids])
        validation_metrics = metrics(dataset.y.loc[validation_ids], validation_prediction, labels)
        for metric, value in validation_metrics.items():
            require(np.isclose(value, chosen["validation_" + metric]), f"Training-only validation {metric} differs from champion report.")
        importance = pd.read_csv(reports / "feature_family_importance.csv")
        require(np.allclose(importance.baseline_validation_f1, validation_metrics["f1_macro"]), "Feature-importance baseline differs from the training-only model.")
        ann_members = verify_ann_average(final, dataset.X.loc[test_ids[:16]], selection.get("ann_seeds", []))
    discovery = verify_discovery_artifacts(
        models, dataset.X.loc[train_ids], dataset.X.loc[validation_ids[:16]],
    )
    candidates = len(pd.read_csv(reports / "cv_results.csv"))
    require(candidates == run["candidate_count"], "Candidate count differs from run metadata.")
    require(len(folds) == run["cv_folds"], "CV fold count differs from run metadata.")
    return {
        "cv_candidates": candidates,
        "cv_folds": len(folds),
        "training_tracks_assessed_exactly_once": fold_rows,
        "artist_overlap_per_fold": 0,
        "official_split_artist_overlap": audit["artist_overlap_counts"],
        "test_metrics_reconciled_models": len(comparison),
        "test_prediction_rows": len(predictions),
        "champion": champion,
        "saved_model_class": type(final).__name__,
        "saved_model_predictions_match_csv": True,
        "saved_model_probabilities_match_csv": True,
        "training_only_validation_metrics_match_report": True,
        "feature_importance_baseline_matches_model": True,
        "ann_members_checked": ann_members,
        "ann_probability_check_rows": min(16, len(test_ids)),
        "ann_probabilities_equal_fixed_seed_average": True if ann_members else None,
        **discovery,
        "model_sha256": {path.name: sha256(path) for path in sorted(models.glob("*")) if path.is_file()},
        "processed_sha256": dataset.manifest["processed_sha256"],
    }


def verify_and_write(root: Path, pytest_report: Path, output: Path) -> dict:
    tests = test_summary(pytest_report)
    result = verify_artifacts(root)
    result.update({
        "automated_tests_passed": tests["passed"],
        "automated_test_outcomes": tests,
        "pytest_report_sha256": sha256(pytest_report),
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_git_worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()),
        "source_sha256": {
            str(path.relative_to(root)): sha256(path)
            for path in sorted([*root.glob("src/musicmap/*.py"), root / "scripts/verify_artifacts.py"])
            if path.is_file()
        },
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--test-report", "--pytest-report", dest="pytest_report", type=Path, required=True, help="Fresh pytest --junitxml result file")
    parser.add_argument("--output", type=Path, help="Default: <root>/reports/verification.json")
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        result = verify_and_write(root, args.pytest_report, args.output or root / "reports/verification.json")
    except (ValueError, KeyError, OSError, ET.ParseError) as error:
        print(f"Verification failed; previous evidence was not replaced: {error}", file=sys.stderr)
        return 1
    print(f"Verified {result['test_metrics_reconciled_models']} result rows, {result['test_prediction_rows']} test predictions, and {result['automated_tests_passed']} passed tests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
