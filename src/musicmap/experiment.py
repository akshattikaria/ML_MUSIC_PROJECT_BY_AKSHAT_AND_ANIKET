"""Tune on the official validation split; lock selection before test evaluation."""

from __future__ import annotations

import json
import platform
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from .data import Dataset, write_json

MODEL_NAMES = ("Decision Tree", "KNN", "SVM (RBF)", "ANN (MLP)")
BASELINE = "Dummy baseline"
SLUGS = {
    "Decision Tree": "decision_tree", "KNN": "knn",
    "SVM (RBF)": "svm", "ANN (MLP)": "ann", BASELINE: "dummy",
}


class LabelEncodedClassifier(ClassifierMixin, BaseEstimator):
    """Keep text labels at the API boundary and numeric labels inside the ANN.

    This also avoids a scikit-learn 1.8 early-stopping path that calls
    numpy.isnan on predicted string labels.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.encoder_ = LabelEncoder()
        encoded = self.encoder_.fit_transform(y)
        self.estimator_ = clone(self.estimator).fit(X, encoded)
        self.classes_ = self.encoder_.classes_
        self.n_features_in_ = self.estimator_.n_features_in_
        return self

    def predict(self, X):
        return self.encoder_.inverse_transform(self.estimator_.predict(X).astype(int))

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def preprocessing() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("variance", VarianceThreshold()),
        ("scaler", StandardScaler()),
    ])


def build_model(name: str, parameters: dict, seed: int = 42) -> Pipeline:
    estimators = {
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "KNN": KNeighborsClassifier(n_jobs=1),
        "SVM (RBF)": SVC(kernel="rbf", cache_size=256),
        "ANN (MLP)": MLPClassifier(
            random_state=seed, activation="relu", solver="adam",
            batch_size=128, max_iter=200, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=15, learning_rate_init=0.001,
        ),
        BASELINE: DummyClassifier(strategy="most_frequent"),
    }
    if name not in estimators:
        raise ValueError(f"Unknown model: {name}")
    estimator = estimators[name].set_params(**parameters)
    if name == "ANN (MLP)":
        estimator = LabelEncodedClassifier(estimator)
    return Pipeline([*preprocessing().steps, ("model", estimator)])


def parameter_grid(name: str, profile: str = "standard") -> list[dict]:
    if profile not in ("quick", "standard"):
        raise ValueError("Profile must be 'quick' or 'standard'.")
    if profile == "quick":
        return {
            "Decision Tree": [{"max_depth": 15, "min_samples_leaf": 5}],
            "KNN": [{"n_neighbors": 15, "weights": "distance"}],
            "SVM (RBF)": [{"C": 10.0, "gamma": "scale"}],
            "ANN (MLP)": [{"hidden_layer_sizes": (128,), "alpha": 0.01}],
            BASELINE: [{}],
        }[name]
    grids = {
        "Decision Tree": {"max_depth": [10, 20, None], "min_samples_leaf": [2, 5]},
        "KNN": {"n_neighbors": [5, 15, 31], "weights": ["uniform", "distance"]},
        "SVM (RBF)": {"C": [1.0, 10.0], "gamma": ["scale", 0.001]},
        "ANN (MLP)": {"hidden_layer_sizes": [(128,), (128, 64)], "alpha": [0.0001, 0.01]},
        BASELINE: {},
    }
    return list(ParameterGrid(grids[name]))


def metrics(y_true, y_pred, labels: list[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }


def _selection_key(candidate: dict) -> tuple:
    return candidate["validation_f1_macro"], candidate["validation_accuracy"]


def run_experiment(
    dataset: Dataset, output: Path, profile: str = "standard",
    seed: int = 42, threads: int = 2,
) -> pd.DataFrame:
    if threads < 1:
        raise ValueError("Threads must be at least 1.")
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model_dir = output.parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    train_ids, val_ids = dataset.ids("training"), dataset.ids("validation")
    labels = sorted(dataset.y.loc[train_ids].unique().tolist())
    X_train, y_train = dataset.X.loc[train_ids], dataset.y.loc[train_ids]
    X_val, y_val = dataset.X.loc[val_ids], dataset.y.loc[val_ids]
    candidates, selected, fitted = [], {}, {}

    with threadpool_limits(limits=threads):
        for name in (*MODEL_NAMES, BASELINE):
            best = None
            for candidate_id, parameters in enumerate(parameter_grid(name, profile)):
                print(f"Tuning {name}: {parameters}", flush=True)
                pipeline = build_model(name, parameters, seed)
                fit_start = time.perf_counter()
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    pipeline.fit(X_train, y_train)
                fit_seconds = time.perf_counter() - fit_start
                predict_start = time.perf_counter()
                predicted = pipeline.predict(X_val)
                row = {
                    "model": name, "candidate_id": candidate_id, "parameters": parameters,
                    "fit_seconds": fit_seconds,
                    "validation_predict_seconds": time.perf_counter() - predict_start,
                    "warnings": [f"{w.category.__name__}: {w.message}" for w in captured],
                    **{f"validation_{k}": v for k, v in metrics(y_val, predicted, labels).items()},
                }
                candidates.append(row)
                if best is None or _selection_key(row) > _selection_key(best):
                    best = row
                    fitted[name] = pipeline
            selected[name] = best
            print(f"  Best validation macro F1: {best['validation_f1_macro']:.4f}", flush=True)

        champion = max(MODEL_NAMES, key=lambda name: _selection_key(selected[name]))
        selection = {
            "criterion": "Validation macro F1; validation accuracy breaks ties; grid order breaks exact ties",
            "champion": champion, "profile": profile, "seed": seed, "threads": threads,
            "test_used_for_selection": False, "selected": selected,
            "training_track_ids": train_ids.tolist(), "validation_track_ids": val_ids.tolist(),
        }
        # This file is written before reading test features, labels, or scores.
        write_json(output / "selection.json", selection)
        write_json(output / "candidate_results.json", candidates)
        candidate_table = pd.DataFrame(candidates)
        for column in ("parameters", "warnings"):
            candidate_table[column] = candidate_table[column].map(json.dumps)
        candidate_table.to_csv(output / "candidate_results.csv", index=False)
        joblib.dump(fitted[champion], model_dir / "selected_training_model.joblib", compress=3)

        print(f"Selection locked: {champion}. Refitting on training + validation.", flush=True)
        dev_ids = train_ids.append(val_ids)
        test_ids = dataset.ids("test")
        results, reports, confusion = [], {}, {}
        predictions = dataset.metadata.loc[test_ids, ["genre", "artist_id", "artist", "title"]].copy()
        predictions = predictions.rename(columns={"genre": "actual"})
        for name in (*MODEL_NAMES, BASELINE):
            tuned = fitted[name]
            train_scores = metrics(y_train, tuned.predict(X_train), labels)
            final = clone(tuned)
            refit_start = time.perf_counter()
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                final.fit(dataset.X.loc[dev_ids], dataset.y.loc[dev_ids])
            refit_seconds = time.perf_counter() - refit_start
            predict_start = time.perf_counter()
            predicted = final.predict(dataset.X.loc[test_ids])
            predict_seconds = time.perf_counter() - predict_start
            results.append({
                "model": name, "selected_by_validation": name == champion,
                "parameters": json.dumps(selected[name]["parameters"]),
                "train_accuracy": train_scores["accuracy"], "train_f1_macro": train_scores["f1_macro"],
                **{k: v for k, v in selected[name].items() if k.startswith("validation_") and k != "validation_predict_seconds"},
                **{f"test_{k}": v for k, v in metrics(dataset.y.loc[test_ids], predicted, labels).items()},
                "tuning_fit_seconds": sum(c["fit_seconds"] for c in candidates if c["model"] == name),
                "refit_seconds": refit_seconds, "test_predict_seconds": predict_seconds,
                "refit_warnings": json.dumps([f"{w.category.__name__}: {w.message}" for w in captured]),
                "ann_iterations": int(final.named_steps["model"].estimator_.n_iter_) if name == "ANN (MLP)" else None,
            })
            reports[name] = classification_report(
                dataset.y.loc[test_ids], predicted, labels=labels, output_dict=True, zero_division=0
            )
            confusion[name] = confusion_matrix(dataset.y.loc[test_ids], predicted, labels=labels).tolist()
            predictions[SLUGS[name]] = predicted
            if name == champion:
                joblib.dump(final, model_dir / "best_model.joblib", compress=3)
            print(f"  {name}: test accuracy {results[-1]['test_accuracy']:.4f}, macro F1 {results[-1]['test_f1_macro']:.4f}", flush=True)

    results = pd.DataFrame(results).sort_values("validation_f1_macro", ascending=False, kind="stable")
    results.to_csv(output / "model_comparison.csv", index=False)
    predictions.to_csv(output / "test_predictions.csv")
    write_json(output / "classification_reports.json", reports)
    write_json(output / "confusion_matrices.json", {"labels": labels, "matrices": confusion})
    write_json(output / "run_info.json", {
        "profile": profile, "seed": seed, "threads": threads, "champion": champion,
        "elapsed_seconds": time.perf_counter() - started,
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl")},
        "dataset_processed_sha256": dataset.manifest.get("processed_sha256", {}),
        "n_train": len(train_ids), "n_validation": len(val_ids), "n_test": len(test_ids),
        "n_features": dataset.X.shape[1], "candidate_count": len(candidates),
        "ann_early_stopping": "10% internal holdout inside each fit; external official validation selects hyperparameters",
    })
    return results
