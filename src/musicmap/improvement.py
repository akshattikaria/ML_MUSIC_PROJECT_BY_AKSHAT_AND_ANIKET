"""Grouped-CV extension of the original MusicMap benchmark.

The old test set has already been viewed: the final evaluation is an extended
benchmark comparison, not a new independent confirmation experiment.
"""
from __future__ import annotations

import json
import platform
import shutil
import tempfile
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from .data import Dataset, write_json
from .experiment import BASELINE, MODEL_NAMES, SLUGS, build_model, metrics

EXTRA_TREES = "Extra Trees"
IMPROVED_MODELS = (*MODEL_NAMES, EXTRA_TREES)
IMPROVED_SLUGS = {**SLUGS, EXTRA_TREES: "extra_trees"}


class AdaptiveReduction(TransformerMixin, BaseEstimator):
    """Cap requested dimensions to the available fold rank, for small subsets."""

    def __init__(self, kind="pca", n_components=128, random_state=42):
        self.kind = kind
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        if self.n_components < 1:
            raise ValueError("Reduction size must be positive.")
        if self.kind == "select":
            self.transformer_ = SelectKBest(f_classif, k=min(self.n_components, X.shape[1]))
        else:
            self.transformer_ = PCA(
                n_components=min(self.n_components, X.shape[1], X.shape[0] - 1),
                svd_solver="randomized", random_state=self.random_state,
            )
        self.transformer_.fit(X, y)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return self.transformer_.transform(X)


class SeedAveragedMLP(ClassifierMixin, BaseEstimator):
    """Average probabilities from predeclared ANN seeds, never select a seed."""

    def __init__(self, parameters=None, seeds=(42, 43, 44)):
        self.parameters = parameters
        self.seeds = seeds

    def fit(self, X, y):
        if not self.seeds:
            raise ValueError("At least one ANN seed is required.")
        self.encoder_ = LabelEncoder().fit(y)
        encoded = self.encoder_.transform(y)
        self.estimators_ = []
        for seed in self.seeds:
            params = {
                "activation": "relu", "solver": "adam", "batch_size": 128,
                "max_iter": 300, "early_stopping": True, "validation_fraction": 0.1,
                "n_iter_no_change": 20, "learning_rate_init": 0.001,
                **(self.parameters or {}), "random_state": seed,
            }
            self.estimators_.append(MLPClassifier(**params).fit(X, encoded))
        self.classes_ = self.encoder_.classes_
        self.n_features_in_ = self.estimators_[0].n_features_in_
        return self

    def predict_proba(self, X):
        return np.mean([estimator.predict_proba(X) for estimator in self.estimators_], axis=0)

    def predict(self, X):
        return self.encoder_.inverse_transform(self.predict_proba(X).argmax(axis=1))


def search_space() -> dict[str, list[dict]]:
    """Fixed affordable grids, saved before search or test evaluation."""
    def config(scaling="standard", reduction="all", n=None, **classifier):
        result = {"scaling": scaling, "reduction": reduction, "classifier": classifier}
        if n is not None:
            result["n_components"] = n
        return result

    spaces = {name: [] for name in (*IMPROVED_MODELS, BASELINE)}
    for depth in (10, 20, None):
        for leaf in (2, 5):
            spaces["Decision Tree"].append(config(max_depth=depth, min_samples_leaf=leaf))
    for scaling in ("standard", "quantile"):
        for reduction, n in (("all", None), ("select", 128), ("pca", 64), ("pca", 128)):
            for k in (5, 15):
                spaces["KNN"].append(config(scaling, reduction, n, n_neighbors=k, weights="distance"))
        for c in (0.1, 1.0, 10.0, 100.0):
            for gamma in ("scale", 0.0003, 0.001, 0.003):
                spaces["SVM (RBF)"].append(config(scaling, C=c, gamma=gamma))
        for n in (128, 256):
            for c in (1.0, 10.0, 100.0):
                spaces["SVM (RBF)"].append(config(scaling, "select", n, C=c, gamma="scale"))
        for n in (64, 128):
            for c in (1.0, 10.0):
                spaces["SVM (RBF)"].append(config(scaling, "pca", n, C=c, gamma="scale"))
    spaces["KNN"].append(config(n_neighbors=31, weights="distance"))
    for hidden in ((128,), (128, 64), (256, 128)):
        for rate in (0.0003, 0.001):
            for alpha in (0.0001, 0.01):
                spaces["ANN (MLP)"].append(config(
                    "quantile", hidden_layer_sizes=hidden, learning_rate_init=rate,
                    alpha=alpha, max_iter=300, n_iter_no_change=20,
                ))
    for alpha in (0.0001, 0.01):
        spaces["ANN (MLP)"].append(config(
            hidden_layer_sizes=(128, 64), alpha=alpha, learning_rate_init=0.001,
            max_iter=300, n_iter_no_change=20,
        ))
    for leaf in (1, 2):
        for max_features in ("sqrt", 0.5):
            spaces[EXTRA_TREES].append(config(
                n_estimators=300, max_features=max_features, min_samples_leaf=leaf,
            ))
    spaces[BASELINE] = [config()]
    return spaces


def build_candidate(name: str, config: dict, seed: int = 42, ann_seeds=None) -> Pipeline:
    scaling, reduction = config["scaling"], config["reduction"]
    if scaling not in ("standard", "quantile") or reduction not in ("all", "select", "pca"):
        raise ValueError("Unknown scaling or reduction method.")
    steps = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("variance", VarianceThreshold()),
        ("scaler", StandardScaler() if scaling == "standard" else QuantileTransformer(
            n_quantiles=256, output_distribution="normal", random_state=seed,
        )),
    ]
    if reduction != "all":
        steps.append(("reduction", AdaptiveReduction(
            reduction, config["n_components"], random_state=seed,
        )))
    parameters = config["classifier"]
    if name == EXTRA_TREES:
        estimator = ExtraTreesClassifier(random_state=seed, n_jobs=1, **parameters)
    elif name == "ANN (MLP)" and ann_seeds is not None:
        estimator = SeedAveragedMLP(parameters=parameters, seeds=tuple(ann_seeds))
    else:
        estimator = build_model(name, parameters, seed).named_steps["model"]
    return Pipeline([*steps, ("model", estimator)])


def make_group_folds(dataset: Dataset, n_splits: int = 3, seed: int = 42):
    ids = dataset.ids("training")
    y = dataset.y.loc[ids]
    groups = dataset.metadata.loc[ids, "artist_id"]
    if groups.isna().any():
        raise ValueError("Artist IDs are required for grouped CV.")
    labels = set(y)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds, audits = [], []
    for number, (fit, validation) in enumerate(cv.split(dataset.X.loc[ids], y, groups)):
        overlap = len(set(groups.iloc[fit]) & set(groups.iloc[validation]))
        if overlap or set(y.iloc[fit]) != labels or set(y.iloc[validation]) != labels:
            raise ValueError("CV fold has artist overlap or is missing a genre. Adjust fold count/seed.")
        folds.append((fit, validation))
        audits.append({
            "fold": number, "artist_overlap": overlap,
            "fit_track_ids": ids[fit].tolist(), "validation_track_ids": ids[validation].tolist(),
            "fit_genres": sorted(set(y.iloc[fit])), "validation_genres": sorted(set(y.iloc[validation])),
        })
    return folds, audits


def _fit(pipeline, X, y):
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        pipeline.fit(X, y)
    return time.perf_counter() - started, [f"{w.category.__name__}: {w.message}" for w in captured]


def preserve_baseline(output: Path) -> None:
    root = output.parent
    original = root / "baseline"
    if original.exists():
        return
    info = output / "run_info.json"
    if info.exists():
        if json.loads(info.read_text())["profile"] == "improved":
            raise ValueError("Cannot archive an improved run as the original baseline.")
        shutil.copytree(output, original / "reports")
        if (root / "models").exists():
            shutil.copytree(root / "models", original / "models")


def run_improvement(
    dataset: Dataset, output: Path, seed: int = 42, threads: int = 2,
    cv_folds: int = 3, spaces: dict | None = None,
) -> pd.DataFrame:
    if threads < 1 or cv_folds < 2:
        raise ValueError("Threads must be positive and CV requires at least two folds.")
    started = time.perf_counter()
    output = Path(output)
    preserve_baseline(output)
    output.mkdir(parents=True, exist_ok=True)
    model_dir = output.parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    spaces = search_space() if spaces is None else spaces
    if not spaces or any(not configs for configs in spaces.values()) or not (set(spaces) - {BASELINE}):
        raise ValueError("At least one non-baseline model and nonempty grids are required.")
    training, validation = dataset.ids("training"), dataset.ids("validation")
    X, y = dataset.X.loc[training], dataset.y.loc[training]
    X_val, y_val = dataset.X.loc[validation], dataset.y.loc[validation]
    labels = sorted(y.unique().tolist())
    folds, audits = make_group_folds(dataset, cv_folds, seed)
    seeds = (seed, seed + 1, seed + 2)
    write_json(output / "cv_splits.json", audits)
    write_json(output / "search_plan.json", {
        "seed": seed, "cv_folds": cv_folds, "ann_seeds": list(seeds), "spaces": spaces,
        "selection_rule": "Config: mean grouped-CV macro F1, lower SD breaks ties. Algorithm: external validation macro F1, accuracy breaks ties.",
        "dataset_processed_sha256": dataset.manifest.get("processed_sha256", {}),
        "evaluation_status": "extended benchmark; original test previously viewed",
    })
    candidates, selected, fitted = [], {}, {}
    total = sum(map(len, spaces.values()))
    with threadpool_limits(limits=threads), tempfile.TemporaryDirectory(prefix="musicmap_cv_") as cache_dir:
        cache = joblib.Memory(cache_dir, verbose=0)
        for name, configs in spaces.items():
            family = []
            for candidate_id, config in enumerate(configs):
                fold_scores, fit_seconds, messages = [], 0.0, []
                for fit_ids, hold_ids in folds:
                    model = build_candidate(name, config, seed)
                    model.memory = cache
                    seconds, captured = _fit(model, X.iloc[fit_ids], y.iloc[fit_ids])
                    fit_seconds += seconds
                    messages.extend(captured)
                    predicted = model.predict(X.iloc[hold_ids])
                    fold_scores.append(float(f1_score(
                        y.iloc[hold_ids], predicted, labels=labels, average="macro", zero_division=0,
                    )))
                row = {
                    "model": name, "candidate_id": candidate_id, "parameters": config,
                    "cv_mean_f1": float(np.mean(fold_scores)),
                    "cv_std_f1": float(np.std(fold_scores, ddof=1)),
                    "cv_fold_f1": fold_scores, "fit_seconds": fit_seconds,
                    "warnings": sorted(set(messages)),
                }
                candidates.append(row)
                family.append(row)
                print(f"CV {len(candidates)}/{total} {name} mean F1={row['cv_mean_f1']:.4f} ± {row['cv_std_f1']:.4f}: {config}", flush=True)
                write_json(output / "cv_results.json", candidates)
                pd.DataFrame([
                    {**r, "parameters": json.dumps(r["parameters"]), "warnings": json.dumps(r["warnings"]),
                     "cv_fold_f1": json.dumps(r["cv_fold_f1"])} for r in candidates
                ]).to_csv(output / "cv_results.csv", index=False)
            selected[name] = max(family, key=lambda row: (row["cv_mean_f1"], -row["cv_std_f1"]))

        # ANN hyperparameters are fixed before repeating seeds or viewing validation.
        if "ANN (MLP)" in spaces:
            config = selected["ANN (MLP)"]["parameters"]
            seed_scores = {s: [] for s in seeds}
            ensemble_scores = []
            for fit_ids, hold_ids in folds:
                probabilities = []
                for s in seeds:
                    model = build_candidate("ANN (MLP)", config, seed, ann_seeds=(s,))
                    model.memory = cache
                    _fit(model, X.iloc[fit_ids], y.iloc[fit_ids])
                    p = model.predict_proba(X.iloc[hold_ids])
                    probabilities.append(p)
                    predicted = np.array(labels)[p.argmax(axis=1)]
                    seed_scores[s].append(float(f1_score(
                        y.iloc[hold_ids], predicted, labels=labels, average="macro", zero_division=0,
                    )))
                predicted = np.array(labels)[np.mean(probabilities, axis=0).argmax(axis=1)]
                ensemble_scores.append(float(f1_score(
                    y.iloc[hold_ids], predicted, labels=labels, average="macro", zero_division=0,
                )))
            ann_model = build_candidate("ANN (MLP)", config, seed, ann_seeds=seeds)
            seconds, messages = _fit(ann_model, X, y)
            fitted["ANN (MLP)"] = ann_model
            selected["ANN (MLP)"] = dict(selected["ANN (MLP)"])
            selected["ANN (MLP)"]["tuning_cv_mean_f1"] = selected["ANN (MLP)"]["cv_mean_f1"]
            selected["ANN (MLP)"]["cv_mean_f1"] = float(np.mean(ensemble_scores))
            selected["ANN (MLP)"]["cv_std_f1"] = float(np.std(ensemble_scores, ddof=1))
            transformed = ann_model[:-1].transform(X_val)
            seed_rows = []
            ensemble = ann_model.named_steps["model"]
            for s, estimator in zip(seeds, ensemble.estimators_):
                pred = ensemble.encoder_.inverse_transform(estimator.predict(transformed))
                scores = metrics(y_val, pred, labels)
                seed_rows.append({
                    "seed": s, "cv_mean_f1": float(np.mean(seed_scores[s])),
                    "cv_std_f1": float(np.std(seed_scores[s], ddof=1)),
                    "validation_f1_macro": scores["f1_macro"],
                    "validation_accuracy": scores["accuracy"],
                })
            scores = metrics(y_val, ann_model.predict(X_val), labels)
            seed_rows.append({
                "seed": "ensemble", "cv_mean_f1": float(np.mean(ensemble_scores)),
                "cv_std_f1": float(np.std(ensemble_scores, ddof=1)),
                "validation_f1_macro": scores["f1_macro"], "validation_accuracy": scores["accuracy"],
            })
            pd.DataFrame(seed_rows).to_csv(output / "ann_seed_results.csv", index=False)

        for name in spaces:
            if name not in fitted:
                model = build_candidate(name, selected[name]["parameters"], seed)
                _fit(model, X, y)
                fitted[name] = model
            row = selected[name]
            prediction = fitted[name].predict(X_val)
            row.update({f"validation_{k}": v for k, v in metrics(y_val, prediction, labels).items()})
            print(f"Validation {name}: macro F1={row['validation_f1_macro']:.4f}", flush=True)

        champion = max(
            (name for name in spaces if name != BASELINE),
            key=lambda name: (selected[name]["validation_f1_macro"], selected[name]["validation_accuracy"]),
        )
        write_json(output / "selection.json", {
            "criterion": "Config by grouped-CV mean macro F1 (lower SD tie); algorithm by validation macro F1 (accuracy tie)",
            "champion": champion, "profile": "improved", "seed": seed, "threads": threads,
            "cv_folds": cv_folds, "ann_seeds": list(seeds), "selected": selected,
            "test_used_for_selection": False, "test_previously_viewed_in_baseline": True,
            "training_track_ids": training.tolist(), "validation_track_ids": validation.tolist(),
        })
        write_json(output / "candidate_results.json", candidates)
        pd.DataFrame([
            {**r, "parameters": json.dumps(r["parameters"]), "warnings": json.dumps(r["warnings"]),
             "cv_fold_f1": json.dumps(r["cv_fold_f1"])} for r in candidates
        ]).to_csv(output / "candidate_results.csv", index=False)
        joblib.dump(fitted[champion], model_dir / "selected_training_model.joblib", compress=3)
        print(f"Selection locked: {champion}. Evaluating extended benchmark.", flush=True)
        # First request for the test partition is after the selection file.
        test_ids = dataset.ids("test")
        development = training.append(validation)
        predictions = dataset.metadata.loc[test_ids, ["genre", "artist_id", "artist", "title"]].copy()
        predictions = predictions.rename(columns={"genre": "actual"})
        results, classification, confusion = [], {}, {}
        for name in spaces:
            model = clone(fitted[name])
            seconds, messages = _fit(model, dataset.X.loc[development], dataset.y.loc[development])
            before = time.perf_counter()
            prediction = model.predict(dataset.X.loc[test_ids])
            predict_seconds = time.perf_counter() - before
            training_scores = metrics(y, fitted[name].predict(X), labels)
            results.append({
                "model": name, "selected_by_validation": name == champion,
                "parameters": json.dumps(selected[name]["parameters"]),
                "cv_mean_f1": selected[name]["cv_mean_f1"], "cv_std_f1": selected[name]["cv_std_f1"],
                "train_accuracy": training_scores["accuracy"], "train_f1_macro": training_scores["f1_macro"],
                **{k: v for k, v in selected[name].items() if k.startswith("validation_")},
                **{f"test_{k}": v for k, v in metrics(dataset.y.loc[test_ids], prediction, labels).items()},
                "tuning_fit_seconds": sum(r["fit_seconds"] for r in candidates if r["model"] == name),
                "refit_seconds": seconds, "test_predict_seconds": predict_seconds,
                "refit_warnings": json.dumps(messages), "ann_iterations": None,
            })
            if name == "ANN (MLP)":
                iterations = [int(m.n_iter_) for m in model.named_steps["model"].estimators_]
                results[-1]["ann_iterations"] = json.dumps(iterations)
            classification[name] = classification_report(
                dataset.y.loc[test_ids], prediction, labels=labels, output_dict=True, zero_division=0,
            )
            confusion[name] = confusion_matrix(dataset.y.loc[test_ids], prediction, labels=labels).tolist()
            predictions[IMPROVED_SLUGS[name]] = prediction
            if name == champion:
                joblib.dump(model, model_dir / "best_model.joblib", compress=3)
            print(f"Test {name}: accuracy={results[-1]['test_accuracy']:.4f}, macro F1={results[-1]['test_f1_macro']:.4f}", flush=True)

    result = pd.DataFrame(results).sort_values("validation_f1_macro", ascending=False, kind="stable")
    result.to_csv(output / "model_comparison.csv", index=False)
    predictions.to_csv(output / "test_predictions.csv")
    write_json(output / "classification_reports.json", classification)
    write_json(output / "confusion_matrices.json", {"labels": labels, "matrices": confusion})
    write_json(output / "run_info.json", {
        "profile": "improved", "seed": seed, "threads": threads, "champion": champion,
        "cv_folds": cv_folds, "ann_seeds": list(seeds), "candidate_count": total,
        "elapsed_seconds": time.perf_counter() - started,
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl")},
        "dataset_processed_sha256": dataset.manifest.get("processed_sha256", {}),
        "n_train": len(training), "n_validation": len(validation), "n_test": len(test_ids),
        "n_features": dataset.X.shape[1], "evaluation_status": "extended benchmark; test previously viewed",
        "ann_early_stopping": "10% track holdout in each fit; probability average of three predeclared seeds",
    })
    summary = {"champion": champion, "evaluation_status": "extended benchmark; original test previously viewed"}
    old_path = output.parent / "baseline/reports/model_comparison.csv"
    if old_path.exists():
        old = pd.read_csv(old_path)
        previous = old.loc[old["selected_by_validation"].astype(str).str.lower().eq("true")].iloc[0]
        current = result.loc[result["model"].eq(champion)].iloc[0]
        summary.update({
            "baseline_champion": previous["model"],
            "baseline_test_accuracy": float(previous["test_accuracy"]),
            "improved_test_accuracy": float(current["test_accuracy"]),
            "test_accuracy_change_percentage_points": float(100 * (current["test_accuracy"] - previous["test_accuracy"])),
            "baseline_test_f1_macro": float(previous["test_f1_macro"]),
            "improved_test_f1_macro": float(current["test_f1_macro"]),
            "test_f1_macro_change": float(current["test_f1_macro"] - previous["test_f1_macro"]),
        })
    write_json(output / "improvement_summary.json", summary)
    return result
