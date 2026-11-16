"""Optimized MusicMap experiment: grouped-CV tuning, more model families, and an ensemble.

Protocol (all decisions are made before the official test partition is requested):

1. Every candidate configuration is scored by artist-grouped K-fold CV inside the
   official 6,400-track training partition (macro F1).
2. The best configuration per model family is kept, together with its out-of-fold
   (OOF) class probabilities.
3. A weighted soft-voting ensemble is built by greedy forward selection on those
   OOF probabilities only (Caruana et al., 2004). No validation or test rows are used.
4. Every family and the ensemble are fitted on training and scored on the official
   800-track validation partition; the champion is chosen by validation macro F1.
5. selection.json is written, then the selected configurations are refitted on
   training + validation and evaluated once on the official test partition.

The official test set was already viewed by the original project, so test scores are
an extended benchmark comparison rather than a pristine confirmation.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, log_loss, top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from .data import Dataset, write_json
from .experiment import BASELINE, MODEL_NAMES, SLUGS, build_model, metrics

EXTRA_TREES = "Extra Trees"
LOGISTIC = "Logistic Regression"
BOOSTING = "Gradient Boosting"
ENSEMBLE = "Soft-voting Ensemble"
STACKED = "Stacked Ensemble"
IMPROVED_MODELS = (*MODEL_NAMES, LOGISTIC, EXTRA_TREES, BOOSTING)
IMPROVED_SLUGS = {
    **SLUGS, EXTRA_TREES: "extra_trees", LOGISTIC: "logistic_regression",
    BOOSTING: "gradient_boosting", ENSEMBLE: "ensemble", STACKED: "stacked",
}
PROFILE = "optimized"
ENSEMBLE_ROUNDS = 25


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

class AdaptiveReduction(TransformerMixin, BaseEstimator):
    """ANOVA selection or PCA, capped to the available rank of the fitting fold."""

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
    """Average probabilities from predeclared ANN seeds; never select a seed."""

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


class WeightedSoftVoter(ClassifierMixin, BaseEstimator):
    """Weighted average of fitted pipelines' class probabilities.

    ``estimators`` is a list of ``(name, pipeline)``; weights are fixed beforehand
    by greedy selection on training-only out-of-fold probabilities.
    """

    def __init__(self, estimators, weights):
        self.estimators = estimators
        self.weights = weights

    def fit(self, X, y):
        self.estimators_ = [(name, clone(model).fit(X, y)) for name, model in self.estimators]
        self._finish()
        return self

    @classmethod
    def from_fitted(cls, estimators, weights):
        voter = cls(estimators=estimators, weights=weights)
        voter.estimators_ = list(estimators)
        voter._finish()
        return voter

    def _finish(self):
        classes = [list(model.classes_) for _, model in self.estimators_]
        if any(c != classes[0] for c in classes):
            raise ValueError("Ensemble members must share class order.")
        self.classes_ = np.asarray(classes[0])
        first = self.estimators_[0][1]
        self.n_features_in_ = first.n_features_in_
        if hasattr(first, "feature_names_in_"):
            self.feature_names_in_ = first.feature_names_in_

    def member_probabilities(self, X) -> dict:
        return {name: model.predict_proba(X) for name, model in self.estimators_}

    def predict_proba(self, X):
        weights = np.asarray(self.weights, dtype=float)
        stacked = [model.predict_proba(X) for _, model in self.estimators_]
        return np.tensordot(weights / weights.sum(), np.asarray(stacked), axes=1)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]



def _stack_features(probabilities: list[np.ndarray]) -> np.ndarray:
    return np.hstack([np.log(np.clip(p, 1e-6, 1.0)) for p in probabilities])


class StackedEnsemble(ClassifierMixin, BaseEstimator):
    """Multinomial logistic meta-learner on members' log-probabilities.

    The meta-learner is trained on training-partition out-of-fold probabilities
    only; members are ordinary fitted pipelines.
    """

    def __init__(self, estimators, meta):
        self.estimators = estimators
        self.meta = meta

    @classmethod
    def from_fitted(cls, estimators, meta):
        model = cls(estimators=estimators, meta=meta)
        model.estimators_ = list(estimators)
        classes = [list(m.classes_) for _, m in model.estimators_]
        if any(c != classes[0] for c in classes) or list(meta.classes_) != classes[0]:
            raise ValueError("Stack members and meta-learner must share class order.")
        model.classes_ = np.asarray(classes[0])
        first = model.estimators_[0][1]
        model.n_features_in_ = first.n_features_in_
        if hasattr(first, "feature_names_in_"):
            model.feature_names_in_ = first.feature_names_in_
        return model

    def fit(self, X, y):  # pragma: no cover - the experiment uses from_fitted
        raise NotImplementedError("Build with StackedEnsemble.from_fitted using OOF-trained meta.")

    def predict_proba(self, X):
        return self.meta.predict_proba(_stack_features([m.predict_proba(X) for _, m in self.estimators_]))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

# ---------------------------------------------------------------------------
# Search space and pipelines
# ---------------------------------------------------------------------------

def search_space() -> dict[str, list[dict]]:
    """Fixed grids, written to search_plan.json before any fitting."""
    def config(scaling="standard", reduction="all", n=None, **classifier):
        result = {"scaling": scaling, "reduction": reduction, "classifier": classifier}
        if n is not None:
            result["n_components"] = n
        return result

    spaces = {name: [] for name in (*IMPROVED_MODELS, BASELINE)}
    for depth in (8, 12, None):
        for leaf in (5, 20):
            spaces["Decision Tree"].append(config(max_depth=depth, min_samples_leaf=leaf))
    for scaling in ("standard", "quantile"):
        for reduction, n in (("all", None), ("select", 128), ("pca", 64)):
            for k in (15, 31):
                spaces["KNN"].append(config(scaling, reduction, n, n_neighbors=k, weights="distance"))
    for scaling in ("standard", "quantile"):
        for reduction, n in (("all", None), ("select", 256)):
            for c in (1.0, 3.0, 10.0):
                spaces["SVM (RBF)"].append(config(scaling, reduction, n, C=c, gamma="scale"))
    for hidden in ((256,), (512,), (256, 128)):
        for alpha in (0.01, 0.1, 1.0):
            spaces["ANN (MLP)"].append(config(
                "quantile", hidden_layer_sizes=hidden, learning_rate_init=0.0003,
                alpha=alpha, max_iter=300, n_iter_no_change=20,
            ))
    for scaling in ("standard", "quantile"):
        for c in (0.001, 0.003, 0.01, 0.03):
            spaces[LOGISTIC].append(config(scaling, C=c, max_iter=3000))
    for max_features in ("sqrt", 0.3):
        for leaf in (1, 2):
            spaces[EXTRA_TREES].append(config(
                n_estimators=500, max_features=max_features, min_samples_leaf=leaf,
            ))
    for leaves in (15, 31):
        for features in (0.2, 0.4):
            spaces[BOOSTING].append(config(
                n_estimators=500, learning_rate=0.03, num_leaves=leaves,
                colsample_bytree=features, subsample=0.8, subsample_freq=1,
                reg_lambda=1.0, min_child_samples=20,
            ))
    spaces[BASELINE] = [config()]
    return spaces


def build_candidate(name: str, config: dict, seed: int = 42, ann_seeds=None,
                    probability: bool = False) -> Pipeline:
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
    parameters = dict(config["classifier"])
    if "hidden_layer_sizes" in parameters:
        parameters["hidden_layer_sizes"] = tuple(parameters["hidden_layer_sizes"])
    if name == EXTRA_TREES:
        estimator = ExtraTreesClassifier(random_state=seed, n_jobs=1, **parameters)
    elif name == LOGISTIC:
        estimator = LogisticRegression(**parameters)
    elif name == BOOSTING:
        from lightgbm import LGBMClassifier
        estimator = LGBMClassifier(random_state=seed, n_jobs=1, verbose=-1, **parameters)
    elif name == "ANN (MLP)" and ann_seeds is not None:
        estimator = SeedAveragedMLP(parameters=parameters, seeds=tuple(ann_seeds))
    else:
        estimator = build_model(name, parameters, seed).named_steps["model"]
        if name == "SVM (RBF)" and probability:
            estimator.set_params(probability=True, random_state=seed)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fit(pipeline, X, y):
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        pipeline.fit(X, y)
    return time.perf_counter() - started, [f"{w.category.__name__}: {w.message}" for w in captured]


def _aligned_proba(model, X, labels) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None
    try:
        probabilities = model.predict_proba(X)
    except AttributeError:  # e.g. SVC without probability=True
        return None
    order = [list(model.classes_).index(label) for label in labels]
    return probabilities[:, order]


def _fold_job(name, config, seed, X_fit, y_fit, X_hold, labels, ann_seeds=None, probability=False):
    """Fit one candidate on one fold in a worker limited to one BLAS thread."""
    with threadpool_limits(limits=1):
        model = build_candidate(name, config, seed, ann_seeds=ann_seeds, probability=probability)
        seconds, messages = _fit(model, X_fit, y_fit)
        predicted = model.predict(X_hold)
        proba = _aligned_proba(model, X_hold, labels)
    return seconds, messages, np.asarray(predicted), proba


def _normalized(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-9, 1.0)
    return clipped / clipped.sum(axis=1, keepdims=True)


def fit_stacker(oof: dict[str, np.ndarray], y_true: np.ndarray, folds: list,
                labels: list[str], grid=(0.01, 0.03, 0.1, 0.3, 1.0), seed: int = 42) -> dict:
    """Choose the meta-learner's C by grouped CV over the OOF predictions, then fit it."""
    names = sorted(oof)
    Z = _stack_features([oof[n] for n in names])
    label_array = np.asarray(labels)
    rows = []
    for c in grid:
        scores = []
        for fit, hold in folds:
            meta = LogisticRegression(C=c, max_iter=3000).fit(Z[fit], y_true[fit])
            scores.append(_macro_f1(y_true[hold], meta.predict(Z[hold]), labels))
        rows.append({"C": c, "cv_mean_f1": float(np.mean(scores)),
                     "cv_std_f1": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                     "cv_fold_f1": scores})
    best = max(rows, key=lambda r: (r["cv_mean_f1"], -r["C"]))
    meta = LogisticRegression(C=best["C"], max_iter=3000).fit(Z, y_true)
    if list(meta.classes_) != list(label_array):
        raise ValueError("Meta-learner classes must follow the label order.")
    return {"members": names, "C": best["C"], "grid": rows, "meta": meta,
            "cv_mean_f1": best["cv_mean_f1"], "cv_std_f1": best["cv_std_f1"], "cv_fold_f1": best["cv_fold_f1"]}


def _top3(y_true, probabilities, labels) -> float | None:
    if len(labels) <= 3:
        return None
    return float(top_k_accuracy_score(y_true, probabilities, k=3, labels=labels))


def _macro_f1(y_true, predicted, labels) -> float:
    return float(f1_score(y_true, predicted, labels=labels, average="macro", zero_division=0))


def greedy_ensemble(oof: dict[str, np.ndarray], y_true: np.ndarray, labels: list[str],
                    folds: list, rounds: int = ENSEMBLE_ROUNDS) -> dict:
    """Caruana forward selection with replacement on OOF probabilities.

    The objective is mean per-fold macro F1 (the same tuning metric), with OOF
    log-loss as a tie-breaker. Uses training-partition OOF predictions only.
    """
    names = sorted(oof)
    label_array = np.asarray(labels)

    def score(probabilities):
        f1 = np.mean([
            _macro_f1(y_true[hold], label_array[probabilities[hold].argmax(1)], labels)
            for _, hold in folds
        ])
        return f1, -log_loss(y_true, _normalized(probabilities), labels=labels)

    counts = {name: 0 for name in names}
    total = np.zeros_like(next(iter(oof.values())))
    history = []
    for step in range(rounds):
        best = None
        for name in names:
            trial = (total * step + oof[name]) / (step + 1)
            key = score(trial)
            if best is None or key > best[0]:
                best = (key, name, trial)
        (f1, neg_loss), name, total = best
        counts[name] += 1
        history.append({"round": step + 1, "added": name, "cv_mean_f1": float(f1), "log_loss": float(-neg_loss)})
    # Keep the best prefix of the selection path (earliest on ties).
    best_round = max(history, key=lambda row: (round(row["cv_mean_f1"], 10), -row["round"]))["round"]
    counts = {name: 0 for name in names}
    for row in history[:best_round]:
        counts[row["added"]] += 1
    weights = {name: count / best_round for name, count in counts.items() if count}
    blended = sum(oof[name] * weight for name, weight in weights.items())
    fold_scores = [
        _macro_f1(y_true[hold], label_array[blended[hold].argmax(1)], labels) for _, hold in folds
    ]
    return {
        "weights": weights, "rounds_used": best_round, "history": history,
        "cv_fold_f1": fold_scores, "cv_mean_f1": float(np.mean(fold_scores)),
        "cv_std_f1": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0,
    }


def bootstrap_interval(y_true, predicted, labels, repeats=2000, seed=42) -> dict:
    """Percentile bootstrap over test tracks for accuracy and macro F1."""
    rng = np.random.default_rng(seed)
    y_true, predicted = np.asarray(y_true), np.asarray(predicted)
    accuracy, f1 = [], []
    for _ in range(repeats):
        sample = rng.integers(0, len(y_true), len(y_true))
        accuracy.append(float(np.mean(y_true[sample] == predicted[sample])))
        f1.append(_macro_f1(y_true[sample], predicted[sample], labels))
    return {
        "accuracy_95ci": [float(np.percentile(accuracy, 2.5)), float(np.percentile(accuracy, 97.5))],
        "f1_macro_95ci": [float(np.percentile(f1, 2.5)), float(np.percentile(f1, 97.5))],
        "repeats": repeats,
    }


def mcnemar(y_true, first, second) -> dict:
    """Exact two-sided McNemar test on paired correctness."""
    from scipy.stats import binomtest
    a = np.asarray(first) == np.asarray(y_true)
    b = np.asarray(second) == np.asarray(y_true)
    only_first, only_second = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = only_first + only_second
    p = float(binomtest(only_first, n, 0.5).pvalue) if n else 1.0
    return {"only_new_correct": only_first, "only_reference_correct": only_second, "p_value": p}


def preserve_baseline(output: Path) -> None:
    """Archive the original run once, and the previous improved run as history."""
    root = output.parent
    original = root / "baseline"
    info = output / "run_info.json"
    if not info.exists():
        return
    profile = json.loads(info.read_text())["profile"]
    if not original.exists():
        if profile in ("improved", PROFILE):
            raise ValueError("Cannot archive an improved run as the original baseline.")
        shutil.copytree(output, original / "reports")
        if (root / "models").exists():
            shutil.copytree(root / "models", original / "models")
    elif profile == "improved" and not (root / "history/improved_v1").exists():
        shutil.copytree(output, root / "history/improved_v1/reports")


def _row_for_csv(row: dict) -> dict:
    return {
        **{k: v for k, v in row.items() if k not in ("oof", "parameters", "warnings", "cv_fold_f1")},
        "parameters": json.dumps(row["parameters"]), "warnings": json.dumps(row["warnings"]),
        "cv_fold_f1": json.dumps(row["cv_fold_f1"]),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_improvement(
    dataset: Dataset, output: Path, seed: int = 42, threads: int = 2,
    cv_folds: int = 3, spaces: dict | None = None, ann_seed_count: int = 5,
) -> pd.DataFrame:
    if threads < 1 or cv_folds < 2:
        raise ValueError("Threads must be positive and CV requires at least two folds.")
    started = time.perf_counter()
    output = Path(output)
    preserve_baseline(output)
    output.mkdir(parents=True, exist_ok=True)
    for stale in ("training_outlier_diagnostics.csv", "training_outlier_summary.json", "verification.json"):
        (output / stale).unlink(missing_ok=True)
    model_dir = output.parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    spaces = search_space() if spaces is None else spaces
    if not spaces or any(not configs for configs in spaces.values()) or not (set(spaces) - {BASELINE}):
        raise ValueError("At least one non-baseline model and nonempty grids are required.")
    training, validation = dataset.ids("training"), dataset.ids("validation")
    X, y = dataset.X.loc[training], dataset.y.loc[training]
    X_val, y_val = dataset.X.loc[validation], dataset.y.loc[validation]
    labels = sorted(y.unique().tolist())
    y_array = y.to_numpy()
    folds, audits = make_group_folds(dataset, cv_folds, seed)
    seeds = tuple(seed + offset for offset in range(ann_seed_count))
    write_json(output / "cv_splits.json", audits)
    write_json(output / "search_plan.json", {
        "profile": PROFILE, "seed": seed, "cv_folds": cv_folds, "ann_seeds": list(seeds),
        "spaces": spaces, "ensemble_rounds": ENSEMBLE_ROUNDS,
        "selection_rule": (
            "Config: mean grouped-CV macro F1 (lower SD breaks ties). Ensemble: greedy forward "
            "selection on training OOF probabilities. Champion: validation macro F1 (accuracy breaks ties)."
        ),
        "dataset_processed_sha256": dataset.manifest.get("processed_sha256", {}),
        "evaluation_status": "extended benchmark; original test previously viewed",
    })
    parallel = Parallel(n_jobs=threads, prefer="processes")
    candidates, selected, fitted = [], {}, {}
    oof_by_family = {}
    total = sum(map(len, spaces.values()))

    # 1. Grouped-CV search, parallel over (candidate, fold).
    for name, configs in spaces.items():
        jobs = [
            (cid, fold_number, delayed(_fold_job)(
                name, config, seed, X.iloc[fit], y.iloc[fit], X.iloc[hold], labels,
            ))
            for cid, config in enumerate(configs) for fold_number, (fit, hold) in enumerate(folds)
        ]
        outputs = parallel(job for *_, job in jobs)
        family = []
        for cid, config in enumerate(configs):
            parts = [out for (c, _, _), out in zip(jobs, outputs) if c == cid]
            scores, oof = [], np.zeros((len(y), len(labels)))
            has_proba = all(part[3] is not None for part in parts)
            for (fit, hold), (_, _, predicted, proba) in zip(folds, parts):
                scores.append(_macro_f1(y_array[hold], predicted, labels))
                if has_proba:
                    oof[hold] = proba
            row = {
                "model": name, "candidate_id": cid, "parameters": config,
                "cv_mean_f1": float(np.mean(scores)),
                "cv_std_f1": float(np.std(scores, ddof=1)),
                "cv_fold_f1": scores, "fit_seconds": float(sum(part[0] for part in parts)),
                "warnings": sorted({m for part in parts for m in part[1]}),
                "oof": oof if has_proba else None,
            }
            candidates.append(row)
            family.append(row)
            print(f"CV {len(candidates)}/{total} {name} mean F1={row['cv_mean_f1']:.4f} ± {row['cv_std_f1']:.4f}: {config}", flush=True)
        selected[name] = max(family, key=lambda row: (row["cv_mean_f1"], -row["cv_std_f1"]))
        pd.DataFrame([_row_for_csv(r) for r in candidates]).to_csv(output / "cv_results.csv", index=False)

    # 2. Probabilistic OOF predictions for the selected configuration of each family.
    seed_rows = []
    for name in spaces:
        if name == BASELINE:
            continue
        config = selected[name]["parameters"]
        if name == "ANN (MLP)":
            jobs = [
                delayed(_fold_job)(name, config, seed, X.iloc[fit], y.iloc[fit], X.iloc[hold], labels, ann_seeds=(s,))
                for s in seeds for fit, hold in folds
            ]
            outputs = parallel(jobs)
            per_seed = {s: outputs[i * len(folds):(i + 1) * len(folds)] for i, s in enumerate(seeds)}
            oof = np.zeros((len(y), len(labels)))
            for s, parts in per_seed.items():
                seed_oof = np.zeros_like(oof)
                for (fit, hold), part in zip(folds, parts):
                    seed_oof[hold] = part[3]
                oof += seed_oof / len(seeds)
                fold_scores = [_macro_f1(y_array[h], np.asarray(labels)[seed_oof[h].argmax(1)], labels) for _, h in folds]
                seed_rows.append({"seed": s, "cv_mean_f1": float(np.mean(fold_scores)),
                                  "cv_std_f1": float(np.std(fold_scores, ddof=1))})
            fold_scores = [_macro_f1(y_array[h], np.asarray(labels)[oof[h].argmax(1)], labels) for _, h in folds]
            seed_rows.append({"seed": "ensemble", "cv_mean_f1": float(np.mean(fold_scores)),
                              "cv_std_f1": float(np.std(fold_scores, ddof=1))})
            selected[name] = {**selected[name], "tuning_cv_mean_f1": selected[name]["cv_mean_f1"],
                              "cv_mean_f1": float(np.mean(fold_scores)),
                              "cv_std_f1": float(np.std(fold_scores, ddof=1)), "cv_fold_f1": fold_scores}
            oof_by_family[name] = oof
        elif selected[name]["oof"] is not None:
            oof_by_family[name] = selected[name]["oof"]
        elif name == "SVM (RBF)":
            outputs = parallel(
                delayed(_fold_job)(name, config, seed, X.iloc[fit], y.iloc[fit], X.iloc[hold], labels, probability=True)
                for fit, hold in folds
            )
            oof = np.zeros((len(y), len(labels)))
            for (fit, hold), part in zip(folds, outputs):
                oof[hold] = part[3]
            oof_by_family[name] = oof
    for row in candidates:
        row.pop("oof", None)
    for row in selected.values():
        row.pop("oof", None)

    # 3. Ensemble weights from training OOF only (exclude weak/non-probabilistic members).
    ensemble_info = None
    if len(oof_by_family) >= 2:
        ensemble_info = greedy_ensemble(oof_by_family, y_array, labels, folds)
        ensemble_info["member_cv_mean_f1"] = {name: selected[name]["cv_mean_f1"] for name in oof_by_family}
        print(f"Ensemble weights: {ensemble_info['weights']} (OOF mean F1={ensemble_info['cv_mean_f1']:.4f})", flush=True)
        selected[ENSEMBLE] = {
            "model": ENSEMBLE, "parameters": {"weights": ensemble_info["weights"], "method": "greedy forward selection on OOF probabilities"},
            "cv_mean_f1": ensemble_info["cv_mean_f1"], "cv_std_f1": ensemble_info["cv_std_f1"],
            "cv_fold_f1": ensemble_info["cv_fold_f1"], "warnings": [],
        }
        stack_info = fit_stacker(oof_by_family, y_array, folds, labels)
        print(f"Stacked ensemble: C={stack_info['C']} (meta-model OOF tuning mean F1={stack_info['cv_mean_f1']:.4f})", flush=True)
        selected[STACKED] = {
            "model": STACKED, "parameters": {"members": stack_info["members"], "meta_C": stack_info["C"],
                                             "method": "logistic regression on member OOF log-probabilities"},
            "cv_mean_f1": stack_info["cv_mean_f1"], "cv_std_f1": stack_info["cv_std_f1"],
            "cv_fold_f1": stack_info["cv_fold_f1"], "warnings": [],
        }
        ensemble_info["stacking"] = {k: v for k, v in stack_info.items() if k != "meta"}
        write_json(output / "ensemble.json", ensemble_info)

    # 4. Fit selected configurations on training and score on official validation.
    def family_model(name):
        if name == "ANN (MLP)":
            return build_candidate(name, selected[name]["parameters"], seed, ann_seeds=seeds)
        return build_candidate(name, selected[name]["parameters"], seed, probability=(name == "SVM (RBF)"))

    def fit_job(name, X_fit, y_fit):
        with threadpool_limits(limits=1):
            model = family_model(name)
            seconds, messages = _fit(model, X_fit, y_fit)
        return name, model, seconds, messages

    families = [name for name in spaces]
    for name, model, _, _ in parallel(delayed(fit_job)(name, X, y) for name in families):
        fitted[name] = model
    if ensemble_info:
        members = [(name, fitted[name]) for name in ensemble_info["weights"]]
        fitted[ENSEMBLE] = WeightedSoftVoter.from_fitted(members, list(ensemble_info["weights"].values()))
        fitted[STACKED] = StackedEnsemble.from_fitted(
            [(name, fitted[name]) for name in stack_info["members"]], stack_info["meta"])
    val_probabilities = {}
    with threadpool_limits(limits=threads):
        for name, model in fitted.items():
            prediction = model.predict(X_val)
            selected[name].update({f"validation_{k}": v for k, v in metrics(y_val, prediction, labels).items()})
            proba = _aligned_proba(model, X_val, labels)
            if proba is not None:
                val_probabilities[name] = proba
                selected[name]["validation_top3_accuracy"] = _top3(y_val, proba, labels)
            print(f"Validation {name}: macro F1={selected[name]['validation_f1_macro']:.4f}", flush=True)
    if seed_rows:
        ann = fitted["ANN (MLP)"]
        transformed = ann[:-1].transform(X_val)
        for row, estimator in zip(seed_rows, ann.named_steps["model"].estimators_):
            pred = ann.named_steps["model"].encoder_.inverse_transform(estimator.predict(transformed))
            scores = metrics(y_val, pred, labels)
            row.update(validation_f1_macro=scores["f1_macro"], validation_accuracy=scores["accuracy"])
        seed_rows[-1].update(validation_f1_macro=selected["ANN (MLP)"]["validation_f1_macro"],
                             validation_accuracy=selected["ANN (MLP)"]["validation_accuracy"])
        pd.DataFrame(seed_rows).to_csv(output / "ann_seed_results.csv", index=False)

    # 5. Lock the selection before requesting test data.
    contenders = [name for name in fitted if name != BASELINE]
    champion = max(contenders, key=lambda n: (selected[n]["validation_f1_macro"], selected[n]["validation_accuracy"]))
    write_json(output / "selection.json", {
        "criterion": "Config by grouped-CV mean macro F1 (lower SD tie); ensemble weights from training OOF; champion by validation macro F1 (accuracy tie)",
        "champion": champion, "profile": PROFILE, "seed": seed, "threads": threads,
        "cv_folds": cv_folds, "ann_seeds": list(seeds), "selected": selected,
        "test_used_for_selection": False, "test_previously_viewed_in_baseline": True,
        "training_track_ids": training.tolist(), "validation_track_ids": validation.tolist(),
    })
    write_json(output / "candidate_results.json", candidates)
    pd.DataFrame([_row_for_csv(r) for r in candidates]).to_csv(output / "candidate_results.csv", index=False)
    joblib.dump(fitted[champion], model_dir / "selected_training_model.joblib", compress=3)
    print(f"Selection locked: {champion}. Refitting on training + validation.", flush=True)

    # 6. Refit on training + validation and evaluate once on test.
    test_ids = dataset.ids("test")
    development = training.append(validation)
    X_dev, y_dev = dataset.X.loc[development], dataset.y.loc[development]
    final, refit_info = {}, {}
    for name, model, seconds, messages in parallel(delayed(fit_job)(name, X_dev, y_dev) for name in families):
        final[name] = model
        refit_info[name] = (seconds, messages)
    if ensemble_info:
        members = [(name, final[name]) for name in ensemble_info["weights"]]
        final[ENSEMBLE] = WeightedSoftVoter.from_fitted(members, list(ensemble_info["weights"].values()))
        refit_info[ENSEMBLE] = (sum(refit_info[n][0] for n in ensemble_info["weights"]), [])
        final[STACKED] = StackedEnsemble.from_fitted(
            [(name, final[name]) for name in stack_info["members"]], stack_info["meta"])
        refit_info[STACKED] = (sum(refit_info[n][0] for n in stack_info["members"]), [])
    X_test, y_test = dataset.X.loc[test_ids], dataset.y.loc[test_ids]
    predictions = dataset.metadata.loc[test_ids, ["genre", "artist_id", "artist", "title"]].copy()
    predictions = predictions.rename(columns={"genre": "actual"})
    results, classification, confusion, intervals = [], {}, {}, {}
    with threadpool_limits(limits=threads):
        for name, model in final.items():
            before = time.perf_counter()
            prediction = model.predict(X_test)
            predict_seconds = time.perf_counter() - before
            proba = _aligned_proba(model, X_test, labels)
            training_scores = metrics(y, fitted[name].predict(X), labels)
            seconds, messages = refit_info[name]
            row = {
                "model": name, "selected_by_validation": name == champion,
                "parameters": json.dumps(selected[name]["parameters"]),
                "cv_mean_f1": selected[name]["cv_mean_f1"], "cv_std_f1": selected[name]["cv_std_f1"],
                "train_accuracy": training_scores["accuracy"], "train_f1_macro": training_scores["f1_macro"],
                **{k: v for k, v in selected[name].items() if k.startswith("validation_")},
                **{f"test_{k}": v for k, v in metrics(y_test, prediction, labels).items()},
                "test_top3_accuracy": _top3(y_test, proba, labels) if proba is not None else None,
                "test_log_loss": float(log_loss(y_test, _normalized(proba), labels=labels)) if proba is not None else None,
                "tuning_fit_seconds": sum(r["fit_seconds"] for r in candidates if r["model"] == name),
                "refit_seconds": seconds, "test_predict_seconds": predict_seconds,
                "refit_warnings": json.dumps(sorted(set(messages))), "ann_iterations": None,
            }
            if name == "ANN (MLP)":
                row["ann_iterations"] = json.dumps([int(m.n_iter_) for m in model.named_steps["model"].estimators_])
            results.append(row)
            classification[name] = classification_report(y_test, prediction, labels=labels, output_dict=True, zero_division=0)
            confusion[name] = confusion_matrix(y_test, prediction, labels=labels).tolist()
            predictions[IMPROVED_SLUGS[name]] = prediction
            if name == champion:
                if proba is not None:
                    predictions["champion_confidence"] = proba.max(axis=1)
                    for i, label in enumerate(labels):
                        predictions[f"p_{label}"] = proba[:, i]
                intervals[name] = bootstrap_interval(y_test, prediction, labels, seed=seed)
                joblib.dump(model, model_dir / "best_model.joblib", compress=3)
            print(f"Test {name}: accuracy={row['test_accuracy']:.4f}, macro F1={row['test_f1_macro']:.4f}", flush=True)

    result = pd.DataFrame(results).sort_values("validation_f1_macro", ascending=False, kind="stable")
    result.to_csv(output / "model_comparison.csv", index=False)
    predictions.to_csv(output / "test_predictions.csv")
    write_json(output / "classification_reports.json", classification)
    write_json(output / "confusion_matrices.json", {"labels": labels, "matrices": confusion})
    write_json(output / "run_info.json", {
        "profile": PROFILE, "seed": seed, "threads": threads, "champion": champion,
        "cv_folds": cv_folds, "ann_seeds": list(seeds), "candidate_count": total,
        "elapsed_seconds": time.perf_counter() - started,
        "python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "versions": {p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl")},
        "dataset_processed_sha256": dataset.manifest.get("processed_sha256", {}),
        "n_train": len(training), "n_validation": len(validation), "n_test": len(test_ids),
        "n_features": dataset.X.shape[1], "evaluation_status": "extended benchmark; test previously viewed",
        "ann_early_stopping": f"10% track holdout in each fit; probability average of {len(seeds)} predeclared seeds",
    })
    summary = {"champion": champion, "evaluation_status": "extended benchmark; original test previously viewed",
               "champion_test_bootstrap": intervals.get(champion)}
    current = result.loc[result["model"].eq(champion)].iloc[0]
    references = []
    for label, folder in (("original", output.parent / "baseline/reports"),
                          ("improved_v1", output.parent / "history/improved_v1/reports")):
        path = folder / "model_comparison.csv"
        if not path.exists():
            continue
        old = pd.read_csv(path)
        previous = old.loc[old["selected_by_validation"].astype(str).str.lower().eq("true")].iloc[0]
        entry = {
            "run": label, "champion": previous["model"],
            "test_accuracy": float(previous["test_accuracy"]), "test_f1_macro": float(previous["test_f1_macro"]),
            "accuracy_change_percentage_points": float(100 * (current["test_accuracy"] - previous["test_accuracy"])),
            "f1_macro_change": float(current["test_f1_macro"] - previous["test_f1_macro"]),
        }
        old_predictions = folder / "test_predictions.csv"
        slug = {**IMPROVED_SLUGS}.get(previous["model"])
        if old_predictions.exists() and slug:
            old_frame = pd.read_csv(old_predictions, index_col="track_id")
            if slug in old_frame and set(old_frame.index) == set(test_ids):
                entry["mcnemar_vs_champion"] = mcnemar(
                    y_test, predictions[IMPROVED_SLUGS[champion]], old_frame.loc[test_ids, slug],
                )
        references.append(entry)
    summary["comparisons"] = references
    write_json(output / "improvement_summary.json", summary)
    return result
