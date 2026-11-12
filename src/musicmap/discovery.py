"""Train-only projections, exploratory clustering, and validation-set attribution."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.metrics import (
    adjusted_rand_score, f1_score, normalized_mutual_info_score, silhouette_score,
)
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from .data import Dataset, FEATURE_FAMILIES, write_json
from .experiment import BASELINE, preprocessing


def grouped_permutation(
    model, X: pd.DataFrame, y: pd.Series, repeats: int = 5, seed: int = 42
) -> pd.DataFrame:
    """Jointly shuffle columns in each family, retaining within-family correlations."""
    if repeats < 2:
        raise ValueError("Use at least two permutation repeats.")
    baseline = f1_score(y, model.predict(X), average="macro", zero_division=0)
    rng = np.random.default_rng(seed)
    permutations = [rng.permutation(len(X)) for _ in range(repeats)]
    families = sorted({name.split("__", 1)[0] for name in X.columns})
    rows = []
    for family in families:
        columns = [name for name in X.columns if name.startswith(family + "__")]
        drops = []
        for order in permutations:
            shuffled = X.copy()
            shuffled.loc[:, columns] = X[columns].to_numpy()[order]
            score = f1_score(y, model.predict(shuffled), average="macro", zero_division=0)
            drops.append(float(baseline - score))
        rows.append({
            "family": family, "n_features": len(columns),
            "mean_f1_drop": float(np.mean(drops)),
            "std_f1_drop": float(np.std(drops, ddof=1)),
            "repeats": repeats, "baseline_validation_f1": float(baseline),
            "interpretation": FEATURE_FAMILIES[family],
        })
        print(f"  Permuted {family}: macro F1 drop {np.mean(drops):.4f}", flush=True)
    return pd.DataFrame(rows).sort_values("mean_f1_drop", ascending=False)


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def comparison_plots(dataset: Dataset, output: Path) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
    comparison = pd.read_csv(output / "model_comparison.csv")
    labels = dataset.manifest["genres"]
    counts = pd.crosstab(dataset.y, dataset.metadata["split"])[["training", "validation", "test"]]
    fig, ax = plt.subplots(figsize=(10, 4))
    counts.plot.bar(stacked=True, ax=ax, color=["#255f85", "#50a995", "#e1a84b"], width=0.7)
    ax.set(xlabel="", ylabel="Tracks", title="FMA-small: genre balance and official splits")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Official split", bbox_to_anchor=(1.02, 1), loc="upper left")
    _save(fig, figures / "genre_balance.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    order = comparison["model"].tolist()
    y = np.arange(len(order))
    axes[0].barh(y - 0.18, comparison["validation_f1_macro"], height=0.34, color="#255f85", label="Validation (select)")
    axes[0].barh(y + 0.18, comparison["test_f1_macro"], height=0.34, color="#50a995", label="Test (evaluate)")
    axes[0].set_yticks(y, order)
    axes[0].invert_yaxis()
    axes[0].set(xlim=(0, 1), xlabel="Macro F1", title="Selection and held-out performance")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[1].barh(y, comparison["test_accuracy"], color="#255f85")
    axes[1].set_yticks(y, order)
    axes[1].invert_yaxis()
    axes[1].set(xlim=(0, 1), xlabel="Accuracy", title="Accuracy on the official test set")
    for i, value in enumerate(comparison["test_accuracy"]):
        axes[1].text(value + 0.015, i, f"{value:.1%}", va="center", fontsize=9)
    fig.tight_layout(w_pad=3)
    _save(fig, figures / "model_comparison.png")

    if "cv_mean_f1" in comparison:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.barh(
            comparison["model"], comparison["cv_mean_f1"],
            xerr=comparison["cv_std_f1"], color="#255f85", capsize=4,
        )
        ax.invert_yaxis()
        ax.set(
            xlim=(0, 1), xlabel="Macro F1 (mean ± fold standard deviation)",
            title="Artist-grouped cross-validation · selected configurations",
        )
        _save(fig, figures / "grouped_cv.png")
        original_path = output.parent / "baseline/reports/model_comparison.csv"
        if original_path.exists():
            old = pd.read_csv(original_path).set_index("model")
            fig, ax = plt.subplots(figsize=(11, 5))
            positions = np.arange(len(comparison))
            original = [old.at[name, "test_accuracy"] if name in old.index else np.nan
                        for name in comparison["model"]]
            ax.barh(positions - .18, original, height=.34, color="#b9c9d3", label="Original experiment")
            ax.barh(positions + .18, comparison["test_accuracy"], height=.34,
                    color="#255f85", label="Improved experiment")
            ax.set_yticks(positions, comparison["model"])
            ax.invert_yaxis()
            ax.set(xlim=(0, 1), xlabel="Test accuracy",
                   title="Extended benchmark comparison on the same official test set")
            ax.legend(loc="lower right")
            _save(fig, figures / "improvement_comparison.png")

    data = json.loads((output / "confusion_matrices.json").read_text())
    names = [name for name in data["matrices"] if name != BASELINE]
    rows = (len(names) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(15, 6.5 * rows), squeeze=False)
    for name, ax in zip(names, axes.flat):
        counts_array = np.array(data["matrices"][name])
        normalized = counts_array / counts_array.sum(axis=1, keepdims=True)
        sns.heatmap(
            normalized, annot=True, fmt=".0%", cmap="Blues", vmin=0, vmax=1,
            xticklabels=labels, yticklabels=labels, ax=ax, cbar=False,
            annot_kws={"size": 8},
        )
        ax.set(title=name, xlabel="Predicted genre", ylabel="Actual genre")
        ax.tick_params(axis="x", rotation=40)
        ax.tick_params(axis="y", rotation=0)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    fig.suptitle("Test confusion matrices · each row sums to 100%", fontsize=17, y=1.01)
    fig.tight_layout(w_pad=3, h_pad=3)
    _save(fig, figures / "confusion_matrices.png")

    selection = json.loads((output / "selection.json").read_text())
    champion = selection["champion"]
    report = json.loads((output / "classification_reports.json").read_text())[champion]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(labels, [report[label]["f1-score"] for label in labels], color="#50a995")
    ax.invert_yaxis()
    ax.set(xlim=(0, 1), xlabel="F1 score", title=f"{champion}: test F1 for each genre")
    _save(fig, figures / "per_genre_f1.png")


def run_discovery(
    dataset: Dataset, output: Path, seed: int = 42, threads: int = 2,
    repeats: int = 5,
) -> dict:
    output = Path(output)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    model_dir = output.parent / "models"
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
    train_ids, val_ids = dataset.ids("training"), dataset.ids("validation")
    X_train, X_val = dataset.X.loc[train_ids], dataset.X.loc[val_ids]
    labels = dataset.manifest["genres"]
    colors = dict(zip(labels, sns.color_palette("colorblind", len(labels)).as_hex()))
    with threadpool_limits(limits=threads):
        print("Fitting train-only PCA and K-means...", flush=True)
        clean = preprocessing()
        scaled_train = clean.fit_transform(X_train)
        scaled_val = clean.transform(X_val)
        n_components = min(50, scaled_train.shape[1], len(train_ids) - 1)
        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
        train_embedding = pca.fit_transform(scaled_train)
        val_embedding = pca.transform(scaled_val)
        transform = Pipeline([("preprocess", clean), ("pca", pca)])
        joblib.dump(transform, model_dir / "discovery_transform.joblib", compress=3)
        np.savez_compressed(
            model_dir / "training_embeddings.npz",
            values=train_embedding, track_ids=train_ids.to_numpy(dtype=np.int64),
        )
        kmeans = KMeans(n_clusters=len(labels), n_init=10, random_state=seed)
        kmeans.fit(train_embedding)
        val_clusters = kmeans.predict(val_embedding)
        cluster_count = len(np.unique(val_clusters))
        silhouette = (
            float(silhouette_score(val_embedding, val_clusters, sample_size=min(800, len(val_ids)), random_state=seed))
            if 1 < cluster_count < len(val_ids) else None
        )
        cluster_table = pd.crosstab(
            pd.Series(val_clusters, index=val_ids, name="cluster"),
            dataset.y.loc[val_ids], dropna=False,
        ).reindex(index=range(len(labels)), columns=labels, fill_value=0)
        cluster_table.to_csv(output / "cluster_genre_counts.csv")
        cluster_rows = dataset.metadata.loc[val_ids, ["genre", "title", "artist"]].copy()
        cluster_rows["cluster"] = val_clusters
        cluster_rows.to_csv(output / "validation_clusters.csv")
        summary = {
            "fit_split": "training",
            "evaluation_split": "validation",
            "test_used_for_discovery": False,
            "pca_components": n_components,
            "pca_2d_explained_variance": float(pca.explained_variance_ratio_[:2].sum()),
            "pca_50d_explained_variance": float(pca.explained_variance_ratio_.sum()),
            "kmeans_k": len(labels),
            "kmeans_adjusted_rand_index": float(adjusted_rand_score(dataset.y.loc[val_ids], val_clusters)),
            "kmeans_normalized_mutual_information": float(normalized_mutual_info_score(dataset.y.loc[val_ids], val_clusters)),
            "kmeans_silhouette": silhouette,
            "permutation_repeats": repeats, "seed": seed,
        }

        map_ids = train_ids.append(val_ids)
        embedding = np.concatenate([train_embedding, val_embedding])
        map_table = dataset.metadata.loc[map_ids].copy()
        map_table["pc1"], map_table["pc2"] = embedding[:, 0], embedding[:, 1]
        map_table.to_csv(output / "song_map.csv")
        interactive = px.scatter(
            map_table.reset_index(), x="pc1", y="pc2", color="genre",
            symbol="split", color_discrete_map=colors,
            hover_name="title", hover_data=["track_id", "artist", "genre", "split"],
            opacity=0.6, template="plotly_white",
            title="MusicMap · train-only PCA, training and validation tracks",
            labels={"pc1": f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
                    "pc2": f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)"},
        )
        interactive.update_traces(marker={"size": 5})
        interactive.update_layout(
            height=730, font={"family": "Arial", "size": 13},
            legend_title_text="Genre / split",
            annotations=[{
                "x": 0, "y": -0.15, "xref": "paper", "yref": "paper",
                "showarrow": False, "align": "left",
                "text": "Drag to zoom; double-click to reset; click legend entries to filter.<br>"
                        "2D projections omit information. Genre labels only color the map.",
            }],
            margin={"b": 120},
        )
        interactive.write_html(output / "song_map.html", include_plotlyjs=True, full_html=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        for label in labels:
            rows = map_table["genre"].eq(label)
            ax.scatter(map_table.loc[rows, "pc1"], map_table.loc[rows, "pc2"],
                       s=9, alpha=0.4, color=colors[label], label=label, rasterized=True)
        ax.set(
            xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
            ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
            title=f"Audio similarity map · first two PCs explain {summary['pca_2d_explained_variance']:.1%}",
        )
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", markerscale=2)
        _save(fig, figures / "pca_map.png")

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(np.arange(1, n_components + 1), np.cumsum(pca.explained_variance_ratio_), color="#255f85", linewidth=2)
        ax.set(xlabel="Number of principal components", ylabel="Cumulative explained variance",
               ylim=(0, 1), title="How much standardized feature variation does PCA retain?")
        _save(fig, figures / "pca_variance.png")

        proportions = cluster_table.div(cluster_table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.heatmap(proportions, annot=True, fmt=".0%", cmap="YlGnBu",
                    vmin=0, vmax=1, ax=ax, annot_kws={"size": 9})
        ax.set(title="K-means cluster composition on validation tracks", xlabel="Known genre", ylabel="Cluster ID")
        ax.tick_params(axis="x", rotation=25)
        _save(fig, figures / "cluster_composition.png")

        f_values, p_values = f_classif(scaled_train, dataset.y.loc[train_ids])
        anova = pd.DataFrame({
            "feature": clean.get_feature_names_out(),
            "training_anova_f": f_values, "training_anova_p": p_values,
        }).sort_values("training_anova_f", ascending=False)
        anova.to_csv(output / "training_feature_associations.csv", index=False)

        print("Measuring feature-family importance on validation tracks...", flush=True)
        training_model = joblib.load(model_dir / "selected_training_model.joblib")
        selected_path = output / "selected_features.csv"
        selected_path.unlink(missing_ok=True)
        reducer = training_model.named_steps.get("reduction")
        selector = getattr(reducer, "transformer_", None)
        if selector is not None and hasattr(selector, "get_support"):
            imputed_names = training_model.named_steps["imputer"].get_feature_names_out()
            varying_names = imputed_names[training_model.named_steps["variance"].get_support()]
            support = selector.get_support()
            selected_names = varying_names[support]
            pd.DataFrame({
                "feature": selected_names,
                "family": [name.split("__", 1)[0] for name in selected_names],
                "selector_training_f": selector.scores_[support],
            }).sort_values("selector_training_f", ascending=False).to_csv(selected_path, index=False)
        importance = grouped_permutation(
            training_model, X_val, dataset.y.loc[val_ids], repeats=repeats, seed=seed
        )
        importance.to_csv(output / "feature_family_importance.csv", index=False)
        fig, ax = plt.subplots(figsize=(11, 6))
        ordered = importance.iloc[::-1]
        ax.barh(ordered["family"], ordered["mean_f1_drop"], xerr=ordered["std_f1_drop"],
                color="#255f85", capsize=3)
        ax.axvline(0, color="#777777", linewidth=0.8)
        ax.set(xlabel="Decrease in validation macro F1 (mean ± repeat standard deviation)",
               title="Which feature families does the selected model rely on?")
        _save(fig, figures / "feature_importance.png")

        display_features = {
            "mfcc__mean__01": "MFCC 1 mean",
            "spectral_centroid__mean__01": "Spectral centroid mean (Hz)",
            "rmse__mean__01": "RMS energy mean",
            "zcr__mean__01": "Zero-crossing rate mean",
        }
        distributions = X_train[list(display_features)].join(dataset.y.loc[train_ids])
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        for (feature, title), ax in zip(display_features.items(), axes.flat):
            sns.boxplot(
                data=distributions, x="genre", y=feature, hue="genre",
                order=labels, hue_order=labels, palette=colors,
                legend=False, showfliers=False, ax=ax,
            )
            ax.set(xlabel="", ylabel=title, title=title)
            ax.tick_params(axis="x", rotation=30)
        fig.suptitle("Training-set feature distributions · outliers omitted from drawing only", y=1.01)
        fig.tight_layout(w_pad=3, h_pad=3)
        _save(fig, figures / "feature_distributions.png")
        write_json(output / "discovery_summary.json", summary)
    comparison_plots(dataset, output)
    return summary


def nearest_tracks(
    dataset: Dataset, model_dir: Path, track_id: int,
    k: int = 5, exclude_same_artist: bool = True,
) -> pd.DataFrame:
    if track_id not in dataset.X.index:
        raise ValueError(f"Track {track_id} is not in the prepared FMA-small subset.")
    if k < 1:
        raise ValueError("k must be positive.")
    transform = joblib.load(Path(model_dir) / "discovery_transform.joblib")
    with np.load(Path(model_dir) / "training_embeddings.npz", allow_pickle=False) as arrays:
        candidates, ids = arrays["values"], arrays["track_ids"]
    query = transform.transform(dataset.X.loc[[track_id]])[0]
    distances = np.linalg.norm(candidates - query, axis=1)
    eligible = ids != track_id
    artist = dataset.metadata.at[track_id, "artist_id"]
    if exclude_same_artist and pd.notna(artist):
        eligible &= dataset.metadata.loc[ids, "artist_id"].ne(artist).fillna(True).to_numpy(dtype=bool)
    ranked = np.flatnonzero(eligible)
    ranked = ranked[np.argsort(distances[ranked], kind="stable")][:k]
    result = dataset.metadata.loc[ids[ranked], ["title", "artist", "genre", "split"]].copy()
    result["distance"] = distances[ranked]
    return result
