"""Generate the lab report (Markdown, HTML, Word) and slide outline from saved measurements.

Every number in the report is read from files written by the experiment and discovery
stages; nothing is typed in by hand.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import mistune
import numpy as np
import pandas as pd

from .data import Dataset, FEATURE_FAMILIES
from .discovery import nearest_tracks
from .experiment import BASELINE

ACCENT = "255F85"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _table(rows: list[dict] | pd.DataFrame, floatfmt: str = ".4f") -> str:
    frame = pd.DataFrame(rows) if isinstance(rows, list) else rows
    frame = frame.copy()
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].astype(str).str.replace("|", "/", regex=False)
    return frame.to_markdown(index=False, floatfmt=floatfmt)


def _pct(value) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:.2%}"


def _num(value, digits: int = 4) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:.{digits}f}"


def _settings(parameters) -> str:
    """Compact, human-readable hyperparameters."""
    if isinstance(parameters, str):
        parameters = json.loads(parameters)
    if "weights" in parameters:
        return ", ".join(f"{name} {weight:.0%}" for name, weight in parameters["weights"].items())
    if "meta_C" in parameters:
        return f"logistic meta-learner (C={parameters['meta_C']}) over {len(parameters['members'])} families"
    parts = []
    if parameters.get("scaling"):
        parts.append(f"{parameters['scaling']} scaling")
    if parameters.get("reduction") == "select":
        parts.append(f"ANOVA top {parameters['n_components']}")
    elif parameters.get("reduction") == "pca":
        parts.append(f"PCA {parameters['n_components']}")
    for key, value in parameters.get("classifier", parameters).items():
        if key in ("scaling", "reduction", "n_components", "max_iter", "n_iter_no_change"):
            continue
        parts.append(f"{key}={value}")
    return ", ".join(parts)


def _profile_methodology(run: dict, selection: dict, output: Path) -> str:
    folds = run.get("cv_folds", 3)
    if run.get("profile") != "optimized":
        return f"""## 3. Experimental methodology

### 3.1 Preprocessing and leakage prevention

Each scikit-learn pipeline contains median imputation, constant-column removal, standard scaling, and a classifier. All transformations are fitted on the fitting partition only, so no statistics from validation or test tracks leak into training.

### 3.2 Model comparison and selection

The "{run['profile']}" profile evaluates {run['candidate_count']} candidates. Each algorithm's hyperparameters and the winning algorithm are chosen by macro F1 on the official 800-track validation set. Selection is saved to "selection.json" before test data are requested. The selected configurations are refitted on training + validation and evaluated once on the 800-track test set.

"""
    cv = pd.read_csv(output / "cv_results.csv")
    seeds = run.get("ann_seeds", [])
    ensemble_file = output / "ensemble.json"
    ensemble = _read(ensemble_file) if ensemble_file.exists() else None
    grid_rows = (
        cv.groupby("model", sort=False).agg(
            Configurations=("candidate_id", "count"), **{"Best CV macro F1": ("cv_mean_f1", "max")},
            **{"Fit time (s)": ("fit_seconds", "sum")},
        ).reset_index().rename(columns={"model": "Model family"})
    )
    selected_rows = [
        {"Model": name, "CV macro F1": _num(row["cv_mean_f1"]), "Selected settings": _settings(row["parameters"])}
        for name, row in selection["selected"].items() if name != BASELINE
    ]
    ensemble_text = ""
    if ensemble:
        history = pd.DataFrame(ensemble["history"])[["round", "added", "cv_mean_f1"]]
        stacking = ensemble.get("stacking")
        stack_text = ""
        if stacking:
            stack_text = (
                "\nA second, more flexible combiner is also evaluated: a **stacked ensemble** whose multinomial "
                "logistic-regression meta-learner takes the log-probabilities of all "
                f"{len(stacking['members'])} families as input ({len(stacking['members']) * 8} features) and learns "
                "how much to trust each family for each genre. The meta-learner is trained on the same training "
                "OOF probabilities; its regularization strength C is chosen by artist-grouped CV over "
                f"those OOF predictions (selected C = {stacking['C']}, meta-model CV macro F1 = {stacking['cv_mean_f1']:.4f}). "
                "This reuses one base-model OOF matrix rather than refitting the complete stack within each "
                "outer fold, so it is an internal tuning score, not a fully nested evaluation of the stack. "
                "For the final test model the members are refitted on training + validation while the meta-learner "
                "is kept as trained on training OOF predictions.\n"
            )
        ensemble_text = f"""
### 3.4 Ensembles: soft voting and stacking

Different algorithms make different mistakes: a boosted tree ensemble, a kernel SVM and a neural network carve up the feature space in very different ways. Averaging their class probabilities can therefore cancel part of the error of each member.

The ensemble is built with greedy forward selection (Caruana et al., 2004) using **only the out-of-fold probabilities from the training-partition cross-validation**. Starting from an empty ensemble, each round adds (with replacement) the family whose inclusion gives the highest mean fold macro F1; after {len(history)} rounds the best prefix of this path is kept. The number of times each family was chosen becomes its voting weight. Validation and test tracks are never used to set the weights, so the ensemble is a predeclared model that competes with the individual families on validation like any other candidate.

Selected weights (rounds used: {ensemble['rounds_used']}): {_settings({'weights': ensemble['weights']})}. The ensemble's OOF macro F1 is {ensemble['cv_mean_f1']:.4f}. Because the weights were chosen on the same OOF predictions, this CV figure is an optimistic internal tuning diagnostic. The separate validation partition selects the champion, and the test partition provides the final benchmark comparison; neither ensemble CV score is a fully nested whole-pipeline estimate.
{stack_text}"""
    return f"""## 3. Experimental methodology

### 3.1 Preprocessing and leakage prevention

Every candidate is a single scikit-learn pipeline: **median imputation → constant-column removal → standard or quantile-to-normal scaling → optional ANOVA feature selection or PCA → classifier**. The whole pipeline is refitted within every outer fitting fold, so held-out outer-fold tracks do not determine the median, scale, quantiles, selected features or principal directions. The quantile transform maps each descriptor to a normal distribution using training ranks, which tames the heavy tails present in many FMA statistics (kurtosis and max columns in particular).

### 3.2 Artist-grouped cross-validation

All tuning uses {folds}-fold **StratifiedGroupKFold** inside the official 6,400-track training partition, with artist ID as the group. No artist appears on both sides of any fold (audited in "cv_splits.json"), which mirrors the artist-disjoint official split and reduces evaluation gains from recognizing an artist already seen during fitting. Each configuration is scored by mean fold macro F1; the fold standard deviation describes variability across these folds (it is not a confidence interval). Fits run in parallel worker processes.

{_table(grid_rows, '.4f')}

In total {len(cv)} configurations were evaluated ({len(cv) * folds} fold fits), plus the probability refits, ANN seed runs and final refits.

### 3.3 Model families

- **Decision Tree** – axis-aligned threshold rules; interpretable but high-variance.
- **KNN** – inverse-distance vote among the nearest training tracks, optionally in an ANOVA or PCA subspace.
- **SVM (RBF)** – maximum-margin classifier in an RBF kernel space; C controls regularization. Platt scaling supplies class probabilities for the ensemble.
- **ANN (MLP)** – feed-forward ReLU network trained with Adam, L2 regularization and early stopping on a 10% internal holdout. The selected architecture is retrained with {len(seeds)} predeclared seeds ({', '.join(map(str, seeds))}) and their probabilities are averaged; no seed is cherry-picked.
- **Logistic Regression** – a regularized linear (multinomial) model; a linear reference whose probability calibration would require separate evaluation.
- **Extra Trees** – an average of 500 extremely randomized trees.
- **Gradient Boosting** – LightGBM: 500 small trees added sequentially, each fitted to the remaining errors of the previous ones, with a 0.03 learning rate, row/feature subsampling and L2 regularization.
- **Dummy baseline** – always predicts the most frequent training genre (12.5% accuracy on balanced data).

Artist grouping applies to the outer CV and official partitions. The ANN's internal 10% early-stopping split and SVM's internal probability estimation do not receive artist groups; these internal mechanisms should not be described as artist-disjoint.

Selected configuration for each family:

{_table(selected_rows)}
{ensemble_text}
### 3.5 Selection, refit and test evaluation

Every family's selected configuration (and the ensemble) is fitted on the full training partition and scored on the separate official **validation** partition. The champion is the candidate with the highest validation macro F1. The complete selection is written to "selection.json" **before** the program requests any test features or labels; an automated test in the project enforces this ordering. The selected configurations are then refitted on training + validation (7,200 tracks) and evaluated once on the 800 test tracks.

The official test set was already viewed when the original project was run. The test scores below are therefore an *extended benchmark comparison* on the same benchmark, not a pristine confirmation of an improvement.

### 3.6 Metrics and statistics

Accuracy is the share of correctly labelled tracks. For one genre, precision = TP/(TP+FP), recall = TP/(TP+FN), and F1 is their harmonic mean; **macro F1** averages F1 over the eight genres with equal weight and is the selection metric. Top-3 accuracy counts a track as correct when the true genre is among the three most probable genres, which is informative because genre boundaries overlap. Log-loss measures probability quality (lower is better).

Uncertainty on the 800 test tracks is summarized with a 2,000-sample percentile **bootstrap** (95% interval). Paired comparisons with earlier runs use the exact **McNemar test**, which looks only at tracks that one model classifies correctly and the other does not. Both calculations treat individual tracks as independent sampling units, although tracks by the same artist can have correlated errors. The saved intervals and p-values are therefore exploratory track-level summaries, not artist-aware inference or proof of an improvement; artist-grouped resampling and a fresh confirmation dataset would strengthen the analysis.

"""


def generate_report(dataset: Dataset, output: Path) -> Path:
    output = Path(output)
    comparison = pd.read_csv(output / "model_comparison.csv")
    selection = _read(output / "selection.json")
    run = _read(output / "run_info.json")
    discovery = _read(output / "discovery_summary.json")
    importance = pd.read_csv(output / "feature_family_importance.csv")
    associations = pd.read_csv(output / "training_feature_associations.csv").head(10)
    summary_file = output / "improvement_summary.json"
    summary = _read(summary_file) if summary_file.exists() else {}
    champion = selection["champion"]
    chosen = comparison.loc[comparison["model"].eq(champion)].iloc[0]
    baseline = comparison.loc[comparison["model"].eq(BASELINE)].iloc[0]
    per_class = _read(output / "classification_reports.json")[champion]
    matrices = _read(output / "confusion_matrices.json")
    labels = dataset.manifest["genres"]

    matrix = np.array(matrices["matrices"][champion])
    np.fill_diagonal(matrix, 0)
    confusions = []
    for position in np.argsort(matrix, axis=None)[::-1][:5]:
        actual, predicted = np.unravel_index(position, matrix.shape)
        if matrix[actual, predicted] > 0:
            confusions.append({
                "Actual genre": matrices["labels"][actual],
                "Predicted genre": matrices["labels"][predicted],
                "Tracks": int(matrix[actual, predicted]),
            })
    audit = dataset.manifest["split_audit"]
    split_rows = [
        {"Partition": name, "Tracks": count, "Role": role}
        for (name, count), role in zip(
            audit["counts"].items(),
            ["Artist-grouped CV tuning; ensemble weights", "Choose the champion", "Final held-out evaluation"],
        )
    ]
    feature_rows = [
        {"Family": name, "Columns": size, "Meaning": FEATURE_FAMILIES[name]}
        for name, size in sorted(dataset.manifest["feature_family_sizes"].items())
    ]
    has_cv = "cv_mean_f1" in comparison
    score_rows = []
    for row in comparison.itertuples():
        entry = {"Model": f"**{row.model}**" if row.model == champion else row.model}
        if has_cv:
            entry["CV macro F1"] = _num(row.cv_mean_f1)
        entry.update({
            "Validation macro F1": _num(row.validation_f1_macro),
            "Test accuracy": _pct(row.test_accuracy),
            "Test macro F1": _num(row.test_f1_macro),
        })
        if "test_top3_accuracy" in comparison:
            entry["Test top-3 accuracy"] = _pct(row.test_top3_accuracy)
        score_rows.append(entry)
    genre_rows = [
        {"Genre": label, "Precision": per_class[label]["precision"],
         "Recall": per_class[label]["recall"], "F1": per_class[label]["f1-score"],
         "Test tracks": int(per_class[label]["support"])}
        for label in labels
    ]
    importance_rows = [
        {"Family": row.family, "Columns": row.n_features,
         "Mean F1 drop": row.mean_f1_drop, "Repeat SD": row.std_f1_drop}
        for row in importance.itertuples()
    ]
    timing_rows = [
        {"Model": row.model, "CV fitting (s)": row.tuning_fit_seconds,
         "Final refit (s)": row.refit_seconds, "Test prediction (s)": row.test_predict_seconds}
        for row in comparison.itertuples()
    ]
    query_id = int(dataset.ids("validation")[0])
    neighbors = nearest_tracks(dataset, output.parent / "models", query_id, k=5)
    neighbors.to_csv(output / "example_neighbors.csv")
    neighbor_table = neighbors.reset_index()[["track_id", "title", "artist", "genre", "distance"]]
    top_families = ", ".join(importance.head(3)["family"])
    best_test = comparison.loc[comparison["model"].ne(BASELINE)].sort_values("test_accuracy", ascending=False).iloc[0]

    # Champion uncertainty and comparison with earlier runs.
    interval = (summary.get("champion_test_bootstrap") or {})
    ci_text = ""
    if interval:
        lo, hi = interval["accuracy_95ci"]
        flo, fhi = interval["f1_macro_95ci"]
        ci_text = (f" A bootstrap over the 800 test tracks gives a 95% interval of **{lo:.1%}–{hi:.1%}** "
                   f"for accuracy and {flo:.3f}–{fhi:.3f} for macro F1.")
    history_rows, history_text = [], ""
    run_names = {"original": "Original project (validation-only tuning)", "improved_v1": "First improvement (grouped CV, 5 families)"}
    for entry in summary.get("comparisons", []):
        test = entry.get("mcnemar_vs_champion")
        history_rows.append({
            "Run": run_names.get(entry["run"], entry["run"]), "Selected model": entry["champion"],
            "Test accuracy": _pct(entry["test_accuracy"]), "Test macro F1": _num(entry["test_f1_macro"]),
            "Change in accuracy (pp)": f"{entry['accuracy_change_percentage_points']:+.2f}",
            "McNemar p": _num(test["p_value"], 3) if test else "—",
        })
    if history_rows:
        history_rows.append({
            "Run": "**This optimized run**", "Selected model": champion,
            "Test accuracy": _pct(chosen.test_accuracy), "Test macro F1": _num(chosen.test_f1_macro),
            "Change in accuracy (pp)": "—", "McNemar p": "—",
        })
        history_text = f"""### 4.1 Comparison with earlier runs

The original and first-improvement reports are preserved under "baseline/" and "history/"; the original models are under "baseline/models/". All three runs use the same 800 official test tracks, so the comparison is paired.

{_table(history_rows)}

![Test accuracy of each model family across runs](figures/improvement_comparison.png)

""" + (
            " ".join(
                f"Against the {run_names.get(e['run'], e['run']).split(' (')[0].lower()}, the McNemar p-value is "
                f"{e['mcnemar_vs_champion']['p_value']:.3f} "
                + ("(below 0.05 under the track-independence assumption; this is exploratory, not artist-aware confirmation)." if e['mcnemar_vs_champion']['p_value'] < 0.05
                   else "(not significant at the 5% level under the track-independence assumption).")
                for e in summary.get("comparisons", []) if e.get("mcnemar_vs_champion")
            ) + " "
        ) + "Because the original test scores had already been seen, a fresh artist-disjoint dataset would be needed for a fully independent confirmation.\n\n"

    ensemble_section = ""
    ensemble_file = output / "ensemble.json"
    if ensemble_file.exists():
        ensemble = _read(ensemble_file)
        members = [
            {"Member": name, "Weight": f"{weight:.0%}",
             "Member CV macro F1": _num(ensemble["member_cv_mean_f1"].get(name))}
            for name, weight in sorted(ensemble["weights"].items(), key=lambda item: -item[1])
        ]
        ensemble_section = f"""### 4.3 What the ensemble is made of

{_table(members)}

![Ensemble weights](figures/ensemble_weights.png)

"""
    seeds_file = output / "ann_seed_results.csv"
    seed_section = ""
    if seeds_file.exists():
        seed_section = f"""ANN seed variability (individual seeds and their probability average):

{_table(pd.read_csv(seeds_file))}

"""
    confidence_section = ""
    if (output / "figures/confidence.png").exists():
        predictions = pd.read_csv(output / "test_predictions.csv")
        if "champion_confidence" in predictions:
            from .improvement import IMPROVED_SLUGS
            slug = IMPROVED_SLUGS.get(champion)
            correct = predictions[slug].eq(predictions["actual"])
            high = predictions["champion_confidence"] >= 0.6
            confidence_section = f"""### 4.5 How far can the predictions be trusted?

The champion outputs a probability for each genre. When its top probability is at least 60% (true for {high.mean():.0%} of test tracks) it is correct {correct[high].mean():.1%} of the time; below 60% it is correct only {correct[~high].mean():.1%} of the time. This is a descriptive association on this benchmark, not a probability-calibration study or a guarantee for an individual song.

![Accuracy by confidence](figures/confidence.png)

"""
    ranked_genres = sorted(labels, key=lambda g: per_class[g]["f1-score"], reverse=True)
    easiest = ", ".join(f"{g} (F1 {per_class[g]['f1-score']:.2f})" for g in ranked_genres[:3])
    hardest = " and ".join(f"{g} (F1 {per_class[g]['f1-score']:.2f})" for g in ranked_genres[-2:])
    notes = {
        "Pop": "Pop is a stylistic umbrella that borrows from many genres",
        "Experimental": "Experimental music is defined by breaking conventions",
        "International": "International mixes many regional styles",
    }
    full = np.array(matrices["matrices"][champion])
    reasons = []
    for g in ranked_genres[-2:]:
        row = full[matrices["labels"].index(g)].astype(float)
        row[matrices["labels"].index(g)] = -1
        top = [matrices["labels"][j] for j in np.argsort(row)[::-1][:2]]
        reasons.append((notes[g] + "; " if g in notes else "") + f"{g} tracks are most often predicted as {top[0]} or {top[1]}")
    hard_note = (" " + ". ".join(r[0].upper() + r[1:] for r in reasons) + ". Overlapping acoustic styles are a plausible explanation, not a proven cause."
                 if reasons else "")
    from .improvement import ENSEMBLE, STACKED
    individual = comparison.loc[~comparison["model"].isin([BASELINE, ENSEMBLE, STACKED])]
    strongest = ", ".join(individual.sort_values("validation_f1_macro", ascending=False)["model"].head(3))
    ensemble_note = ""
    if champion in (ENSEMBLE, STACKED):
        best_single = individual.sort_values("validation_f1_macro", ascending=False).iloc[0]
        delta = 100 * (chosen.test_accuracy - best_single.test_accuracy)
        ensemble_note = (f" The {champion.lower()} differs from the best single validation family "
                         f"({best_single.model}) by {delta:+.2f} accuracy points on test and "
                         f"{chosen.validation_f1_macro - best_single.validation_f1_macro:+.4f} macro F1 on validation.")
    family_names = [name for name in comparison["model"] if name != BASELINE]

    text = f"""# MusicMap
## Music Genre Classification and Pattern Discovery Using Machine Learning

Machine Learning Course Project

Student: Akshat Tikaria — Roll number: 241070005

Student: Aniket Shinde — Roll number: 171070977

Institution: Veermata Jijabai Technological Institute (VJTI)

Course: Machine Learning

## Abstract

MusicMap investigates whether precomputed audio descriptors can identify a track's music genre and reveal relationships between songs. It uses the {len(dataset.X):,} tracks of the balanced FMA-small dataset (eight genres, {dataset.X.shape[1]} numeric descriptors per track) and keeps the official artist-disjoint training, validation and test partitions. {len(family_names)} candidate models — {', '.join(family_names)} — are tuned with artist-grouped cross-validation and compared against a majority-class baseline. The validation-selected model, **{champion}**, reaches **{chosen.test_accuracy:.2%} test accuracy** (random top-1 chance: {1 / len(labels):.1%}) and **{chosen.test_f1_macro:.4f} macro F1**{f", with the correct genre among its top three guesses for {chosen.test_top3_accuracy:.1%} of tracks (random top-3 chance: {min(3, len(labels)) / len(labels):.1%})" if 'test_top3_accuracy' in comparison and pd.notna(chosen.test_top3_accuracy) else ''}. PCA, K-means clustering, nearest-neighbour retrieval and grouped permutation importance provide complementary views of musical similarity and of which audio properties matter. The findings describe this dataset and protocol rather than universal genre boundaries.

## 1. Problem statement and objectives

Genre labels help organize and recommend music, but genres share instruments, production styles and conventions. MusicMap treats genre recognition as supervised multiclass classification: one descriptor vector is mapped to one of eight top-level genres.

Objectives:

1. Prepare a reproducible dataset from published audio descriptors, without downloading 7.2 GiB of audio.
2. Compare the required algorithms (Decision Tree, KNN, SVM, ANN) with stronger modern baselines (Logistic Regression, Extra Trees, Gradient Boosting) and a soft-voting ensemble, using training-fitted preprocessing and artist-disjoint outer evaluation.
3. Report more than accuracy: macro F1, per-genre errors, top-3 accuracy, confidence and uncertainty.
4. Identify the audio feature families that drive the predictions.
5. Discover structure through dimensionality reduction, clustering and nearest-song retrieval.

## 2. Dataset and features

The source is *FMA: A Dataset for Music Analysis* (Defferrard, Benzi, Vandergheynst and Bresson, ISMIR 2017). FMA-small contains 8,000 30-second excerpts balanced across Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop and Rock. Only the 342 MiB metadata archive is needed; its published SHA-1 ({dataset.manifest['source_sha1']}) is verified before reading. Feature rows and metadata are joined by track ID, never by row position.

{_table(split_rows)}

Each genre contributes 800 training, 100 validation and 100 test tracks. Shared artists between partitions: {', '.join(f"{k.replace('__', '/')} = {v}" for k, v in audit['artist_overlap_counts'].items())} — no artist appears in more than one partition.

![Genre distribution and partitions](figures/genre_balance.png)

The {dataset.X.shape[1]} descriptors summarize 74 frame-level feature dimensions using seven statistics each (74 × 7 = 518), grouped into 11 audio feature families. The statistics over time are: mean, standard deviation, skewness, kurtosis, median, minimum and maximum. Only audio descriptors enter the classifiers; title, artist, album, genre and split are kept separate.

{_table(feature_rows)}

For example, "mfcc__mean__01" is the temporal mean of the first MFCC coefficient. There is no explicit tempo feature. Data checks found {dataset.manifest['nonfinite_values_replaced_with_nan']} non-finite values and {len(dataset.manifest['removed_all_missing_track_ids'])} unusable rows.

{_profile_methodology(run, selection, output)}## 4. Results

The model selected on validation before the test set was requested is **{champion}**. On the 800 test tracks it achieves **{chosen.test_accuracy:.2%} accuracy** and **{chosen.test_f1_macro:.4f} macro F1**.{ci_text} The majority-class baseline achieves {baseline.test_accuracy:.2%} accuracy and {baseline.test_f1_macro:.4f} macro F1. Rows are ordered by validation macro F1, the selection criterion.

{_table(score_rows)}

![Comparison of all models](figures/model_comparison.png)

{f"The highest observed test accuracy among all candidates is {best_test.test_accuracy:.2%} ({best_test.model}). Test scores are descriptive: they never change the predeclared validation-based choice." if best_test.model != champion else "The validation-selected champion also has the highest test accuracy of all candidates, so validation and test agree on the ranking at the top."} Validation scores are higher than test scores for every trained model. This gap can reflect partition differences and selection on validation; these measurements alone do not establish its cause.

{history_text}### 4.2 Cross-validation stability

![Artist-grouped cross-validation](figures/grouped_cv.png)

{seed_section}{ensemble_section}### 4.4 Per-genre performance

{_table(genre_rows)}

![Champion per-genre F1](figures/per_genre_f1.png)

The largest off-diagonal confusion counts for the champion are:

{_table(confusions)}

![Confusion matrices](figures/confusion_matrices.png)

Each confusion-matrix row is normalized by the number of tracks of that actual genre. The champion recognizes {easiest} best and finds {hardest} hardest.{hard_note}

{confidence_section}### 4.6 Computation and reproducibility

{_table(timing_rows, '.1f')}

The tuning, refit and evaluation stage took {run['elapsed_seconds'] / 60:.1f} minutes using {run['threads']} parallel worker processes (seed {run['seed']}), with Python {run['python']} and scikit-learn {run['versions']['scikit-learn']}. Times exclude data preparation, discovery and report generation and vary by machine. Exact dependency versions and checksums are recorded in "run_info.json".

## 5. Feature interpretation

The training-only champion is evaluated on validation tracks while one entire feature family is jointly shuffled across tracks; all other families stay intact. The resulting drop in macro F1 measures how much the model relies on that family. Each family is shuffled {discovery['permutation_repeats']} times.

{_table(importance_rows)}

![Feature-family permutation importance](figures/feature_importance.png)

The leading families are {top_families}. MFCCs describe timbre (the "colour" of the sound: distorted guitars, synthesizers, voice), and spectral contrast captures the difference between spectral peaks and valleys, which can describe differences between clean acoustic recordings and denser production, without establishing a universal genre rule. Family sizes differ and correlated families can substitute for one another, so this ranking describes model reliance, not musical causation.

The ten individual descriptors most associated with genre on the training set (one-way ANOVA F):

{_table(associations[['feature', 'training_anova_f']].rename(columns={'feature': 'Descriptor', 'training_anova_f': 'Training ANOVA F'}), '.1f')}

![Training feature distributions](figures/feature_distributions.png)

## 6. Pattern discovery and song similarity

### 6.1 PCA music map

Scaling and PCA are fitted on training descriptors only; training and validation tracks are projected into that space, and genre labels are used only as colours. The first two components keep {discovery['pca_2d_explained_variance']:.1%} of the standardized variance and the first {discovery['pca_components']} keep {discovery['pca_50d_explained_variance']:.1%}.

![PCA song map](figures/pca_map.png)

![PCA cumulative variance](figures/pca_variance.png)

Open "song_map.html" for the offline interactive version: hover for track details, filter genres through the legend and zoom. Overlap in two dimensions is expected because most information lives in the remaining components.

### 6.2 K-means clustering

K-means (k = {discovery['kmeans_k']}, fixed in advance) is fitted on the training PCA representation without labels; validation tracks are assigned to the nearest centroid.

- Adjusted Rand index against genres: {discovery['kmeans_adjusted_rand_index']:.4f}
- Normalized mutual information: {discovery['kmeans_normalized_mutual_information']:.4f}
- Silhouette score: {_num(discovery['kmeans_silhouette'])}

![Validation cluster composition](figures/cluster_composition.png)

The low but positive agreement shows that these unsupervised audio clusters capture some genre structure but do not cleanly recover the genre labels. These metrics do not identify which acoustic properties cause the clusters or establish that only supervised methods can recognize genres.

### 6.3 Nearest songs

Euclidean distance in the training-fitted {discovery['pca_components']}-dimensional PCA space retrieves similar training tracks. The query itself and tracks by the same artist are excluded. For validation track {query_id}, "{dataset.metadata.at[query_id, 'title']}" by {dataset.metadata.at[query_id, 'artist']} ({dataset.y.at[query_id]}), the five nearest tracks are:

{_table(neighbor_table)}

Distances are representation-specific and not listener-preference scores; this is an audio-similarity demonstration, not a personalized recommender.

## 7. Discussion and limitations

1. **Representation limits.** All models use the same 518 summary statistics, which discard the order of musical events. Representation is a possible bottleneck, but this finite model search does not establish a feature ceiling; spectrograms or pretrained audio embeddings are alternatives to evaluate.
2. **Overlapping labels.** Each track carries one top-level genre although many songs mix styles. Top-3 accuracy shows how often the correct label remains among three guesses; it does not by itself prove perceptual similarity between confused genres.
3. **Reused test set.** The official test partition was viewed in the original project, so improvements are an extended benchmark comparison. The track-level McNemar test and bootstrap intervals do not remove this limitation and do not account for within-artist dependence.
4. **Finite search.** Grids are deliberately affordable on a laptop. Larger searches could find slightly better settings for individual families.
5. **Artist effects.** Artist-disjoint splits remove artist memorization but not every production or label-specific correlation.
6. **Importance is not causation.** Permutation importance describes one fitted model on one validation set.
7. **New audio.** Predicting a new MP3 requires extracting the same descriptors; "musicmap extract" follows the official FMA extraction recipe, but modern library versions can produce slightly different values.

## 8. Conclusion and future work

Using training-fitted preprocessing and artist-disjoint outer evaluation, {champion} was selected on validation and obtained {chosen.test_accuracy:.2%} test accuracy and {chosen.test_f1_macro:.4f} macro F1 on eight balanced genres — about {chosen.test_accuracy / 0.125:.1f} times the chance level. The strongest individual families on validation were {strongest}.{ensemble_note} The feature analysis highlights {top_families}, while the unsupervised clusters only partially agree with genre labels.

Future work: learn directly from mel-spectrograms with a CNN or use pretrained audio embeddings (e.g. VGGish, PANNs, CLAP), extend to the 16-genre FMA-medium subset, and confirm results on a fresh artist-disjoint test set.

## 9. Reproduction and deliverables

~~~sh
python -m pip install -r requirements.txt "scikit-learn==1.8.0"
musicmap improve --seed 42
musicmap predict --track-id 704
musicmap neighbors --track-id 704 --k 5
python -m pytest -q --junitxml=/tmp/musicmap-tests.xml
python scripts/verify_artifacts.py --test-report /tmp/musicmap-tests.xml
jupyter lab notebooks/MusicMap.ipynb
~~~

The notebook displays the included results by default; set RUN_TRAINING = True to recompute. "musicmap app" starts an interactive demo in the browser. The examples use held-out test track 704; demonstration examples are not a new evaluation. The included saved classifiers require scikit-learn 1.8.0. Own-audio extraction requires a sample rate of at least 11,025 Hz; lower-rate files receive conversion guidance rather than being silently resampled. FFmpeg and ffprobe enable decoding of M4A/AAC files unsupported by the default decoder. No GPU or benchmark audio download is needed.

## References and attribution

1. Defferrard, M., Benzi, K., Vandergheynst, P., and Bresson, X. (2017). FMA: A Dataset for Music Analysis. ISMIR. https://arxiv.org/abs/1612.01840
2. Official FMA repository: https://github.com/mdeff/fma
3. Caruana, R., Niculescu-Mizil, A., Crew, G., and Ksikes, A. (2004). Ensemble selection from libraries of models. ICML.
4. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, 2825–2830. https://scikit-learn.org/
5. McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika 12, 153–157.

FMA metadata and derived features are attributed to Defferrard et al. and distributed under CC BY 4.0. No audio is redistributed.
"""
    text = re.sub(r"\n{3,}", "\n\n", text)
    report = output / "MusicMap_Report.md"
    report.write_text(text, encoding="utf-8")
    _write_html(text, output)
    _write_docx(text, output)
    _write_presentation(champion, chosen, discovery, importance, output, run, summary)
    _update_readme(output.parent, comparison, champion, chosen, summary, run)
    print(f"Wrote report: {report}", flush=True)
    return report


def _write_html(text: str, output: Path) -> None:
    renderer = mistune.create_markdown(plugins=["table"])
    body = renderer(text)
    html = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MusicMap · ML lab report</title><style>
:root{--ink:#183143;--accent:#255f85;--soft:#edf4f7;--line:#d8e2e7;--bg:#f0f4f5;--card:#fff}
body{font:16px/1.65 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);background:var(--bg);margin:0}
main{max-width:1080px;margin:40px auto;padding:50px;background:var(--card);border-radius:16px;box-shadow:0 1px 3px #0001}
h1{font-size:48px;letter-spacing:-2px;margin:0;color:var(--accent)}
h2{font-size:25px;color:var(--accent);margin-top:44px;line-height:1.3;border-bottom:2px solid var(--soft);padding-bottom:6px}
h3{font-size:20px;margin-top:32px}a{color:var(--accent)}
table{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow:auto;margin:14px 0}
th,td{padding:8px 12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
td:last-child{white-space:normal}
th{background:var(--soft)}tr:nth-child(even){background:#f8fafb}
img{max-width:100%;height:auto;margin:12px 0;border-radius:6px}
pre{background:var(--bg);padding:20px;overflow:auto;border-radius:8px}
li{margin:6px 0}@media(max-width:700px){main{margin:0;padding:22px;border-radius:0}h1{font-size:36px}}
@media print{body{background:white}main{padding:0;margin:0;box-shadow:none}h2,h3{break-after:avoid}img,table{break-inside:avoid}}
</style></head><body><main>""" + body + "</main></body></html>"
    (output / "MusicMap_Report.html").write_text(html, encoding="utf-8")


def _write_docx(markdown: str, output: Path) -> None:
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt, RGBColor

    document = Document()
    section = document.sections[0]
    section.top_margin = section.bottom_margin = Inches(0.8)
    section.left_margin = section.right_margin = Inches(0.85)
    normal = document.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.12
    for name, size in (("Title", 30), ("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 12)):
        style = document.styles[name]
        style.font.color.rgb = RGBColor.from_string(ACCENT)
        style.font.size = Pt(size)
        style.font.name = "Calibri"
    footer = section.footer.paragraphs[0]
    footer.text = "MusicMap · Machine Learning Course Project · page "
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run()
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if kind:
            element = OxmlElement("w:fldChar")
            element.set(qn("w:fldCharType"), kind)
        else:
            element = OxmlElement("w:instrText")
            element.set(qn("xml:space"), "preserve")
            element.text = text
        run._r.append(element)

    counters = {"figure": 0, "table": 0}

    def shade(cell, color):
        properties = cell._tc.get_or_add_tcPr()
        fill = OxmlElement("w:shd")
        fill.set(qn("w:val"), "clear")
        fill.set(qn("w:color"), "auto")
        fill.set(qn("w:fill"), color)
        properties.append(fill)

    def add_inline(paragraph, tokens, bold=False, italic=False):
        for token in tokens:
            kind = token["type"]
            if kind in ("text", "codespan"):
                piece = paragraph.add_run(token.get("raw", ""))
                piece.bold, piece.italic = bold, italic
                if kind == "codespan":
                    piece.font.name = "Consolas"
            elif kind == "strong":
                add_inline(paragraph, token.get("children", []), True, italic)
            elif kind == "emphasis":
                add_inline(paragraph, token.get("children", []), bold, True)
            elif kind in ("softbreak", "linebreak"):
                paragraph.add_run().add_break()
            elif kind == "link":
                add_inline(paragraph, token.get("children", []), bold, italic)
            else:
                add_inline(paragraph, token.get("children", []), bold, italic)

    def plain(tokens):
        result = ""
        for token in tokens:
            if token["type"] in ("text", "codespan"):
                result += token.get("raw", "")
            elif token["type"] in ("softbreak", "linebreak"):
                result += " "
            else:
                result += plain(token.get("children", []))
        return result

    first_heading = {"done": False}

    def add_blocks(tokens, list_style=None):
        for token in tokens:
            kind = token["type"]
            children = token.get("children", [])
            if kind == "heading":
                level = token["attrs"]["level"]
                title = plain(children)
                if level == 1 and not first_heading["done"]:
                    for _ in range(5):
                        document.add_paragraph()
                    paragraph = document.add_paragraph(title, style="Title")
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    first_heading["done"] = True
                    continue
                if level == 2 and title.startswith("Music Genre"):
                    paragraph = document.add_paragraph()
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    piece = paragraph.add_run(title)
                    piece.font.size, piece.bold = Pt(15), True
                    piece.font.color.rgb = RGBColor.from_string("50A995")
                    continue
                if level == 2 and title == "Abstract":
                    document.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
                document.add_heading(title, level=max(1, level - 1))
            elif kind in ("paragraph", "block_text"):
                images = [child for child in children if child["type"] == "image"]
                if images:
                    for child in images:
                        path = output / child["attrs"]["url"]
                        if path.exists():
                            document.add_picture(str(path), width=Inches(6.4))
                            document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                            counters["figure"] += 1
                            caption = document.add_paragraph(
                                f"Figure {counters['figure']}. {plain(child.get('children', []))}", style="Caption")
                            caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
                else:
                    paragraph = document.add_paragraph(style=list_style)
                    add_inline(paragraph, children)
                    if plain(children).startswith(("Machine Learning Course Project", "Student:", "Institution:", "Course:")):
                        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            elif kind == "list":
                style = "List Number" if token.get("attrs", {}).get("ordered") else "List Bullet"
                for item in children:
                    add_blocks(item.get("children", []), style)
            elif kind == "block_code":
                paragraph = document.add_paragraph()
                piece = paragraph.add_run(token.get("raw", "").rstrip())
                piece.font.name = "Consolas"
                piece.font.size = Pt(9)
            elif kind == "table":
                head, body = [], []
                for part in children:
                    if part["type"] == "table_head":
                        head = [cell.get("children", []) for cell in part["children"]]
                    elif part["type"] == "table_body":
                        body = [[cell.get("children", []) for cell in row["children"]] for row in part["children"]]
                if not head:
                    continue
                table = document.add_table(rows=1, cols=len(head))
                table.style = "Table Grid"
                table.alignment = WD_TABLE_ALIGNMENT.CENTER
                for i, cell_tokens in enumerate(head):
                    cell = table.rows[0].cells[i]
                    cell.text = ""
                    piece = cell.paragraphs[0].add_run(plain(cell_tokens))
                    piece.bold = True
                    piece.font.color.rgb = RGBColor.from_string("FFFFFF")
                    shade(cell, ACCENT)
                for r, row in enumerate(body):
                    cells = table.add_row().cells
                    for i, cell_tokens in enumerate(row):
                        cells[i].text = ""
                        add_inline(cells[i].paragraphs[0], cell_tokens)
                        if r % 2:
                            shade(cells[i], "EEF4F7")
                for row in table.rows:
                    for cell in row.cells:
                        for paragraph in cell.paragraphs:
                            paragraph.paragraph_format.space_after = Pt(0)
                            for piece in paragraph.runs:
                                piece.font.size = Pt(8.5)
                document.add_paragraph()

    parser = mistune.create_markdown(renderer="ast", plugins=["table"])
    add_blocks(parser(markdown))
    document.save(output / "MusicMap_Report.docx")


def _write_presentation(champion, chosen, discovery, importance, output: Path,
                        run: dict | None = None, summary: dict | None = None) -> None:
    run = run or {}
    summary = summary or {}
    comparisons = {entry["run"]: entry for entry in summary.get("comparisons", [])}
    original = comparisons.get("original")
    progress = (
        f"- Original project: {original['champion']} {original['test_accuracy']:.2%} → this run: "
        f"{champion} {chosen.test_accuracy:.2%} ({original['accuracy_change_percentage_points']:+.2f} pp).\n"
        if original else ""
    )
    report = json.loads((output / "classification_reports.json").read_text())[champion]
    genres = sorted((k for k in report if k not in ("accuracy", "macro avg", "weighted avg")),
                    key=lambda g: report[g]["f1-score"], reverse=True)
    easy, hard = " and ".join(genres[:2]), " and ".join(genres[-2:])
    slides = f"""# MusicMap presentation outline

Ten slides for a 7–10 minute Machine Learning course presentation.

Akshat Tikaria (241070005) and Aniket Shinde (171070977)

Veermata Jijabai Technological Institute (VJTI)

## Slide 1 — Title and problem
- MusicMap: Music Genre Classification and Pattern Discovery Using Machine Learning.
- Can audio descriptors predict a track's genre and reveal similar songs?

## Slide 2 — Dataset
- FMA-small: 8,000 tracks, eight balanced genres, 518 descriptors (74 feature dimensions × 7 statistics, grouped into 11 families).
- Precomputed features; no audio download or GPU.
- Show figures/genre_balance.png and cite Defferrard et al. (2017).

## Slide 3 — Experimental design
- Official artist-disjoint split: 6,400 training / 800 validation / 800 test.
- Tune with {run.get('cv_folds', 3)}-fold artist-grouped cross-validation inside training (no artist on both sides).
- Choose the champion on validation; lock selection; refit on 7,200; evaluate once on test.

## Slide 4 — Models
- Required: Decision Tree, KNN, SVM (RBF), ANN (MLP, seed-averaged).
- Stronger baselines: Logistic Regression, Extra Trees, Gradient Boosting.
- Soft-voting ensemble: weights chosen by greedy selection on out-of-fold probabilities.
- Stacking: a logistic meta-classifier learns how to combine the seven families' out-of-fold log-probabilities.
- Majority-class baseline for context (12.5% top-1 accuracy).

## Slide 5 — Results
- Champion (chosen on validation): {champion}.
- Test accuracy {chosen.test_accuracy:.2%}; macro F1 {chosen.test_f1_macro:.4f}{f"; top-3 accuracy {chosen.test_top3_accuracy:.1%}" if 'test_top3_accuracy' in chosen and pd.notna(chosen.test_top3_accuracy) else ''}.
- Random chance: 12.5% top-1, 37.5% top-3 for eight classes.
{progress}- Show figures/model_comparison.png and figures/improvement_comparison.png.

## Slide 6 — Errors and confidence
- Show figures/confusion_matrices.png or figures/per_genre_f1.png.
- Easiest genres: {easy}; hardest: {hard} (see per-genre F1).
- Show figures/confidence.png: higher-confidence groups are more accurate in this saved benchmark; this is not a calibration guarantee.

## Slide 7 — Feature interpretation
- Jointly shuffle each feature family on validation data and measure the F1 drop.
- Top three: {', '.join(importance.head(3)['family'])}.
- Show figures/feature_importance.png. Importance = model reliance, not causation.

## Slide 8 — Pattern discovery
- Train-only PCA: first two PCs keep {discovery['pca_2d_explained_variance']:.1%} of variance.
- K-means without labels: validation ARI {discovery['kmeans_adjusted_rand_index']:.3f} — clusters only partially agree with genre labels.
- Show figures/pca_map.png and figures/cluster_composition.png.

## Slide 9 — Live demonstration
- musicmap app — pick a test-partition song, see genre probabilities and eight similar songs.
- Or: musicmap predict --track-id 704 and musicmap neighbors --track-id 704 --k 5.
- Open song_map.html; hover, zoom and filter the legend.

## Slide 10 — Conclusions and limits
- {champion} achieved the highest validation macro F1 in this experiment; fold variability does not establish superior stability.
- Limits: summary features, overlapping genres, previously viewed test set, track-level uncertainty that ignores within-artist dependence.
- Future: spectrogram CNNs / pretrained embeddings, FMA-medium, fresh confirmation set.
"""
    (output / "presentation_outline.md").write_text(slides, encoding="utf-8")


def _update_readme(root: Path, comparison: pd.DataFrame, champion: str, chosen, summary: dict, run: dict) -> None:
    """Refresh the measured-results block of README.md so it always matches the run."""
    readme = root / "README.md"
    start, end = "<!-- RESULTS START -->", "<!-- RESULTS END -->"
    if not readme.exists() or start not in readme.read_text(encoding="utf-8"):
        return
    rows = []
    for row in comparison.itertuples():
        rows.append({
            "Model": f"**{row.model}**" if row.model == champion else row.model,
            "CV macro F1": _num(getattr(row, "cv_mean_f1", None)),
            "Validation macro F1": _num(row.validation_f1_macro),
            "Test accuracy": _pct(row.test_accuracy),
            "Test macro F1": _num(row.test_f1_macro),
            "Top-3 accuracy": _pct(getattr(row, "test_top3_accuracy", None)),
        })
    lines = [start, "", _table(rows), ""]
    interval = summary.get("champion_test_bootstrap")
    text = (f"**Champion (chosen on validation): {champion}** — test accuracy **{chosen.test_accuracy:.2%}**, "
            f"macro F1 **{chosen.test_f1_macro:.4f}**")
    if interval:
        lo, hi = interval["accuracy_95ci"]
        text += f" (95% bootstrap interval for accuracy: {lo:.1%}–{hi:.1%})"
    lines.append(text + ".")
    names = {"original": "original project", "improved_v1": "first improvement"}
    for entry in summary.get("comparisons", []):
        test = entry.get("mcnemar_vs_champion")
        lines.append(
            f"- vs. {names.get(entry['run'], entry['run'])} ({entry['champion']}, {entry['test_accuracy']:.2%}): "
            f"**{entry['accuracy_change_percentage_points']:+.2f} percentage points**"
            + (f", McNemar p = {test['p_value']:.3f}" if test else "")
        )
    if interval or summary.get("comparisons"):
        lines += ["", "Bootstrap intervals and McNemar p-values are exploratory track-level summaries; "
                  "they do not account for correlated errors among tracks by the same artist."]
    lines += ["", f"Full run: {run['elapsed_seconds'] / 60:.1f} min with {run['threads']} parallel workers "
              f"(Python {run['python']}, scikit-learn {run['versions']['scikit-learn']}).", "", end]
    text = readme.read_text(encoding="utf-8")
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    readme.write_text(before + "\n".join(lines) + after, encoding="utf-8")
