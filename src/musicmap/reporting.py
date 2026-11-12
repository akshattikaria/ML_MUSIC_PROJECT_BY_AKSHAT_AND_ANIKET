"""Create a report from saved measurements, with editable Word and offline HTML copies."""

from __future__ import annotations

import json
from pathlib import Path

import mistune
import numpy as np
import pandas as pd

from .data import Dataset, FEATURE_FAMILIES
from .discovery import nearest_tracks
from .experiment import BASELINE


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _table(rows: list[dict] | pd.DataFrame) -> str:
    frame = pd.DataFrame(rows) if isinstance(rows, list) else rows
    return frame.to_markdown(index=False, floatfmt=".4f")


def generate_report(dataset: Dataset, output: Path) -> Path:
    output = Path(output)
    comparison = pd.read_csv(output / "model_comparison.csv")
    selection = _read(output / "selection.json")
    run = _read(output / "run_info.json")
    discovery = _read(output / "discovery_summary.json")
    importance = pd.read_csv(output / "feature_family_importance.csv")
    associations = pd.read_csv(output / "training_feature_associations.csv").head(10)
    champion = selection["champion"]
    chosen = comparison.loc[comparison["model"].eq(champion)].iloc[0]
    baseline = comparison.loc[comparison["model"].eq(BASELINE)].iloc[0]
    per_class = _read(output / "classification_reports.json")[champion]
    matrices = _read(output / "confusion_matrices.json")
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
    split_rows = [
        {"Partition": name, "Tracks": count, "Role": role}
        for (name, count), role in zip(
            dataset.manifest["split_audit"]["counts"].items(),
            ["Fit candidate models and preprocessing", "Choose hyperparameters and model", "Final held-out evaluation"],
        )
    ]
    feature_rows = [
        {"Family": name, "Columns": size, "Meaning": FEATURE_FAMILIES[name]}
        for name, size in sorted(dataset.manifest["feature_family_sizes"].items())
    ]
    score_rows = []
    for row in comparison.itertuples():
        score_rows.append({
            "Model": row.model,
            "Validation F1": f"{row.validation_f1_macro:.4f}",
            "Test accuracy": f"{row.test_accuracy:.2%}",
            "Test precision": f"{row.test_precision_macro:.4f}",
            "Test recall": f"{row.test_recall_macro:.4f}",
            "Test F1": f"{row.test_f1_macro:.4f}",
        })
    selected_rows = [
        {"Model": name, "Selected hyperparameters": json.dumps(selection["selected"][name]["parameters"])}
        for name in selection["selected"] if name != BASELINE
    ]
    genre_rows = [
        {"Genre": label, "Precision": per_class[label]["precision"],
         "Recall": per_class[label]["recall"], "F1": per_class[label]["f1-score"],
         "Test tracks": int(per_class[label]["support"])}
        for label in dataset.manifest["genres"]
    ]
    importance_rows = [
        {"Family": row.family, "Columns": row.n_features,
         "Mean F1 drop": row.mean_f1_drop, "Repeat SD": row.std_f1_drop}
        for row in importance.itertuples()
    ]
    timing_rows = [
        {"Model": row.model, "Candidate fitting (s)": row.tuning_fit_seconds,
         "Final refit (s)": row.refit_seconds, "Test prediction (s)": row.test_predict_seconds}
        for row in comparison.itertuples()
    ]
    query_id = int(dataset.ids("validation")[0])
    neighbors = nearest_tracks(dataset, output.parent / "models", query_id, k=5)
    neighbors.to_csv(output / "example_neighbors.csv")
    neighbor_table = neighbors.reset_index()[["track_id", "title", "artist", "genre", "distance"]].copy()
    for column in ("title", "artist"):
        neighbor_table[column] = neighbor_table[column].astype(str).str.replace("|", "/", regex=False)
    top_families = ", ".join(importance.head(3)["family"])
    all_warnings = [
        f"{row['model']} candidate {row['candidate_id']}: {warning}"
        for row in _read(output / "candidate_results.json")
        for warning in row["warnings"]
    ]
    all_warnings.extend(
        f"{row.model} final refit: {warning}"
        for row in comparison.itertuples() for warning in json.loads(row.refit_warnings)
    )
    warning_text = (
        "\n".join(f"- {warning}" for warning in all_warnings)
        if all_warnings else "No fitting warnings were recorded."
    )
    audit = dataset.manifest["split_audit"]
    text = f"""# MusicMap
## Music Genre Classification and Pattern Discovery Using Machine Learning

ML Laboratory Project

Student: [Enter your name]  
Roll number: [Enter your roll number]  
Department / institution: [Enter details]  
Course / instructor: [Enter details]

## Abstract

MusicMap investigates whether precomputed audio descriptors can distinguish music genres and reveal relationships between songs. The experiment uses {len(dataset.X):,} tracks from the balanced FMA-small subset, with eight genres and {dataset.X.shape[1]} numeric features. Decision Tree, K-nearest neighbors (KNN), radial-basis-function support vector machine (SVM), and a feed-forward artificial neural network (ANN) are compared with a majority-class baseline. The official artist-disjoint training, validation, and test partitions are preserved. Hyperparameters and the winning model are chosen using validation macro F1 before evaluating the test set. The selected model, {champion}, achieves {chosen.test_accuracy:.2%} test accuracy and {chosen.test_f1_macro:.4f} test macro F1 in this run. PCA, K-means, nearest-neighbor retrieval, and grouped permutation importance provide complementary views of musical similarity and feature relevance. The findings describe this dataset and experimental configuration rather than universal genre boundaries.

## 1. Problem statement and objectives

Genre labels can help organize large music collections, but genres share instruments, production styles, and musical conventions. The project treats genre recognition as supervised multiclass classification: one feature vector is mapped to one of eight top-level genre labels.

The objectives are:

1. Prepare a manageable, reproducible dataset using audio features rather than downloading or decoding thousands of recordings.
2. Compare Decision Tree, KNN, SVM, and ANN on identical external partitions and report more than accuracy alone.
3. Identify audio feature families associated with genre separation.
4. Discover similarities through dimensionality reduction, clustering, and nearest-neighbor retrieval.

The research questions are which model performs best under this protocol, which genres it confuses, and whether audio-based clusters agree with the supplied labels.

## 2. Dataset and features

The source is FMA: A Dataset for Music Analysis by Defferrard, Benzi, Vandergheynst, and Bresson (ISMIR 2017). FMA-small contains 8,000 30-second track excerpts, balanced across Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, and Rock. This project retains {len(dataset.X):,} tracks.

Only the official 342 MiB metadata archive is required. The 7.2 GiB FMA-small audio archive is unnecessary for these experiments. The published archive SHA-1 is verified as {dataset.manifest['source_sha1']}. The CSVs have multiple header rows; feature rows and metadata are joined by track ID, never by assumed row position.

{_table(split_rows)}

Each genre contributes 800 training, 100 validation, and 100 test tracks. The recorded overlap counts for artist IDs are {audit['artist_overlap_counts']}; missing artist IDs: {audit['missing_artist_ids']}. Thus this prepared release has no artists shared across partitions.

![Genre distribution and partitions](figures/genre_balance.png)

The 518 descriptors belong to 11 audio feature families. For each feature dimension, the dataset provides mean, standard deviation, skewness, kurtosis, median, minimum, and maximum over analysis frames. Predictors contain audio descriptors only. Titles, track IDs, artist IDs, album IDs, genre labels, and split labels are kept separately and never enter classifiers.

{_table(feature_rows)}

For example, "mfcc__mean__01" means the temporal mean of the first MFCC coefficient. The table has no explicit tempo/BPM feature; this project does not claim to measure tempo directly.

Data-quality checks found {dataset.manifest['nonfinite_values_replaced_with_nan']} non-finite feature values, {dataset.manifest['rows_needing_partial_imputation']} tracks requiring partial imputation, and {len(dataset.manifest['removed_all_missing_track_ids'])} completely missing feature rows. Stored features use float32 to reduce memory. Prepared files also have SHA-256 integrity checks.

## 3. Experimental methodology

### 3.1 Preprocessing and leakage prevention

Candidate fitting uses the 6,400 official training tracks. Each scikit-learn pipeline contains median imputation, removal of constant columns, standard scaling, and a classifier. Imputation and scaling are fitted inside the pipeline on training data. The transformation is z = (x − training mean) / training standard deviation. Scaling matters especially for distance-based KNN, SVM, and ANN; applying it to a tree is harmless.

The current subset has complete features, so imputation does not change it. The same implementation can handle partially missing values. Entirely unusable rows are explicitly recorded if encountered during preparation.

All 518 input descriptors are available to each classifier. PCA is used for exploratory analysis and similarity retrieval, not as classifier preprocessing. No feature ranking is used to select predictors after viewing test results.

### 3.2 Model comparison and selection

The "{run['profile']}" profile evaluates {run['candidate_count']} candidates including the baseline. In the standard profile, the grids are:

- Decision Tree: maximum depth in 10, 20, or unlimited; minimum leaf size in 2 or 5 (6 candidates).
- KNN: k in 5, 15, or 31; uniform or inverse-distance votes (6 candidates).
- RBF SVM: C in 1 or 10; gamma equal to the training-data scale heuristic or 0.001 (4 candidates).
- ANN: one 128-unit layer or two layers of 128 and 64 units; L2 alpha in 0.0001 or 0.01 (4 candidates).
- Dummy baseline: always predict the most frequent training label (1 candidate).

The ANN is a feed-forward multilayer perceptron trained with Adam, ReLU activation, batch size 128, initial learning rate 0.001, and at most 200 epochs. It uses an internal 10% holdout for early stopping, with patience of 15 epochs. That internal holdout is a random track split within the current fitting partition; the official external validation and test sets remain artist-disjoint. The ANN encodes genre strings numerically during fitting and decodes predictions back to text.

Each algorithm's hyperparameters are chosen by macro F1 on the official 800-track validation set. Validation accuracy breaks ties; an exact tie retains the first grid candidate. The winning algorithm is also chosen by validation macro F1. Selection is saved to "selection.json" before test features or labels are requested. Each selected configuration is then freshly fitted on the combined 7,200 training and validation tracks; ANN early stopping still reserves 10% internally. These refitted models are evaluated once on the 800-track test set. The baseline is reported for context and is excluded from the four-model winner selection.

There is no cross-validation in this baseline experiment. A fixed official validation partition provides an affordable laptop workflow. Candidate scores are available in "candidate_results.csv", and all final test predictions are saved in "test_predictions.csv".

### 3.3 Algorithms

- Decision Tree learns threshold rules that reduce Gini impurity. It captures nonlinear interactions and is interpretable, but deep trees can overfit.
- KNN assigns a label from nearby standardized feature vectors using Euclidean distance. It is simple but sensitive to scaling, redundant features, and high dimensionality.
- SVM learns separating boundaries in an RBF feature space. Its kernel is exp(−gamma × squared Euclidean distance); C controls the penalty for misclassification.
- ANN learns nonlinear combinations of descriptors through fully connected layers. Regularization and early stopping limit overfitting, but results can vary with initialization.

Selected hyperparameters:

{_table(selected_rows)}

### 3.4 Evaluation metrics

Accuracy is the fraction of correctly classified tracks. For a genre, precision is TP / (TP + FP), recall is TP / (TP + FN), and F1 is 2 × precision × recall / (precision + recall). Macro F1 averages per-genre F1 with equal weight for every genre. Balanced accuracy is mean recall. Undefined precision or F1 is assigned zero.

The test partition is balanced, so balanced accuracy equals accuracy here. Confusion matrices show which genres are mistaken for which others. Validation scores measure training-only candidates; test scores measure their refitted versions, so the columns refer to different fitted models on different partitions.

## 4. Results

The algorithm selected before test evaluation is **{champion}**. Its test accuracy is **{chosen.test_accuracy:.2%}** and macro F1 is **{chosen.test_f1_macro:.4f}**. The dummy baseline reaches {baseline.test_accuracy:.2%} accuracy and {baseline.test_f1_macro:.4f} macro F1.

{_table(score_rows)}

![Comparison of all models](figures/model_comparison.png)

The model ordering follows validation macro F1. A higher test score from another model would be a descriptive observation, not a reason to change the preselected winner. These are actual measurements from the included run; scores from a different machine or dependency version may differ slightly.

### 4.1 Per-genre performance

{_table(genre_rows)}

![Selected model per-genre F1](figures/per_genre_f1.png)

The largest off-diagonal confusion counts for the selected model are:

{_table(confusions)}

![All four confusion matrices](figures/confusion_matrices.png)

Each confusion-matrix row is normalized by its actual genre count. These errors show overlap in the supplied acoustic representation; they do not prove that a genre label is wrong.

### 4.2 Computation and reproducibility

{_table(timing_rows)}

The full tuning/refitting/evaluation stage took {run['elapsed_seconds']:.1f} seconds in the execution environment, using a limit of {run['threads']} numerical-library threads and seed {run['seed']}. Times exclude downloading, preparation, visualization, and report generation. Candidate fitting time sums fitting only; prediction and other overhead account for the remaining elapsed time. This timing is not a laptop performance guarantee.

The run used Python {run['python']} and scikit-learn {run['versions']['scikit-learn']}. Exact direct dependency versions are recorded in "requirements-tested.txt"; runtime details are in "run_info.json". A fixed seed improves repeatability but does not establish robustness across random seeds.

Fitting diagnostics:

{warning_text}

## 5. Feature interpretation

The selected training-only model is evaluated on validation tracks. One entire feature family is jointly shuffled across tracks while all other families remain unchanged. The drop in macro F1 estimates how much this fitted model relies on that family. Joint shuffling preserves correlations among columns within the family. Each family uses {discovery['permutation_repeats']} repeated permutations with a fixed seed.

{_table(importance_rows)}

![Feature-family permutation importance](figures/feature_importance.png)

The leading families in this run are {top_families}. The audio meanings of these families are listed in Section 2. Positive drops indicate reliance; near-zero or negative drops can reflect redundancy, sampling variation, or features that the model does not use effectively. The error bars show variability across permutations, not confidence intervals across new datasets. Family sizes differ, and correlated families can substitute for one another, so the ranking is neither causal nor an equal-size feature comparison.

Individual descriptors can also be ranked by a training-only one-way ANOVA F score:

{_table(associations[['feature', 'training_anova_f']].rename(columns={'feature': 'Descriptor', 'training_anova_f': 'Training ANOVA F'}))}

This ranking is an exploratory association measure. It ignores multivariate interactions and correlated predictors, and it is not a multiple-testing-adjusted significance claim or a feature-selection step.

![Training feature distributions](figures/feature_distributions.png)

Boxes show medians and interquartile ranges. Outlier points are omitted only from the drawing; they remain in the training data. Overlapping distributions explain why no single simple descriptor is expected to identify every genre.

## 6. Pattern discovery and song similarity

### 6.1 PCA music map

Preprocessing and PCA are fitted only on training audio features. Training and validation tracks are projected into that space; genre labels are added only as display colors. The first two principal components retain {discovery['pca_2d_explained_variance']:.2%} of standardized feature variance, and the first {discovery['pca_components']} retain {discovery['pca_50d_explained_variance']:.2%}.

![PCA song map](figures/pca_map.png)

![PCA cumulative variance](figures/pca_variance.png)

Open "song_map.html" for the offline interactive map: hover for track details, filter genres through the legend, and zoom. Nearby points share aspects of their measured audio descriptors. Since two dimensions discard information, separation or overlap in this drawing is not a classifier accuracy result.

### 6.2 K-means clustering

K-means uses the training-set {discovery['pca_components']}-dimensional PCA representation and k = {discovery['kmeans_k']}, fixed in advance to match the number of known genres. It does not use genre labels during fitting. Validation tracks receive the nearest learned cluster centroid.

- Adjusted Rand index against validation genres: {discovery['kmeans_adjusted_rand_index']:.4f}.
- Normalized mutual information against validation genres: {discovery['kmeans_normalized_mutual_information']:.4f}.
- Validation silhouette score in PCA space: {discovery['kmeans_silhouette'] if discovery['kmeans_silhouette'] is not None else 'undefined'}.

ARI measures agreement after accounting for chance, NMI measures shared label information, and silhouette measures geometric compactness/separation. Cluster numbers are arbitrary and cannot be interpreted as genre IDs. A cluster can contain multiple genres, and one genre can span multiple clusters.

![Validation cluster composition](figures/cluster_composition.png)

The validation metrics and composition chart are descriptive; they are not used to tune k or select a classifier. Test tracks are excluded from discovery analyses.

### 6.3 Nearest songs

Euclidean distance in the training-fitted {discovery['pca_components']}-dimensional PCA space retrieves similar tracks from the training catalog. The query itself and tracks with the same artist ID are excluded by default. Genre labels are shown for interpretation and are not part of the distance.

For validation query ID {query_id}, "{dataset.metadata.at[query_id, 'title']}" by {dataset.metadata.at[query_id, 'artist']} ({dataset.y.at[query_id]}), the five nearest eligible training tracks are:

{_table(neighbor_table)}

Distances are representation-specific values, not probabilities or listener-preference scores. This is an audio similarity demonstration rather than a personalized recommendation system.

## 7. Discussion and limitations

1. The four algorithms receive the same external partitions, but their small hyperparameter grids do not establish the best possible performance for each model family.
2. Artist-disjoint splits reduce artist memorization. They do not remove every potential recording, production, album, or dataset-specific correlation.
3. Genre boundaries are subjective and sometimes overlapping. This task uses one top-level label per track and only eight FMA-small genres.
4. Summary descriptors discard musical sequence and detailed temporal structure. The model does not directly process waveforms or learn from spectrograms.
5. The ANN uses an internal track-level early-stopping split, one initialization seed, and at most 200 epochs. Repeated artist-aware experiments could change the comparison.
6. Standard scaling can remain sensitive to extreme values. PCA maximizes variance rather than genre discrimination, and the 2D map discards most dimensions.
7. Grouped permutation importance describes reliance of one tuned model on one validation set. The validation partition has already influenced model selection, so it is not an independent confirmatory interpretation dataset.
8. Results come from a single official holdout protocol. A larger evaluation or repeated group-aware validation would better quantify uncertainty.
9. Predicting a genuinely new audio file requires compatible extraction of the same descriptors. The delivered predictor accepts prepared FMA feature rows or a matching flat feature CSV; raw MP3 feature extraction is outside this implementation.

## 8. Conclusion and future work

MusicMap provides a reproducible comparison of Decision Tree, KNN, SVM, and ANN for feature-based music genre classification. Under the recorded protocol, {champion} was selected on validation data and obtained {chosen.test_accuracy:.2%} test accuracy with {chosen.test_f1_macro:.4f} macro F1. The feature analysis highlights {top_families}, while PCA, K-means, and nearest-neighbor retrieval reveal relationships that are only partly aligned with genre labels.

Future experiments can use artist-grouped cross-validation within the training partition, multiple ANN seeds, carefully validated feature selection, broader genres, or pretrained audio embeddings. Any such changes should be selected without reusing this test set for tuning.

## 9. Reproduction and deliverables

From the project directory, install the requirements and run:

~~~sh
python -m pip install -r requirements-tested.txt
musicmap run --profile standard --seed 42 --threads 2
jupyter lab notebooks/MusicMap.ipynb
~~~

The notebook defaults to displaying the included results. Set its RUN_TRAINING and RUN_DISCOVERY switches to True to recompute. Prepared subset data are bundled in the complete package; if absent, preparation downloads the official metadata archive automatically. No audio download or GPU is required.

Deliverables include the Python package, executed notebook, metric tables, all test predictions, selected model, interactive map, figures, this report in Markdown/HTML/Word, a presentation outline, and viva preparation notes. Fill in the student and institution fields and adapt the report to the instructor's required format.

## References and attribution

1. Defferrard, M., Benzi, K., Vandergheynst, P., and Bresson, X. (2017). FMA: A Dataset for Music Analysis. Proceedings of the 18th International Society for Music Information Retrieval Conference. https://arxiv.org/abs/1612.01840
2. Official FMA repository and release documentation: https://github.com/mdeff/fma
3. Official audio descriptor extraction implementation: https://github.com/mdeff/fma/blob/master/features.py
4. Scikit-learn user guide and API documentation: https://scikit-learn.org/stable/

FMA metadata and derived feature data are attributed to Defferrard et al. and distributed under CC BY 4.0: https://creativecommons.org/licenses/by/4.0/. This project uses the original 2017 metadata release and does not redistribute audio recordings.
"""
    if run.get("profile") == "improved":
        text = _improved_report(text, output, comparison, selection, run)
    report = output / "MusicMap_Report.md"
    report.write_text(text, encoding="utf-8")
    renderer = mistune.create_markdown(plugins=["table"])
    body = renderer(text)
    html = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MusicMap · ML lab report</title><style>
body{font:16px/1.65 system-ui,sans-serif;color:#183143;background:#f0f4f5;margin:0}
main{max-width:1080px;margin:40px auto;padding:50px;background:white;border-radius:16px}
h1{font-size:48px;letter-spacing:-2px;margin:0;color:#255f85}
h2{font-size:25px;color:#255f85;margin-top:44px;line-height:1.3}
h3{font-size:20px;margin-top:32px}a{color:#255f85}
table{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow:auto}
th,td{padding:9px 12px;border-bottom:1px solid #d8e2e7;text-align:left}
th{background:#edf4f7}tr:nth-child(even){background:#f8fafb}
img{max-width:100%;height:auto;margin:12px 0}
pre{background:#f0f4f5;padding:20px;overflow:auto;border-radius:8px}
li{margin:6px 0}@media(max-width:700px){main{margin:0;padding:22px}h1{font-size:38px}}
@media print{body{background:white}main{padding:0;margin:0}h2,h3{break-after:avoid}img,table{break-inside:avoid}}
</style></head><body><main>""" + body + "</main></body></html>"
    (output / "MusicMap_Report.html").write_text(html, encoding="utf-8")
    _write_docx(text, output)
    _write_presentation(champion, chosen, discovery, importance, output, run)
    print(f"Wrote report: {report}", flush=True)
    return report


def _improved_report(text: str, output: Path, comparison: pd.DataFrame, selection: dict, run: dict) -> str:
    """Replace baseline-specific prose with the recorded improved protocol."""
    champion = selection["champion"]
    chosen = comparison.loc[comparison["model"].eq(champion)].iloc[0]
    cv = pd.read_csv(output / "cv_results.csv")
    seeds = pd.read_csv(output / "ann_seed_results.csv")
    cv_columns = [name for name in ("model", "cv_mean_f1", "cv_std_f1", "parameters") if name in cv]
    cv_best = cv.sort_values(
        ["cv_mean_f1", "cv_std_f1"], ascending=[False, True], kind="stable"
    ).groupby("model", sort=False).head(1)
    baseline_file = output.parent / "baseline/reports/model_comparison.csv"
    baseline_comparison = ""
    if baseline_file.exists():
        original = pd.read_csv(baseline_file)
        original_selection = _read(baseline_file.parent / "selection.json")
        old = original.loc[original["model"].eq(original_selection["champion"])].iloc[0]
        rows = [{
            "Run": "Original standard profile", "Selected model": old.model,
            "Validation macro F1": old.validation_f1_macro, "Test accuracy": old.test_accuracy,
            "Test macro F1": old.test_f1_macro,
        }, {
            "Run": "Improved experiment", "Selected model": champion,
            "Validation macro F1": chosen.validation_f1_macro, "Test accuracy": chosen.test_accuracy,
            "Test macro F1": chosen.test_f1_macro,
        }]
        delta = 100 * (chosen.test_accuracy - old.test_accuracy)
        baseline_comparison = (
            "\n### 4.1 Comparison with the preserved original run\n\n" + _table(rows) +
            f"\n\nThe selected model's test accuracy changes by {delta:+.2f} percentage points, "
            f"and macro F1 changes by {chosen.test_f1_macro - old.test_f1_macro:+.4f}. "
            "The original reports, predictions, and models are preserved under baseline/. "
            "This is an extended benchmark comparison: the official test performance was already "
            "seen during the original project. It is not an independent, newly unseen confirmation "
            "of the improvement. A fresh artist-disjoint dataset would provide stronger confirmation.\n"
            "\n![Original and improved test accuracy by model](figures/improvement_comparison.png)\n"
        )
        original_by_model = original.set_index("model")
        changes = []
        for row in comparison.itertuples():
            before = original_by_model.loc[row.model] if row.model in original_by_model.index else None
            changes.append({
                "Model": row.model,
                "Original test accuracy": f"{before.test_accuracy:.2%}" if before is not None else "New benchmark",
                "Updated test accuracy": f"{row.test_accuracy:.2%}",
                "Change (percentage points)": f"{100 * (row.test_accuracy - before.test_accuracy):+.3f}" if before is not None else "—",
                "Updated test macro F1": f"{row.test_f1_macro:.4f}",
            })
        highest_test = comparison.loc[comparison["model"].ne(BASELINE)].sort_values(
            "test_accuracy", ascending=False
        ).iloc[0]
        baseline_comparison += (
            "\nSame-family comparisons:\n\n" + _table(changes) +
            f"\n\nThe validation-selected model is {champion}; the highest observed updated test accuracy "
            f"is {highest_test.test_accuracy:.2%} from {highest_test.model}. Test performance is descriptive "
            "and does not change the predeclared validation-based selection. Broader tuning improves some "
            "same-family scores in this run, but it does not establish a general improvement of the "
            "selection procedure. Small differences on 800 tracks do not by themselves demonstrate "
            "statistically significant gains.\n"
        )
    folds = run.get("cv_folds", selection.get("cv_folds", 3))
    seed = run["seed"]
    seed_names = ", ".join(str(value) for value in run.get("ann_seeds", [seed, seed + 1, seed + 2]))
    outlier_note = ""
    outlier_file = output / "training_outlier_summary.json"
    if outlier_file.exists():
        outliers = _read(outlier_file)
        outlier_note = (
            f"Training-only diagnostics found {outliers['features_with_values_above_20_sigma']} "
            f"descriptors with an absolute standardized value above 20, with maximum absolute z-score "
            f"{outliers['max_abs_z']:.2f}. These observations motivated testing quantile normalization. "
            "A quantile transform changes distances and can discard useful scale information; it is "
            "not universally beneficial. Its settings are chosen using training CV, and this diagnostic "
            "does not use validation or test rows."
        )
    methodology = f"""## 3. Experimental methodology

### 3.1 Preprocessing and candidate representations

The official 6,400-track training partition is the only source of cross-validation data. Every fitted pipeline learns median imputation, constant-column removal, scaling, and any feature reduction from the current fitting fold. Fold-assessment features and labels are not used to fit those transformations.

The improved experiment compares the full descriptor representation with training-fitted feature selection and PCA. ANOVA selection ranks individual descriptors by their association with the training labels; PCA learns directions of high transformed-feature variance without labels. The candidate search also compares standard scaling with a quantile-to-normal transformation to handle skewed feature distributions and extreme values. These transformations are fitted within each fold. This supervised candidate search is separate from the fixed standardized 50-component PCA representation used for the similarity map and clustering.

{outlier_note}

### 3.2 Artist-grouped cross-validation and selection

The search evaluates {len(cv)} recorded configurations with {folds}-fold artist-grouped cross-validation inside the official training partition. The fold builder preserves artist groups: no artist occurs in both fitting and assessment tracks in a fold. "cv_splits.json" records the assignments and audits. Each candidate's mean macro F1 across these folds is the tuning criterion; its standard deviation describes fold-to-fold variability, not a confidence interval.

For each model family, the configuration with the highest mean cross-validation macro F1 advances to the official 800-track validation partition. The final winning family is chosen using validation macro F1, with the recorded tie-breaking rules in "selection.json". Validation is outside the cross-validation folds. The official test set contributes no features or labels to this run's search or selection.

The improved search broadens SVM regularization/kernel settings, compares feature representations, tunes ANN size/learning rate/regularization, and adds Extra Trees as an ensemble benchmark. The four original required algorithms remain present. "cv_results.csv" contains all candidate settings and scores; the following summarizes the best cross-validation row per family:

{_table(cv_best[cv_columns])}

![Selected models in artist-grouped cross-validation](figures/grouped_cv.png)

The CV chart uses the selected family configurations and displays fold standard deviations. For ANN, it shows the fixed three-seed ensemble's CV score; the table above shows the base-seed architecture search.

The ANN is a feed-forward network operating on descriptors. The architecture, learning rate, and regularization grid is tuned using base seed {seed}, with at most 300 epochs and early-stopping patience of 20 epochs. Its selected configuration is evaluated with seeds {seed_names}, and the final ANN uses a fixed average of all three models' class probabilities. No best-performing seed is cherry-picked. The individual and ensemble scores are retained in "ann_seed_results.csv"; the base-seed tuning scores and final ensemble CV score answer different questions. The numeric label encoder converts predictions back to readable genres. Each network uses an internal random 10% track holdout for early stopping, within its current fitting partition; that internal split is distinct from the artist-disjoint outer folds and official validation/test partitions.

{_table(seeds)}

Selected settings for each family:

{_table([{"Model": name, "Settings": json.dumps(row["parameters"])} for name, row in selection["selected"].items()])}

### 3.3 Refit and extended test comparison

The complete selection is saved before this run requests test features or labels. Selected configurations are freshly fitted on the combined 7,200 training and validation tracks. Their predictions on the 800 official test tracks are recorded once for this run. The dummy model is a contextual baseline and cannot win the trained-model selection.

The same official test set was already viewed in the original experiment. The updated scores therefore provide an extended benchmark comparison, not a pristine new confirmatory holdout. Decisions in the implemented search use training folds and validation scores; a fresh artist-disjoint confirmation set would strengthen claims about generalization gains.

### 3.4 Algorithms and evaluation

- Decision Tree learns threshold rules that reduce class impurity.
- KNN uses neighboring transformed feature representations and inverse-distance votes.
- RBF SVM learns a nonlinear separating boundary; C and gamma control regularization and kernel locality.
- ANN learns nonlinear combinations of descriptors using fully connected layers and regularization.
- Extra Trees averages randomized decision trees to reduce sensitivity to a single fitted tree.

Accuracy is correct labels divided by track count. Per-genre precision is TP / (TP + FP), recall is TP / (TP + FN), and F1 is their harmonic mean. Macro F1 gives all eight genres equal weight. Undefined scores are assigned zero. With balanced test genres, balanced accuracy equals accuracy.

Cross-validation scores describe fold-fitted candidates. Validation scores describe training-partition fits, and test scores describe models refitted on training plus validation. They concern different fitted models and different samples.

"""
    before, remainder = text.split("## 3. Experimental methodology", 1)
    _, after = remainder.split("## 4. Results", 1)
    text = before + methodology + "## 4. Results" + after
    text = text.replace(
        "and a feed-forward artificial neural network (ANN) are compared with a majority-class baseline.",
        "and a feed-forward artificial neural network (ANN) are compared with Extra Trees and a majority-class baseline.",
    ).replace(
        "Hyperparameters and the winning model are chosen using validation macro F1 before evaluating the test set.",
        "Hyperparameters are tuned using artist-grouped cross-validation within training, and the winning family is chosen using validation macro F1 before an extended comparison on the previously viewed official test set.",
    )
    text = text.replace("### 4.1 Per-genre performance", baseline_comparison + "\n### 4.2 Per-genre performance")
    text = text.replace("### 4.2 Computation and reproducibility", "### 4.3 Computation and reproducibility")
    text = text.replace("![All four confusion matrices]", "![Confusion matrices for the trained models]")
    text = text.replace(
        "Candidate fitting time sums fitting only; prediction and other overhead account for the remaining elapsed time.",
        "Candidate fitting time sums primary candidate cross-validation fits only. It excludes the additional ANN seed diagnostics and full-training fits; those stages, predictions, and overhead are included in the total elapsed time.",
    )
    text = text.replace(
        "Compare Decision Tree, KNN, SVM, and ANN on identical external partitions",
        "Compare Decision Tree, KNN, SVM, ANN, and Extra Trees on identical external partitions",
    ).replace(
        "MusicMap provides a reproducible comparison of Decision Tree, KNN, SVM, and ANN for feature-based music genre classification.",
        "MusicMap provides a reproducible comparison of Decision Tree, KNN, SVM, ANN, and Extra Trees for feature-based music genre classification.",
    )
    text = text.replace(
        "This ranking is an exploratory association measure. It ignores multivariate interactions and correlated predictors, and it is not a multiple-testing-adjusted significance claim or a feature-selection step.",
        "This displayed full-training ranking is an exploratory association measure. It ignores multivariate interactions and correlated predictors and is not a multiple-testing-adjusted significance claim. Any ANOVA selector in a supervised candidate is separately fitted within its own training fold; the displayed ranking is not reused across CV folds.",
    )
    text = text.replace(
        "1. The four algorithms receive the same external partitions, but their small hyperparameter grids do not establish the best possible performance for each model family.",
        "1. The five trained algorithms receive the same external partitions and artist-grouped fold assignments. The finite candidate search still does not establish the best possible performance for each model family.",
    ).replace(
        "5. The ANN uses an internal track-level early-stopping split, one initialization seed, and at most 200 epochs. Repeated artist-aware experiments could change the comparison.",
        "5. Multiple ANN initialization seeds are examined, but this limited seed set does not fully characterize optimization variability. Consult the recorded ANN settings and seed results rather than treating one fit as universal.",
    ).replace(
        "8. Results come from a single official holdout protocol. A larger evaluation or repeated group-aware validation would better quantify uncertainty.",
        "8. The official test set was already viewed before this improvement study. Updated scores are an extended benchmark comparison and require a fresh artist-disjoint confirmation set for stronger generalization claims.",
    ).replace(
        "Future experiments can use artist-grouped cross-validation within the training partition, multiple ANN seeds, carefully validated feature selection, broader genres, or pretrained audio embeddings. Any such changes should be selected without reusing this test set for tuning.",
        "Artist-grouped cross-validation, feature reduction, multiple ANN seeds, and an ensemble benchmark are now implemented. Further work can use more genres, pretrained audio embeddings, and a fresh artist-disjoint confirmation set. Future decisions should use development data rather than the reported test scores.",
    ).replace(
        "musicmap run --profile standard --seed 42 --threads 2",
        "musicmap improve --seed 42 --threads 2 --cv-folds 3",
    )
    return text


def _write_docx(markdown: str, output: Path) -> None:
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor

    document = Document()
    section = document.sections[0]
    section.top_margin = section.bottom_margin = Inches(0.75)
    section.left_margin = section.right_margin = Inches(0.8)
    document.styles["Normal"].font.name = "Calibri"
    document.styles["Normal"].font.size = Pt(10)
    for name in ("Title", "Heading 1", "Heading 2", "Heading 3"):
        document.styles[name].font.color.rgb = RGBColor.from_string("255F85")
    section.footer.paragraphs[0].text = "MusicMap | Machine Learning Laboratory Project"

    def plain(tokens):
        result = ""
        for token in tokens:
            if token["type"] in ("text", "codespan"):
                result += token.get("raw", "")
            elif token["type"] in ("softbreak", "linebreak"):
                result += "\n"
            elif token["type"] == "link":
                label = plain(token.get("children", []))
                url = token.get("attrs", {}).get("url", "")
                result += label if label == url else f"{label} ({url})"
            else:
                result += plain(token.get("children", []))
        return result

    def add_blocks(tokens, list_style=None):
        for token in tokens:
            kind = token["type"]
            children = token.get("children", [])
            if kind == "heading":
                level = token["attrs"]["level"]
                document.add_heading(plain(children), level=max(0, level - 1))
            elif kind in ("paragraph", "block_text"):
                images = [child for child in children if child["type"] == "image"]
                if images:
                    for child in images:
                        path = output / child["attrs"]["url"]
                        if path.exists():
                            document.add_picture(str(path), width=Inches(6.5))
                            document.add_paragraph(plain(child.get("children", [])), style="Caption")
                else:
                    document.add_paragraph(plain(children), style=list_style)
            elif kind == "list":
                style = "List Number" if token.get("attrs", {}).get("ordered") else "List Bullet"
                for item in children:
                    add_blocks(item.get("children", []), style)
            elif kind == "block_code":
                paragraph = document.add_paragraph()
                run = paragraph.add_run(token.get("raw", ""))
                run.font.name = "Consolas"
                run.font.size = Pt(9)
            elif kind == "table":
                rows = []
                for part in children:
                    if part["type"] == "table_head":
                        rows.append([plain(cell.get("children", [])) for cell in part["children"]])
                    elif part["type"] == "table_body":
                        for row in part["children"]:
                            rows.append([plain(cell.get("children", [])) for cell in row["children"]])
                if rows:
                    table = document.add_table(rows=1, cols=len(rows[0]))
                    table.style = "Light Shading Accent 1"
                    for i, value in enumerate(rows[0]):
                        table.rows[0].cells[i].text = value
                    for row in rows[1:]:
                        cells = table.add_row().cells
                        for i, value in enumerate(row):
                            cells[i].text = value
                    for row in table.rows:
                        for cell in row.cells:
                            for paragraph in cell.paragraphs:
                                for run in paragraph.runs:
                                    run.font.size = Pt(8)
                    document.add_paragraph()

    parser = mistune.create_markdown(renderer="ast", plugins=["table"])
    add_blocks(parser(markdown))
    document.save(output / "MusicMap_Report.docx")


def _write_presentation(champion, chosen, discovery, importance, output: Path, run: dict | None = None) -> None:
    slides = f"""# MusicMap presentation outline

Use these ten slides for a 7–10 minute lab presentation. Add your name, roll number, and institution.

## Slide 1 — Title and problem
- MusicMap: Music Genre Classification and Pattern Discovery Using Machine Learning.
- Can audio descriptors predict a track's genre and reveal similar songs?

## Slide 2 — Dataset
- FMA-small: 8,000 tracks, eight balanced genres, 518 descriptors.
- Precomputed features; no audio download or GPU.
- Show figures/genre_balance.png and cite Defferrard et al. (2017).

## Slide 3 — Experimental design
- Official artist-disjoint split: 6,400 training / 800 validation / 800 test.
- Fit preprocessing on training; choose on validation; refit on 7,200; evaluate on test.
- Explain why artist leakage would inflate a random track split.

## Slide 4 — Models
- Decision Tree: threshold rules.
- KNN: nearby standardized feature vectors.
- SVM: nonlinear RBF boundary.
- ANN: feed-forward ReLU network with early stopping.
- Include the majority-class baseline.

## Slide 5 — Evaluation and results
- Macro F1 chooses the winner; also report accuracy and per-genre scores.
- Validation-selected model: {champion}.
- Test accuracy: {chosen.test_accuracy:.2%}; macro F1: {chosen.test_f1_macro:.4f}.
- Show figures/model_comparison.png.

## Slide 6 — Errors
- Show figures/confusion_matrices.png or figures/per_genre_f1.png.
- Explain two observed genre confusions using the report's counts.
- Distinguish measured errors from speculation about their causes.

## Slide 7 — Feature interpretation
- Jointly shuffle feature families on validation data.
- Highest three mean drops: {', '.join(importance.head(3)['family'])}.
- Show figures/feature_importance.png.
- Importance describes model reliance, not musical causality.

## Slide 8 — Pattern discovery
- PCA fitted on training features; first two PCs retain {discovery['pca_2d_explained_variance']:.2%} variance.
- K-means fitted without genres; validation ARI: {discovery['kmeans_adjusted_rand_index']:.4f}.
- Show figures/pca_map.png and figures/cluster_composition.png.

## Slide 9 — Demonstration
- Open song_map.html; hover, zoom, and filter the genre legend.
- Run musicmap neighbors --track-id [a valid FMA-small ID] --k 5.
- Show that neighbors use audio descriptors and a training-only catalog.

## Slide 10 — Conclusions and limits
- State the measured winner and the strongest feature families.
- Eight genres, one fixed split, one ANN seed, modest tuning.
- New raw audio requires compatible feature extraction.
- Future work: grouped cross-validation, repeated seeds, pretrained embeddings.
"""
    if run and run.get("profile") == "improved":
        slides = slides.replace(
            "- Fit preprocessing on training; choose on validation; refit on 7,200; evaluate on test.",
            "- Tune with artist-grouped cross-validation inside training; choose the algorithm on validation.\n"
            "- Refit on training + validation; report an extended comparison on the previously viewed test set.",
        ).replace("- Include the majority-class baseline.",
                  "- Extra Trees supplies an ensemble benchmark; include the majority-class baseline.")
        slides = slides.replace("- Eight genres, one fixed split, one ANN seed, modest tuning.",
                                "- Eight genres, finite search, several ANN seeds, and one previously viewed official test set.")
        slides = slides.replace("- Future work: grouped cross-validation, repeated seeds, pretrained embeddings.",
                                "- Future work: a fresh confirmation set, more genres, and pretrained embeddings.")
    (output / "presentation_outline.md").write_text(slides, encoding="utf-8")
