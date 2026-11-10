# MusicMap

**Music Genre Classification and Pattern Discovery Using Machine Learning**

A complete ML lab project using FMA-small and precomputed audio features. It compares Decision Tree, KNN, SVM, an ANN ensemble, and Extra Trees, then explores song similarity with PCA, K-means, and feature-family permutation importance. Python/scikit-learn runs on a CPU; TensorFlow, audio decoding, and a GPU are unnecessary.

Start with [the executed notebook](notebooks/MusicMap.ipynb) or [the lab report](reports/MusicMap_Report.html). An editable [Word report](reports/MusicMap_Report.docx), [presentation outline](reports/presentation_outline.md), and [viva questions](docs/VIVA.md) are included.

## Improved experiment

The improved experiment implements the recommended changes:

- Three-fold artist-grouped cross-validation inside the 6,400 official training tracks.
- A broader SVM search, including regularization, kernel settings, and standard or quantile-to-normal scaling.
- Full descriptors compared with training-fitted ANOVA feature selection and PCA.
- ANN architecture, learning-rate, and regularization tuning, followed by seeds 42/43/44 and a fixed mean-probability ensemble.
- Extra Trees as an additional benchmark, retaining all four required algorithms and the dummy baseline.

One CV-winning configuration per model family advances to the separate 800-track validation set. The winning family is selected by validation macro F1, then selected configurations are refitted on training plus validation. The updated test results are an **extended benchmark comparison**: the official test set was already viewed during the original project. A fresh artist-disjoint confirmation set would strengthen claims about improvement.

The original reports and models are preserved in baseline/. The original SVM achieved 49.125% test accuracy and approximately 0.4885 macro F1. See the updated report and reports/model_comparison.csv for the active measurements, reports/cv_results.csv for all tuning scores, and reports/ann_seed_results.csv for ANN variability.

<!-- RESULTS START -->
### Recorded results

| Model          | Original test accuracy   | Updated test accuracy   | Change (percentage points)   |   Validation macro F1 |   Updated test macro F1 |
|:---------------|:-------------------------|:------------------------|:-----------------------------|----------------------:|------------------------:|
| ANN (MLP)      | 45.875%                  | 47.250%                 | +1.375                       |                0.5613 |                  0.4695 |
| SVM (RBF)      | 49.125%                  | 49.750%                 | +0.625                       |                0.5575 |                  0.4925 |
| Extra Trees    | New benchmark            | 48.125%                 | —                            |                0.543  |                  0.476  |
| KNN            | 33.375%                  | 38.250%                 | +4.875                       |                0.4944 |                  0.3674 |
| Decision Tree  | 34.250%                  | 34.250%                 | +0.000                       |                0.3656 |                  0.3437 |
| Dummy baseline | 12.500%                  | 12.500%                 | +0.000                       |                0.0278 |                  0.0278 |

**The validation-selected model is ANN (MLP)**: validation macro F1 0.5613, test accuracy 47.25%, and test macro F1 0.4695. This is **1.875 percentage points below** the original validation-selected SVM's test accuracy. The recommended changes strengthened the experiment, but did not improve the final selection procedure's observed test score.

The updated **SVM** has the highest observed test accuracy, **49.75%** (macro F1 **0.4925**), compared with its original 49.125%: a **0.625 percentage point** increase. KNN and ANN improve relative to their own original versions. These modest differences on 800 test tracks do not establish statistically significant gains, and the test ranking is not used to replace the validation-selected winner.

The full search/refit/evaluation took about **19.7 minutes with two numerical-library threads** in the execution environment, excluding data preparation and discovery/report generation. Laptop runtime varies. The ANN is a fixed probability average of seeds 42, 43, and 44, not a best-seed selection.
<!-- RESULTS END -->

## Setup

Python **3.12** is recommended and was used for the included run; the source supports Python 3.11+. Extract the complete project ZIP, open a terminal in the extracted musicmap directory, and create a virtual environment.

~~~sh
python -m venv .venv
~~~

Activate it on macOS/Linux:

~~~sh
source .venv/bin/activate
~~~

Or in Windows PowerShell:

~~~powershell
.\.venv\Scripts\Activate.ps1
~~~

Install the exact direct dependency versions used in this project:

~~~sh
python -m pip install -r requirements-tested.txt
~~~

If you prefer compatible versions selected for your Python installation, use requirements.txt instead. Saved scikit-learn models should be loaded with the tested scikit-learn version.

## Review the completed project

~~~sh
jupyter lab notebooks/MusicMap.ipynb
~~~

The notebook shows the included results by default. Its RUN_TRAINING and RUN_DISCOVERY switches control whether expensive stages are recomputed. It explains each stage and reads the same reusable Python modules as the CLI.

The smaller MusicMap_Improved_Code_and_Results ZIP omits data and fitted models. To execute its notebook, first run musicmap improve or set RUN_TRAINING = True; its exported HTML results can be viewed immediately.

You can open reports/MusicMap_Report.html or reports/song_map.html directly in a browser. The interactive map works offline because its JavaScript is included. The Word report can be edited in Word or LibreOffice.

## Reproduce from start to finish

~~~sh
musicmap improve --seed 42 --threads 2 --cv-folds 3
~~~

This verifies/prepares data, runs artist-grouped tuning, refits and evaluates models, generates discovery plots, and rebuilds Markdown/HTML/Word reports. It replaces the active results and preserves the original baseline. The full improved grid has 94 configurations: Decision Tree 6, KNN 17, SVM 52, ANN 14, Extra Trees 4, and dummy 1. Each has three CV fits, with additional ANN seed diagnostics and final refits; runtime varies by laptop. A GPU is unnecessary.

The complete package includes the prepared 8,000 × 518 feature subset, so reproduction can work without downloading data. For a source-only checkout, preparation downloads the official **342 MiB metadata archive** automatically and streams only selected rows from the compressed CSVs. Allow about 2 GiB of working disk space plus the Python environment. The grouped-CV search temporarily caches fitted preprocessing, which may use 0.5–1 GiB and is removed when the run finishes. Memory is bounded by chunked CSV reading and the small model dataset. Do not download fma_small.zip; its 7.2 GiB of audio are not needed.

To rebuild preparation from the original archive:

~~~sh
musicmap prepare --force
~~~

Or use an already downloaded official archive:

~~~sh
musicmap prepare --archive /path/to/fma_metadata.zip --force
~~~

The official SHA-1 is verified before reading the archive. The prepared features and metadata have SHA-256 checksums recorded in data/processed/manifest.json.

## Run individual stages

~~~sh
musicmap prepare
musicmap improve --seed 42 --threads 2 --cv-folds 3
musicmap analyze --repeats 5 --threads 2
musicmap report
python -m pytest -q
~~~

A quick-profile run uses one configuration per algorithm on the same complete partitions:

~~~sh
musicmap run --profile quick --seed 42 --threads 2
~~~

Quick and standard runs reproduce the original simpler protocol and replace the active results. Use musicmap improve for the updated experiment; baseline/ preserves the original completed run.

All commands assume the project root is the current directory. From another directory, use the global option before the subcommand: musicmap --root /path/to/musicmap report. Python module invocation is also supported: python -m musicmap report.

## Demonstrate predictions and similarity

Use a track ID from data/processed/tracks.csv, the notebook, or the interactive map:

~~~sh
musicmap predict --track-id 2
musicmap neighbors --track-id 2 --k 5
~~~

Prediction prints the final model's label, actual dataset label, and official partition. A prediction for a training/validation track is in-sample for the final refit; use the recorded test metrics when discussing generalization. The neighbors command searches only the training catalog in a training-fitted 50-dimensional PCA space and excludes the same artist by default. Add --allow-same-artist to change that filter.

For new, compatible audio descriptors, use a flat CSV with track_id as its first column and the exact 518 feature names shown in the prepared dataset:

~~~sh
musicmap predict --features-csv my_audio_features.csv --output predictions.csv
~~~

This project uses precomputed descriptors. It does **not** extract those descriptors from an arbitrary MP3. Modern librosa defaults may differ from the original FMA extraction code, so raw-audio support requires a separately validated extraction pipeline.

## Methodology

- Preserve the official artist-disjoint split; join features and metadata by ID.
- Keep genre and all non-audio metadata out of the predictor matrix.
- Fit imputation, constant-column removal, and scaling within each training pipeline.
- Tune configurations by mean macro F1 across artist-grouped CV folds in the official training partition.
- Evaluate one CV-winning configuration per family on official validation; choose the winning family by validation macro F1.
- Save model selection before requesting the test partition.
- Refit chosen configurations on training plus validation; record one test evaluation per run.
- Report the reused official test set as an extended comparison, since its original scores were previously viewed.
- Use a training-only tuned model for validation-set permutation importance.
- Fit PCA and K-means on training features; use validation for discovery evaluation.
- Fit candidate feature selectors and PCA transformations inside the supervised fitting folds.
- Keep the fixed similarity-map PCA separate from supervised candidate transformations.
- Save exact metrics, predictions, diagnostics, seeds, versions, and source checksums.

ANN means a feed-forward multilayer perceptron here. It uses a 10% internal track holdout for early stopping, distinct from the official artist-disjoint validation set. Explicit label encoding supports string genres with scikit-learn 1.8's early-stopping implementation.

## Project files

~~~text
musicmap/
  notebooks/MusicMap.ipynb         Executed, explained lab notebook
  src/musicmap/
    data.py                       Download, integrity checks, chunked parsing
    experiment.py                 Original baseline models and shared helpers
    improvement.py                Grouped CV, reductions, seed ensemble, benchmark
    discovery.py                  PCA, clustering, importance, nearest tracks
    reporting.py                  Reports generated from measured results
    cli.py                        Commands and prediction
  data/processed/
    features.npz                  Prepared numeric feature subset
    tracks.csv                    Separate labels and display metadata
    manifest.json                 Provenance, checksums, split/data audit
  models/
    best_model.joblib             Validation-selected model, refit on 7,200
    selected_training_model.joblib Training-only model for interpretation
    discovery_transform.joblib    Train-fitted preprocessing and PCA
    training_embeddings.npz       Training-only similarity catalog
  reports/
    MusicMap_Report.md/html/docx  Actual lab report in three formats
    model_comparison.csv          All model metrics and timings
    candidate_results.csv/json    CV candidates, with validation scores for winners
    cv_results.csv                All artist-grouped tuning candidates
    cv_splits.json                Fold membership and artist overlap audit
    ann_seed_results.csv          Individual seeds and fixed ANN ensemble
    improvement_summary.json      Comparison against the original baseline
    selection.json                Selection saved before test evaluation
    test_predictions.csv          Every held-out track and prediction
    classification_reports.json   Per-class precision, recall, and F1
    confusion_matrices.json       Raw counts for all models
    feature_family_importance.csv Grouped permutation results
    training_feature_associations.csv  Exploratory ANOVA ranking
    discovery_summary.json        PCA and clustering metrics
    song_map.html                 Offline interactive map
    figures/                      Report-quality PNG figures
    presentation_outline.md       Ten-slide presentation guide
  baseline/reports/               Preserved original standard experiment
  baseline/models/                Preserved original models (complete ZIP only)
  docs/VIVA.md                    Preparation questions with explanations
  tests/test_pipeline.py          Split, preprocessing, ANN, importance checks
  scripts/build_notebook.py       Recreate the documented notebook
  scripts/package_project.py      Create source/results and complete ZIPs
~~~

## Submission checklist

1. Read and run the notebook so you can explain the actual implementation.
2. Add your name, roll number, institution, and course details to the Word report.
3. Adapt sections and formatting to your instructor's rubric.
4. Use the recorded metrics as measured, including the modest accuracy and limitations.
5. Prepare the ten-slide outline and practice the viva questions.
6. Cite the dataset authors using CITATION.bib.

## Attribution

Defferrard, M., Benzi, K., Vandergheynst, P., and Bresson, X. (2017). **FMA: A Dataset for Music Analysis**. ISMIR. [Paper](https://arxiv.org/abs/1612.01840), [official repository](https://github.com/mdeff/fma), [feature extraction](https://github.com/mdeff/fma/blob/master/features.py).

FMA metadata and derived features are distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This package includes a prepared derivative of the original metadata/features with attribution; no audio is redistributed. Original project source code is provided under the MIT license in LICENSE.
