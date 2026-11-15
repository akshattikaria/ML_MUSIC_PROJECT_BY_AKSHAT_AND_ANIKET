# MusicMap

**Music Genre Classification and Pattern Discovery Using Machine Learning** · version 2 (optimized)

MusicMap predicts the genre of a song (Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock) from 518 audio descriptors of the FMA-small dataset, explains which audio properties matter, and maps songs by sound similarity. Everything runs on an ordinary laptop CPU; no GPU or audio download is required.

Start with [the executed notebook](notebooks/MusicMap.ipynb), [the lab report](reports/MusicMap_Report.html) (also as an editable [Word file](reports/MusicMap_Report.docx)), or launch the interactive demo with `musicmap app`.

## What's new in version 2

| Area | Version 1 | Version 2 |
|:--|:--|:--|
| Models | Decision Tree, KNN, SVM, ANN, Extra Trees | + **Logistic Regression**, **LightGBM gradient boosting**, a **soft-voting ensemble** and a **stacked ensemble** |
| Ensembles | none (ANN seed average only) | Weights / meta-learner learned from artist-grouped **out-of-fold** probabilities — training data only |
| ANN | 3 seeds, 128-unit layers | 5 seeds, wider layers and stronger regularization |
| Speed | sequential, ≈20 min for 94 configs | fits run in **parallel worker processes** |
| Outputs | label only | genre **probabilities**, top-3 guesses, confidence analysis, top-3 accuracy, log-loss |
| Statistics | point estimates | 95% **bootstrap intervals** and exact **McNemar tests** against earlier runs |
| Your own music | not supported | `musicmap predict --audio song.mp3` — FMA-compatible feature extraction |
| Demo | static HTML map | **Streamlit app**: explore tracks, upload audio, interactive map, results |
| Compatibility | Python 3.12, scikit-learn 1.8, pandas 2 | Python 3.11–3.13, scikit-learn 1.5–1.9, pandas 2–3 |
| Reports | prose patched by string replacement | regenerated cleanly from measurements; numbers in this README are filled in automatically |

The protocol stays strict: every tuning and ensembling decision uses only the official training partition (artist-grouped CV); validation picks the champion; `selection.json` is saved **before** test data are requested (an automated test enforces this).

## Results

<!-- RESULTS START -->

| Model                |   CV macro F1 |   Validation macro F1 | Test accuracy   |   Test macro F1 | Top-3 accuracy   |
|:---------------------|--------------:|----------------------:|:----------------|----------------:|:-----------------|
| **Stacked Ensemble** |        0.5244 |                0.5912 | 50.38%          |          0.4987 | 80.38%           |
| Gradient Boosting    |        0.5387 |                0.5871 | 47.00%          |          0.4708 | 77.25%           |
| Soft-voting Ensemble |        0.5543 |                0.5824 | 50.75%          |          0.5030 | 79.75%           |
| ANN (MLP)            |        0.5238 |                0.5688 | 48.62%          |          0.4825 | 78.12%           |
| SVM (RBF)            |        0.5410 |                0.5575 | 49.75%          |          0.4925 | 78.88%           |
| Extra Trees          |        0.5009 |                0.5467 | 48.25%          |          0.4784 | 77.38%           |
| Logistic Regression  |        0.5115 |                0.5336 | 46.88%          |          0.4667 | 77.00%           |
| KNN                  |        0.4233 |                0.4944 | 38.25%          |          0.3674 | 69.50%           |
| Decision Tree        |        0.3478 |                0.3839 | 37.25%          |          0.3676 | 67.50%           |
| Dummy baseline       |        0.0277 |                0.0278 | 12.50%          |          0.0278 | 37.50%           |

**Champion (chosen on validation): Stacked Ensemble** — test accuracy **50.38%**, macro F1 **0.4987** (95% bootstrap interval for accuracy: 47.1%–53.9%).
- vs. original project (SVM (RBF), 49.12%): **+1.25 percentage points**, McNemar p = 0.378
- vs. first improvement (ANN (MLP), 47.25%): **+3.13 percentage points**, McNemar p = 0.037

Full run: 19.9 min with 2 parallel workers (Python 3.13.18, scikit-learn 1.8.0).

<!-- RESULTS END -->

The official test set was already viewed in the original project, so these scores are an **extended benchmark comparison** on the same benchmark, not an independent confirmation. Earlier runs are preserved in `baseline/` (original) and `history/improved_v1/` (first improvement).

## Setup

Python 3.11–3.13. From the extracted `musicmap` folder:

~~~sh
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
~~~

`requirements.txt` installs the package plus the notebook, test, demo-app and audio extras. For the exact versions used to produce the included results, use `requirements-tested.txt` instead. A minimal install is `python -m pip install -e .`.

## Use it

~~~sh
musicmap app                                   # interactive demo in your browser
musicmap predict --track-id 2                  # genre + confidence + top-3 for a dataset track
musicmap predict --audio my_song.mp3           # classify your own audio file(s)
musicmap neighbors --track-id 2 --k 5          # most similar songs by other artists
musicmap extract song1.mp3 song2.wav --output features.csv
musicmap predict --features-csv features.csv --output predictions.csv
jupyter lab notebooks/MusicMap.ipynb           # the explained, executed notebook
~~~

Audio classification extracts the same 518 descriptors as FMA from a 30-second excerpt in the middle of the file, following the official FMA extraction recipe. Modern librosa versions produce values close to, but not identical with, the 2017 features, so treat predictions for your own music as a demonstration rather than a benchmark.

## Reproduce from scratch

~~~sh
musicmap improve --seed 42        # tuning, ensembles, refit, test, discovery, reports
python -m pytest -q               # 20+ tests: leakage isolation, selection lock, ensembles, audio schema
python scripts/build_notebook.py --execute
~~~

`musicmap improve` evaluates 56 configurations with 3-fold artist-grouped CV, builds both ensembles, picks the champion on validation, refits on training + validation, evaluates on test, regenerates every figure, the Markdown/HTML/Word report, the slide outline and the results block above. It uses up to 4 parallel workers by default (`--threads N`). Runtime is roughly 20–40 minutes depending on the CPU.

The complete package includes the prepared 8,000 × 518 feature subset. In a source-only checkout, `musicmap prepare` downloads the official 342 MiB metadata archive, verifies its SHA-1 and streams only the needed rows (`--archive path/to/fma_metadata.zip` uses a local copy). The 7.2 GiB audio archive is never needed.

Other stages:

~~~sh
musicmap analyze --repeats 5      # discovery: PCA map, K-means, permutation importance, figures
musicmap report                   # rebuild reports from saved measurements
musicmap run --profile standard   # legacy validation-only protocol of the original project
~~~

All commands assume the project root is the current directory; otherwise pass `musicmap --root /path/to/musicmap <command>`.

## Methodology in brief

1. Keep the official artist-disjoint split (6,400 / 800 / 800); join features and metadata by track ID; keep genre, artist and title out of the predictors.
2. Every candidate is one pipeline (imputation → constant-column removal → standard or quantile scaling → optional ANOVA selection / PCA → classifier) refitted inside every fold.
3. Tune each family by mean macro F1 over 3 artist-grouped folds of the training partition.
4. Collect out-of-fold probabilities of each family's best configuration; learn soft-voting weights by greedy forward selection (Caruana et al., 2004) and a logistic-regression stacker (C chosen by inner grouped CV) from them.
5. Score all families and both ensembles on validation; the best validation macro F1 becomes the champion; save `selection.json`.
6. Refit on training + validation; evaluate once on test; report bootstrap intervals and McNemar tests.
7. Discovery uses training-fitted PCA/K-means and validation tracks only; permutation importance uses the training-only champion.

## Project layout

~~~text
musicmap/
  notebooks/MusicMap.ipynb          Executed, explained lab notebook
  src/musicmap/
    data.py                         Download, integrity checks, chunked parsing
    experiment.py                   Original (legacy) protocol and shared helpers
    improvement.py                  Grouped CV, 7 families, ensembles, statistics
    discovery.py                    PCA map, clustering, importance, similar songs, figures
    reporting.py                    Markdown/HTML/Word report, slides, README results
    audio.py                        FMA-compatible feature extraction from audio files
    app.py                          Streamlit demo
    cli.py                          Command-line interface
  data/processed/                   Prepared features, metadata, manifest with checksums
  models/
    best_model.joblib               Champion refitted on training + validation
    selected_training_model.joblib  Training-only champion (for interpretation)
    discovery_transform.joblib      Train-fitted scaling + PCA for the similarity map
    training_embeddings.npz         Training-only similarity catalog
  reports/
    MusicMap_Report.md/html/docx    The lab report in three formats
    model_comparison.csv            All metrics for every model
    cv_results.csv, cv_splits.json  Every tuning score; fold membership and artist audit
    ensemble.json                   Ensemble weights, selection path, stacking CV
    selection.json                  Champion and settings, saved before test evaluation
    improvement_summary.json        Bootstrap interval and McNemar tests vs earlier runs
    test_predictions.csv            Every test track: predictions and champion probabilities
    song_map.html                   Offline interactive music map
    figures/                        All report figures
    presentation_outline.md         Ten-slide presentation guide
  baseline/                         Original project (reports and models)
  history/improved_v1/              First improvement run (reports)
  docs/VIVA.md                      Viva preparation questions with answers
  tests/                            Automated checks
  scripts/                          Notebook builder and packager
~~~

## Submission checklist

1. Run the notebook and the demo so you can explain the implementation.
2. Add your name, roll number, institution and course details to the Word report.
3. Adapt sections and formatting to your instructor's rubric.
4. Quote the measured numbers as they are, including limitations.
5. Prepare the ten-slide outline and practise the viva questions.
6. Cite the dataset authors using CITATION.bib.

## Attribution

Defferrard, M., Benzi, K., Vandergheynst, P., and Bresson, X. (2017). **FMA: A Dataset for Music Analysis**. ISMIR. [Paper](https://arxiv.org/abs/1612.01840), [official repository](https://github.com/mdeff/fma).

FMA metadata and derived features are distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This package includes a prepared derivative of the metadata/features with attribution; no audio is redistributed. Project source code is MIT-licensed (see LICENSE).
