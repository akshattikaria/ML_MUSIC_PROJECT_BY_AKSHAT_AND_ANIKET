# MusicMap presentation outline

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
- Tune with artist-grouped cross-validation inside training; choose the algorithm on validation.
- Refit on training + validation; report an extended comparison on the previously viewed test set.
- Explain why artist leakage would inflate a random track split.

## Slide 4 — Models
- Decision Tree: threshold rules.
- KNN: nearby standardized feature vectors.
- SVM: nonlinear RBF boundary.
- ANN: feed-forward ReLU network with early stopping.
- Extra Trees supplies an ensemble benchmark; include the majority-class baseline.

## Slide 5 — Evaluation and results
- Macro F1 chooses the winner; also report accuracy and per-genre scores.
- Validation-selected model: ANN (MLP).
- Test accuracy: 47.25%; macro F1: 0.4695.
- Show figures/model_comparison.png.

## Slide 6 — Errors
- Show figures/confusion_matrices.png or figures/per_genre_f1.png.
- Explain two observed genre confusions using the report's counts.
- Distinguish measured errors from speculation about their causes.

## Slide 7 — Feature interpretation
- Jointly shuffle feature families on validation data.
- Highest three mean drops: mfcc, spectral_contrast, chroma_cqt.
- Show figures/feature_importance.png.
- Importance describes model reliance, not musical causality.

## Slide 8 — Pattern discovery
- PCA fitted on training features; first two PCs retain 18.91% variance.
- K-means fitted without genres; validation ARI: 0.1403.
- Show figures/pca_map.png and figures/cluster_composition.png.

## Slide 9 — Demonstration
- Open song_map.html; hover, zoom, and filter the genre legend.
- Run musicmap neighbors --track-id [a valid FMA-small ID] --k 5.
- Show that neighbors use audio descriptors and a training-only catalog.

## Slide 10 — Conclusions and limits
- State the measured winner and the strongest feature families.
- Eight genres, finite search, several ANN seeds, and one previously viewed official test set.
- New raw audio requires compatible feature extraction.
- Future work: a fresh confirmation set, more genres, and pretrained embeddings.
