# MusicMap presentation outline

Ten slides for a 7–10 minute lab presentation. Add your name, roll number and institution.

## Slide 1 — Title and problem
- MusicMap: Music Genre Classification and Pattern Discovery Using Machine Learning.
- Can audio descriptors predict a track's genre and reveal similar songs?

## Slide 2 — Dataset
- FMA-small: 8,000 tracks, eight balanced genres, 518 descriptors (11 feature families × 7 statistics).
- Precomputed features; no audio download or GPU.
- Show figures/genre_balance.png and cite Defferrard et al. (2017).

## Slide 3 — Experimental design
- Official artist-disjoint split: 6,400 training / 800 validation / 800 test.
- Tune with 3-fold artist-grouped cross-validation inside training (no artist on both sides).
- Choose the champion on validation; lock selection; refit on 7,200; evaluate once on test.

## Slide 4 — Models
- Required: Decision Tree, KNN, SVM (RBF), ANN (MLP, seed-averaged).
- Stronger baselines: Logistic Regression, Extra Trees, Gradient Boosting.
- Soft-voting ensemble: weights chosen by greedy selection on out-of-fold probabilities.
- Majority-class baseline for context (12.5%).

## Slide 5 — Results
- Champion (chosen on validation): Stacked Ensemble.
- Test accuracy 50.38%; macro F1 0.4987; top-3 accuracy 80.4%.
- Original project: SVM (RBF) 49.12% → this run: Stacked Ensemble 50.38% (+1.25 pp).
- Show figures/model_comparison.png and figures/improvement_comparison.png.

## Slide 6 — Errors and confidence
- Show figures/confusion_matrices.png or figures/per_genre_f1.png.
- Easiest genres: Hip-Hop and Rock; hardest: Experimental and Folk (see per-genre F1).
- Show figures/confidence.png: high-confidence predictions are much more accurate.

## Slide 7 — Feature interpretation
- Jointly shuffle each feature family on validation data and measure the F1 drop.
- Top three: mfcc, spectral_contrast, chroma_stft.
- Show figures/feature_importance.png. Importance = model reliance, not causation.

## Slide 8 — Pattern discovery
- Train-only PCA: first two PCs keep 18.9% of variance.
- K-means without labels: validation ARI 0.140 — clusters follow timbre more than genre.
- Show figures/pca_map.png and figures/cluster_composition.png.

## Slide 9 — Live demonstration
- musicmap app — pick a song, see genre probabilities and the five most similar songs.
- Or: musicmap predict --track-id 2 and musicmap neighbors --track-id 2.
- Open song_map.html; hover, zoom and filter the legend.

## Slide 10 — Conclusions and limits
- Ensembles of diverse models give the best, most stable results on summary features.
- Limits: summary features, overlapping genres, previously viewed test set.
- Future: spectrogram CNNs / pretrained embeddings, FMA-medium, fresh confirmation set.
