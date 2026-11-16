# MusicMap viva preparation

Use these notes to understand the implementation and explain the measured results in your own words.

Machine Learning course project, Veermata Jijabai Technological Institute (VJTI).

Akshat Tikaria — 241070005; Aniket Shinde — 171070977.

1. **What is the project trying to do?**  
   Predict one of eight music genres from precomputed audio descriptors, compare the four required classifiers plus an Extra Trees benchmark, and investigate audio-based similarities and feature relevance.

2. **Why use FMA-small?**  
   It contains 8,000 tracks in eight balanced genres and has official metadata, precomputed features, and supplied partitions. It is manageable on a standard laptop.

3. **Did you download and process 8,000 audio files?**  
   The implemented experiment uses the official precomputed feature table. The metadata archive is 342 MiB; the 7.2 GiB audio archive is unnecessary. Describe this choice clearly.

4. **What is an input sample?**  
   One track represented by 518 numeric descriptors: 74 frame-level feature dimensions summarized by seven statistics, grouped into 11 families. Genre is the target. Track ID, title, artist ID, album ID, and split metadata are separate and do not enter the classifiers.

5. **What are MFCCs?**  
   Mel-frequency cepstral coefficients summarize the shape of a spectrum after mel-frequency analysis and a cepstral transformation. They are useful descriptors of timbre. FMA supplies temporal summary statistics for 20 MFCC coefficients.

6. **What do chroma, spectral centroid, and zero-crossing rate describe?**  
   Chroma summarizes energy in pitch classes. Spectral centroid describes the center of spectral energy and relates to brightness. Zero-crossing rate counts rapid sign changes and can reflect noisiness. Their relationships to genres are statistical rather than exact rules.

7. **Why standardize?**  
   Descriptors have different numerical scales. Large-valued features could dominate distance or optimization without representing greater musical importance. Fit the mean and standard deviation on the fitting partition only.

8. **What is data leakage?**  
   Information unavailable at prediction time influences training or selection. Examples include scaling on the entire dataset, using artist IDs as predictors, or choosing parameters by test performance.

9. **Why preserve the official artist-disjoint splits?**  
   Recordings by the same artist may share production and performance cues. A random track split can make evaluation easier through artist memorization. The prepared dataset has zero overlapping artist IDs across its 6,400/800/800 partitions.

10. **What is the difference between training, validation, and test?**  
    Training-fold data fit parameters and preprocessing. Artist-grouped cross-validation chooses a configuration within each family. The separate official validation set chooses the family. Selected configurations are refitted on training plus validation before the recorded test evaluation. Because the original test scores were already viewed, updated scores are an extended benchmark comparison.

11. **Why use macro F1?**  
    It averages F1 equally across genres and requires both precision and recall. It makes class-specific failure visible even when a single accuracy number is misleading.

12. **Why is the dummy macro F1 much lower than 12.5%?**  
    A majority-class predictor chooses one label for every track. It has 12.5% accuracy on eight balanced genres, but seven genres have zero F1. The predicted genre has F1 about 0.2222; averaging over eight produces about 0.0278.

13. **How do the four algorithms work?**  
    A Decision Tree learns feature thresholds. KNN votes among nearby training vectors. An RBF SVM learns a nonlinear separating boundary. An ANN learns nonlinear weighted transformations through fully connected layers.

14. **Is the ANN a deep audio model?**  
    It is a feed-forward multilayer perceptron operating on summary descriptors, with either one or two hidden layers in the grid. It does not process waveforms or spectrograms.

15. **How does ANN early stopping work here?**  
    Each network reserves 10% of its current fitting partition internally and stops when its internal score stops improving, subject to the configured patience and iteration limit. The exact settings are recorded with the selected parameters. This internal track split is separate from the official external validation set and is not artist-grouped. SVM probability estimation also uses internal splits without artist groups. Our artist-disjoint guarantees apply to outer CV and the official partitions. Genre labels are encoded numerically inside the ANN wrapper.

16. **Which model won?**  
    In version 2, Stacked Ensemble wins validation macro F1 (0.5912), with 50.375% test accuracy and 0.4987 test macro F1. Among the four required families, ANN wins validation while SVM has the highest observed test accuracy (49.75%). Soft voting has the highest observed test accuracy across all candidates (50.75%), but test scores do not change the validation-selected champion. The original standard SVM scored 49.125%; the v2 gain of 1.25 percentage points is not statistically significant under the reported test (p = 0.378).

17. **Why are the scores modest?**  
    The setup uses summary features, overlapping genre labels, a finite tuning search, and an artist-disjoint evaluation. These are plausible constraints, not proven explanations for every error. Additional controlled experiments would be needed to attribute causes or establish performance limits.

18. **What does permutation importance mean?**  
    Shuffle a feature family across validation tracks while holding other families fixed and measure the macro F1 decrease. A larger positive drop suggests the selected training-only model relies more on that family. It is not causal evidence.

19. **Which features were most important?**  
    MFCC, spectral contrast, and STFT chroma lead the v2 feature-family permutation ranking. Group sizes differ, and correlated features can substitute for one another, so avoid claiming that every MFCC coefficient is important.

20. **What does PCA do?**  
    It finds orthogonal directions with high variance in standardized training features. Two components provide a visualization; 50 provide the clustering/similarity representation. The included 2D map retains about 18.9% of variance, so it omits substantial information.

21. **Does PCA use genre labels?**  
    The map PCA does not use genres, which only color the drawing after projection. In the improved experiment, PCA is also tested as a supervised pipeline transformation, fitted only on each training fold. ANOVA feature selection does use training labels and is separately fitted within each fold.

22. **Why does K-means not produce eight clean genre groups?**  
    It minimizes within-cluster squared distances in PCA space, without learning from genre labels. Acoustic geometry need not coincide with human genre categories. Cluster IDs have arbitrary numbering.

23. **What do ARI, NMI, and silhouette measure?**  
    ARI measures label agreement after accounting for chance, NMI measures shared information between labelings, and silhouette measures geometric compactness and separation. A silhouette score is not a genre-classification accuracy.

24. **How are similar tracks retrieved?**  
    By Euclidean distance in the training-fitted 50-component PCA space, searching the training catalog. The query and same-artist tracks are excluded by default. Genre is displayed but not used in the distance.

25. **Can the project classify a new MP3 directly?**  
    Yes, as a demonstration. musicmap predict --audio song.mp3 (or the Upload tab of musicmap app) extracts the same 518 descriptors with librosa, following the official FMA features.py recipe: 74 component dimensions in 11 families summarized by seven statistics over a 30-second excerpt. M4A and other files unsupported by the default decoder use FFmpeg when installed. Library versions differ from the 2017 extraction, so predictions for new audio are not part of the benchmark.

26. **What would you improve next?**  
    Version 2 adds stronger models, ensembles, probability outputs and exploratory statistical comparisons. Summary statistics discard temporal order, so spectrogram CNNs or pretrained audio embeddings are useful alternatives to test. This experiment does not prove a feature ceiling. Further work includes FMA-medium, artist-aware uncertainty estimates and a fresh artist-disjoint test set.

27. **How do you know the results are reproducible?**  
    The code fixes seeds, records versions and candidate scores, verifies checksums, preserves track IDs and saves final predictions. Tests check preprocessing isolation, the selection lock, serialization, alignment, audio handling and artifact verification. scripts/verify_artifacts.py reconciles actual saved models against the prediction CSV and reports, so an outdated model file cannot silently pass merely because the app opens.

28. **What should you demonstrate during the lab?**  
    Run the notebook, explain the split audit and metrics, show a confusion matrix and feature importance, zoom the interactive music map, retrieve nearest tracks, and describe limitations without overstating the scores.

29. **How does artist-grouped cross-validation work?**  
    StratifiedGroupKFold divides only the official training partition into folds while keeping an artist's tracks together. Every fold learns its imputer, scaler, feature selector/PCA, and classifier from fitting tracks. No artist occurs on both sides of a fold assessment.

30. **Why use a quantile-to-normal transformation?**  
    Many audio statistics are skewed or have extreme values. This training-fitted transformation maps their empirical quantiles toward a normal distribution. It changes geometry and can affect SVM and KNN; whether it helps is decided using training CV scores.

31. **Why average ANN seeds rather than select the best seed?**  
    Selecting the luckiest initialization can give an optimistic impression. Version 2 uses five fixed seeds (42–46) with equal probability averaging. Individual seed scores show variability; all five contribute to the reported ANN.

32. **What does Extra Trees add?**  
    It averages many randomized decision trees, giving a useful ensemble benchmark against a single tree and the distance/network models. It still uses the same audio descriptors and external partitions.

33. **Can we call an updated test score independent confirmation?**  
    The official test scores were already viewed during the original work. The updated model is selected without using test scores in its implemented search, but the comparison is still an extended benchmark study. A fresh artist-disjoint confirmation dataset would support a stronger generalization claim.

34. **What happened in the first improvement run?**  
    Its ANN won on validation but scored 47.25% on test, below the original SVM's 49.125%. That run showed that a single validation winner can be noisy on 800 tracks — one motivation for version 2's ensembles, which average out the idiosyncratic errors of individual models.

35. **Do the outlier diagnostics prove quantile transformation is better?**  
    No. Training-only diagnostics found 83 descriptors with values above 20 absolute standard deviations. This motivates testing an alternative transformation; quantile normalization can also distort useful relationships. Its benefit must be assessed using the declared development protocol, and small benchmark gains do not establish statistical significance.

## Version 2 additions

36. **What is gradient boosting (LightGBM)?**  
    It builds hundreds of small decision trees one after another; each new tree is fitted to the errors (gradients of the log-loss) of the current ensemble, and a small learning rate shrinks every step. Feature and row subsampling plus L2 regularization limit overfitting. Unlike a single tree it models subtle interactions, and unlike Extra Trees its trees cooperate rather than vote independently.

37. **Why add Logistic Regression?**  
    It provides a regularized linear baseline and is fast. Comparing it with nonlinear families shows their observed benefit in this finite search. It produces probabilities for ensembling, but calibration must be measured rather than assumed from the algorithm name.

38. **What are out-of-fold (OOF) predictions?**  
    In K-fold CV every training track is predicted once by a fold model that did not fit on that row. These probabilities let us learn ensemble weights without validation or test data. Because base configurations were selected using these folds, the collected OOF scores are tuning diagnostics rather than a fresh independent evaluation of the full selection procedure.

39. **How does the soft-voting ensemble choose its weights?**  
    Greedy forward selection (Caruana et al., 2004): start empty; in each round add (with replacement) the family whose inclusion gives the best mean fold macro F1 on the OOF probabilities; keep the best prefix of 25 rounds. The count of each family becomes its weight. Because members can be added repeatedly, strong families get more weight and useless ones get none.

40. **What is stacking, and how is it different?**  
    A stacked ensemble trains a second-level model — here multinomial logistic regression — on the members' OOF log-probabilities (7 families × 8 genres = 56 inputs). It can learn, for example, to trust the SVM for Folk but gradient boosting for Electronic. Its regularization C is chosen by an inner grouped CV over the OOF predictions. More flexible than voting, but with more parameters to estimate.

41. **Why can an ensemble beat its best member?**  
    Members with similar accuracy but different errors (a kernel method, a tree ensemble, a neural network) partly cancel each other's mistakes when their probabilities are averaged. The gain is largest when errors are weakly correlated.

42. **Are ensemble CV scores trustworthy?**  
    Treat them as internal tuning diagnostics. Soft-voting weights are chosen on the OOF predictions being scored. The stacker reuses one global base-model OOF matrix for meta-model CV, so this is not fully nested evaluation of the entire stack. Official validation and test rows remain outside base OOF fitting, but validation selects the champion and the historical test set was already viewed.

43. **What is a bootstrap confidence interval?**  
    The implemented interval resamples the 800 test tracks with replacement 2,000 times and takes the 2.5th and 97.5th percentiles of the recomputed metric. Its accuracy interval is 47.1%–53.9%. This assumes track-level sampling and does not account for correlated songs by the same artist; artist-cluster resampling would better address that dependence. Do not interpret it as a definitive new-artist generalization interval.

44. **What does the McNemar test check?**  
    It tests whether paired correctness differs between two classifiers: the "only A correct" and "only B correct" counts should balance under the null hypothesis. The implementation uses an exact binomial test. Ordinary McNemar assumes independent pairs across tracks, which can be violated by songs sharing an artist, so these p-values are exploratory. The reported p = 0.378 does not establish an improvement over the original SVM.

45. **What is top-3 accuracy and why report it?**  
    The share of tracks whose true genre is among the model's three most probable genres. Version 2 scores 80.375% top-3 versus 37.5% random top-3 chance. Its single-label accuracy is 50.375% versus 12.5% random top-1 chance. These are different metrics; do not call the system an 80%-accurate single-label classifier.

46. **What does the confidence chart show?**  
    Test tracks are grouped by the champion's top probability. Higher-confidence tracks are more often correct in this recorded run, but that is not a complete calibration study or a guarantee for any individual track. Confident mistakes still occur.

47. **Why parallel worker processes?**  
    Each (configuration, fold) fit is independent, so joblib runs several at once, one BLAS thread each. This shortens the run roughly in proportion to the CPU cores without changing any result.
