# MusicMap viva preparation

Use these notes to understand the implementation and explain the measured results in your own words.

1. **What is the project trying to do?**  
   Predict one of eight music genres from precomputed audio descriptors, compare the four required classifiers plus an Extra Trees benchmark, and investigate audio-based similarities and feature relevance.

2. **Why use FMA-small?**  
   It contains 8,000 tracks in eight balanced genres and has official metadata, precomputed features, and supplied partitions. It is manageable on a standard laptop.

3. **Did you download and process 8,000 audio files?**  
   The implemented experiment uses the official precomputed feature table. The metadata archive is 342 MiB; the 7.2 GiB audio archive is unnecessary. Describe this choice clearly.

4. **What is an input sample?**  
   One track represented by 518 numeric descriptors. Genre is the target. Track ID, title, artist ID, album ID, and split metadata are separate and do not enter the classifiers.

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
    Each network reserves 10% of its current fitting partition internally and stops when its internal score stops improving, subject to the configured patience and iteration limit. The exact settings are recorded with the selected parameters. This internal track split is separate from the official external validation set. Genre labels are encoded numerically inside the ANN wrapper.

16. **Which model won?**  
    In the preserved original standard run, SVM achieved approximately 49.1% test accuracy and 0.4885 macro F1. For the improved run, read the validation-selected winner and exact measurements in the active report and model_comparison.csv. A higher test score must not be used to change the preselected winner.

17. **Why are the scores modest?**  
    The setup uses summary features, overlapping genre labels, a finite tuning search, and an artist-disjoint evaluation. These are plausible constraints, not proven explanations for every error. Additional controlled experiments would be needed to attribute causes or establish performance limits.

18. **What does permutation importance mean?**  
    Shuffle a feature family across validation tracks while holding other families fixed and measure the macro F1 decrease. A larger positive drop suggests the selected training-only model relies more on that family. It is not causal evidence.

19. **Which features were most important?**  
    MFCC, spectral contrast, and STFT chroma led the original baseline ranking. Inspect the updated feature-family table for the improved selected model. Group sizes differ, and correlated features can substitute for one another, so avoid claiming that every MFCC coefficient is important.

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
    The delivered predictor takes prepared feature rows or a matching 518-column feature CSV. Supporting a raw MP3 requires compatible extraction and separate validation against the original FMA descriptor implementation.

26. **What would you improve next?**  
    The improved version already implements artist-grouped cross-validation, broader tuning, feature reduction, and multiple ANN seeds. Next steps are broader genres, pretrained representations, and a fresh artist-disjoint confirmation set.

27. **How do you know the results are reproducible?**  
    The code fixes seeds, records versions and all candidate scores, verifies source/processed checksums, preserves track IDs, saves final predictions, and provides a one-command run. Tests check preprocessing isolation, the selection lock, ANN serialization, data alignment, and feature-family shuffling.

28. **What should you demonstrate during the lab?**  
    Run the notebook, explain the split audit and metrics, show a confusion matrix and feature importance, zoom the interactive music map, retrieve nearest tracks, and describe limitations without overstating the scores.

29. **How does artist-grouped cross-validation work?**  
    StratifiedGroupKFold divides only the official training partition into folds while keeping an artist's tracks together. Every fold learns its imputer, scaler, feature selector/PCA, and classifier from fitting tracks. No artist occurs on both sides of a fold assessment.

30. **Why use a quantile-to-normal transformation?**  
    Many audio statistics are skewed or have extreme values. This training-fitted transformation maps their empirical quantiles toward a normal distribution. It changes geometry and can affect SVM and KNN; whether it helps is decided using training CV scores.

31. **Why average ANN seeds rather than select the best seed?**  
    Selecting the luckiest initialization can give an optimistic impression. The improved ANN uses the fixed seeds 42, 43, and 44 with equal probability averaging. Individual seed scores show variability; all three contribute to the reported ensemble.

32. **What does Extra Trees add?**  
    It averages many randomized decision trees, giving a useful ensemble benchmark against a single tree and the distance/network models. It still uses the same audio descriptors and external partitions.

33. **Can we call an updated test score independent confirmation?**  
    The official test scores were already viewed during the original work. The updated model is selected without using test scores in its implemented search, but the comparison is still an extended benchmark study. A fresh artist-disjoint confirmation dataset would support a stronger generalization claim.

34. **Did the recommended changes improve the selected winner's test accuracy?**  
    No. The improved ANN ensemble won on validation macro F1, but its 47.25% test accuracy was 1.875 percentage points below the original selected SVM's 49.125%. The updated SVM reached 49.75%, the highest updated test score, but switching to it because of that test result would break the declared selection rule. Some same-family scores improved, while the final selection outcome did not.

35. **Do the outlier diagnostics prove quantile transformation is better?**  
    No. Training-only diagnostics found 83 descriptors with values above 20 absolute standard deviations. This motivates testing an alternative transformation; quantile normalization can also distort useful relationships. Its benefit must be assessed using the declared development protocol, and small benchmark gains do not establish statistical significance.
