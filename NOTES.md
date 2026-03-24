# Pipeline Notes — Ambient MS Metabolomics Toolkit

Detailed notes on what each pipeline step does, why the defaults were chosen, and how to interpret the outputs. This is a reference document — you do not need to read it to run the pipeline, but it explains the reasoning behind every decision.

---

## What This Pipeline Does

### 1. Loading

Your raw spectra are read in from CSV or TXT files. Each file is one sample, and each subfolder represents one group (e.g. control vs treatment). Because different instruments or export settings can produce spectra with slightly different m/z axes, all samples are interpolated onto a shared m/z axis so they can be compared directly.

### 2. Binning

Raw spectra often have m/z values measured at slightly different points across samples. Binning groups nearby m/z values into fixed-width windows and sums their intensities, making the data consistent and reducing noise from small m/z shifts between runs.

**Default: 0.5 Da bins.** Using a 0.5 Da bin width for triple quadrupole (QqQ) data is a common approach in metabolomics, particularly for low-resolution instruments, to ensure isotopic peaks are correctly accounted for and to improve spectral consistency across samples. Although the SCIEX 4500 has a native step size of 0.1 Da, 0.5 Da bins were used to account for m/z drift across runs. If your instrument has higher resolution or more stable m/z calibration, you may want to reduce the bin width.

### 3. Filtering

Two filters are applied to remove uninformative features before any statistics:

- **Low variance filter** — removes features whose intensity barely changes across samples. If a feature looks the same in every sample, it cannot help distinguish your groups.
- **Low abundance filter** — removes features with very low mean intensity across all samples. These are likely noise rather than real signal.

**Default: 25% variance filter, 5% abundance filter.** These thresholds are taken from MetaboAnalyst defaults. They may not be optimal for every dataset — if too many or too few features are being removed, these are the first parameters to adjust in `config.py`.

### 4. Normalization, Transformation, and Scaling

Three steps are applied to make samples comparable to each other. Each can be configured independently in `config.py`.

**Normalization** corrects for differences in sample amount or injection volume between runs:

- **TIC** (default) — divides each sample by its total ion current and rescales to the median. Standard for ambient MS and direct infusion experiments.
- **Median** — divides by the sample median rather than the sum. More robust when a few very abundant features dominate the signal.
- **PQN** — Probabilistic Quotient Normalization. Compares each sample to a reference spectrum and estimates a dilution factor. Handles sample-to-sample variation well. Recommended if TIC gives poor results.
- **Quantile** — forces all samples to have the same intensity distribution. Aggressive — use only if you are confident distributional differences are technical, not biological.
- **None** — no normalization.

**Transformation** compresses the wide dynamic range of MS data:

- **Log10** (default) — standard for MS metabolomics
- **Log2**, **sqrt**, or **none**

**Scaling** adjusts for differences in feature magnitude so all features contribute equally. Van den Berg et al. (2006, *BMC Genomics* 7:142) compared scaling methods and found autoscaling and range scaling best removed the dependence of feature rankings on average concentration:

- **Autoscale** (default) — subtract mean, divide by standard deviation. All features equally weighted.
- **Pareto** — divide by square root of standard deviation. A compromise between autoscaling and no scaling.
- **Range** — divide by biological range (max - min). Scales relative to observed biological variation.
- **Vast** — downweights features with high relative variation. Focuses on stable, consistently-changing features.
- **Level** — divide by mean. Converts to fold changes relative to average. Useful for biomarker discovery.
- **None** — no scaling applied. High-abundance features will dominate.

### 5. PLS-DA (Partial Least Squares Discriminant Analysis)

PLS-DA is a supervised dimensionality reduction method. "Supervised" means it uses your group labels (e.g. control vs treatment) to guide the analysis — it is specifically looking for m/z features that differ between your groups, rather than just describing overall variation.

It works by finding combinations of m/z features (called components) that best separate your groups. The output is a 3D scores plot where each dot is one sample. Samples that cluster together are metabolically similar. Clear separation between group clusters means the pipeline has found m/z features that reliably distinguish your groups. Overlapping clusters suggest the groups are metabolically similar or that there is too much within-group variation.

**Default: 8 components.** More components capture more variation but can overfit. For the scores plot, 8 components are used. For VIP scores (below), only 1 component is used because the Python implementation matches MetaboAnalyst exactly at component 1 but diverges at higher components due to differences between scikit-learn and R's ropls package.

### 6. VIP Scores (Variable Importance in Projection)

VIP scores are produced by PLS-DA and rank each m/z feature by how much it contributed to the group separation. A VIP score above 1.0 is the standard threshold for a feature being considered important — this is not arbitrary, it reflects features that contribute more than average to the model.

The output is a dot plot of the top 30 features alongside a heatmap showing the mean intensity of each feature per group. The heatmap uses a red-blue colour scale where red = high intensity and blue = low intensity. This lets you immediately see not just which features are important, but whether they are higher or lower in each group.

### 7. Classifier Comparison

Six machine learning classifiers are trained and evaluated on your data:

- **Random Forest** — builds many decision trees on random subsets of the data and averages their predictions. Robust to noise and works well with high-dimensional data like MS.
- **SVM (Support Vector Machine)** — finds the boundary that best separates your groups in feature space. The linear kernel is used here, which is standard for metabolomics.
- **Gradient Boosting** — builds trees sequentially, each one correcting the errors of the previous. Often very accurate but can overfit.
- **Logistic Regression** — a simple linear model that predicts group membership. Fast and interpretable. A good baseline.
- **LDA (Linear Discriminant Analysis)** — conceptually similar to PLS-DA. Finds linear combinations of features that best separate groups. A classical approach in metabolomics.
- **Ridge Regression** — a regularized linear model that produces clean per-feature coefficients. Stable, fast, and well-suited to high-dimensional correlated data like MS. A strong choice when the primary question is which m/z values matter most.

Each classifier is evaluated using **5-fold stratified cross-validation**. This means the data is randomly split into 5 equal parts. The model trains on 4 parts and is tested on the 1 part it has never seen. This is repeated 5 times so every sample gets tested exactly once. The result is 5 accuracy scores, one per fold.

**Reading the comparison plot:**

The plot has two panels:

- **Top panel** — shows each fold's test accuracy as a dot, with the mean shown as a horizontal line and a bar for visual reference. Dots spread far apart indicate the model performs inconsistently across folds, which may reflect a small dataset or unstable model.
- **Bottom panel** — shows mean train accuracy vs mean test accuracy side by side for each model. A large gap between train and test accuracy (train is much higher) indicates **overfitting** — the model has memorised the training data but does not generalise well to new samples. A small gap indicates the model is learning something real.

**What counts as a good accuracy?** This depends entirely on how many groups you have. With 2 groups, random chance gives 50% accuracy. With 3 groups, chance is 33%. A well-performing model should be meaningfully above chance. Perfect accuracy (1.0) on a small dataset should be treated with caution — it may indicate overfitting.

### 8. Feature Importance Overlap

Six interpretable methods rank all m/z features by how important they were to group separation: Random Forest, SVM, Gradient Boosting, Logistic Regression, Ridge Regression, and PLS-DA VIP. This step finds features that appear in the top 50 of at least 2 of these 6 methods.


The output is saved as a CSV file with the following columns:

| Column | What it means | How to use it |
|--------|--------------|---------------|
| `mz` | The m/z value of the feature | Use this to look up the feature in your data or a database |
| `rf_importance` | Random Forest feature importance (mean decrease in impurity) | Higher = that m/z split the data more cleanly across all trees. Relative values with no fixed scale. |
| `svm_importance` | Mean absolute SVM coefficient across classes | Higher = that m/z pulled samples further apart at the decision boundary. Relative values. |
| `gb_importance` | Gradient Boosting feature importance | Same interpretation as RF but from a different tree-building strategy. |
| `lr_importance` | Mean absolute Logistic Regression coefficient | Higher = stronger linear association with group membership. |
| `ridge_importance` | Mean absolute Ridge Regression coefficient | Higher = stronger regularized linear association with group membership. Complementary to `lr_importance`. |
| `vip_score` | PLS-DA Variable Importance in Projection | The only column with a meaningful threshold: values above 1.0 are considered important. |
| `n_methods` | How many of the 6 methods ranked this feature in their top 50 | **Start here.** This is the most useful column for prioritisation. |

**Important: do not compare numbers across columns.** Each importance metric is on a completely different scale — a `rf_importance` of 0.02 and a `vip_score` of 1.5 cannot be directly compared. Instead, use each column to rank features within that method, and use `n_methods` to identify the most consistently important features across all methods.

**How to prioritise candidates:**

1. Sort by `n_methods` descending — features appearing in 5 or 6 methods are your strongest candidates
2. Among features with the same `n_methods`, check whether `vip_score` is above 1.0
3. Look at the spectrum plot (`spectrum_features_*.png`) to see where these features sit in the raw data

---

## Configuration Reference

All pipeline settings are controlled from `config.py`. This is the **only file you need to edit**.

| Setting | Default | What it controls |
|---------|---------|-----------------|
| `EXPERIMENT` | `'your_experiment_folder'` | Your experiment folder name |
| `MZ_MIN` / `MZ_MAX` | `100` / `1000` | m/z range to keep (Da) |
| `BIN_WIDTH` | `0.5` | Bin width in Da |
| `VARIANCE_PERCENTILE` | `25` | Low variance filter threshold (0 = off) |
| `ABUNDANCE_PERCENTILE` | `5` | Low abundance filter threshold (0 = off) |
| `NORMALIZATION` | `'tic'` | Sample normalization: `'tic'`, `'median'`, `'pqn'`, `'quantile'`, `'none'` |
| `LOG_TRANSFORM` | `'log10'` | Transformation: `'log10'`, `'log2'`, `'sqrt'`, `'none'` |
| `SCALING` | `'autoscale'` | Scaling: `'autoscale'`, `'pareto'`, `'range'`, `'vast'`, `'level'`, `'none'` |
| `N_PLSDA_COMPONENTS` | `8` | Number of PLS-DA components |
| `N_TOP_VIP` | `30` | Number of VIP features to plot |
| `USE_*` | `True` | Toggle individual classifiers on/off |
| `CV_FOLDS` | `5` | Number of cross-validation folds |
| `TOP_N_FEATURES` | `50` | Top N features per method for overlap analysis |
