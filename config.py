"""
config.py — Pipeline Configuration
====================================
This is the only file you need to edit to customise the pipeline.
Change the settings below to match your data and research question.

For guidance on which settings to choose, see the README.

Reference: van den Berg et al. (2006) BMC Genomics 7:142
           doi: 10.1186/1471-2164-7-142
"""

# ── Experiment ─────────────────────────────────────────────────────────────────
# The name of your experiment folder. Must be in the same directory as the
# .py files. Include spaces and special characters exactly as they appear.
EXPERIMENT = 'your_experiment_folder'


# ── m/z Range ──────────────────────────────────────────────────────────────────
# Features outside this range are removed before any analysis.
# Default reflects the informative range for fungal metabolomics on a SCIEX 4500.
# Adjust to match the range where your instrument produces reliable signal.
MZ_MIN = 100   # lower m/z cutoff (Da)
MZ_MAX = 1000  # upper m/z cutoff (Da)


# ── Binning ────────────────────────────────────────────────────────────────────
# Groups nearby m/z values into fixed-width windows and sums their intensities.
# 0.5 Da is appropriate for low-resolution QqQ instruments to account for m/z
# drift. Use a smaller value (e.g. 0.1) for higher-resolution instruments.
BIN_WIDTH = 0.5  # bin width in Da


# ── Filtering ──────────────────────────────────────────────────────────────────
# Removes uninformative features before statistics.
# Increase the percentile to remove more features; set to 0 to disable.
VARIANCE_PERCENTILE = 25  # remove bottom X% of features by variance (0 = off)
ABUNDANCE_PERCENTILE = 5  # remove bottom X% of features by abundance (0 = off)


# ── Normalization ──────────────────────────────────────────────────────────────
# Corrects for differences in sample amount, injection volume, or instrument
# sensitivity between runs. Choose based on your experimental setup.
#
# Options:
#   'tic'      — divide each sample by its total ion current (sum of all
#                intensities) and rescale to the median TIC. The most common
#                approach for ambient MS and direct infusion experiments.
#                Default and recommended for LMJ-SSP data.
#
#   'median'   — divide each sample by its median feature intensity and rescale
#                to the global median. More robust than TIC when a small number
#                of very abundant features dominate the signal.
#
#   'pqn'      — Probabilistic Quotient Normalization. Compares each sample to
#                a reference spectrum (median of all samples) and estimates a
#                dilution factor. Handles sample-to-sample variation well.
#                Recommended if TIC gives poor results.
#
#   'quantile' — forces all samples to have the same intensity distribution.
#                Very aggressive normalization — use only if you are confident
#                that systematic distributional differences between samples are
#                technical, not biological.
#
#   'none'     — no normalization applied. Use only if your samples were
#                collected under strictly identical conditions and you trust
#                the raw signal intensities.
NORMALIZATION = 'tic'


# ── Transformation ─────────────────────────────────────────────────────────────
# Log10 transformation compresses the wide dynamic range of MS data so that
# high-intensity features do not dominate the analysis.
# Options:
#   'log10'  — recommended for most MS metabolomics data (default)
#   'log2'   — similar to log10 but with base 2; common in transcriptomics
#   'sqrt'   — square root transform; gentler compression than log
#   'none'   — no transformation applied
LOG_TRANSFORM = 'log10'


# ── Scaling ────────────────────────────────────────────────────────────────────
# Scaling adjusts for differences in the magnitude of features so that all
# features contribute equally to the analysis regardless of their raw intensity.
#
# Van den Berg et al. (2006) compared scaling methods for metabolomics and found
# that autoscaling and range scaling best removed the dependence of feature
# rankings on average concentration. Choose based on your biological question:
#
#   'autoscale'  — subtract mean, divide by std dev. All features equally
#                  weighted. Best for comparing relative changes across features.
#                  Default and recommended for exploratory analysis.
#
#   'pareto'     — divide by square root of std dev. Large fold changes are
#                  reduced but the data is not made dimensionless. A compromise
#                  between autoscaling and no scaling.
#
#   'range'      — divide by the biological range (max - min). Scales features
#                  relative to their observed biological variation. Good when
#                  the biological range is meaningful.
#
#   'vast'       — variable stability scaling. Downweights features with high
#                  relative variation. Useful when you want to focus on stable,
#                  consistently-changing features.
#
#   'level'      — divide by the mean. Converts intensities to fold changes
#                  relative to the average. Useful for identifying biomarkers
#                  with large relative changes.
#
#   'none'       — no scaling applied. High-abundance features will dominate
#                  the analysis. Not recommended unless you have a specific reason.
SCALING = 'autoscale'


# ── PLS-DA ─────────────────────────────────────────────────────────────────────
# Number of components for the PLS-DA scores plot.
# More components capture more variation but can overfit.
# 8 is a reasonable default for exploratory analysis.
# Note: VIP scores always use 1 component regardless of this setting.
N_PLSDA_COMPONENTS = 8


# ── VIP Plot ───────────────────────────────────────────────────────────────────
# How many top VIP features to show in the dot plot and heatmap.
N_TOP_VIP = 30


# ── Classifiers ────────────────────────────────────────────────────────────────
# Set to True to include a classifier, False to skip it.
# Disabling slow classifiers (SVM, Gradient Boosting) can speed up the pipeline
# on large datasets.
USE_RANDOM_FOREST       = True
USE_SVM                 = True
USE_GRADIENT_BOOSTING   = True
USE_LOGISTIC_REGRESSION = True
USE_KNN                 = True
USE_LDA                 = True
USE_RIDGE               = True


# ── Cross-validation ───────────────────────────────────────────────────────────
# Number of folds for stratified k-fold cross-validation.
# 5 is standard. Use a smaller number (e.g. 3) if you have very few samples.
CV_FOLDS = 5


# ── Feature Importance Overlap ─────────────────────────────────────────────────
# How many top features to take from each method when looking for overlap.
# Features appearing in at least 2 of the 5 interpretable methods are reported.
TOP_N_FEATURES = 50
