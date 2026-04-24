# Changelog

## [Unreleased] — Per-group attribution for ensemble features

### Added
- `feature_overlap_<experiment>.csv` now includes per-group attribution columns
  that answer the question "which condition is this feature coming from?":
  - `mean_<group>` — mean log-normalised intensity in each group (univariate,
    raw-biology view)
  - `ridge_<group>` — signed one-vs-rest Ridge coefficient in each group
    (multivariate, model-attribution view; positive = associated with that
    group, negative = suppressed)
  - `top_condition_mean` — group with the highest mean intensity
  - `top_condition_ridge` — group with the largest positive Ridge coefficient
  - `mean_margin` — top mean ÷ second-highest mean; values close to 1.0 flag
    features that are similarly abundant in two or more groups
  - `ridge_direction` — `'elevated'`, `'suppressed'`, or `'mixed'` based on
    the sign pattern of the Ridge coefficients across groups

  Reporting both `mean` and `ridge` makes it easy to spot the interesting case
  where they disagree: when a feature ranks highly because of its multivariate
  structure rather than raw abundance.

### Changed
- `feature_importance_analysis()` (in both `shared/classifier_comparison.py`
  and `shared/classifier_comparison_standard.py`) now accepts an optional
  `X_norm` argument: the same matrix as `X` but normalised + log-transformed
  only (no scaling). This is what the `mean_<group>` columns are computed on,
  so values are interpretable on the raw log-intensity axis. If `X_norm` is
  not provided the function still runs and falls back to the scaled matrix
  (with a warning) for backward compatibility.
- `standard/run_analysis.py` and `r_comparable/run_analysis.py` now compute
  `X_norm` alongside `X` in the preprocessing step and pass it through.
- `scripts/condition_abundance.py` simplified: it no longer re-fits Ridge or
  re-loads the experiment. It reads the per-group columns directly from
  `feature_overlap_<experiment>.csv` and produces the side-by-side
  mean-abundance / Ridge-coefficient heatmap for high-confidence features
  (n_methods ≥ MIN_N_METHODS).
- `README.md` — output-files table updated to describe the enriched CSV.
- `NOTES.md` — feature-importance section restructured to document the new
  columns and how to interpret `top_condition_mean` vs `top_condition_ridge`
  agreement and disagreement.

### Migration notes
- Old `feature_overlap_<experiment>.csv` files generated before this change
  will not have the new columns. Re-run `python -m standard.run_analysis`
  (or the r_comparable equivalent) to regenerate them.
- `scripts/condition_abundance.py` will detect missing columns and print a
  message asking the user to re-run the main pipeline.
