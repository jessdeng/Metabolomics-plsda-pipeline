import os
import sys
# Ensure repo root is on the path so config.py and sibling packages are found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
extras.py — Optional Analysis Extras
======================================
Run this file after run_analysis.py to generate additional outputs.
All settings are controlled from config.py.

Usage:
    python extras.py
"""

import os
import glob
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import config
from r_comparable.preprocessing import (
    load_experiment, bin_features,
    filter_low_variance, filter_low_abundance, preprocess
)
from r_comparable.pipeline import compute_vip_1comp, fit_plsda
from shared.classifier_comparison import (
    RandomForest, svm_classify, gradient_boosting,
    logistic_regression, knn_classify, lda_classify, ridge_classify,
    feature_importance_analysis
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── Shared data loading ────────────────────────────────────────────────────────

def _load_and_preprocess(experiment_name):
    """Load, filter, and preprocess one experiment. Returns X, y, mz, X_filt_raw."""
    experiment_dir = os.path.join(BASE_DIR, experiment_name)
    assert os.path.isdir(experiment_dir), (
        f"Experiment folder not found: {experiment_dir!r}"
    )

    X_raw, y_labels, _, mz = load_experiment(experiment_dir)
    X_binned, mz = bin_features(X_raw, mz, bin_width=config.BIN_WIDTH)

    if config.VARIANCE_PERCENTILE > 0:
        X_filt, mz = filter_low_variance(X_binned, mz, percentile=config.VARIANCE_PERCENTILE)
    else:
        X_filt = X_binned.copy()

    if config.ABUNDANCE_PERCENTILE > 0:
        X_filt, mz = filter_low_abundance(X_filt, mz, percentile=config.ABUNDANCE_PERCENTILE)

    X_filt_raw = X_filt.copy()
    X = preprocess(
        X_filt,
        normalization=config.NORMALIZATION,
        log_transform=config.LOG_TRANSFORM,
        scaling=config.SCALING
    )
    return X, y_labels, mz, X_filt_raw


# ── 1. Summary Report ─────────────────────────────────────────────────────────

def run_summary_report(X, y_labels, mz, safe_name, out_dir, classifier_results=None):
    """
    Save a plain text summary of the pipeline settings and results.
    Accepts pre-computed classifier_results dict {name: (test_accs, train_accs)}
    to avoid re-running cross-validation. If not provided, runs classifiers fresh.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("AMBIENT MS METABOLOMICS TOOLKIT — RUN SUMMARY")
    lines.append(f"Pipeline        : MetaboAnalyst-compatible")
    lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 60)

    lines.append("\n── Experiment ──────────────────────────────────────────")
    lines.append(f"  Name        : {config.EXPERIMENT}")
    lines.append(f"  Samples     : {X.shape[0]}")
    lines.append(f"  Features    : {X.shape[1]} (after filtering)")
    lines.append(f"  m/z range   : {mz.min():.1f} – {mz.max():.1f} Da")

    groups, counts = np.unique(y_labels, return_counts=True)
    lines.append(f"  Groups      :")
    for g, c in zip(groups, counts):
        lines.append(f"    {g:30s}  n={c}")

    if len(groups) < 2:
        lines.append("\n  [warning] Only 1 group detected — cannot run classification.")
    elif any(c < 5 for c in counts):
        lines.append("\n  [warning] At least one group has fewer than 5 samples.")
        lines.append("  Cross-validation results may be unreliable at this sample size.")

    lines.append("\n── Preprocessing Settings ──────────────────────────────")
    lines.append(f"  Bin width           : {config.BIN_WIDTH} Da")
    lines.append(f"  Variance filter     : {config.VARIANCE_PERCENTILE}%")
    lines.append(f"  Abundance filter    : {config.ABUNDANCE_PERCENTILE}%")
    lines.append(f"  Normalization       : {config.NORMALIZATION}")
    lines.append(f"  Transformation      : {config.LOG_TRANSFORM}")
    lines.append(f"  Scaling             : {config.SCALING}")
    lines.append(f"  PLS-DA components   : {config.N_PLSDA_COMPONENTS}")
    lines.append(f"  Top VIP features    : {config.N_TOP_VIP}")
    lines.append(f"  CV folds            : {config.CV_FOLDS}")
    lines.append(f"  Top N for overlap   : {config.TOP_N_FEATURES}")

    lines.append("\n── Classifiers ─────────────────────────────────────────")
    classifier_fns = {
        'Random Forest':       (config.USE_RANDOM_FOREST,       RandomForest),
        'SVM':                 (config.USE_SVM,                 svm_classify),
        'Gradient Boosting':   (config.USE_GRADIENT_BOOSTING,   gradient_boosting),
        'Logistic Regression': (config.USE_LOGISTIC_REGRESSION, logistic_regression),
        'KNN':                 (config.USE_KNN,                 knn_classify),
        'LDA':                 (config.USE_LDA,                 lda_classify),
        'Ridge':               (config.USE_RIDGE,               ridge_classify),
    }
    for name, (enabled, fn) in classifier_fns.items():
        if not enabled:
            lines.append(f"  {name:22s}  [disabled]")
        elif classifier_results is not None and name in classifier_results:
            test_accs, train_accs = classifier_results[name]
            lines.append(f"  {name:22s}  test={test_accs.mean():.3f} ± {test_accs.std():.3f}  "
                         f"train={train_accs.mean():.3f} ± {train_accs.std():.3f}")
        else:
            test_accs, train_accs = fn(X, y_labels, n_splits=config.CV_FOLDS)
            lines.append(f"  {name:22s}  test={test_accs.mean():.3f} ± {test_accs.std():.3f}  "
                         f"train={train_accs.mean():.3f} ± {train_accs.std():.3f}")

    lines.append("\n── Output Files ────────────────────────────────────────")
    output_patterns = [
        os.path.join(out_dir, f"plsda_scores_3d_{safe_name}.html"),
        os.path.join(out_dir, f"vip_scores_{safe_name}.png"),
        os.path.join(out_dir, f"classifier_comparison_{safe_name}.png"),
        os.path.join(out_dir, f"spectrum_features_{safe_name}.png"),
        os.path.join(out_dir, f"feature_overlap_{safe_name}.csv"),
    ]
    for fpath in output_patterns:
        exists = "[ok]" if os.path.exists(fpath) else "[missing]"
        lines.append(f"  {exists}  {fpath}")

    lines.append("\n" + "=" * 60)

    out_path = os.path.join(out_dir, f"summary_{safe_name}.txt")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Saved → {out_path}")


# ── 2. Within-Group Variation Plot ────────────────────────────────────────────

def run_variation_plot(X, y_labels, safe_name, out_dir):
    """
    Box plot showing the distribution of preprocessed feature intensities
    per sample, grouped by class. Helps identify outlier samples.
    """
    groups = sorted(np.unique(y_labels))
    palette = sns.color_palette('colorblind', n_colors=len(groups))

    # Build a tidy dataframe: one row per (sample, feature)
    records = []
    for i, (label, row) in enumerate(zip(y_labels, X)):
        for val in row:
            records.append({'group': label, 'intensity': val})
    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 2), 5))
    sns.boxplot(data=df, x='group', y='intensity', palette=palette, ax=ax)
    ax.set_xlabel('Group')
    ax.set_ylabel('Preprocessed Intensity')
    ax.set_title(f'Within-Group Feature Intensity Distribution — {config.EXPERIMENT}')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"variation_plot_{safe_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")


# ── 3. Feature Correlation Heatmap ────────────────────────────────────────────

def run_correlation_heatmap(X, y_labels, mz, safe_name, out_dir):
    """
    Pearson correlation heatmap of the top N VIP features.
    Highly correlated features may represent the same compound at different
    charge states or isotopes.
    """
    vip = compute_vip_1comp(X, y_labels)
    top_idx = np.argsort(vip)[::-1][:config.N_CORR_FEATURES]
    top_mz = mz[top_idx]
    X_top = X[:, top_idx]

    corr = np.corrcoef(X_top.T)
    labels = [f"{v:.1f}" for v in top_mz]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        xticklabels=labels,
        yticklabels=labels,
        cmap='RdBu_r',
        center=0,
        vmin=-1, vmax=1,
        annot=False,
        ax=ax,
        linewidths=0.3
    )
    ax.set_title(f'Feature Correlation — Top {config.N_CORR_FEATURES} VIP Features\n{config.EXPERIMENT}',
                 fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"correlation_heatmap_{safe_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")


# ── 4. Cross-Experiment Comparison ────────────────────────────────────────────

def run_cross_experiment_comparison(out_dir):
    """
    Loads two experiments, runs feature importance on each, and identifies
    m/z features that appear in the top features of both experiments.
    """
    safe_a = config.EXPERIMENT_A.strip().replace(' ', '_').replace(':', '')
    safe_b = config.EXPERIMENT_B.strip().replace(' ', '_').replace(':', '')

    print(f"  Loading experiment A: {config.EXPERIMENT_A!r}")
    X_a, y_a, mz_a, _ = _load_and_preprocess(config.EXPERIMENT_A)

    print(f"  Loading experiment B: {config.EXPERIMENT_B!r}")
    X_b, y_b, mz_b, _ = _load_and_preprocess(config.EXPERIMENT_B)

    print("  Computing top features for experiment A...")
    overlap_a, _ = feature_importance_analysis(X_a, y_a, mz_a, safe_a + '_comparison', out_dir,
                                                top_n=config.TOP_N_FEATURES)

    print("  Computing top features for experiment B...")
    overlap_b, _ = feature_importance_analysis(X_b, y_b, mz_b, safe_b + '_comparison', out_dir,
                                                top_n=config.TOP_N_FEATURES)

    # Match features within 0.6 Da tolerance (slightly above bin width)
    tolerance = config.BIN_WIDTH * 1.2
    shared = []
    for mz_val_a in overlap_a['mz'].values:
        matches = overlap_b[np.abs(overlap_b['mz'].values - mz_val_a) <= tolerance]
        if len(matches) > 0:
            shared.append({
                'mz': mz_val_a,
                'n_methods_A': overlap_a.loc[overlap_a['mz'] == mz_val_a, 'n_methods'].values[0],
                'n_methods_B': matches['n_methods'].values[0],
            })

    if shared:
        shared_df = pd.DataFrame(shared).sort_values('n_methods_A', ascending=False)
        out_path = os.path.join(out_dir, f"comparison_{safe_a}_vs_{safe_b}.csv")
        shared_df.to_csv(out_path, index=False, encoding='utf-8')
        print(f"  Shared features found: {len(shared_df)}")
        print(f"  Saved → {out_path}")

        # Plot shared features
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.vlines(shared_df['mz'], 0, 1, color=sns.color_palette('colorblind')[0], alpha=0.7, linewidth=1)
        ax.set_xlabel('m/z (Da)')
        ax.set_yticks([])
        ax.set_title(f'Shared Important Features\n{config.EXPERIMENT_A}  vs  {config.EXPERIMENT_B}')
        ax.set_xlim(config.MZ_MIN, config.MZ_MAX)
        plt.tight_layout()
        plot_path = os.path.join(out_dir, f"comparison_{safe_a}_vs_{safe_b}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved → {plot_path}")
    else:
        print("  No shared features found within tolerance.")


# ── 5. Reproducibility Report ─────────────────────────────────────────────────

def run_reproducibility_report(X, y_labels, mz, safe_name, out_dir):
    """
    Runs Random Forest and Logistic Regression twice with different random seeds
    and reports what fraction of top features are consistent across both runs.
    A high overlap means the feature rankings are stable and trustworthy.
    """
    top_n = config.TOP_N_FEATURES
    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    results = []
    for name, ModelClass, kwargs in [
        ('Random Forest',       RandomForestClassifier,  {'n_estimators': 100}),
        ('Logistic Regression', LogisticRegression,      {'max_iter': 1000}),
        ('Ridge',               RidgeClassifier,         {}),
        ('SVM',                 SVC,                     {'kernel': 'linear'}),
    ]:
        model_a = ModelClass(random_state=42, **kwargs)
        model_b = ModelClass(random_state=99, **kwargs)
        model_a.fit(X, y)
        model_b.fit(X, y)

        if hasattr(model_a, 'feature_importances_'):
            imp_a = model_a.feature_importances_
            imp_b = model_b.feature_importances_
        else:
            imp_a = np.abs(model_a.coef_).mean(axis=0)
            imp_b = np.abs(model_b.coef_).mean(axis=0)

        top_a = set(np.argsort(imp_a)[::-1][:top_n])
        top_b = set(np.argsort(imp_b)[::-1][:top_n])
        overlap = len(top_a & top_b)
        stability = overlap / top_n * 100

        results.append({
            'Model': name,
            f'Top {top_n} overlap (%)': f"{stability:.1f}%",
            'Shared features': overlap,
        })
        print(f"  {name:22s}  {stability:.1f}% of top {top_n} features stable across seeds")

    df = pd.DataFrame(results)
    out_path = os.path.join(out_dir, f"reproducibility_{safe_name}.csv")
    df.to_csv(out_path, index=False, encoding='utf-8')
    print(f"  Saved → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    experiment_name = config.EXPERIMENT.strip()
    safe_name = experiment_name.replace(' ', '_').replace(':', '')

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = os.path.join(BASE_DIR, 'output_r_comparable')
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nLoading data for: {experiment_name!r}")
    X, y_labels, mz, X_filt_raw = _load_and_preprocess(experiment_name)
    print(f"  {X.shape[0]} samples, {X.shape[1]} features")

    # ── Load or compute classifier results ───────────────────────────────────
    classifier_results = {}
    results_path = os.path.join(out_dir, f"classifier_results_{safe_name}.npz")
    if config.RUN_SUMMARY_REPORT:
        if os.path.exists(results_path):
            print(f"\n[Extras] Loading saved classifier results from {results_path}")
            saved = np.load(results_path)
            names = set(k.replace('__test', '').replace('__train', '') for k in saved.files)
            for name in names:
                classifier_results[name] = (saved[f"{name}__test"], saved[f"{name}__train"])
        else:
            print(f"\n[Extras] No saved classifier results found ({results_path})")
            print("  Running classifiers now — run run_analysis.py first to avoid this.")
            classifier_fns = {
                'Random Forest':       (config.USE_RANDOM_FOREST,       RandomForest),
                'SVM':                 (config.USE_SVM,                 svm_classify),
                'Gradient Boosting':   (config.USE_GRADIENT_BOOSTING,   gradient_boosting),
                'Logistic Regression': (config.USE_LOGISTIC_REGRESSION, logistic_regression),
                'KNN':                 (config.USE_KNN,                 knn_classify),
                'LDA':                 (config.USE_LDA,                 lda_classify),
                'Ridge':               (config.USE_RIDGE,               ridge_classify),
            }
            for name, (enabled, fn) in classifier_fns.items():
                if enabled:
                    print(f"  {name}...")
                    classifier_results[name] = fn(X, y_labels, n_splits=config.CV_FOLDS)

    if config.RUN_SUMMARY_REPORT:
        print("\n[Extras] Summary Report")
        run_summary_report(X, y_labels, mz, safe_name, out_dir, classifier_results=classifier_results)

    if config.RUN_VARIATION_PLOT:
        print("\n[Extras] Within-Group Variation Plot")
        run_variation_plot(X, y_labels, safe_name, out_dir)

    if config.RUN_CORRELATION_HEATMAP:
        print("\n[Extras] Feature Correlation Heatmap")
        run_correlation_heatmap(X, y_labels, mz, safe_name, out_dir)

    if config.RUN_COMPARISON:
        print(f"\n[Extras] Cross-Experiment Comparison")
        print(f"  A: {config.EXPERIMENT_A!r}")
        print(f"  B: {config.EXPERIMENT_B!r}")
        run_cross_experiment_comparison(out_dir)

    if config.RUN_REPRODUCIBILITY:
        print("\n[Extras] Reproducibility Report")
        run_reproducibility_report(X, y_labels, mz, safe_name, out_dir)

    print("\nDone.")

if __name__ == '__main__':
    main()
