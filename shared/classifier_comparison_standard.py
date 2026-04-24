import os
"""
Classifier Comparison for Mass Spectrometry Metabolomics Data
=============================================================
Trains and evaluates 7 supervised classifiers using 5-fold stratified
cross-validation, then plots per-fold accuracy, mean accuracy, and a
train vs test comparison to flag potential overfitting.

Classifiers:
    - Random Forest
    - Support Vector Machine (linear kernel)
    - Gradient Boosting
    - Logistic Regression
    - Linear Discriminant Analysis
    - Ridge Regression

Usage:
    python run_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from standard.pipeline import compute_vip_1comp


# ── Shared cross-validation runner ────────────────────────────────────────────

def _run_cv(model_fn, X, y, n_splits=5, random_state=42):
    """
    Run stratified k-fold CV for any model.
    Returns (test_accs, train_accs) as numpy arrays.
    model_fn: callable that returns a fresh unfitted model instance.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    test_accs, train_accs = [], []

    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model = model_fn()
        model.fit(X_tr, y_tr)

        train_accs.append(accuracy_score(y_tr, model.predict(X_tr)))
        test_accs.append(accuracy_score(y_te, model.predict(X_te)))

    return np.array(test_accs), np.array(train_accs)


def _encode(y_labels):
    le = LabelEncoder()
    return le.fit_transform(y_labels)


# ── Individual classifiers ────────────────────────────────────────────────────

def RandomForest(X, y_labels, n_splits=5, random_state=42):
    y = _encode(y_labels)
    return _run_cv(
        lambda: RandomForestClassifier(n_estimators=100, random_state=random_state),
        X, y, n_splits, random_state
    )


def svm_classify(X, y_labels, n_splits=5, random_state=42):
    y = _encode(y_labels)
    return _run_cv(
        lambda: SVC(kernel='linear', random_state=random_state),
        X, y, n_splits, random_state
    )


def gradient_boosting(X, y_labels, n_splits=5, random_state=42):
    y = _encode(y_labels)
    return _run_cv(
        lambda: GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state
        ),
        X, y, n_splits, random_state
    )


def logistic_regression(X, y_labels, n_splits=5, random_state=42):
    y = _encode(y_labels)
    return _run_cv(
        lambda: LogisticRegression(max_iter=1000, random_state=random_state),
        X, y, n_splits, random_state
    )


def lda_classify(X, y_labels, n_splits=5, random_state=42):
    y = _encode(y_labels)
    return _run_cv(
        lambda: LinearDiscriminantAnalysis(),
        X, y, n_splits, random_state
    )


def ridge_classify(X, y_labels, n_splits=5, random_state=42):
    y = _encode(y_labels)
    return _run_cv(
        lambda: RidgeClassifier(),
        X, y, n_splits, random_state
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_accuracy_comparison(results, experiment_name, out_path):
    """
    results: dict of {model_name: (test_accs, train_accs)}
    Produces two panels:
      - Top: per-fold test accuracy dots + mean bar
      - Bottom: mean train vs test accuracy to flag overfitting
    """
    names = list(results.keys())
    n = len(names)
    palette = sns.color_palette('colorblind', n_colors=n)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(max(10, n * 1.4), 10),
        gridspec_kw={'height_ratios': [2, 1]}
    )

    # ── Top panel: per-fold dots + mean bar ───────────────────────────────────
    for i, (name, (test_accs, train_accs)) in enumerate(results.items()):
        color = palette[i]
        # Mean bar
        ax_top.bar(i, test_accs.mean(), color=color, alpha=0.5, width=0.6, zorder=1)
        # Per-fold dots
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(test_accs))
        ax_top.scatter(
            np.full(len(test_accs), i) + jitter,
            test_accs,
            color=color, edgecolors='black', linewidths=0.5,
            s=50, zorder=2
        )
        # Mean line
        ax_top.hlines(
            test_accs.mean(), i - 0.3, i + 0.3,
            colors='black', linewidths=1.5, zorder=3
        )

    ax_top.set_xticks(range(n))
    ax_top.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax_top.set_ylabel('Test Accuracy (5-fold CV)')
    ax_top.set_ylim(0, 1.05)
    ax_top.set_title(f'Classifier Comparison — {experiment_name}')
    ax_top.axhline(0.5, color='grey', linestyle='--', linewidth=0.8, alpha=0.5,
                   label='Chance (0.5)')
    ax_top.legend(fontsize=8)

    # ── Bottom panel: train vs test mean (overfitting check) ──────────────────
    x = np.arange(n)
    width = 0.35
    train_means = [results[name][1].mean() for name in names]
    test_means  = [results[name][0].mean() for name in names]

    cb = sns.color_palette('colorblind')
    ax_bot.bar(x - width / 2, train_means, width, label='Train', color=cb[0], alpha=0.7)
    ax_bot.bar(x + width / 2, test_means,  width, label='Test',  color=cb[1], alpha=0.7)

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax_bot.set_ylabel('Mean Accuracy')
    ax_bot.set_ylim(0, 1.05)
    ax_bot.set_title('Train vs Test Accuracy (overfitting check)')
    ax_bot.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")


# ── Feature importance overlap ────────────────────────────────────────────────

def feature_importance_analysis(X, y_labels, mz, safe_name, out_dir,
                                top_n=50, X_norm=None, log_transform='log10'):
    """
    Fits RF, SVM, GB, LR, Ridge, and PLS-DA (VIP) on the full dataset and finds
    m/z features that appear in the top `top_n` of at least 2 of these 6 methods.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
        Fully preprocessed feature matrix (normalised + transformed + scaled).
        Used by all classifiers and for the Ridge one-vs-rest coefficients.
    y_labels : ndarray of str
        Group labels per sample.
    mz : ndarray
        m/z value for each feature.
    safe_name : str
        Filesystem-safe experiment name (used in output filename).
    out_dir : str
        Directory where the feature_overlap CSV is written.
    top_n : int
        How many top-ranked features per method to count for the overlap.
    X_norm : ndarray (n_samples, n_features), optional
        Same shape as X, but normalised + log-transformed only (no scaling).
        Used for per-group mean intensities so the values are comparable to
        the raw spectrum. If None, mean columns are skipped (back-compat).
    log_transform : str
        Transformation used to produce X_norm — one of 'log10', 'log2', 'sqrt',
        'none'. Used to compute mean_margin in linear space (a ratio) rather
        than as a quotient on the transformed axis (which would be meaningless).

    Output CSV columns
    ------------------
    Identification:
        mz, n_methods
    Per-method importances:
        rf_importance, svm_importance, gb_importance, lr_importance,
        ridge_importance, vip_score
    Per-group attribution (NEW):
        mean_<group>          — mean log-normalised intensity per group
        ridge_<group>         — signed one-vs-rest Ridge coefficient per group
        top_condition_mean    — group with the highest mean intensity
        top_condition_ridge   — group with the largest positive Ridge coefficient
        mean_margin           — top mean / second-highest mean in LINEAR space
                                (>=1; values close to 1 = ambiguous call)
        ridge_direction       — 'elevated' / 'suppressed' / 'mixed'
    """
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    classes = le.classes_

    # Fit each model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_imp = rf.feature_importances_

    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X, y)
    svm_imp = np.abs(svm.coef_).mean(axis=0) if svm.coef_.ndim > 1 else np.abs(svm.coef_).ravel()

    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                    max_depth=3, random_state=42)
    gb.fit(X, y)
    gb_imp = gb.feature_importances_

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X, y)
    lr_imp = np.abs(lr.coef_).mean(axis=0) if lr.coef_.ndim > 1 else np.abs(lr.coef_).ravel()

    ridge = RidgeClassifier()
    ridge.fit(X, y)
    ridge_imp = np.abs(ridge.coef_).mean(axis=0) if ridge.coef_.ndim > 1 else np.abs(ridge.coef_).ravel()

    vip_imp = compute_vip_1comp(X, y_labels)

    # Find top features per method
    tops = [
        set(np.argsort(imp)[::-1][:top_n])
        for imp in [rf_imp, svm_imp, gb_imp, lr_imp, ridge_imp, vip_imp]
    ]
    counts = Counter(idx for top in tops for idx in top)

    overlap_2plus = {idx for idx, c in counts.items() if c >= 2}
    overlap_3plus = {idx for idx, c in counts.items() if c >= 3}

    print(f"  Features in at least 2 of 6 methods: {len(overlap_2plus)}")
    print(f"  Features in at least 3 of 6 methods: {len(overlap_3plus)}")

    overlap_list = sorted(overlap_2plus)
    overlap_df = pd.DataFrame({
        'mz':              mz[overlap_list],
        'rf_importance':   rf_imp[overlap_list],
        'svm_importance':  svm_imp[overlap_list],
        'gb_importance':   gb_imp[overlap_list],
        'lr_importance':   lr_imp[overlap_list],
        'ridge_importance': ridge_imp[overlap_list],
        'vip_score':       vip_imp[overlap_list],
        'n_methods':       [counts[idx] for idx in overlap_list],
    })

    # ── Per-group attribution columns ─────────────────────────────────────────
    print(f"  Adding per-group attribution: mean intensity + Ridge one-vs-rest")

    # Ridge one-vs-rest coefficients on the SCALED matrix (one row per class).
    # ridge.coef_ shape: (n_classes, n_features) when n_classes > 2,
    #                    (1, n_features)         when binary.
    if ridge.coef_.shape[0] == 1 and len(classes) == 2:
        # Binary case: scikit-learn returns one row representing class 1.
        # Mirror it so we have a positive/negative coefficient per class.
        ridge_signed = np.vstack([-ridge.coef_[0], ridge.coef_[0]])
    else:
        ridge_signed = ridge.coef_  # (n_classes, n_features)

    # Per-group mean intensity on the unscaled (but normalised + logged) matrix.
    # Falls back to the scaled matrix if X_norm not supplied (less interpretable
    # but keeps the function callable in legacy contexts).
    X_for_means = X_norm if X_norm is not None else X
    if X_norm is None:
        print("  WARNING: X_norm not provided — mean_<group> columns will be on "
              "the scaled axis, not raw log-intensity")

    for j, group in enumerate(classes):
        mask = (y_labels == group)
        means_full = X_for_means[mask].mean(axis=0)              # (n_features,)
        overlap_df[f'mean_{group}']  = means_full[overlap_list]
        overlap_df[f'ridge_{group}'] = ridge_signed[j, overlap_list]

    mean_cols  = [f'mean_{g}'  for g in classes]
    ridge_cols = [f'ridge_{g}' for g in classes]

    # Top condition by mean: highest mean intensity wins
    mean_arr = overlap_df[mean_cols].values
    top_idx_mean = np.argmax(mean_arr, axis=1)
    overlap_df['top_condition_mean'] = [classes[i] for i in top_idx_mean]

    # Mean margin: ratio of top mean to second-highest mean, in LINEAR space.
    # A simple quotient of the transformed values is meaningless (and on a log
    # axis can flip sign whenever the second-highest value is negative — which
    # is normal after log10 of a low intensity). We invert the transform first.
    sorted_means = np.sort(mean_arr, axis=1)[:, ::-1]            # descending
    top1 = sorted_means[:, 0]
    top2 = sorted_means[:, 1] if sorted_means.shape[1] > 1 else top1

    if log_transform == 'log10':
        overlap_df['mean_margin'] = 10.0 ** (top1 - top2)
    elif log_transform == 'log2':
        overlap_df['mean_margin'] = 2.0 ** (top1 - top2)
    elif log_transform == 'sqrt':
        # Invert sqrt by squaring, then take ratio in linear space.
        # Uses np.where to avoid division warnings; tiny squares -> nan margin.
        lin1 = top1 ** 2
        lin2 = top2 ** 2
        overlap_df['mean_margin'] = np.where(lin2 > 0, lin1 / lin2, np.nan)
    else:
        # No transform applied — values are already linear intensities.
        overlap_df['mean_margin'] = np.where(top2 > 0, top1 / top2, np.nan)

    # Top condition by Ridge: largest POSITIVE coefficient.
    # Direction tag interprets the sign pattern across groups.
    ridge_arr = overlap_df[ridge_cols].values
    top_idx_ridge = np.argmax(ridge_arr, axis=1)
    overlap_df['top_condition_ridge'] = [classes[i] for i in top_idx_ridge]

    n_pos = (ridge_arr > 0).sum(axis=1)
    n_neg = (ridge_arr < 0).sum(axis=1)
    direction = []
    for p, n in zip(n_pos, n_neg):
        if p == 1 and n >= 1:
            direction.append('elevated')      # single group up, others down
        elif n == 1 and p >= 1:
            direction.append('suppressed')    # single group down, others up
        else:
            direction.append('mixed')         # multiple groups same sign
    overlap_df['ridge_direction'] = direction

    overlap_df = overlap_df.sort_values('n_methods', ascending=False)

    csv_path = os.path.join(out_dir, f'feature_overlap_{safe_name}.csv')
    overlap_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"  Saved → {csv_path}")

    return overlap_df, counts
