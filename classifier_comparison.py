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
    - K-Nearest Neighbors
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from metaboanalyst_pipeline import compute_vip_1comp


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


def knn_classify(X, y_labels, n_splits=5, random_state=42):
    y = _encode(y_labels)
    return _run_cv(
        lambda: KNeighborsClassifier(n_neighbors=5),
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

def feature_importance_analysis(X, y_labels, mz, safe_name, top_n=50):
    """
    Fits RF, SVM, GB, LR, and PLS-DA (VIP) on the full dataset and finds
    m/z features that appear in the top 50 of at least 2 methods.

    Note: KNN is excluded here because it is distance-based and does not
    produce interpretable per-feature importance scores. It contributes to
    the accuracy comparison only. Ridge Regression is included via its
    linear coefficients.
    """
    le = LabelEncoder()
    y = le.fit_transform(y_labels)

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
    }).sort_values('n_methods', ascending=False)

    overlap_df.to_csv(f'feature_overlap_{safe_name}.csv', index=False, encoding='utf-8')
    print(f"  Saved → feature_overlap_{safe_name}.csv")

    return overlap_df, counts
