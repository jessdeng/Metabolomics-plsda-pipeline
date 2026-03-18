"""
Classifier Comparison for Mass Spectrometry Data
=================================================
Pipeline: raw spectra → bin → filter → sum norm → log10 → auto-scale
          → Random Forest / SVM / Gradient Boosting

Usage:
    python classifier_comparison.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from preprocessing import load_experiment, bin_features, filter_low_variance, filter_low_abundance, preprocess
from metaboanalyst_pipeline import compute_vip_1comp
from collections import Counter

def RandomForest(X, y_labels, n_splits=5, random_state=42):
    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_accs = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        fold_accs.append(acc)
        print(f"    Fold {fold + 1}: {acc:.3f}")

    return np.array(fold_accs)


def svm_classify(X, y_labels, n_splits=5, random_state=42):
    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_accs = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model = SVC(kernel='linear', random_state=random_state)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        fold_accs.append(acc)
        print(f"    Fold {fold + 1}: {acc:.3f}")

    return np.array(fold_accs)


def gradient_boosting(X, y_labels, n_splits=5, random_state=42):
    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_accs = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=random_state, max_depth=3)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        fold_accs.append(acc)
        print(f"    Fold {fold + 1}: {acc:.3f}")

    return np.array(fold_accs)


def plot_accuracy_comparison(rf_accs, svm_accs, gb_accs, experiment_name, out_path):
    names = ['Random Forest', 'SVM', 'Gradient Boosting']
    means = [rf_accs.mean(), svm_accs.mean(), gb_accs.mean()]
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(names, means)
    ax.set_ylabel('Mean Accuracy')
    ax.set_title(f'Classifier Comparison — {experiment_name}')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")


def feature_importance_analysis(X, y_labels, mz, safe_name):
    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_importance = rf.feature_importances_

    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X, y)
    svm_importance = np.abs(svm.coef_).mean(axis=0)

    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gb.fit(X, y)
    gb_importance = gb.feature_importances_

    vip_importance = compute_vip_1comp(X, y_labels)

    top_n = 50
    rf_top = set(np.argsort(rf_importance)[::-1][:top_n])
    svm_top = set(np.argsort(svm_importance)[::-1][:top_n])
    gb_top = set(np.argsort(gb_importance)[::-1][:top_n])
    vip_top = set(np.argsort(vip_importance)[::-1][:top_n])

    all_tops = list(rf_top) + list(svm_top) + list(gb_top) + list(vip_top)
    counts = Counter(all_tops)

    overlap_2plus = {idx for idx, count in counts.items() if count >= 2}
    overlap_3plus = {idx for idx, count in counts.items() if count >= 3}

    print(f"  Features in at least 2 of 4 methods: {len(overlap_2plus)}")
    print(f"  Features in at least 3 of 4 methods: {len(overlap_3plus)}")

    overlap_list = sorted(overlap_2plus)
    overlap_df = pd.DataFrame({
        'mz': mz[overlap_list],
        'rf_importance': rf_importance[overlap_list],
        'svm_importance': svm_importance[overlap_list],
        'gb_importance': gb_importance[overlap_list],
        'vip_score': vip_importance[overlap_list],
        'n_methods': [counts[idx] for idx in overlap_list]
    }).sort_values('n_methods', ascending=False)
    overlap_df.to_csv(f'feature_overlap_{safe_name}.csv', index=False)
    print(f"  Saved → feature_overlap_{safe_name}.csv")
    
    return overlap_df, counts