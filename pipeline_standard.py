"""
Standard Pipeline — no MetaboAnalyst offset
============================================
Identical to metaboanalyst_pipeline.py except it imports from preprocessing_standard,
which labels bins at their true geometric centers (no -0.05 Da offset).

Use this version when MetaboAnalyst replication is not the goal.
Use metaboanalyst_pipeline.py when you need output to match MetaboAnalyst exactly.

Usage:
    python run_analysis_standard.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.graph_objects as go 

from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from preprocessing_standard import load_experiment, bin_features, filter_low_variance, filter_low_abundance, preprocess

# ── Configuration ─────────────────────────────────────────────────────────────
N_COMPONENTS = 8   # number of PLS-DA components for scores plot and cross-validation
N_TOP_VIP   = 30   # how many top VIP features to show in the bar chart
# ──────────────────────────────────────────────────────────────────────────────
def fit_plsda(X, y_labels, n_components):
    """
    Fit PLS-DA with one-hot encoded Y.
    Used for scores plot and cross-validation (8 components).
    """
    le  = LabelEncoder()
    y   = le.fit_transform(y_labels)
    Y   = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))

    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(X, Y)
    T = pls.transform(X)

    return pls, T, y, Y, le.classes_


def cross_validate(X, y, n_components, n_splits=5, random_state=42):
    """Stratified k-fold cross-validation. Returns per-fold accuracies."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_accs = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        Y_tr = OneHotEncoder(sparse_output=False).fit_transform(y_tr.reshape(-1, 1))
        model = PLSRegression(n_components=n_components, scale=False)
        model.fit(X_tr, Y_tr)

        y_pred = model.predict(X_te).argmax(axis=1)
        acc = accuracy_score(y_te, y_pred)
        fold_accs.append(acc)
        print(f"    Fold {fold + 1}: {acc:.3f}")

    return np.array(fold_accs)


def compute_vip_1comp(X, y_labels):
    """
    Compute VIP scores using only 1 PLS-DA component.
    This matches MetaboAnalyst's component 1 VIP exactly.
    """
    le = LabelEncoder()
    y  = le.fit_transform(y_labels)
    Y  = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))

    pls = PLSRegression(n_components=1, scale=False)
    pls.fit(X, Y)

    T = pls.x_scores_       # (n_samples, 1)
    W = pls.x_weights_      # (n_features, 1)
    Q = pls.y_loadings_     # (n_groups, 1)

    n_features = X.shape[1]
    SS     = np.sum(T ** 2, axis=0) * np.sum(Q ** 2, axis=0)
    W_norm = W / np.sqrt(np.sum(W ** 2, axis=0))
    vip    = np.sqrt(n_features * (W_norm ** 2 @ SS) / SS.sum())

    return vip


def plot_scores_3d(T, pls, y_labels, classes, experiment_name, out_path):
    T_all = pls.x_scores_
    Q = pls.y_loadings_
    SS = np.sum(T_all ** 2, axis=0) * np.sum(Q ** 2, axis=0)
    pct_cov = SS / SS.sum() * 100

    fig = go.Figure()
    palette = sns.color_palette('colorblind', n_colors=len(classes))

    for i, group in enumerate(classes):
        mask = y_labels == group
        r, g, b = [int(c * 255) for c in palette[i][:3]]
        fig.add_trace(go.Scatter3d(
            x=T[mask, 0],
            y=T[mask, 1],
            z=T[mask, 2],
            mode='markers',
            name=group,
            marker=dict(size=5, color=f'rgb({r},{g},{b})')
        ))

    fig.update_layout(
        title=f'PLS-DA Scores — {experiment_name}',
        scene=dict(
            xaxis_title=f'Component 1 ({pct_cov[0]:.1f}%)',
            yaxis_title=f'Component 2 ({pct_cov[1]:.1f}%)',
            zaxis_title=f'Component 3 ({pct_cov[2]:.1f}%)'
        )
    )
    fig.write_html(out_path)
    print(f"  Saved → {out_path}")

def plot_vip(vip, mz, X, y_labels, n_top, experiment_name, out_path):
    top_idx = np.argsort(vip)[::-1][:n_top]
    top_mz = mz[top_idx]
    top_vip = vip[top_idx]

    top_idx = top_idx[::-1]
    top_mz = top_mz[::-1]
    top_vip = top_vip[::-1]

    groups = sorted(np.unique(y_labels))
    n_groups = len(groups)

    heatmap_data = np.zeros((n_top, n_groups))
    for i, idx in enumerate(top_idx):
        for j, group in enumerate(groups):
            mask = y_labels == group
            heatmap_data[i, j] = X[mask, idx].mean()

    row_mins = heatmap_data.min(axis=1, keepdims=True)
    row_maxs = heatmap_data.max(axis=1, keepdims=True)
    row_range = row_maxs - row_mins
    row_range[row_range == 0] = 1
    heatmap_norm = (heatmap_data - row_mins) / row_range

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1.2], wspace=0.05)

    x_min = top_vip.min() * 0.95

    ax_dot = fig.add_subplot(gs[0])
    for i in range(n_top):
        ax_dot.plot([1.7, top_vip[i]], [i, i], color='grey', linewidth=0.5)
        ax_dot.plot(top_vip[i], i, 'o', color='#555555', markersize=7)

    ax_dot.set_yticks(range(n_top))
    ax_dot.set_yticklabels([f"{v:.1f}" for v in top_mz], fontsize=9)
    ax_dot.set_xlabel('VIP Scores')
    ax_dot.set_xlim(top_vip.min() * 0.95, top_vip.max() * 1.05)

    ax_heat = fig.add_subplot(gs[1])
    cmap = plt.cm.RdBu_r
    ax_heat.imshow(heatmap_norm, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    for i in range(n_top + 1):
        ax_heat.axhline(i - 0.5, color='white', linewidth=1)
    for j in range(n_groups + 1):
        ax_heat.axvline(j - 0.5, color='white', linewidth=1)

    ax_heat.set_xticks(range(n_groups))
    ax_heat.set_xticklabels(groups, rotation=45, ha='right', fontsize=8)
    ax_heat.set_yticks([])

    ax_dot.set_ylim(-0.5, n_top - 0.5)
    ax_heat.set_ylim(-0.5, n_top - 0.5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_heat, shrink=0.4, aspect=15, pad=0.02)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Low', 'High'])

    fig.suptitle(f'Top {n_top} VIP Features — {experiment_name}', fontsize=13)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")