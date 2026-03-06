"""
MetaboAnalyst Pipeline — rebuilt in Python
==========================================
Pipeline: raw spectra → bin → filter → sum norm → log10 → auto-scale → PLS-DA → VIP scores

Notes:
  - 8 components are used for the scores plot and cross-validation.
  - 1 component is used for VIP scores, because Python's PLS algorithm
    matches MetaboAnalyst exactly at component 1 but diverges at higher
    components due to internal algorithm differences between scikit-learn
    and R's ropls package.

Usage:
    python metaboanalyst_pipeline.py
"""

import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Change EXPERIMENT to whichever folder you want to analyse.
# Options (copy exactly, including any trailing spaces):
#   'S9 Carbon with Media '   ← note the trailing space
#   'S9 Lights with Media'
#   'S9 Nitrogen w: Media'
EXPERIMENT = 'S9 Carbon with Media '

N_COMPONENTS = 8   # number of PLS-DA components for scores plot and cross-validation
N_TOP_VIP   = 30   # how many top VIP features to show in the bar chart
# ──────────────────────────────────────────────────────────────────────────────


def load_experiment(experiment_dir):
    """
    Read all CSVs from the experiment folder.
    Each direct subfolder = one class/group.
    Each CSV inside = one sample.
    Returns X (n_samples × n_features), labels array, sample names, mz axis.
    """
    samples, labels, names = [], [], []
    mz = None

    group_folders = sorted(
        d for d in os.listdir(experiment_dir)
        if os.path.isdir(os.path.join(experiment_dir, d)) and not d.startswith('.')
    )

    for group in group_folders:
        csv_files = sorted(glob.glob(os.path.join(experiment_dir, group, '*.csv')))
        if not csv_files:
            print(f"  [warning] No CSV files in '{group}', skipping.")
            continue

        for csv_path in csv_files:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.lstrip('\ufeff').str.strip()
            col_map = {c.lower(): c for c in df.columns}

            if mz is None:
                mz = df[col_map['mz']].values

            samples.append(df[col_map['int']].values)
            labels.append(group.strip())
            names.append(os.path.basename(csv_path))

    return np.array(samples, dtype=float), np.array(labels), names, mz


def bin_features(X, mz, bin_width=0.5):
    """
    Bin m/z features into fixed-width bins by summing intensities.
    Bin centers are offset to match MetaboAnalyst's .2 / .7 labeling.
    """
    bin_edges = np.arange(mz.min(), mz.max() + bin_width, bin_width)
    bin_labels = bin_edges[:-1] + (bin_width / 2 - 0.05)

    X_binned = np.zeros((X.shape[0], len(bin_labels)))

    for i, (low, high) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (mz >= low) & (mz < high)
        if mask.any():
            X_binned[:, i] = X[:, mask].sum(axis=1)

    non_empty = X_binned.sum(axis=0) > 0
    return X_binned[:, non_empty], bin_labels[non_empty]


def filter_low_variance(X, mz, percentile=25):
    """
    Remove features with low relative standard deviation (RSD).
    Matches MetaboAnalyst's 'Interquartile range' filter at 25%.
    """
    rsd = X.std(axis=0) / X.mean(axis=0)
    threshold = np.percentile(rsd, percentile)
    keep = rsd > threshold
    return X[:, keep], mz[keep]


def filter_low_abundance(X, mz, percentile=5):
    """
    Remove features with low mean intensity.
    Uses mean (not median) and 5% cutoff to match MetaboAnalyst.
    """
    mean_intensity = np.mean(X, axis=0)
    threshold = np.percentile(mean_intensity, percentile)
    keep = mean_intensity > threshold
    return X[:, keep], mz[keep]


def preprocess(X):
    """Sum normalization → log10 → auto-scaling."""
    # Step 1: sum normalization, scaled by median total intensity
    row_sums = X.sum(axis=1, keepdims=True)
    X_norm = X / row_sums * np.median(row_sums)

    # Step 2: log10 with half-minimum imputation for zeros
    min_positive = X_norm[X_norm > 0].min()
    X_log = np.log10(X_norm + min_positive / 2)

    # Step 3: auto-scaling (mean-center, divide by std)
    feat_mean = X_log.mean(axis=0)
    feat_std  = X_log.std(axis=0, ddof=1)
    feat_std[feat_std == 0] = 1
    X_scaled = (X_log - feat_mean) / feat_std

    return X_scaled


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


def plot_scores(T, pls, X, y_labels, classes, experiment_name, out_path):
    """
    Scores plot with % covariance explained on each axis.
    The covariance explained is calculated as the proportion of the total
    X–Y covariance captured by each component.
    """
    # Calculate % covariance explained per component
    # SS per component = t't * q'q (how much X–Y relationship each component captures)
    T_all = pls.x_scores_
    Q     = pls.y_loadings_
    SS    = np.sum(T_all ** 2, axis=0) * np.sum(Q ** 2, axis=0)
    pct_cov = SS / SS.sum() * 100

    palette   = sns.color_palette('Set1', n_colors=len(classes))
    color_map = dict(zip(classes, palette))

    fig, ax = plt.subplots(figsize=(8, 6))
    for group in classes:
        mask = y_labels == group
        ax.scatter(T[mask, 0], T[mask, 1], label=group,
                   color=color_map[group], s=60, edgecolors='white', linewidths=0.5)

    ax.axhline(0, color='grey', lw=0.5, ls='--')
    ax.axvline(0, color='grey', lw=0.5, ls='--')
    ax.set_xlabel(f'Component 1 ({pct_cov[0]:.1f}%)', fontsize=12)
    ax.set_ylabel(f'Component 2 ({pct_cov[1]:.1f}%)', fontsize=12)
    ax.set_title(f'PLS-DA Scores — {experiment_name}', fontsize=13)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")


def plot_vip(vip, mz, n_top, experiment_name, out_path):
    top_idx = np.argsort(vip)[::-1][:n_top]
    top_mz  = mz[top_idx]
    top_vip = vip[top_idx]

    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(range(n_top), top_vip, edgecolor='white')
    for bar, v in zip(bars, top_vip):
        bar.set_color('#d62728' if v > 1 else 'steelblue')

    ax.axhline(y=1, color='black', lw=1, ls='--', label='VIP = 1 threshold')
    ax.set_xticks(range(n_top))
    ax.set_xticklabels([f"{v:.1f}" for v in top_mz], rotation=90, fontsize=7)
    ax.set_xlabel('m/z')
    ax.set_ylabel('VIP Score (Component 1)')
    ax.set_title(f'Top {n_top} VIP Features — {experiment_name}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")


def main():
    sns.set_theme(style='whitegrid')
    plt.rcParams['figure.dpi'] = 120

    experiment_dir = os.path.join(BASE_DIR, EXPERIMENT)
    experiment_name = EXPERIMENT.strip()
    safe_name = experiment_name.replace(' ', '_').replace(':', '')

    assert os.path.isdir(experiment_dir), (
        f"Experiment folder not found: {experiment_dir!r}\n"
        f"Available options: {[d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d)) and not d.startswith('.')]}"
    )

    # ── 1. Load & filter ──────────────────────────────────────────────────────
    print(f"\n[1/5] Loading data: {experiment_name!r}")
    X_raw, y_labels, sample_names, mz = load_experiment(experiment_dir)
    print(f"  Raw samples : {X_raw.shape[0]}")
    print(f"  Raw features: {X_raw.shape[1]}")

    X_binned, mz = bin_features(X_raw, mz, bin_width=0.5)
    print(f"  After binning: {X_binned.shape[1]} features")

    X_filt, mz = filter_low_variance(X_binned, mz, percentile=25)
    print(f"  After variance filter: {X_filt.shape[1]} features")

    X_filt, mz = filter_low_abundance(X_filt, mz, percentile=5)
    print(f"  After abundance filter: {X_filt.shape[1]} features")

    print("  Groups:")
    for g, count in zip(*np.unique(y_labels, return_counts=True)):
        print(f"    {g:25s}  n={count}")

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    print("\n[2/5] Preprocessing (sum norm → log10 → auto-scale)")
    X = preprocess(X_filt)

    # ── 3. PLS-DA (8 components for scores plot) ─────────────────────────────
    print(f"\n[3/5] Fitting PLS-DA ({N_COMPONENTS} components for scores plot)")
    pls, T, y, Y, classes = fit_plsda(X, y_labels, N_COMPONENTS)
    print(f"  Classes: {list(classes)}")

    # ── 4. Cross-validation ───────────────────────────────────────────────────
    print("\n[4/5] 5-fold cross-validation")
    fold_accs = cross_validate(X, y, n_components=N_COMPONENTS)
    print(f"  Mean accuracy : {fold_accs.mean():.3f} ± {fold_accs.std():.3f}")
    print(f"  Chance level  : {1 / len(classes):.3f}")

    # ── 5. VIP scores (1 component to match MetaboAnalyst) ───────────────────
    print("\n[5/5] Computing VIP scores (1 component) & saving outputs")
    vip = compute_vip_1comp(X, y_labels)
    print(f"  VIP > 1: {(vip > 1).sum()} / {len(vip)} features")
    print(f"  Top m/z: {mz[vip.argmax()]:.1f}  (VIP = {vip.max():.3f})")

    # Save scores plot
    plot_scores(T, pls, X, y_labels, classes, experiment_name,
                out_path=f"plsda_scores_{safe_name}.png")

    # Save VIP bar chart
    plot_vip(vip, mz, N_TOP_VIP, experiment_name,
             out_path=f"vip_scores_{safe_name}.png")

    # Save VIP table (sorted by score, no true/false column)
    vip_csv = f"vip_table_{safe_name}.csv"
    pd.DataFrame({'mz': mz, 'vip_score': vip}) \
      .sort_values('vip_score', ascending=False) \
      .to_csv(vip_csv, index=False)
    print(f"  Saved → {vip_csv}")

    print("\nDone.")


if __name__ == '__main__':
    main()
