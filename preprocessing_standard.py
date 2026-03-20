"""
Preprocessing functions for mass spectrometry data — standard version.

This is a drop-in replacement for preprocessing.py with one difference:
bin_features() labels each bin at its true geometric center (bin_edge + bin_width/2)
rather than applying MetaboAnalyst's -0.05 Da cosmetic offset.

Use this version when MetaboAnalyst validation is not the goal.
Use preprocessing.py when you need output labels to match MetaboAnalyst exactly.
"""

import os
import glob
import numpy as np
import pandas as pd


def load_experiment(experiment_dir):
    """
    Read all CSVs and TXTs from the experiment folder.
    Each direct subfolder = one class/group.
    Each CSV/TXT inside = one sample.
    Interpolates all samples onto a common m/z axis.
    Returns X (n_samples × n_features), labels array, sample names, mz axis.
    """
    samples, labels, names = [], [], []
    raw_mz_list = []

    group_folders = sorted([
        d for d in os.listdir(experiment_dir)
        if os.path.isdir(os.path.join(experiment_dir, d))
    ])

    for group in group_folders:
        data_files = sorted(
            glob.glob(os.path.join(experiment_dir, group, '*.csv')) +
            glob.glob(os.path.join(experiment_dir, group, '*.txt'))
        )
        if not data_files:
            print(f"  [warning] No CSV or TXT files in '{group}', skipping.")
            continue

        for file_path in data_files:
            sep = '\t' if file_path.endswith('.txt') else ','
            df = pd.read_csv(file_path, sep=sep, encoding='utf-8-sig')
            df.columns = df.columns.str.lstrip('\ufeff').str.strip()
            col_map = {c.lower(): c for c in df.columns}
            col_map['mz'] = col_map.get('mz') or col_map.get('mass/charge')
            col_map['int'] = col_map.get('int') or col_map.get('intensity')

            if col_map['mz'] is None or col_map['int'] is None:
                print(f"  [warning] Skipping {os.path.basename(file_path)}: "
                      f"could not find m/z or intensity columns. "
                      f"Found: {list(df.columns)}")
                continue

            raw_mz_list.append(df[col_map['mz']].values)
            samples.append(df[col_map['int']].values)
            labels.append(group.strip())
            names.append(os.path.basename(file_path))

    # Build a common m/z axis covering the overlap range across all files
    mz_min = max(mz.min() for mz in raw_mz_list)
    mz_max = min(mz.max() for mz in raw_mz_list)
    common_mz = np.linspace(mz_min, mz_max, num=5000)

    # Interpolate every sample onto the common axis
    interpolated = []
    for mz_i, int_i in zip(raw_mz_list, samples):
        interp_int = np.interp(common_mz, mz_i, int_i)
        interpolated.append(interp_int)

    return np.array(interpolated, dtype=float), np.array(labels), names, common_mz

def filter_mass_range(X, mz, mz_min=100, mz_max=1000):
    keep = (mz >= mz_min) & (mz <= mz_max)
    return X[:, keep], mz[keep]

def bin_features(X, mz, bin_width=0.5):
    """
    Bin m/z features into fixed-width bins by summing intensities.
    Bin centers are the true geometric centers of each bin (bin_edge + bin_width/2).

    Note: preprocessing.py applies a -0.05 Da offset to match MetaboAnalyst's
    internal .2/.7 labeling convention. That offset has no physical basis — it is
    a cosmetic artifact of MetaboAnalyst's output format. This version labels bins
    correctly as the midpoint of each window.
    """
    bin_edges = np.arange(mz.min(), mz.max() + bin_width, bin_width)
    bin_labels = bin_edges[:-1] + bin_width / 2

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


def preprocess(X, normalization='tic', log_transform='log10', scaling='autoscale'):
    """
    Normalize, transform, and scale the data.

    Parameters
    ----------
    normalization : str
        Sample normalization method. Options: 'tic', 'median', 'pqn', 'quantile', 'none'.
        See Wu & Li (2016) Sci Rep 6:38881 for a comparison of methods.
    log_transform : str
        Transformation to apply. Options: 'log10', 'log2', 'sqrt', 'none'.
    scaling : str
        Scaling method. Options: 'autoscale', 'pareto', 'range', 'vast', 'level', 'none'.
        See van den Berg et al. (2006) BMC Genomics 7:142 for guidance.
    """

    # ── Normalization ─────────────────────────────────────────────────────────
    if normalization == 'tic':
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        X = X / row_sums * np.median(row_sums)

    elif normalization == 'median':
        row_medians = np.median(X, axis=1, keepdims=True)
        row_medians[row_medians == 0] = 1
        X = X / row_medians * np.median(row_medians)

    elif normalization == 'pqn':
        # Probabilistic Quotient Normalization
        # Reference: Dieterle et al. (2006) Anal Chem 78:4281
        # 1. TIC-normalize first as a preliminary step
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        X_tic = X / row_sums
        # 2. Reference spectrum = median of all TIC-normalized samples
        reference = np.median(X_tic, axis=0)
        reference[reference == 0] = 1
        # 3. Quotients = each sample divided by reference
        quotients = X_tic / reference
        # 4. Dilution factor = median of quotients per sample
        dilution = np.median(quotients, axis=1, keepdims=True)
        dilution[dilution == 0] = 1
        X = X_tic / dilution

    elif normalization == 'quantile':
        # Quantile normalization: force all samples to same distribution
        ranks = np.argsort(np.argsort(X, axis=1), axis=1)
        sorted_X = np.sort(X, axis=1)
        row_means = sorted_X.mean(axis=0)
        X = row_means[ranks]

    elif normalization == 'none':
        pass

    else:
        raise ValueError(f"Unknown normalization: '{normalization}'. "
                         f"Choose from: 'tic', 'median', 'pqn', 'quantile', 'none'.")

    # ── Transformation ────────────────────────────────────────────────────────
    min_positive = X[X > 0].min() if (X > 0).any() else 1e-6
    half_min = min_positive / 2

    if log_transform == 'log10':
        X = np.log10(X + half_min)
    elif log_transform == 'log2':
        X = np.log2(X + half_min)
    elif log_transform == 'sqrt':
        X = np.sqrt(X)
    elif log_transform == 'none':
        pass
    else:
        raise ValueError(f"Unknown log_transform: '{log_transform}'. "
                         f"Choose from: 'log10', 'log2', 'sqrt', 'none'.")

    # ── Scaling ───────────────────────────────────────────────────────────────
    feat_mean = X.mean(axis=0)
    feat_std  = X.std(axis=0, ddof=1)
    feat_std[feat_std == 0] = 1

    if scaling == 'autoscale':
        X = (X - feat_mean) / feat_std
    elif scaling == 'pareto':
        X = (X - feat_mean) / np.sqrt(feat_std)
    elif scaling == 'range':
        feat_range = X.max(axis=0) - X.min(axis=0)
        feat_range[feat_range == 0] = 1
        X = (X - feat_mean) / feat_range
    elif scaling == 'vast':
        X = ((X - feat_mean) / feat_std) * (feat_mean / (feat_std + 1e-10))
    elif scaling == 'level':
        level = np.abs(feat_mean)
        level[level == 0] = 1
        X = (X - feat_mean) / level
    elif scaling == 'none':
        X = X - feat_mean
    else:
        raise ValueError(f"Unknown scaling: '{scaling}'. "
                         f"Choose from: 'autoscale', 'pareto', 'range', 'vast', 'level', 'none'.")

    return X
