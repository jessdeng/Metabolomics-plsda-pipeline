"""
Preprocessing functions for mass spectrometry data.
Used by both the MetaboAnalyst PLS-DA pipeline and the classifier comparison.
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
            df = pd.read_csv(file_path, sep=sep)
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
    Bin centers are offset to match MetaboAnalyst's .2 / .7 labeling.
    """
    bin_edges = np.arange(mz.min(), mz.max() + bin_width, bin_width)
    bin_labels = bin_edges[:-1] + bin_width / 2 - 0.05

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
    # TIC Normalization
    row_sums = X.sum(axis=1, keepdims=True)
    X_norm = X / row_sums * np.median(row_sums)

    # Log10 Transformation
    min_positive = X_norm[X_norm > 0].min()
    X_log = np.log10(X_norm + min_positive / 2)
    feat_mean = X_log.mean(axis=0)

    # Auto Scaling
    feat_std = X_log.std(axis=0, ddof=1)
    feat_std[feat_std == 0] = 1
    X_scaled = (X_log - feat_mean) / feat_std

    return X_scaled
