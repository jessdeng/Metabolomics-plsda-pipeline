"""
condition_abundance.py
======================
For each high-confidence ensemble feature (n_methods >= 4):
  1. Computes mean intensity per condition group (log10 normalised, pre-scaling)
  2. Fits a one-vs-rest Ridge regression and extracts signed coefficients
     per condition — showing which conditions each feature is associated with
  3. Produces two side-by-side plots: mean abundance heatmap and Ridge
     coefficient heatmap

Usage (from repo root):
    python scripts/condition_abundance.py

Configure the settings below. Set PIPELINE to 'standard' or 'r_comparable'
to match which pipeline you ran.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import LabelEncoder

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ── Configuration ─────────────────────────────────────────────────────────────
PIPELINE     = 'standard'    # 'standard' or 'r_comparable'
MIN_N_METHODS = 4            # minimum ensemble count to include
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, f'output_{PIPELINE}')


def main():
    experiment    = config.EXPERIMENT.strip()
    safe_name     = experiment.replace(' ', '_').replace(':', '')
    experiment_dir = os.path.join(BASE_DIR, experiment)

    # Import preprocessing from the chosen pipeline
    if PIPELINE == 'standard':
        from standard.preprocessing import (
            load_experiment, bin_features,
            filter_low_variance, filter_low_abundance, preprocess
        )
    else:
        from r_comparable.preprocessing import (
            load_experiment, bin_features,
            filter_low_variance, filter_low_abundance, preprocess
        )

    # ── Load feature overlap CSV ──────────────────────────────────────────────
    overlap_path = os.path.join(OUTPUT_DIR, f'feature_overlap_{safe_name}.csv')
    if not os.path.exists(overlap_path):
        print(f"feature_overlap CSV not found: {overlap_path}")
        print("Run run_analysis.py first.")
        return

    overlap_df = pd.read_csv(overlap_path)
    candidates = overlap_df[overlap_df['n_methods'] >= MIN_N_METHODS].copy()

    if candidates.empty:
        print(f"No features with n_methods >= {MIN_N_METHODS}.")
        return

    print(f"\nHigh-confidence features (n_methods >= {MIN_N_METHODS}):")
    print(candidates[['mz', 'n_methods', 'vip_score']].to_string(index=False))

    # ── Load and preprocess ───────────────────────────────────────────────────
    print(f"\nLoading: {experiment}")
    X_raw, y_labels, _, mz = load_experiment(experiment_dir)
    X_binned, mz = bin_features(X_raw, mz, bin_width=config.BIN_WIDTH)

    if config.VARIANCE_PERCENTILE > 0:
        X_filt, mz = filter_low_variance(X_binned, mz,
                                          percentile=config.VARIANCE_PERCENTILE)
    else:
        X_filt = X_binned.copy()

    if config.ABUNDANCE_PERCENTILE > 0:
        X_filt, mz = filter_low_abundance(X_filt, mz,
                                           percentile=config.ABUNDANCE_PERCENTILE)

    # Log-normalised only (no scaling) for abundance values
    X_norm   = preprocess(X_filt.copy(),
                          normalization=config.NORMALIZATION,
                          log_transform=config.LOG_TRANSFORM,
                          scaling='none')
    # Fully scaled for Ridge
    X_scaled = preprocess(X_filt.copy(),
                          normalization=config.NORMALIZATION,
                          log_transform=config.LOG_TRANSFORM,
                          scaling=config.SCALING)

    groups = sorted(np.unique(y_labels))
    le = LabelEncoder()
    y  = le.fit_transform(y_labels)

    # Fit one-vs-rest Ridge on full feature matrix
    ridge = RidgeClassifier()
    ridge.fit(X_scaled, y)

    # ── Match candidates to bins ──────────────────────────────────────────────
    records = []
    for _, row in candidates.iterrows():
        target_mz = row['mz']
        idx       = np.argmin(np.abs(mz - target_mz))
        actual_mz = mz[idx]

        if abs(actual_mz - target_mz) > config.BIN_WIDTH:
            print(f"  WARNING: m/z {target_mz:.2f} — nearest bin {actual_mz:.2f} "
                  f"is too far, skipping")
            continue

        group_means = {g: X_norm[y_labels == g, idx].mean() for g in groups}
        ridge_coefs = {le.classes_[c]: ridge.coef_[c, idx]
                       for c in range(len(le.classes_))}

        records.append({
            'mz_candidate': round(target_mz, 2),
            'mz_bin':       round(actual_mz, 2),
            'n_methods':    int(row['n_methods']),
            'vip_score':    round(row['vip_score'], 3),
            **{f'mean_{g}':  round(group_means[g], 4) for g in groups},
            **{f'ridge_{g}': round(ridge_coefs[g], 4) for g in groups},
        })

    if not records:
        print("No candidates could be matched to bins.")
        return

    df = pd.DataFrame(records)

    # Save table
    csv_path = os.path.join(OUTPUT_DIR, f'condition_abundance_{safe_name}.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\nSaved table → {csv_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    mean_cols  = [f'mean_{g}'  for g in groups]
    ridge_cols = [f'ridge_{g}' for g in groups]
    n_feat     = len(df)

    fig, axes = plt.subplots(1, 2,
                              figsize=(max(12, len(groups) * 2.5),
                                       max(3, n_feat * 0.8 + 2)))

    ylabels = [f"m/z {r['mz_candidate']}  (n={r['n_methods']}, VIP={r['vip_score']})"
               for _, r in df.iterrows()]

    # Left: mean abundance (row-normalised)
    abundance     = df[mean_cols].values
    row_min       = abundance.min(axis=1, keepdims=True)
    row_max       = abundance.max(axis=1, keepdims=True)
    row_rng       = np.where(row_max - row_min == 0, 1, row_max - row_min)
    abundance_norm = (abundance - row_min) / row_rng

    ax1 = axes[0]
    ax1.imshow(abundance_norm, cmap='RdBu_r', aspect='auto', vmin=0, vmax=1)
    for i in range(n_feat + 1):
        ax1.axhline(i - 0.5, color='white', linewidth=1)
    for j in range(len(groups) + 1):
        ax1.axvline(j - 0.5, color='white', linewidth=1)
    ax1.set_xticks(range(len(groups)))
    ax1.set_xticklabels(groups, rotation=35, ha='right', fontsize=9)
    ax1.set_yticks(range(n_feat))
    ax1.set_yticklabels(ylabels, fontsize=9)
    ax1.set_title('Mean Abundance per Condition\n(log\u2081\u2080 TIC-normalised, row-scaled)',
                  fontsize=10)
    sm1 = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(0, 1))
    sm1.set_array([])
    cb1 = fig.colorbar(sm1, ax=ax1, shrink=0.4, aspect=12, pad=0.02)
    cb1.set_ticks([0, 1])
    cb1.set_ticklabels(['Low', 'High'])

    # Right: Ridge coefficients
    ridge_vals = df[ridge_cols].values
    vmax       = np.abs(ridge_vals).max()

    ax2 = axes[1]
    ax2.imshow(ridge_vals, cmap='coolwarm', aspect='auto', vmin=-vmax, vmax=vmax)
    for i in range(n_feat + 1):
        ax2.axhline(i - 0.5, color='white', linewidth=1)
    for j in range(len(groups) + 1):
        ax2.axvline(j - 0.5, color='white', linewidth=1)
    ax2.set_xticks(range(len(groups)))
    ax2.set_xticklabels(groups, rotation=35, ha='right', fontsize=9)
    ax2.set_yticks(range(n_feat))
    ax2.set_yticklabels([], fontsize=9)
    ax2.set_title('Ridge Coefficient per Condition\n'
                  '(positive = feature associated with that group)', fontsize=10)
    sm2 = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(-vmax, vmax))
    sm2.set_array([])
    cb2 = fig.colorbar(sm2, ax=ax2, shrink=0.4, aspect=12, pad=0.02)
    cb2.set_ticks([-vmax, 0, vmax])
    cb2.set_ticklabels(['Suppressed', '0', 'Elevated'])

    fig.suptitle(
        f'{experiment}  |  High-confidence features (n_methods \u2265 {MIN_N_METHODS})',
        fontsize=11, y=1.01
    )

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f'condition_abundance_{safe_name}.png')
    plt.savefig(plot_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved plot  → {plot_path}")

    # Print summary
    print("\n── Ridge coefficients (positive = associated with that condition) ──")
    print(df[['mz_candidate', 'n_methods'] + ridge_cols].to_string(index=False))


if __name__ == '__main__':
    main()
