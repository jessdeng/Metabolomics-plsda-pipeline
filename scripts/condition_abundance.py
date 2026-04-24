"""
condition_abundance.py
======================
Plots a side-by-side heatmap of mean intensity and Ridge one-vs-rest
coefficients for high-confidence ensemble features (n_methods >= MIN_N_METHODS).

As of the per-group attribution update, the underlying numbers (mean_<group>,
ridge_<group>) are written directly to feature_overlap_<experiment>.csv by
run_analysis.py. This script no longer recomputes them — it just reads the
CSV and produces the heatmap figure.

Usage (from repo root):
    python scripts/condition_abundance.py

Configure PIPELINE and MIN_N_METHODS below.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ── Configuration ─────────────────────────────────────────────────────────────
PIPELINE      = 'standard'    # 'standard' or 'r_comparable'
MIN_N_METHODS = 4             # minimum ensemble count to include
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, f'output_{PIPELINE}')


def main():
    experiment = config.EXPERIMENT.strip()
    safe_name  = experiment.replace(' ', '_').replace(':', '')

    overlap_path = os.path.join(OUTPUT_DIR, f'feature_overlap_{safe_name}.csv')
    if not os.path.exists(overlap_path):
        print(f"feature_overlap CSV not found: {overlap_path}")
        print("Run run_analysis.py first.")
        return

    df = pd.read_csv(overlap_path)

    mean_cols  = [c for c in df.columns
                  if c.startswith('mean_') and c != 'mean_margin']
    ridge_cols_all = [c for c in df.columns
                      if c.startswith('ridge_') and c != 'ridge_importance']

    if not mean_cols or not ridge_cols_all:
        print("CSV does not contain per-group attribution columns.")
        print("Re-run run_analysis.py to regenerate the CSV with the new schema.")
        return

    # Strip prefixes to get group names — order matches both heatmaps
    groups = [c.replace('mean_', '') for c in mean_cols]
    expected_ridge_cols = [f'ridge_{g}' for g in groups]
    missing = [c for c in expected_ridge_cols if c not in df.columns]
    if missing:
        print(f"Missing expected Ridge columns: {missing}")
        return
    ridge_cols = expected_ridge_cols

    candidates = df[df['n_methods'] >= MIN_N_METHODS].copy()
    if candidates.empty:
        print(f"No features with n_methods >= {MIN_N_METHODS}.")
        return

    print(f"\nHigh-confidence features (n_methods >= {MIN_N_METHODS}):")
    summary_cols = ['mz', 'n_methods', 'vip_score',
                    'top_condition_mean', 'top_condition_ridge', 'ridge_direction']
    summary_cols = [c for c in summary_cols if c in candidates.columns]
    print(candidates[summary_cols].to_string(index=False))

    # ── Plot ──────────────────────────────────────────────────────────────────
    n_feat = len(candidates)
    fig, axes = plt.subplots(
        1, 2,
        figsize=(max(12, len(groups) * 2.5), max(3, n_feat * 0.8 + 2))
    )

    ylabels = [f"m/z {row['mz']:.2f}  (n={int(row['n_methods'])}, "
               f"VIP={row['vip_score']:.2f})"
               for _, row in candidates.iterrows()]

    # Left: row-normalised mean abundance
    abundance      = candidates[mean_cols].values
    row_min        = abundance.min(axis=1, keepdims=True)
    row_max        = abundance.max(axis=1, keepdims=True)
    row_rng        = np.where(row_max - row_min == 0, 1, row_max - row_min)
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
    ax1.set_title('Mean Abundance per Condition\n(row-scaled)', fontsize=10)
    sm1 = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(0, 1))
    sm1.set_array([])
    cb1 = fig.colorbar(sm1, ax=ax1, shrink=0.4, aspect=12, pad=0.02)
    cb1.set_ticks([0, 1])
    cb1.set_ticklabels(['Low', 'High'])

    # Right: Ridge coefficients (signed, symmetric scale)
    ridge_vals = candidates[ridge_cols].values
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
                  '(positive = associated with that group)', fontsize=10)
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
    print(f"\nSaved plot → {plot_path}")


if __name__ == '__main__':
    main()
