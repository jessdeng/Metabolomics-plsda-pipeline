"""
vip_vs_nmethods.py
==================
Plots ensemble confidence (n_methods) vs PLS-DA VIP score for all
feature overlap candidates across experiments.

Highlights features that are high-confidence by ensemble voting but
below the conventional VIP > 1.0 threshold — i.e. features the
ensemble catches that PLS-DA alone would miss.

Usage (from repo root):
    python scripts/vip_vs_nmethods.py

Reads feature_overlap_*.csv files from your output folder.
Configure OUTPUT_DIR below to match which pipeline you ran.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ── Configuration ─────────────────────────────────────────────────────────────
# Change to 'output_r_comparable' if you ran the R-comparable pipeline
OUTPUT_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'output_standard')
MIN_N_LABEL  = 4    # label features with this many methods or more on the plot
# ──────────────────────────────────────────────────────────────────────────────


def main():
    # Find all feature overlap CSVs in the output folder
    csv_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, 'feature_overlap_*.csv')))

    if not csv_files:
        print(f"No feature_overlap_*.csv files found in {OUTPUT_DIR}")
        print("Run run_analysis.py first to generate them.")
        return

    # Build experiment name from filename
    experiments = {}
    for f in csv_files:
        name = os.path.basename(f).replace('feature_overlap_', '').replace('.csv', '').replace('_', ' ')
        experiments[name] = f

    n = len(experiments)
    ncols = 2
    nrows = (n + 1) // 2

    palette = sns.color_palette('colorblind', n_colors=n)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.5 * nrows))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, (name, fpath), color in zip(axes, experiments.items(), palette):
        df = pd.read_csv(fpath)

        rng    = np.random.default_rng(42)
        jitter = rng.uniform(-0.12, 0.12, size=len(df))
        sizes  = df['n_methods'].map({2: 40, 3: 80, 4: 140, 5: 220})

        ax.scatter(
            df['n_methods'] + jitter,
            df['vip_score'],
            s=sizes, color=color, alpha=0.6,
            edgecolors='white', linewidths=0.4, zorder=2
        )

        ax.axhline(1.0, color='#888888', linestyle='--',
                   linewidth=0.9, alpha=0.7, zorder=1)

        # Label high-confidence features
        for _, row in df[df['n_methods'] >= MIN_N_LABEL].iterrows():
            ax.annotate(
                f"{row['mz']:.1f}",
                xy=(row['n_methods'], row['vip_score']),
                xytext=(6, 3), textcoords='offset points',
                fontsize=7, color='#333333', fontweight='bold'
            )

        ax.set_title(name, fontsize=10, fontweight='bold', color='#1a1a1a')
        ax.set_xlabel('Ensemble count (number of methods)', fontsize=8)
        ax.set_ylabel('PLS-DA VIP score', fontsize=8)
        ax.set_xticks([2, 3, 4, 5])
        ax.tick_params(labelsize=8)
        ax.set_xlim(1.5, 5.8)

    # Shading and n counts (after ylim is set)
    for ax, (name, fpath) in zip(axes, experiments.items()):
        df   = pd.read_csv(fpath)
        ymax = ax.get_ylim()[1]
        ax.axvspan(3.5, 5.8, ymin=0,
                   ymax=1.0 / ymax if ymax > 1.0 else 1.0,
                   color='#FDEBD0', alpha=0.35, zorder=0)
        for n_m in [2, 3, 4, 5]:
            count = len(df[df['n_methods'] == n_m])
            if count > 0:
                ax.text(n_m, ax.get_ylim()[0] + 0.02, f'n={count}',
                        ha='center', fontsize=7, color='#666666', style='italic')

    # Hide unused axes
    for ax in axes[len(experiments):]:
        ax.set_visible(False)

    # Legend
    legend_handles = [
        plt.scatter([], [], s=40,  color='grey', alpha=0.6, label='n = 2 methods'),
        plt.scatter([], [], s=80,  color='grey', alpha=0.6, label='n = 3 methods'),
        plt.scatter([], [], s=140, color='grey', alpha=0.6, label='n = 4 methods'),
        plt.scatter([], [], s=220, color='grey', alpha=0.6, label='n = 5 methods'),
        mpatches.Patch(color='#FDEBD0', alpha=0.5, label='High confidence, VIP < 1.0'),
        plt.Line2D([0], [0], color='#888888', linestyle='--',
                   linewidth=0.9, label='VIP = 1.0 threshold'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle('Ensemble confidence vs. PLS-DA VIP score',
                 fontsize=12, fontweight='bold', y=1.01)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'vip_vs_nmethods.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
