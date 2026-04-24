"""
condition_spectrum.py
=====================
Builds an MS-native visual that answers the question
"which condition does each ensemble feature come from?"

For each ensemble feature in feature_overlap_<experiment>.csv, places a
coloured vertical bar at its m/z on top of the overall average spectrum.
Bar colour encodes the feature's top_condition_mean (which group has
the highest mean intensity at that m/z).  Bar height encodes mean
intensity in the top group.

Ambiguous features (mean_margin < 1.5x) are drawn in grey so the user
can immediately see which calls are confident and which are not.

Two panels:
  Top    — high-confidence features (n_methods >= 4)
  Bottom — all overlap features (n_methods >= 2)

Two outputs:
  PNG  via matplotlib   — drop into reports/slides
  HTML via plotly       — hover for full per-feature details

Usage (from repo root):
    python scripts/condition_spectrum.py

Configure PIPELINE and AMBIGUITY_CUTOFF below.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ── Configuration ─────────────────────────────────────────────────────────────
PIPELINE         = 'standard'   # 'standard' or 'r_comparable'
HIGH_CONF_N      = 4            # n_methods threshold for "high confidence"
AMBIGUITY_CUTOFF = 1.5          # mean_margin below this is treated as ambiguous
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, f'output_{PIPELINE}')


def _load_data():
    """Reload the binned spectrum so we have something to plot under the bars."""
    experiment = config.EXPERIMENT.strip()
    safe_name  = experiment.replace(' ', '_').replace(':', '')
    experiment_dir = os.path.join(BASE_DIR, experiment)

    if PIPELINE == 'standard':
        from standard.preprocessing import (
            load_experiment, bin_features, filter_low_variance, filter_low_abundance
        )
    else:
        from r_comparable.preprocessing import (
            load_experiment, bin_features, filter_low_variance, filter_low_abundance
        )

    X_raw, y_labels, _, mz = load_experiment(experiment_dir)
    X_binned, mz_binned    = bin_features(X_raw, mz, bin_width=config.BIN_WIDTH)
    return X_binned, mz_binned, y_labels, experiment, safe_name


def _build_palette(groups):
    """Return a {group: rgb tuple} dict using a colorblind-friendly palette."""
    import seaborn as sns
    cb = sns.color_palette('colorblind', n_colors=len(groups))
    return {g: cb[i] for i, g in enumerate(groups)}


def _filter_subset(df, threshold):
    return df[df['n_methods'] >= threshold].copy()


# ── PNG (matplotlib) ──────────────────────────────────────────────────────────

def _draw_panel(ax, mz_axis, avg_spectrum, subset_df, palette, title):
    """Draw average spectrum + colored bars for one subset of features."""
    ax.plot(mz_axis, avg_spectrum, color='lightgrey', linewidth=0.7, zorder=1)

    # Plot one set of bars per group so legend entries are clean
    bar_height = avg_spectrum.max() * 1.05

    for _, row in subset_df.iterrows():
        mz_val = row['mz']
        group  = row['top_condition_mean']
        margin = row['mean_margin']

        # Snap to the nearest bin so the bar height = real spectrum value
        idx = np.argmin(np.abs(mz_axis - mz_val))
        height = avg_spectrum[idx]

        if pd.isna(margin) or margin < AMBIGUITY_CUTOFF:
            # Ambiguous — draw in grey, dashed, so the user knows
            color, alpha, ls = '#888888', 0.6, '--'
        else:
            color, alpha, ls = palette[group], 0.85, '-'

        ax.vlines(mz_val, 0, height, colors=[color], alpha=alpha,
                  linewidth=1.5, linestyles=ls, zorder=3)

    ax.set_xlabel('m/z')
    ax.set_ylabel('Mean intensity (cps)')
    ax.set_title(title, fontsize=11)
    ax.set_xlim(mz_axis.min(), mz_axis.max())
    ax.set_ylim(bottom=0)


def make_png(df, X_binned, mz_axis, y_labels, experiment, safe_name):
    avg_spectrum = X_binned.mean(axis=0)
    groups       = sorted(np.unique(y_labels))
    palette      = _build_palette(groups)

    high_conf = _filter_subset(df, HIGH_CONF_N)
    full      = _filter_subset(df, 2)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    _draw_panel(ax_top, mz_axis, avg_spectrum, high_conf, palette,
                f'High-confidence features (n_methods \u2265 {HIGH_CONF_N}) — '
                f'{len(high_conf)} features')
    _draw_panel(ax_bot, mz_axis, avg_spectrum, full, palette,
                f'All overlap features (n_methods \u2265 2) — '
                f'{len(full)} features')

    # Legend: one entry per group + ambiguous
    legend_handles = [mpatches.Patch(color=palette[g], label=g) for g in groups]
    legend_handles.append(mpatches.Patch(color='#888888', alpha=0.6,
                          label=f'Ambiguous (margin < {AMBIGUITY_CUTOFF}\u00d7)'))
    fig.legend(handles=legend_handles, loc='upper right',
               bbox_to_anchor=(0.99, 0.98), fontsize=9, framealpha=0.9)

    fig.suptitle(f'Ensemble features by top condition — {experiment}',
                 fontsize=13, y=1.00)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f'condition_spectrum_{safe_name}.png')
    plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved → {out_path}")


# ── HTML (plotly) ─────────────────────────────────────────────────────────────

def _rgb_str(rgb_tuple):
    r, g, b = (int(c * 255) for c in rgb_tuple[:3])
    return f'rgb({r},{g},{b})'


def _add_panel_traces(fig, mz_axis, avg_spectrum, subset_df, palette,
                      groups, panel_label, row):
    """Add background spectrum + per-feature bars to a specific subplot row."""
    # Background spectrum
    fig.add_trace(
        go.Scatter(x=mz_axis, y=avg_spectrum, mode='lines',
                   line=dict(color='lightgrey', width=1),
                   name=f'{panel_label} avg spectrum',
                   hoverinfo='skip', showlegend=False),
        row=row, col=1
    )

    # One trace per group (so legend toggles work) + one trace for ambiguous
    for group in groups:
        sub = subset_df[
            (subset_df['top_condition_mean'] == group) &
            (subset_df['mean_margin'] >= AMBIGUITY_CUTOFF)
        ]
        if len(sub) == 0:
            continue
        # Snap bars to actual spectrum heights
        heights = [avg_spectrum[np.argmin(np.abs(mz_axis - m))] for m in sub['mz']]
        custom = sub[['n_methods','vip_score','top_condition_ridge',
                      'mean_margin','ridge_direction']].values

        fig.add_trace(
            go.Bar(x=sub['mz'], y=heights,
                   marker_color=_rgb_str(palette[group]),
                   marker_line_color=_rgb_str(palette[group]),
                   width=0.4,
                   name=f'{panel_label}: {group}',
                   legendgroup=group,
                   showlegend=(panel_label == 'High'),  # only show in legend once
                   customdata=custom,
                   hovertemplate=(
                       'm/z %{x:.3f}<br>'
                       'Top condition: ' + group + '<br>'
                       'n_methods: %{customdata[0]}<br>'
                       'VIP: %{customdata[1]:.2f}<br>'
                       'Ridge top condition: %{customdata[2]}<br>'
                       'Mean margin: %{customdata[3]:.2f}\u00d7<br>'
                       'Ridge direction: %{customdata[4]}<extra></extra>'
                   )),
            row=row, col=1
        )

    # Ambiguous features
    amb = subset_df[subset_df['mean_margin'] < AMBIGUITY_CUTOFF]
    if len(amb) > 0:
        heights = [avg_spectrum[np.argmin(np.abs(mz_axis - m))] for m in amb['mz']]
        custom = amb[['n_methods','vip_score','top_condition_mean',
                      'top_condition_ridge','mean_margin','ridge_direction']].values

        fig.add_trace(
            go.Bar(x=amb['mz'], y=heights,
                   marker_color='#888888', marker_line_color='#888888',
                   width=0.4,
                   name=f'{panel_label}: Ambiguous',
                   legendgroup='ambiguous',
                   showlegend=(panel_label == 'High'),
                   customdata=custom,
                   hovertemplate=(
                       'm/z %{x:.3f}<br>'
                       '<b>AMBIGUOUS</b> (margin < ' + str(AMBIGUITY_CUTOFF) + '\u00d7)<br>'
                       'Top mean: %{customdata[2]}<br>'
                       'Top ridge: %{customdata[3]}<br>'
                       'n_methods: %{customdata[0]}<br>'
                       'VIP: %{customdata[1]:.2f}<br>'
                       'Mean margin: %{customdata[4]:.2f}\u00d7<br>'
                       'Ridge direction: %{customdata[5]}<extra></extra>'
                   )),
            row=row, col=1
        )


def make_html(df, X_binned, mz_axis, y_labels, experiment, safe_name):
    from plotly.subplots import make_subplots

    avg_spectrum = X_binned.mean(axis=0)
    groups       = sorted(np.unique(y_labels))
    palette      = _build_palette(groups)

    high_conf = _filter_subset(df, HIGH_CONF_N)
    full      = _filter_subset(df, 2)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=(
            f'High-confidence features (n_methods \u2265 {HIGH_CONF_N}) — {len(high_conf)} features',
            f'All overlap features (n_methods \u2265 2) — {len(full)} features',
        ),
        vertical_spacing=0.10,
    )

    _add_panel_traces(fig, mz_axis, avg_spectrum, high_conf, palette,
                      groups, 'High', row=1)
    _add_panel_traces(fig, mz_axis, avg_spectrum, full, palette,
                      groups, 'All', row=2)

    fig.update_xaxes(title_text='m/z', row=2, col=1)
    fig.update_yaxes(title_text='Mean intensity (cps)', row=1, col=1)
    fig.update_yaxes(title_text='Mean intensity (cps)', row=2, col=1)
    fig.update_layout(
        title=f'Ensemble features by top condition — {experiment}',
        barmode='overlay',
        height=800, width=1200,
        legend=dict(itemclick='toggle'),
        hovermode='closest',
    )

    out_path = os.path.join(OUTPUT_DIR, f'condition_spectrum_{safe_name}.html')
    fig.write_html(out_path)
    print(f"  Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.path.isdir(OUTPUT_DIR):
        print(f"Output folder not found: {OUTPUT_DIR}")
        print("Run run_analysis.py first.")
        return

    X_binned, mz_axis, y_labels, experiment, safe_name = _load_data()

    overlap_path = os.path.join(OUTPUT_DIR, f'feature_overlap_{safe_name}.csv')
    if not os.path.exists(overlap_path):
        print(f"feature_overlap CSV not found: {overlap_path}")
        return

    df = pd.read_csv(overlap_path)
    needed = ['mz', 'n_methods', 'top_condition_mean',
              'top_condition_ridge', 'mean_margin', 'ridge_direction']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"CSV missing required columns: {missing}")
        print("Re-run run_analysis.py to regenerate the CSV with the new schema.")
        return

    print(f"Loaded {len(df)} ensemble features from {overlap_path}")
    print(f"  High-confidence (n_methods >= {HIGH_CONF_N}): "
          f"{(df['n_methods'] >= HIGH_CONF_N).sum()}")

    print("\nMaking PNG ...")
    make_png(df, X_binned, mz_axis, y_labels, experiment, safe_name)
    print("\nMaking HTML ...")
    make_html(df, X_binned, mz_axis, y_labels, experiment, safe_name)


if __name__ == '__main__':
    main()
