import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_spectrum_with_features(X_binned, mz, y_labels, overlap_df, experiment_name, out_path):
    groups = sorted(np.unique(y_labels))
    n_groups = len(groups)
    palette = sns.color_palette('muted', n_colors=n_groups)

    fig, axes = plt.subplots(n_groups, 1, figsize=(14, 3 * n_groups), sharex=True)

    important_mz = overlap_df['mz'].values

    for i, group in enumerate(groups):
        ax = axes[i]
        mask = y_labels == group
        avg_spectrum = X_binned[mask].mean(axis=0)

        ax.plot(mz, avg_spectrum, color=palette[i], linewidth=0.5)

        for mz_val in important_mz:
            ax.axvline(x=mz_val, color='red', alpha=0.3, linewidth=0.5)

        ax.set_ylabel('Intensity (cps)')
        ax.set_title(group, fontsize=10)
        ax.set_xlim(mz.min(), mz.max())

    axes[-1].set_xlabel('m/z')
    fig.suptitle(f'Group Spectra with Important Features — {experiment_name}', fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")