import os
import sys
# Ensure repo root is on the path so config.py and sibling packages are found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import numpy as np

import config
from standard.preprocessing import load_experiment, bin_features, filter_low_variance, filter_low_abundance, preprocess
from standard.pipeline import compute_vip_1comp, fit_plsda, plot_scores_3d, plot_vip
from shared.classifier_comparison_standard import (
    RandomForest, svm_classify, gradient_boosting,
    logistic_regression, lda_classify, ridge_classify,
    plot_accuracy_comparison, feature_importance_analysis
)
from shared.visualization import plot_spectrum_with_features

# ── Load settings from config.py ──────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENT = config.EXPERIMENT
# ──────────────────────────────────────────────────────────────────────────────

def main():
    experiment_dir  = os.path.join(BASE_DIR, EXPERIMENT)
    experiment_name = EXPERIMENT.strip()
    safe_name       = experiment_name.replace(' ', '_').replace(':', '')

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = os.path.join(BASE_DIR, 'output_standard')
    os.makedirs(out_dir, exist_ok=True)

    assert os.path.isdir(experiment_dir), (
        f"Experiment folder not found: {experiment_dir!r}\n"
        f"Available options: {[d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d)) and not d.startswith('.')]}"
    )

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"\n[1/12] Loading data: {experiment_name!r}")
    X_raw, y_labels, sample_names, mz = load_experiment(experiment_dir)
    print(f"  Raw samples : {X_raw.shape[0]}")
    print(f"  Raw features: {X_raw.shape[1]}")

    # ── 2. Bin ────────────────────────────────────────────────────────────────
    X_binned, mz = bin_features(X_raw, mz, bin_width=config.BIN_WIDTH)
    mz_binned = mz.copy()
    print(f"\n[2/12] Binning ({config.BIN_WIDTH} Da bins): {X_binned.shape[1]} features")

    # ── 3. Filter ─────────────────────────────────────────────────────────────
    print(f"\n[3/12] Filtering")
    if config.VARIANCE_PERCENTILE > 0:
        X_filt, mz = filter_low_variance(X_binned, mz, percentile=config.VARIANCE_PERCENTILE)
        print(f"  After variance filter ({config.VARIANCE_PERCENTILE}%): {X_filt.shape[1]} features")
    else:
        X_filt = X_binned.copy()
        print(f"  Variance filter disabled")

    if config.ABUNDANCE_PERCENTILE > 0:
        X_filt, mz = filter_low_abundance(X_filt, mz, percentile=config.ABUNDANCE_PERCENTILE)
        print(f"  After abundance filter ({config.ABUNDANCE_PERCENTILE}%): {X_filt.shape[1]} features")
    else:
        print(f"  Abundance filter disabled")

    X_filt_raw = X_filt.copy()

    print("  Groups:")
    for g, count in zip(*np.unique(y_labels, return_counts=True)):
        print(f"    {g:25s}  n={count}")

    # ── 4. Preprocess ─────────────────────────────────────────────────────────
    print(f"\n[4/12] Preprocessing")
    print(f"  Normalization     : {config.NORMALIZATION}")
    print(f"  Transformation    : {config.LOG_TRANSFORM}")
    print(f"  Scaling           : {config.SCALING}")
    X = preprocess(
        X_filt,
        normalization=config.NORMALIZATION,
        log_transform=config.LOG_TRANSFORM,
        scaling=config.SCALING
    )

    # ── 5. PLS-DA ─────────────────────────────────────────────────────────────
    print(f"\n[5/12] Fitting PLS-DA ({config.N_PLSDA_COMPONENTS} components)")
    pls, T, y, Y, classes = fit_plsda(X, y_labels, config.N_PLSDA_COMPONENTS)
    print(f"  Classes: {list(classes)}")

    # ── 6. VIP scores ─────────────────────────────────────────────────────────
    print(f"\n[6/12] Computing VIP scores (1 component, top {config.N_TOP_VIP})")
    vip = compute_vip_1comp(X, y_labels)
    plot_scores_3d(T, pls, y_labels, classes, experiment_name,
                   out_path=os.path.join(out_dir, f"plsda_scores_3d_{safe_name}.html"))
    plot_vip(vip, mz, X_filt_raw, y_labels, config.N_TOP_VIP, experiment_name,
             out_path=os.path.join(out_dir, f"vip_scores_{safe_name}.png"))

    # ── 7–11. Classifiers ─────────────────────────────────────────────────────
    all_classifiers = {
        'Random Forest':       (config.USE_RANDOM_FOREST,       RandomForest),
        'SVM':                 (config.USE_SVM,                 svm_classify),
        'Gradient Boosting':   (config.USE_GRADIENT_BOOSTING,   gradient_boosting),
        'Logistic Regression': (config.USE_LOGISTIC_REGRESSION, logistic_regression),
        'LDA':                 (config.USE_LDA,                 lda_classify),
        'Ridge':               (config.USE_RIDGE,               ridge_classify),
    }

    results = {}
    step = 7
    for name, (enabled, fn) in all_classifiers.items():
        if enabled:
            print(f"\n[{step}/12] {name}")
            test_accs, train_accs = fn(X, y_labels, n_splits=config.CV_FOLDS)
            results[name] = (test_accs, train_accs)
            print(f"  Test  accuracy: {test_accs.mean():.3f} ± {test_accs.std():.3f}")
            print(f"  Train accuracy: {train_accs.mean():.3f} ± {train_accs.std():.3f}")
            step += 1

    # ── Save classifier results for reuse in extras.py ───────────────────────
    results_path = os.path.join(out_dir, f"classifier_results_{safe_name}.npz")
    np.savez(results_path, **{
        f"{name}__test":  test_accs
        for name, (test_accs, _) in results.items()
    }, **{
        f"{name}__train": train_accs
        for name, (_, train_accs) in results.items()
    })
    print(f"\n  Classifier results saved → {results_path}")

    # ── 11. Plot comparison ───────────────────────────────────────────────────
    print("\n[11/12] Plot Comparison")
    plot_accuracy_comparison(results, experiment_name,
                             out_path=os.path.join(out_dir, f"classifier_comparison_{safe_name}.png"))

    # ── 12. Feature Importance ────────────────────────────────────────────────
    print("\n[12/12] Feature Importance Overlap Analysis")
    overlap_df, counts = feature_importance_analysis(
        X, y_labels, mz, safe_name, out_dir,
        top_n=config.TOP_N_FEATURES
    )
    plot_spectrum_with_features(X_binned, mz_binned, y_labels, overlap_df, experiment_name,
                                out_path=os.path.join(out_dir, f"spectrum_features_{safe_name}.png"))

    print("\nDone.")

if __name__ == '__main__':
    main()
