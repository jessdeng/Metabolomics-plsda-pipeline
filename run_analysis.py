import os
import numpy as np

from preprocessing import load_experiment, bin_features, filter_low_variance, filter_low_abundance, preprocess
from metaboanalyst_pipeline import compute_vip_1comp, fit_plsda, plot_scores_3d, plot_vip
from classifier_comparison import (
    RandomForest, svm_classify, gradient_boosting,
    logistic_regression, knn_classify, lda_classify, elasticnet_classify,
    plot_accuracy_comparison, feature_importance_analysis
)
from visualization import plot_spectrum_with_features

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT = 'your_experiment_folder'  # ← change this to your experiment folder name# ──────────────────────────────────────────────────────────────────────────────

def main():
    experiment_dir = os.path.join(BASE_DIR, EXPERIMENT)
    experiment_name = EXPERIMENT.strip()
    safe_name = experiment_name.replace(' ', '_').replace(':', '')

    assert os.path.isdir(experiment_dir), (
        f"Experiment folder not found: {experiment_dir!r}\n"
        f"Available options: {[d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d)) and not d.startswith('.')]}"
    )

    # ── 1. Load & filter ──────────────────────────────────────────────────────
    print(f"\n[1/12] Loading data: {experiment_name!r}")
    X_raw, y_labels, sample_names, mz = load_experiment(experiment_dir)
    print(f"  Raw samples : {X_raw.shape[0]}")
    print(f"  Raw features: {X_raw.shape[1]}")

    X_binned, mz = bin_features(X_raw, mz, bin_width=0.5)
    mz_binned = mz.copy()  # save for spectrum plot later
    print(f"  After binning: {X_binned.shape[1]} features")

    X_filt, mz = filter_low_variance(X_binned, mz, percentile=25)
    print(f"  After variance filter: {X_filt.shape[1]} features")

    X_filt, mz = filter_low_abundance(X_filt, mz, percentile=5)
    X_filt_raw = X_filt.copy()
    print(f"  After abundance filter: {X_filt.shape[1]} features")
    

    print("  Groups:")
    for g, count in zip(*np.unique(y_labels, return_counts=True)):
        print(f"    {g:25s}  n={count}")

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    print("\n[2/12] Preprocessing (sum norm → log10 → auto-scale)")
    X = preprocess(X_filt)

    # ── 3. PLS-DA ─────────────────────────────────────────────────────────────
    print("\n[3/12] Fitting PLS-DA (8 components for scores plot)")
    pls, T, y, Y, classes = fit_plsda(X, y_labels, 8)
    print(f"  Classes: {list(classes)}")

    # ── 4. VIP scores ─────────────────────────────────────────────────────────
    print("\n[4/12] Computing VIP scores (1 component)")
    vip = compute_vip_1comp(X, y_labels)
    plot_scores_3d(T, pls, y_labels, classes, experiment_name,
                   out_path=f"plsda_scores_3d_{safe_name}.html")
    plot_vip(vip, mz, X_filt_raw, y_labels, 30, experiment_name,
             out_path=f"vip_scores_{safe_name}.png")

    # ── 5–8. Classifiers ──────────────────────────────────────────────────────
    classifier_fns = {
        'Random Forest':       RandomForest,
        'SVM':                 svm_classify,
        'Gradient Boosting':   gradient_boosting,
        'Logistic Regression': logistic_regression,
        'KNN':                 knn_classify,
        'LDA':                 lda_classify,
        'ElasticNet':          elasticnet_classify,
    }

    results = {}
    for i, (name, fn) in enumerate(classifier_fns.items(), start=1):
        print(f"\n[{i + 4}/12] {name}")
        test_accs, train_accs = fn(X, y_labels)
        results[name] = (test_accs, train_accs)
        print(f"  Test  accuracy: {test_accs.mean():.3f} ± {test_accs.std():.3f}")
        print(f"  Train accuracy: {train_accs.mean():.3f} ± {train_accs.std():.3f}")

    # ── 8. Plot comparison ────────────────────────────────────────────────────
    print("\n[11/12] Plot Comparison")
    plot_accuracy_comparison(results, experiment_name,
                             out_path=f"classifier_comparison_{safe_name}.png")

    # ── 9. Feature Importance ─────────────────────────────────────────────────
    print("\n[12/12] Feature Importance Overlap Analysis")
    overlap_df, counts = feature_importance_analysis(X, y_labels, mz, safe_name)
    plot_spectrum_with_features(X_binned, mz_binned, y_labels, overlap_df, experiment_name,
                                out_path=f"spectrum_features_{safe_name}.png")

    print("\nDone.")

if __name__ == '__main__':
    main()
