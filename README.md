# Ambient MS Metabolomics Toolkit

A Python pipeline for ambient MS metabolomics data. Includes a full classifier comparison (6 ML models), ensemble feature importance voting, and cross-validated PLS-DA. Two parallel pipelines are provided — one with bin labeling compatible with R-based PLS-DA packages, one with data-driven bin labels for accurate compound identification.

For detailed notes on what each step does, why the defaults were chosen, and how to interpret the outputs, see [NOTES.md](NOTES.md).

> ⚠️ **Important:** The default parameters were designed for fungal metabolomics data collected on a SCIEX 4500 triple quadrupole in MS1-only mode using the LMJ-SSP. If you are using different instrumentation or sample types, adjust the parameters in `config.py`.

---

## Which version should I use?

**If you are not trying to match R package PLS-DA output, use the standard version.**

| | R-comparable | Standard |
|---|---|---|
| **Run with** | `python -m r_comparable.run_analysis` | `python -m standard.run_analysis` |
| **Extras with** | `python -m r_comparable.extras` | `python -m standard.extras` |
| **Outputs to** | `output_r_comparable/` | `output_standard/` |
| **Bin labels** | Fixed arithmetic offset matching R package convention | Mean of actual m/z values within each bin |
| **Use when** | Comparing results against R-based PLS-DA pipelines | General use — accurate m/z labels for database lookup |

Classifier accuracies, VIP scores, and PLS-DA plots are numerically identical between the two versions — the models only see the binned intensity matrix, not the m/z labels. The difference is in the reported m/z for each important feature.

---

## Repository Structure

```
ambient-ms-metabolomics-toolkit/
├── config.py               ← only file you need to edit
├── requirements.txt
├── setup.py
├── README.md
├── NOTES.md
│
├── shared/                 ← shared modules (classifiers, visualization)
│   ├── classifier_comparison.py
│   ├── classifier_comparison_standard.py
│   └── visualization.py
│
├── r_comparable/           ← R package-compatible pipeline
│   ├── preprocessing.py
│   ├── pipeline.py
│   ├── run_analysis.py
│   └── extras.py
│
├── standard/               ← standard pipeline
│   ├── preprocessing.py
│   ├── pipeline.py
│   ├── run_analysis.py
│   └── extras.py
│
└── scripts/                ← additional analysis scripts
    └── condition_abundance.py
```

---

## Setup Instructions

### Step 1 — Install Python

1. Go to [https://www.python.org/downloads](https://www.python.org/downloads)
2. Download and run the installer
3. ⚠️ Check **"Add Python to PATH"** before clicking Install

```bash
python --version
```

---

### Step 2 — Install VS Code

1. Go to [https://code.visualstudio.com](https://code.visualstudio.com)
2. Install and open VS Code
3. Install the **Python** extension by Microsoft

---

### Step 3 — Download This Repository

```bash
git clone https://github.com/jessdeng/ambient-ms-metabolomics-toolkit.git
```

Or click **Code → Download ZIP** and unzip.

---

### Step 4 — Install Required Packages

```bash
python setup.py
```

---

### Step 5 — Set Up Your Data

```
your_experiment_folder/
├── Group1/
│   ├── sample1.csv
│   ├── sample2.csv
├── Group2/
│   ├── sample1.csv
│   ├── sample2.csv
```

- Each subfolder = one group
- Each file = one sample
- Supported: `.csv` (comma-separated) or `.txt` (tab-separated)
- Required columns: `mz` or `Mass/Charge`, and `int` or `Intensity`

Place your experiment folder in the **root of the repository** — the same level as `config.py`. The folder name becomes your experiment identifier throughout the pipeline.

```
ambient-ms-metabolomics-toolkit/
├── config.py
├── your_experiment_folder/    ← your data goes here
│   ├── Control/
│   │   ├── sample1.csv
│   │   ├── sample2.csv
│   ├── Treatment/
│   │   ├── sample1.csv
│   │   ├── sample2.csv
├── r_comparable/
├── standard/
├── shared/
└── scripts/
```

---

### Step 6 — Configure and Run

1. Open `config.py` and set your experiment folder name:

```python
EXPERIMENT = 'your_experiment_folder'
```

2. Run from the repository root:

```bash
# Standard pipeline (recommended)
python -m standard.run_analysis

# R-compatible pipeline
python -m r_comparable.run_analysis
```

---

## Output Files

Outputs are saved to `output_standard/` or `output_r_comparable/` depending on which pipeline you run.

| File | Description |
|------|-------------|
| `plsda_scores_3d_*.html` | Interactive 3D PLS-DA scores plot |
| `vip_scores_*.png` | Top VIP features with intensity heatmap |
| `classifier_comparison_*.png` | Per-fold accuracy and overfitting panel |
| `spectrum_features_*.png` | Average spectrum with important features marked |
| `feature_overlap_*.csv` | Ensemble feature candidates ranked by method agreement, with per-group mean intensity, signed Ridge coefficients, and a "top condition" call for each feature |
| `classifier_results_*.npz` | Saved CV results — loaded automatically by extras |

---

## Requirements

- Python 3.8 or higher
- See `requirements.txt` for package versions
