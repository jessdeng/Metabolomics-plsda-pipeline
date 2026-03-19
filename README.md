# Ambient MS Metabolomics Toolkit

This toolkit started as a Python recreation of the [MetaboAnalyst](https://www.metaboanalyst.ca) preprocessing and PLS-DA pipeline, built to give direct agency over my own mass spectrometry data without relying on a web interface. The core preprocessing steps — binning, filtering, normalization, log transformation, and auto-scaling — were validated against MetaboAnalyst's own output using the same dataset and settings, matching to 4 decimal places at the PLS-DA component 1 level.

From that foundation, the toolkit has expanded to include classifier comparison (Random Forest, SVM, Gradient Boosting) and feature importance overlap analysis, going beyond what MetaboAnalyst offers out of the box.

> ⚠️ **Important:** The default parameters in this pipeline were designed for a specific dataset — fungal metabolomics data collected on a SCIEX 4500 triple quadrupole mass spectrometer in MS1-only mode using liquid microjunction surface sampling probe (LMJ-SSP). If you are using different instrumentation or sample types, you will likely need to adjust the parameters. Guidance on what to change and why is provided inline below.

---

## What This Pipeline Does

### 1. Loading

Your raw spectra are read in from CSV or TXT files. Each file is one sample, and each subfolder represents one group (e.g. control vs treatment). Because different instruments or export settings can produce spectra with slightly different m/z axes, all samples are interpolated onto a shared m/z axis so they can be compared directly.

### 2. Binning

Raw spectra often have m/z values measured at slightly different points across samples. Binning groups nearby m/z values into fixed-width windows and sums their intensities, making the data consistent and reducing noise from small m/z shifts between runs.

**Default: 0.5 Da bins.** Using a 0.5 Da bin width for triple quadrupole (QqQ) data is a common approach in metabolomics, particularly for low-resolution instruments, to ensure isotopic peaks are correctly accounted for and to improve spectral consistency across samples. Although the SCIEX 4500 has a native step size of 0.1 Da, 0.5 Da bins were used to account for m/z drift across runs. If your instrument has higher resolution or more stable m/z calibration, you may want to reduce the bin width.

### 3. Filtering

Two filters are applied to remove uninformative features before any statistics:

- **Low variance filter** — removes features whose intensity barely changes across samples. If a feature looks the same in every sample, it cannot help distinguish your groups.
- **Low abundance filter** — removes features with very low mean intensity across all samples. These are likely noise rather than real signal.

**Default: 25% variance filter, 5% abundance filter.** These thresholds are taken from MetaboAnalyst defaults. They may not be optimal for every dataset — if too many or too few features are being removed, these are the first parameters to adjust.

### 4. Normalization and Scaling

Three transformations are applied to make samples comparable to each other:

- **Sum normalization** — divides each sample by its total ion current (TIC) and rescales to the median, correcting for differences in how much sample was injected.
- **Log10 transformation** — compresses the wide dynamic range of mass spec data so that very high-intensity features do not dominate the analysis.
- **Auto-scaling** — subtracts the mean and divides by the standard deviation for each feature, so all features contribute equally regardless of their absolute intensity.

### 5. PLS-DA (Partial Least Squares Discriminant Analysis)

PLS-DA is a supervised dimensionality reduction method. It finds combinations of your m/z features (components) that best separate your experimental groups. The 3D scores plot shows where each sample sits in this reduced space — samples that cluster together are metabolically similar, and separation between groups indicates the pipeline has found distinguishing features.

### 6. VIP Scores (Variable Importance in Projection)

VIP scores rank each m/z feature by how much it contributed to the group separation found by PLS-DA. Features with a VIP score above 1.0 are generally considered important. The top 30 features are shown in a dot plot alongside a heatmap of their mean intensity per group, so you can immediately see which features are high or low in which group.

### 7. Classifier Comparison

Three machine learning classifiers (Random Forest, Support Vector Machine, Gradient Boosting) are each trained and evaluated using 5-fold cross-validation. This means the data is split into 5 parts, and each classifier is tested on data it has never seen. The accuracy scores tell you how well your m/z features can predict group membership, and comparing classifiers helps confirm whether the separation is robust.

### 8. Feature Importance Overlap

The top features from each classifier are compared to identify which m/z values appear as important across multiple methods. Features that show up in PLS-DA VIP scores AND multiple classifiers are the most reliable candidates for further investigation.

---

## Setup Instructions (Start Here)

### Step 1 — Install Python

1. Go to [https://www.python.org/downloads](https://www.python.org/downloads)
2. Click the big yellow **Download Python** button
3. Run the installer
4. ⚠️ **Important:** Check the box that says **"Add Python to PATH"** before clicking Install

To verify it worked, open a terminal and run:

```bash
python --version
```

You should see something like `Python 3.12.0`.

---

### Step 2 — Install VS Code

1. Go to [https://code.visualstudio.com](https://code.visualstudio.com)
2. Download and install for your operating system
3. Open VS Code
4. Go to the Extensions panel (left sidebar, looks like four squares)
5. Search for **Python** and install the extension by Microsoft

---

### Step 3 — Download This Repository

1. Click the green **Code** button at the top of this GitHub page
2. Click **Download ZIP**
3. Unzip the folder somewhere on your computer (e.g. Desktop)

Or if you have Git installed:

```bash
git clone https://github.com/jessdeng/ambient-ms-metabolomics-toolkit.git
```

---

### Step 4 — Install Required Packages

1. Open the project folder in VS Code: **File → Open Folder**
2. Open a terminal in VS Code: **Terminal → New Terminal**
3. Run:

```bash
python setup.py
```

This installs all the Python packages the pipeline needs. You only need to do this once.

---

### Step 5 — Set Up Your Data

Organize your experiment data in the following folder structure:

```
your_experiment_folder/
├── Group1/
│   ├── sample1.csv   or   sample1.txt
│   ├── sample2.csv   or   sample2.txt
├── Group2/
│   ├── sample1.csv   or   sample1.txt
│   ├── sample2.csv   or   sample2.txt
```

- Each **subfolder** = one group/class (e.g. `Control`, `Treatment`)
- Each **file** = one sample
- Supported formats: `.csv` (comma-separated) or `.txt` (tab-separated)
- Required columns:
  - `mz` or `Mass/Charge` — m/z values
  - `int` or `Intensity` — intensity values

**Default m/z range: 100–1000 Da.** This reflects the range where meaningful fungal metabolite signal was observed on the SCIEX 4500. Data above 1000 Da was collected but found to be uninformative for this experiment. Adjust this range in `run_analysis.py` to suit your own data.

---

### Step 6 — Configure and Run

1. Open `run_analysis.py` in VS Code
2. At the top of the file, set the experiment folder name:

```python
EXPERIMENT = 'your_experiment_folder'
```

3. Make sure your experiment folder is in the **same directory** as the `.py` files
4. In the terminal, run:

```bash
python run_analysis.py
```

---

## Output Files

After running, the following files will be saved in the project folder:

| File | Description |
|------|-------------|
| `plsda_scores_3d_*.html` | Interactive 3D PLS-DA scores plot |
| `vip_scores_*.png` | Top 30 VIP features with heatmap |
| `classifier_comparison_*.png` | Accuracy comparison across classifiers |
| `spectrum_features_*.png` | Mass spectrum with important features highlighted |
| `feature_importance_*.csv` | Top features from each classifier exported as CSV |

---

## Requirements

- Python 3.8 or higher
- See `requirements.txt` for package versions
