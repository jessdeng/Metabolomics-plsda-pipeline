# Ambient MS Metabolomics Toolkit

A Python pipeline for ambient MS metabolomics data. Preprocessing and PLS-DA are validated against MetaboAnalyst to 4 decimal places at component 1. The toolkit also includes a full classifier comparison (7 ML models) and feature importance overlap analysis, going beyond what MetaboAnalyst offers out of the box.

For detailed notes on what each step does, why the defaults were chosen, and how to interpret the outputs, see [NOTES.md](NOTES.md).

> ⚠️ **Important:** The default parameters were designed for fungal metabolomics data collected on a SCIEX 4500 triple quadrupole in MS1-only mode using the LMJ-SSP. If you are using different instrumentation or sample types, you will likely need to adjust the parameters in `config.py`.

---

## Which version should I use?

This toolkit has two parallel pipelines. **If you are not trying to replicate MetaboAnalyst, use the standard version.**

| | MetaboAnalyst version | Standard version |
|---|---|---|
| **Run with** | `python run_analysis.py` | `python run_analysis_standard.py` |
| **Extras with** | `python extras.py` | `python extras_standard.py` |
| **Outputs to** | `output_metaboanalyst/` | `output_standard/` |
| **Bin labels** | Offset by -0.05 Da to match MetaboAnalyst's internal convention | Mean of actual m/z values within each bin — physically meaningful |
| **Use when** | You need results to match MetaboAnalyst exactly for validation or comparison | You want accurate m/z labels for database lookup and reporting |

The classifier accuracies, VIP scores, and PLS-DA plots are numerically identical between the two versions — the models only see the binned intensity matrix, not the m/z labels. The difference is in the reported m/z for each important feature, which matters when identifying candidate compounds.

---

## Setup Instructions

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

VS Code is a free code editor that makes it easy to open, edit, and run Python files.

1. Go to [https://code.visualstudio.com](https://code.visualstudio.com)
2. Download and install for your operating system
3. Open VS Code and install the **Python** extension by Microsoft (Extensions panel, left sidebar)

---

### Step 3 — Download This Repository

Click the green **Code** button at the top of this page → **Download ZIP**, then unzip the folder somewhere on your computer.

Or if you have Git installed:

```bash
git clone https://github.com/jessdeng/ambient-ms-metabolomics-toolkit.git
```

---

### Step 4 — Install Required Packages

1. Open the project folder in VS Code: **File → Open Folder**
2. Open a terminal: **Terminal → New Terminal**
3. Run:

```bash
python setup.py
```

This only needs to be done once.

---

### Step 5 — Set Up Your Data

Organize your data in the following folder structure. Subfolder names become your group labels in all plots and outputs.

```
your_experiment_folder/
├── Group1/
│   ├── sample1.csv
│   ├── sample2.csv
├── Group2/
│   ├── sample1.csv
│   ├── sample2.csv
```

- Each **subfolder** = one group/class (e.g. `Control`, `Treatment`)
- Each **file** = one sample
- Supported formats: `.csv` (comma-separated) or `.txt` (tab-separated)
- Required columns: `mz` or `Mass/Charge`, and `int` or `Intensity`

---

### Step 6 — Configure and Run

1. Open `config.py` in VS Code
2. Set your experiment folder name:

```python
EXPERIMENT = 'your_experiment_folder'
```

3. Make sure your experiment folder is in the **same directory** as the `.py` files
4. Run the standard version:

```bash
python run_analysis_standard.py
```

Or the MetaboAnalyst-compatible version:

```bash
python run_analysis.py
```

The terminal will print progress as each step runs. See [NOTES.md](NOTES.md) for guidance on adjusting parameters.

---

## Output Files

All outputs are saved to `output_standard/` or `output_metaboanalyst/` depending on which version you run.

| File | Description |
|------|-------------|
| `plsda_scores_3d_*.html` | Interactive 3D PLS-DA scores plot — open in any web browser |
| `vip_scores_*.png` | Top VIP features with intensity heatmap per group |
| `classifier_comparison_*.png` | Per-fold accuracy dots and train vs test overfitting panel |
| `spectrum_features_*.png` | Average mass spectrum per group with important features marked |
| `feature_overlap_*.csv` | Important m/z features ranked by how many methods identified them |
| `classifier_results_*.npz` | Saved classifier CV results — loaded automatically by `extras.py` |

---

## Requirements

- Python 3.8 or higher
- See `requirements.txt` for package versions
