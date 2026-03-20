# Ambient MS Metabolomics Toolkit

A Python pipeline for ambient MS metabolomics data. Preprocessing and PLS-DA are validated against MetaboAnalyst to 4 decimal places at component 1. The toolkit also includes a full classifier comparison (7 ML models) and feature importance overlap analysis, going beyond what MetaboAnalyst offers out of the box.

For detailed notes on what each step does, why the defaults were chosen, and how to interpret the outputs, see [NOTES.md](NOTES.md).

> ⚠️ **Important:** The default parameters were designed for fungal metabolomics data collected on a SCIEX 4500 triple quadrupole in MS1-only mode using the LMJ-SSP. If you are using different instrumentation or sample types, you will likely need to adjust the parameters in `config.py`.

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
4. Run:

```bash
python run_analysis.py
```

The terminal will print progress as each step runs. See [NOTES.md](NOTES.md) for guidance on adjusting parameters.

---

## Output Files

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
