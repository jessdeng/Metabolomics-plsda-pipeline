# Metabolomics PLS-DA Pipeline

A Python pipeline for supervised multivariate analysis of mass spectrometry data using Partial Least Squares Discriminant Analysis (PLS-DA). Built to reproduce and validate results from [MetaboAnalyst](https://www.metaboanalyst.ca/), a widely used web-based metabolomics platform.

## Background

This project originated from my PhD research analyzing metabolic differences of fungal species across different experimental conditions using ambient ionization mass spectrometry. The original analysis was performed in MetaboAnalyst. I rebuilt the pipeline in Python to deepen my understanding of the underlying algorithms, gain full control over each processing step, and create a reproducible, scriptable workflow.

## Pipeline Overview

The pipeline follows the same sequence as MetaboAnalyst:

1. **Data loading** — Reads raw CSV files organized by experimental group
2. **m/z binning** — Combines features into 0.5 Da bins by summing intensities
3. **Data filtering** — Removes low-variance features (bottom 25% by relative standard deviation) and low-abundance features (bottom 5% by mean intensity)
4. **Sum normalization** — Scales each sample by its total intensity, adjusted to the median across all samples
5. **Log10 transformation** — Applies log base 10 with half-minimum imputation for zero values
6. **Auto-scaling** — Mean-centers and divides by the standard deviation of each feature (z-score), following the rationale in [van den Berg et al. (2006)](https://doi.org/10.1186/1471-2164-7-142)
7. **PLS-DA** — Fits a Partial Least Squares Discriminant Analysis model to separate experimental groups
8. **Cross-validation** — 5-fold stratified cross-validation to assess model performance
9. **VIP scores** — Variable Importance in Projection scores to identify the most discriminating m/z features

## Validation Against MetaboAnalyst

The component 1 VIP scores from this pipeline were validated against MetaboAnalyst's output across all 2707 features:

- Maximum absolute difference: 0.00005
- Ranking correlation: 0.9999999808
- Identical number of features above the VIP > 1 threshold

VIP scores are computed using 1 component because Python's scikit-learn PLS implementation and MetaboAnalyst's R-based ropls package produce identical first components but diverge at higher components due to differences in the internal deflation algorithms. This is a known cross-platform numerical issue, not a bug.

## Outputs

- **PLS-DA scores plot** — 2D scatter plot of component 1 vs component 2 with % covariance explained on each axis
- **VIP bar chart** — Top 30 features ranked by VIP score
- **VIP table** — CSV file with all m/z features and their VIP scores, sorted by importance

## Project Structure

```
metabolomics-plsda-pipeline/
├── README.md
├── metaboanalyst_pipeline.py      # Main pipeline script
├── requirements.txt               # Python dependencies
└── S9 Carbon with Media/          # Example experiment folder
    ├── Control Media/             #   Group subfolder containing sample CSVs
    ├── Glycerol Media/
    ├── S9 Control/
    └── ...
```

## Usage

```bash
python metaboanalyst_pipeline.py
```

To analyze a different experiment, change the `EXPERIMENT` variable at the top of the script.

## Requirements

- Python 3.8+
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
```

## What I Learned

- How PLS-DA works internally: the NIPALS algorithm, weight vectors, score deflation, and VIP computation
- Why different software implementations (Python vs R) can produce different multivariate results despite using theoretically equivalent algorithms
- The importance of validating computational pipelines against established tools before drawing scientific conclusions
- Practical Python skills: numpy array operations, pandas data wrangling, matplotlib visualization, and scikit-learn's machine learning API
