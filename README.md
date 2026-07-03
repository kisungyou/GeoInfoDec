# Geometric Information Decomposition on the Sphere

This repository contains compact, reproducible examples for the manuscript titled

> Geometric Information Decomposition for Weighted Empirical Measures on the Sphere.

The main contribution is to propose **geometric information decomposition (GID)**: a maximum-entropy uncertainty profile for weighted empirical probability measures on the unit sphere.

The observed object is a weighted empirical probability measure on the sphere. Equal weights recover the usual empirical distribution; nonuniform weights arise in particles, importance samples, quadrature rules, reliability-weighted directional observations, and attention/retrieval-weighted normalized embeddings.

GID replaces a single von Mises--Fisher concentration summary with entropy gaps from nested maximum-entropy projections:

- **Level 1:** mean-direction / vMF information.
- **Level 2:** axial, antipodal, elliptic, and girdle structure.
- **Higher levels:** finer angular structure, including multimodality.

## Repository contents

```text
.
├── README.md
├── requirements.txt
├── src.py
├── LICENSE
└── notebooks/
    ├── 01_s1_fourier_hierarchy.ipynb
    ├── 02_s2_low_order_gaps.ipynb
    ├── 03_null_calibration_s2.ipynb
    └── 04_query_weighted_digits.ipynb
```

The notebooks are designed to be read directly on GitHub. They include saved outputs, so GitHub will render the tables and figures without executing the code. To recompute the examples, run the notebooks locally.

## Installation

```bash
git clone https://github.com/kisungyou/GeoInfoDec
cd GeoInfoDec
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then start Jupyter:

```bash
jupyter lab
```

## Core code

The file `src.py` contains reusable computational tools rather than complete examples. The notebooks call these functions to produce demonstrations. The main groups of functions are:

- weighted empirical utilities;
- sampling routines on `S^1` and `S^2`;
- quadrature rules;
- Fourier and low-order spherical feature maps;
- maximum-entropy fitting by quadrature;
- information gaps and effective uncertainty;
- vMF summaries;
- second-order null-calibration tools.

## Notebooks

1. **[`01_s1_fourier_hierarchy.ipynb`](notebooks/01_s1_fourier_hierarchy.ipynb)**  
   Demonstrates Fourier maximum-entropy gaps on the circle. A unimodal distribution activates the first gap, an antipodal mixture activates the second, and a trimodal mixture activates the third.

2. **[`02_s2_low_order_gaps.ipynb`](notebooks/02_s2_low_order_gaps.ipynb)**  
   Demonstrates vMF, antipodal, and girdle examples on the ordinary sphere. The notebook compares GID gaps with the mean-resultant length, fitted vMF concentration, and raw second-moment anisotropy.

3. **[`03_null_calibration_s2.ipynb`](notebooks/03_null_calibration_s2.ipynb)**  
   Shows why the null calibration must account for weights. Equal weights are well calibrated by a chi-square law, while informative importance weights require the quadratic-form calibration.

4. **[`04_query_weighted_digits.ipynb`](notebooks/04_query_weighted_digits.ipynb)**  
   A small real-data illustration using scikit-learn digits. Images are projected to three dimensions, normalized to `S^2`, and assigned query-specific softmax weights.

## Reproducibility notes

All notebooks use fixed random seeds. The examples are intentionally small enough to run on a laptop. They are meant to demonstrate the methodology, not to be a full optimized software implementation for high-dimensional spherical harmonics.