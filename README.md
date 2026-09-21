# Geometric Information Decomposition for Directional Data

Code and reproducibility records for the manuscript **Geometric Information Decomposition for Directional Data**.

A directional mean can cancel when observations concentrate around opposing directions. Geometric information decomposition compares nested maximum-entropy models to measure structure left after matching lower-order moments. This repository implements unpenalized fits, sampling-design-aware calibration, residual-moment comparisons, and a held-out digit-data application on the circle and sphere.

## Install

Use Python 3.11 and install the pinned dependencies in a virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`. No dataset download, account, accelerator, or external model is needed after installation. Study C uses the digit data bundled with scikit-learn.

## Run

From this directory:

```sh
# Exercise every experiment and numerical check with a small number of repetitions.
python run.py --mode smoke

# Reproduce the manuscript-scale experiments and compare them with saved records.
python run.py --mode paper --verify

# Also regenerate all manuscript figures as matched PGF, PDF, and PNG files.
python run.py --mode paper --figures --verify
```

Generated files go to `results/smoke/` or `results/paper/`. These directories are ignored by Git. Frozen manuscript outputs remain in `reference_results/` and are never overwritten. `--output-root PATH` chooses a separate output directory; a directory already used by one mode cannot be reused for the other. To run one component, add `--study a`, `--study b`, `--study c`, or `--study checks`; `--verify` requires a complete paper run.

Figure generation requires a TeX installation providing `pdflatex`, `amsmath`, and `amssymb`, plus a PDF-to-PNG converter supported by Matplotlib (Poppler or Ghostscript). Numerical runs do not require TeX. All figures share `code/figure_style.py`: Matplotlib's PGF backend, Computer Modern, 10-point text, and a natural width of 6.75 inches matching the AISTATS manuscript. Include their PDF or PGF files at natural size. PNG copies are provided for notebook and browser viewing. The introductory illustration uses retained clouds and estimates; the experimental figures use the selected run's fresh results.

## Notebooks

Start Jupyter with `jupyter lab`, then open:

1. [Numerical validity](notebooks/01_numerical_checks.ipynb): identities, invariance, invalid-input behavior, and refinement gates.
2. [Study A calibration](notebooks/02_study_a_calibration.ipynb): fitted-gap, score, Wald, naive, and Kish-rescaled reference tests.
3. [Study B information and inference](notebooks/03_study_b_information_and_inference.ipynb): residual structure, estimation, local alternatives, heavy weights, and a fixed Holm family.
4. [Study C held-out digits](notebooks/04_study_c_heldout_digits.ipynb): preprocessing fitted on a training partition, query-weighted fits, and separate held-out log scores.

Every notebook has `MODE = 'smoke'` and `FIGURES = False` near the top. Change `MODE` to `'paper'` to run the manuscript protocol; set `FIGURES = True` to regenerate figures when TeX is available. Notebooks call the same workflow as the command line and display their own generated results. Saved notebook outputs show smoke runs and are execution demonstrations, not evidence for the manuscript's claims.

| Component | Smoke mode | Paper mode |
|---|---:|---:|
| A: null calibration, 13 cells | 5 repetitions per cell; 65 total | 1,000 per circle cell and 400 per sphere cell; 11,200 total |
| B: fixed/local alternatives, 14 cells | 5 per cell; 70 total | 600 per cell; 8,400 total |
| B: heavy-weight stress, 3 laws | 5 paired samples per law; 30 weighted profiles | 100 paired samples per law; 600 profiles |
| B: finite-tail Holm family | 5 samples | 400 samples |
| C: held-out query analysis | First 4 frozen queries at 3 temperatures; 12 fits | All 40 queries at 3 temperatures; 120 fits |
| Numerical checks | 32 identities and 9 refinement-gate checks | The same 41 checks |

Smoke retains the full cell designs, sample sizes, fitting tolerances, integration rules, and seed conventions. Each simulation replicate has a fixed seed, so smoke outputs reproduce the first five paper replicates in every cell. Study C preserves the full disjoint 600/40/600/557 training/query/reference/evaluation split in both modes; only the evaluated query count changes. Smoke rates and correlations should not be interpreted as estimates of the paper's findings.

## Map to the manuscript

| Manuscript material | Implementation | Frozen evidence |
|---|---|---|
| Fitting and design-aware residual inference | `code/gid_pipeline.py` | Per-fit diagnostics in the study CSVs |
| Study A and calibration table/figure | `code/study_ab.py a` | `reference_results/study_a/` |
| Study B, estimation, and population comparisons | `code/study_ab.py b` | `reference_results/study_b/` |
| Heavy-weight stress and finite-tail Holm experiment | `code/study_b_supplement.py` | `pareto_stress_*`, `finite_tail_*`, and population CSVs in Study B |
| Study C and held-out figure | `code/study_c_digits.py`, `code/study_c_validate.py` | `reference_results/study_c_*` |
| Numerical evidence | `code/numerical_checks.py`, `code/refinement_gate_checks.py` | Numerical/check manifests in Study B |
| Introductory illustration | `code/make_problem_figure.py` | `reference_results/problem_*`, `reference_figures/problem.*` |

`protocols/` contains the frozen paper settings. Each run writes its selected settings and execution provenance alongside its numerical outputs. NPZ files preserve observations, weights, partitions, and fitted quantities; CSV files retain individual results and statuses. `reference_results/` is a snapshot from the manuscript's original numerical execution, so its timing and source-hash metadata describe that historical run. Current verification is documented in `checks/VALIDATION.md` and its machine-readable reports.

The verifier compares all regenerated scientific CSV fields and retained arrays, aggregate held-out results, Holm results, and numerical diagnostics. It excludes runtime timings and historical source/environment metadata. Floating-point comparisons allow relative tolerance `2e-10` and absolute tolerance `2e-11`; discrete values and array shapes must agree. Missing outputs or changed statuses fail verification.

## Scope of the evidence

The experiments support design-aware fitted-gap calibration in the reported settings and show how residual structure can be useful for held-out modeling. They do not establish universal superiority over residual-moment baselines. Wald intervals under-cover in some finite samples, the local power approximation can be slow, and the finite-tail experiment does not establish exact family-wise error control. Pareto-weight experiments are descriptive because the weights have infinite second moment. Study C's queries share reference and evaluation pools, so the query-level comparisons are dependent and descriptive.

Failed fits and failed checked refinements make inferential outputs unavailable rather than turning them into zero p-values. Unchecked refinements are labeled separately. Grid refinement is numerical evidence for these runs, not a uniform integration-error certificate. See the manuscript for assumptions and asymptotic statements.

## License and data attribution

Research code is distributed under the existing [MIT license](LICENSE). The digit benchmark is the Optical Recognition of Handwritten Digits dataset by E. Alpaydin and C. Kaynak (1998), distributed by UCI under CC BY 4.0: [dataset DOI](https://doi.org/10.24432/C50P49). Dependency and dataset notices appear in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/`.
