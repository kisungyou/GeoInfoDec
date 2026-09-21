# Running Studies A and B

The default **smoke** mode checks that the complete fitting, inference, and reporting workflow executes. It uses five replicates in every simulation cell while preserving the paper's cell definitions, sample sizes, seeds, quadrature, solver tolerances, covariance estimators, and numerical checks. Smoke rates are not paper results or evidence of calibration, coverage, or power.

From the repository root, run:

```sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export GID_MODE=smoke
python code/study_ab.py
python code/study_b_supplement.py
python code/numerical_checks.py
python code/refinement_gate_checks.py
python code/summarize_ab.py
```

This writes to `results/smoke/`: 65 Study A replicates, 70 Study B replicates, 30 descriptive weighted profiles (three laws, five paired samples, two weight choices), and five fixed-family finite-tail analyses. All 32 numerical identity checks and nine refinement-gate checks run in both modes. Deterministic population calculations also retain full resolution in both modes.

For the complete paper protocol, set `GID_MODE=paper` and run the same five commands. The default destination is then `results/paper/`. Paper mode retains 11,200 Study A replicates, 8,400 Study B replicates, 600 descriptive weighted profiles, and 400 finite-tail analyses. Replicate ordering and per-cell seeds match the frozen protocol. `study_ab.py a` or `study_ab.py b` runs either principal study separately.

`GID_OUTPUT_ROOT` overrides the generated-output directory. Use a separate directory for each mode and use the same environment for every command in a run; the report generator rejects mixed-mode inputs. `reference_results/` contains frozen paper outputs and is never a generated-output destination. Each new run writes its mode and actual counts to configuration and manifest files. Rejection rates, Monte Carlo standard errors, Wilson intervals, and report statements use the actual configured replication count, retaining failures in denominators.

NumPy and SciPy are required for the numerical studies. Plots are disabled by default; set `GID_PLOTS=1` to enable the study plotting calls. Plotting additionally uses Matplotlib and the shared PGF/pdflatex style, so a working TeX installation is required. Generated tables do not require TeX to be installed. The environment manifest records the versions and source hashes used for each run.

## Retained data and interpretation

Configuration JSON files are saved before drawing data. NPZ files retain observations and raw weights. Per-replicate CSV files retain seeds, fitted gaps, uncertainty calculations, p-values, numerical diagnostics, and failure statuses. Generated reports are labelled by mode and calculate empirical statements from that run's outputs; they do not copy conclusions from frozen reference reports.

The Pareto stress study is descriptive: shape 1.5 gives infinite second moment, so it supplies no Gaussian inferential validation. Fixed nonzero-gap intervals are pointwise asymptotic approximations. Near-null interval results are diagnostic only. A finite family of valid tests can use Holm adjustment; stopping at the first nonsignificant gap is not justified.

## Shared fitting API

`gid_pipeline.py` defines positive normalized quadrature, raw circle Fourier features, and linear plus Frobenius-orthonormal traceless quadratic sphere features. `fit_dual` uses unpenalized damped Newton iteration and returns parameters, empirical dual value, density entropy, moment mismatch, curvature, timing, and status. Inferential gaps use dual differences; final fits use neither ridge nor gap clipping.

`analyze` fits one nested pair, computes its reduced-model residual and Schur complement, and estimates the covariance for deterministic or self-normalized importance weights. `design='iid'` uses equal weights in the deterministic-weight formula. Both covariance estimators center at the empirical full moment. A singular but nonzero residual covariance can retain quadratic-form calibration while leaving Wald inference unavailable. `expanded_counts_moments` preserves the original independent observation count for compressed integer frequencies. `holm_adjust` handles a fixed, complete family without assuming independence.

The stored `score` is a²rᵀS⁻¹r. It and 2a²I use the spectrum of S⁻¹ᐟ²ΩS⁻¹ᐟ². The half-quadratic score in the local expansion is `score/2`; the saved remainder is a²I−score/2. Nonzero-gap standard errors use the natural-parameter contrast and the design-specific sampling covariance.

Circle reference tails use angular quadrature. Higher-dimensional tails use fixed scrambled Sobol Gaussian integration with an add-one exceedance calculation. Neither is a finite-sample exact test. The node-count refinement records numerical sensitivity of the spherical reference probabilities.

## Numerical checks and failure handling

The identity checks include derivatives, basis changes, rotations, nested KL/log-score identities, the exact fourth-harmonic covariance, and frequency-count compression. Deliberate boundary and coarse-grid infeasibility examples must return unsuccessful fits.

Grid refinement checks up to the first 25 available replicates per Study A cell and up to the first ten per Study B cell. The operational criterion requires successful finite refined fits and n times the absolute gap change at most 0.001. Failed checks make inferential outputs unavailable; unchecked rows remain explicitly marked. The threshold was added during the final implementation audit without tuning statistical results. These checks are evidence about computed runs, not certified uniform integration-error bounds.

`refinement_gate_checks.py` injects failed, missing, nonfinite, and excessive-difference cases and verifies that unavailable inference is retained as such. It also verifies that successful or unchecked fits preserve their outputs.
