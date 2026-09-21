# Study A: executed fitted-gap calibration

All values below come from the retained per-replicate records. The sampling protocol was saved before execution. No fits, gaps, or p-values were excluded or clipped. A numerical refinement gate was added during the final implementation audit, without tuning statistical results.

Executed 11200 replicates: 1,000 for each circle cell at n=200 and n=800, and 400 for each spherical cell at n=400. All fits succeeded.

## Primary results

| Cell | Fitted gap rate (95% Wilson interval) | Score | Wald | Naive | Kish |
|---|---:|---:|---:|---:|---:|
| circle_uniform_direct_n200 | 0.061 (0.048, 0.078) | 0.060 | 0.061 | 0.059 | 0.059 |
| circle_vm2_direct_n200 | 0.063 (0.050, 0.080) | 0.056 | 0.065 | 0.056 | 0.056 |
| circle_cos4_direct_n200 | 0.053 (0.041, 0.069) | 0.053 | 0.056 | 0.059 | 0.059 |
| circle_uniform_importance_n200 | 0.053 (0.041, 0.069) | 0.052 | 0.058 | 0.209 | 0.048 |
| circle_vm2_importance_n200 | 0.040 (0.030, 0.054) | 0.039 | 0.040 | 0.018 | 0.007 |
| circle_uniform_direct_n800 | 0.055 (0.042, 0.071) | 0.055 | 0.055 | 0.055 | 0.055 |
| circle_vm2_direct_n800 | 0.054 (0.042, 0.070) | 0.050 | 0.055 | 0.051 | 0.051 |
| circle_cos4_direct_n800 | 0.043 (0.032, 0.057) | 0.043 | 0.045 | 0.047 | 0.047 |
| circle_uniform_importance_n800 | 0.051 (0.039, 0.066) | 0.051 | 0.057 | 0.221 | 0.056 |
| circle_vm2_importance_n800 | 0.046 (0.035, 0.061) | 0.046 | 0.048 | 0.020 | 0.009 |
| sphere_uniform_direct_n400 | 0.062 (0.043, 0.091) | 0.062 | 0.065 | 0.062 | 0.062 |
| sphere_vm8_direct_n400 | 0.062 (0.043, 0.091) | 0.043 | 0.085 | 0.055 | 0.055 |
| sphere_uniform_importance_n400 | 0.052 (0.035, 0.079) | 0.055 | 0.087 | 0.247 | 0.055 |

The circle informative-weight examples show both liberal and conservative failure of an ordinary chi-square calibration. Kish scaling happens to work well for the uniform target here, but is strongly conservative for the nonuniform importance target. This does not imply that scalar ESS is generally sufficient.

The exact fourth-harmonic example has reference eigenvalues 1.4 and 0.6. Its empirical difference from naive chi-square rejection is small at this sample precision. The analytic covariance difference is exact; the simulation does not resolve a large practical rejection-rate difference in this cell.

The spherical Wald rates are 0.085 under vMF(8) and 0.0875 under informative sampling. This finite-sample inflation should be reported, not described as exact calibration.

## Numerical evidence

Maximum fitted moment residual: 2e-10. Maximum observed Hessian condition: 3.48e+03. Maximum absolute gap versus adjacent KL discrepancy: 2.82e-10.

On the first 25 replicates per cell, the maximum n-scaled grid refinement difference was 2.13e-12. On the first ten rotated replicates per cell, the maximum n-scaled rotation difference was 1.42e-12. These are observed diagnostics, not uniform error bounds.

Spherical reference probabilities use an add-one scrambled Sobol Gaussian rule with 16,384 nodes. Increasing this to 65,536 on refinement replicates changed p-values by at most 0.002108. This is numerical integration of an asymptotic reference law, not an exact randomization test. Circle two-eigenvalue tails use deterministic 256-angle integration.

Checked rows require successful finite refined fits and n times the absolute gap change at most 0.001. Failure makes all inferential outputs unavailable, retaining raw diagnostics. Unchecked rows are explicitly marked not_checked. This threshold is an operational diagnostic, not a uniform error certificate; it was added during the final implementation audit. All checked rows passed.

Cumulative solver time: 13.49 seconds; complete study time (excluding plotting) 22.71 seconds on this run. Hardware-dependent timings are descriptive.

The stored score is a² rᵀS⁻¹r, twice the half-quadratic convention in REVIEW. Both it and 2a²I use eigenvalues of S⁻¹ᐟ²ΩS⁻¹ᐟ². The saved discrepancy is a²I minus score/2.

Per-cell NPZ files preserve observations and unnormalized weights. CSV files preserve seeds, all p-values, numerical diagnostics, and fit status. Config files document the exact target and proposal ratios.
