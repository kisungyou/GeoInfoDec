# Study C: frozen held-out digits analysis

## Protocol and scope

The protocol was written before the first run. A fixed permutation assigns 600 images to representation training, 40 to queries, 600 to reference fitting, and 557 to held-out evaluation. The four sets are pairwise disjoint. All 40 query outcomes are retained. Standardization and three-dimensional PCA use only the representation set. Projected vectors are normalized onto the sphere; query weights use cosine similarity in the original standardized 64-dimensional feature space. The primary inverse temperature is 6; 3 and 9 are sensitivity analyses.

The estimand concerns this constructed spherical representation. Queries share reference and evaluation images. The source contains repeated handwriting observations, and writer identifiers are unavailable. Results are descriptive: no population p-values, independent-query standard errors, or population confidence intervals are reported. Labels identify queries but do not define geometric adequacy. The protocol is prespecified within this revision, not externally preregistered.

## Main results

| Inverse temperature | Positive gains | Mean gain | Median gain | Range | Median second gap |
|---:|---:|---:|---:|---|---:|
| 6 | 39/40 | 0.1528 | 0.1522 | -0.0007 to 0.4366 | 0.1587 |
| 3 | 40/40 | 0.0554 | 0.0451 | 0.0079 to 0.1390 | 0.0668 |
| 9 | 38/40 | 0.1404 | 0.1310 | -0.1179 to 0.4418 | 0.1785 |

All gains and information gaps use natural logarithms. Negative test gains are preserved.

| Reference summary | Descriptive Spearman association with held-out gain | Top-half mean gain | Negative gains among selected |
|---|---:|---:|---:|
| I1 | 0.490 | 0.1795 | 0/20 |
| I2 | 0.701 | 0.2033 | 0/20 |
| raw_second_norm | 0.605 | 0.1923 | 0/20 |
| residual_second_norm | 0.217 | 0.1714 | 1/20 |
| model_score | 0.677 | 0.1921 | 0/20 |
| sandwich_wald | 0.493 | 0.1823 | 0/20 |

The lower first-gap quartile contains 10 queries, with a maximum first gap of 0.6988 nats. Its mean held-out gain is 0.1273 nats, and 9 gains are positive. Its median second gap is 0.0775 nats.

A fixed top-half budget comparison selects 20 queries using reference information only. The complete policy results include always-vMF and always-Fisher-Bingham. This comparison describes one fixed finite benchmark. It is not a tuned or validated deployment rule. The information gap is more closely associated with primary held-out gains than the unscaled residual norm. The curvature score is close, and its selected set shares 18 of 20 queries with the gap selection. At inverse temperature 3, the Wald descriptor has a stronger rank association than the gap. The benchmark therefore does not establish a unique predictive advantage for GID. Since the richer model improves 39 of 40 primary queries, always using Fisher-Bingham has greater average log-score gain than selecting only half. No useful acceptance cutoff or optimal complexity penalty is established.

## Numerical and reproducibility records

Successful query-temperature fits: 120/120. Maximum moment residual: 1.87e-10. Maximum coarse/fine change in the second gap: 1.72e-11. Maximum coarse/fine change in test gain: 3.28e-10.

The shared solver fits unpenalized dual objectives. The integration rule combines Gauss-Legendre nodes in the vertical coordinate with equally spaced azimuths, using 24 by 48 nodes and a 48 by 96 refinement. Each query records both fit status and numerical errors. No ridge, target shrinkage, silent gap clipping, or significance threshold is used.

The raw second-moment norm is the Frobenius norm of the second moment minus one third of the identity. The residual norm subtracts the fitted vMF second moment. The model score uses the reduced-model Schur complement and the reference sample count. The sandwich Wald descriptor uses the normalized squared-weight empirical covariance. Its numerical value is reported without claiming a population null distribution.

Seed: 20260916. Data SHA-256: `20def7f70a702f0af9732fbba4375e147a7d54fe70d8c45569b8e7c1c7010c10`. Split SHA-256: `39793955dce7ea8ea2c5bb6c0b884e27826b6957f39f73dc114616a640c9fd61`. Analysis and numerical-data writing time: 0.68 seconds (excludes imports, report writing, and plotting). Total optimization iterations: 1359.

Machine-readable files contain every query-temperature result, all policy comparisons, partition IDs, raw and normalized weights, frozen standardization/PCA parameters, spherical coordinates, fitted natural parameters, moment covariances, and evaluation log-score differences. The failure manifest records all unsuccessful analyses; an empty list explicitly records no failures.

## Data attribution

E. Alpaydin and C. Kaynak (1998), Optical Recognition of Handwritten Digits, UCI Machine Learning Repository, DOI: [10.24432/C50P49](https://doi.org/10.24432/C50P49). The UCI repository identifies the dataset license as [CC BY 4.0](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits). The scikit-learn [load_digits documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) identifies its bundled 1,797-image data as the original UCI test subset. These metadata were checked on 2026-09-16.

The figure is generated by this analysis from the derived spherical coordinates. It contains no externally sourced artwork or copied publication graphics. The data remain attributable to Alpaydin and Kaynak under the UCI dataset license. No new license is asserted for the manuscript or its code.

Reproduce with `python code/study_c_digits.py` using the versions in `study_c_metadata.json`. No data download is required once scikit-learn is installed.

Independent validation passed for partition disjointness, Frobenius identities, the analytic vMF normalizer, the empirical log-score identity, shared-pipeline matrix scaling, and an additional 64 by 128 integration rule. A complete rerun reproduced all 120 substantive query rows and all 1,936 saved arrays exactly; timing fields were excluded. See `study_c_validation.json`, `study_c_validation_history.json`, and `study_c_reproducibility.json`.
