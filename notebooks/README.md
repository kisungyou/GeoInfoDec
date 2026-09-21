# Study notebooks

These notebooks use the same `workflow.run_study` entry point as the repository workflow. They read the generated CSV, JSON, and array files rather than implementing separate estimators.

| Notebook | Content |
| --- | --- |
| [01_numerical_checks.ipynb](01_numerical_checks.ipynb) | Numerical identities, boundary diagnostics, and refinement failure handling |
| [02_study_a_calibration.ipynb](02_study_a_calibration.ipynb) | Moment-null calibration under direct and informative-weight sampling |
| [03_study_b_information_and_inference.ipynb](03_study_b_information_and_inference.ipynb) | Population geometry, fixed alternatives, local power, descriptive stress, and Holm tests |
| [04_study_c_heldout_digits.ipynb](04_study_c_heldout_digits.ipynb) | Fixed-split held-out digit analysis and descriptive query comparisons |

The saved notebook outputs come from successful smoke-mode executions. All four notebooks were also executed at paper scale in separate validation copies.

Install the repository requirements and open Jupyter from the repository root or this directory. Each notebook locates `workflow.py` from the current directory or its ancestors.

The first code cell contains two settings:

```python
MODE = "smoke"  # Change to "paper" for the complete experiment.
FIGURES = False
```

Restart and run all cells after changing settings. Both modes use the canonical fitting engine, feature definitions, sampling designs, and numerical checks. Smoke mode reduces simulation repetitions and the number of analyzed Study C queries. It preserves the sample sizes within simulations and the complete Study C data partition.

| Component | Smoke | Paper |
| --- | ---: | ---: |
| Numerical checks | 32 identity checks and 9 failure-handling checks | Same checks |
| Study A | 65 replicates across 13 cells | 11,200 replicates |
| Study B main paths | 70 replicates across 14 cells | 8,400 replicates |
| Study B descriptive profiles | 30 profiles | 600 profiles |
| Study B finite family | 5 replicates | 400 replicates |
| Study C | 4 frozen queries at 3 temperatures | 40 frozen queries at 3 temperatures |

Generated files are written to `results/smoke` or `results/paper`. The frozen `reference_results` directory is separate and is not overwritten. Every displayed result comes from the selected run. No notebook substitutes a frozen paper figure for a smoke result.

The default run needs no TeX installation. Set `FIGURES = True` to request the shared publication-figure workflow, which requires LaTeX and its image-conversion dependencies. The notebooks display the resulting PNG companions from the same output directory as the tables.

Smoke outputs demonstrate that the workflow executes. They are not estimates precise enough to support paper claims about calibration, coverage, or power. Study C remains a descriptive finite-benchmark analysis in both modes because queries share fitting and evaluation pools.
