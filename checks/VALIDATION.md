# Repository validation

Validated on 2026-09-21 with Python 3.11.15 and the pinned scientific dependencies, CPU computation, and one BLAS thread.

- The complete CLI ran successfully in smoke and paper modes, including optional figure generation.
- Paper mode regenerated 11,200 Study A replicates, 8,400 Study B replicates, 600 heavy-weight/equal-weight profiles, 400 Holm-family samples, and 120 held-out query-temperature analyses.
- Reference comparison passed for 74 scientific files, 788,102 CSV fields, and 1,983 retained arrays. Timing fields and historical environment/source metadata are excluded. See `paper_reference_comparison.json` for tolerances and scope.
- Study C additionally reproduced its 1,936 original arrays exactly, with the extra `evaluated_query_ids` array documenting the selected mode. The smoke analysis preserved the complete disjoint data split and used the first four frozen queries. Four corruption checks (missing row, wrong query pairing, nonempty failure manifest, and wrong mode) were rejected.
- All 32 numerical identity checks and nine refinement/failure-gate checks passed in both modes. These diagnostics do not establish uniform integration-error bounds.
- All four notebooks executed successfully in both modes. Delivered notebooks retain smoke settings and their executed outputs; paper validation used temporary copies and a separate output directory. See `notebook_execution.json`.
- All four figures in both modes passed PGF font-size, natural-width, asset, and text-clipping checks. Every prose, tick, and legend size is 10 TeX points in Computer Modern. The figure renderer left scientific inputs unchanged. Representative outputs were also visually reviewed.
- Regression tests reject mixed-mode output reuse, attempted writes to frozen references, and altered discrete comparison fields. Run `python checks/test_workflow.py` to repeat them.

The numerical core `code/gid_pipeline.py` is unchanged from the manuscript execution (SHA-256 `e816eb0a983addcb8b423798bf0566f9809bfc023fc247d3798cd320c0772d1b`). The source manifest records the repository adapters and workflow used for this release. Original timing and source hashes inside `reference_results/` describe the historical manuscript run, not this repository validation.

To repeat the complete scientific comparison:

```sh
python run.py --mode paper --figures --verify
```

To check execution without TeX or large simulation counts:

```sh
python run.py --mode smoke
```

Smoke outputs are not intended to substantiate empirical manuscript claims. Paper results retain the manuscript's finite-sample coverage, power, heavy-weight, and shared-query-pool limitations.
