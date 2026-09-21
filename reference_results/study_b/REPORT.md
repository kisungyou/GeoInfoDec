# Study B: executed interpretation, power, and estimation

The main simulation contains 8,400 independent repetitions, with 600 for each fixed or local alternative cell. All fits succeeded. Circle alternatives have density proportional to exp(2 cos(theta) + beta cos(2 theta)).

## Fixed alternatives

| n | beta | Population I2 | Bias | RMSE | 95% interval coverage |
|---:|---:|---:|---:|---:|---:|
| 200 | 0.3 | 0.0084 | 0.0050 | 0.0128 | 0.9267 |
| 200 | 0.6 | 0.0294 | 0.0027 | 0.0187 | 0.9100 |
| 800 | 0.3 | 0.0084 | 0.0011 | 0.0050 | 0.9383 |
| 800 | 0.6 | 0.0294 | 0.0009 | 0.0095 | 0.9350 |

Coverage falls below its nominal level, especially at n=200. The manuscript should not claim accurate finite-sample coverage from these runs. The intervals are justified asymptotically only at fixed nonzero gaps; local-sequence interval results in the full CSV are diagnostic only.

## Local alternatives

At b=4 with beta=b/sqrt(n), fitted-gap rejection was 0.203 at n=200 and 0.280 at n=800, compared with the limiting noncentral chi-square power 0.359. The model chi-square test had corresponding rejection 0.332 and 0.310. This path gives no claim of power dominance and shows slow approach to the local approximation.

## Population interpretation

The vMF(8) and balanced antipodal vMF(8) laws have the same raw quadratic anisotropy, about 0.5486. Their residual quadratic gaps are zero and 1.0659 nats, respectively. The axial girdle proportional to exp(-12z²) has gap 0.8632 nats. These are deterministic population calculations, independently refined from 48x96 to 96x192 quadrature.

For 1+0.8cos(4theta), the first three population gaps are zero and the fourth is 0.1670 nats. A fixed family testing I1, I2, I3, I4, and D4-D1 with Holm adjustment detected I4 and the finite-tail contrast in all 400 samples of size800. Stopping after the first nonsignificant gap would have stopped in371 samples.

Holm rejected at least one of the three true nulls in28/400 samples (0.070). Thus these finite-sample results do not establish exact family-wise error control; the stated result remains asymptotic. All planned tests and failures are retained.

## Descriptive heavy-weight stress

Three circle laws were each evaluated in100 paired samples of size400, with equal weights and independent Pareto weights of tail index1.5. The latter have infinite second moment. The stored effect-size and ESS summaries are descriptive and supply no Gaussian inferential validation.

## Numerical audit

The first ten replicates per cell were checked against a refined grid. The final implementation audit added an operational gate at n times the absolute gap change at most 0.001, with successful finite refined fits required. Failed checks make p-values, standard errors, and intervals unavailable. This addition did not tune statistical results and is not a uniform error certificate. All checked rows passed; unchecked rows remain explicitly marked.

32 of 32 numerical identity checks passed, including basis and rotation invariance, the analytic fourth-harmonic spectrum, moment derivatives, likelihood-ratio scaling, count compression, and explicit boundary/grid-infeasibility failure status.

Maximum moment residual over the main B simulation: 2e-10; maximum absolute adjacent-KL discrepancy: 1.32e-10; maximum n-scaled refinement change: 1.07e-12. These checks do not prove uniform numerical error bounds or coverage.
