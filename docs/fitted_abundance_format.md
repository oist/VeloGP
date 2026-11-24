# Fitted abundance file format

Produced by the `--out-abundance` option of VeloGP (`python -m velogp`) and
VeloMix (`python -m velogp.velomix`). In
[MFASeqPipeline](https://github.com/oist/MFASeqPipeline/blob/master/docs/mfa_pipeline.md#velogp)
these are `velogp_abundance_file` (`{sample}.velogp.abd.gz`) and
`velomix_abundance_file` (`{sample}.velomix.abd.gz`). The file compares the
observed abundances with the abundances predicted by the fitted model.

* Tab-separated text, gzip- or bz2-compressed if the file name ends in `.gz` or
  `.bz2`.
* The first line is a header with the column names below.
* One row per window of the input
  [abundance file](https://github.com/oist/MFASeqPipeline/blob/master/docs/abundance_format.md)
  that was used for the fit, in the order of the input. Windows with abundance
  ≤ 0, standard error ≤ 0 or validity 0 (by default `ratio_mappable = 0`) are
  left out.
* VeloMix writes the same windows in the same order as the VeloGP fit it
  started from.

## Columns

| # | Column | Type | Description |
|---|---|---|---|
| 1 | `chr` | string | Chromosome name |
| 2 | `pos` | int | Window position, from the input's position column (by default `midpoint`, 0-based), truncated to an integer |
| 3 | `y_obs` | float | Observed abundance, from the input's abundance column (by default `coverage_ratio`) |
| 4 | `y_obs_stderr` | float | Standard error s_obs of `y_obs` used in the fit. Without `--error-model` this is the input's standard error column (by default `coverage_ratio_stderr`); with `--error-model` it also includes the overdispersion, `sqrt(q² + d · y_obs²)` |
| 5 | `y_pred` | float | Predicted abundance, `scale · p_pred`, where `scale = mean(y_obs) / mean(p_pred)` |
| 6 | `p_pred` | float | Unscaled predicted abundance P(x) = exp(−λ · t(x)), where t(x) is the time at which `pos` is replicated. P = 1 for positions replicated at time 0 |

## VeloMix

The VeloMix file has the same six columns, but `y_pred` and `p_pred` refer to
the mixture over all components. They are followed by two columns for each of
the K = 2ⁿ components of a mixture with n stochastic events (see
[velomix.md](velomix.md#event-order-and-component-labels) for the component
labels):

| # | Column | Type | Description |
|---|---|---|---|
| 7 … 6+K | `y.<label>` | float | Predicted abundance of the component, `scale · P_k(x)`, using the mixture's scale |
| 7+K … 6+2K | `w.<label>` | float | Weight w_k of the component; the same in all rows |

The prediction of the mixture is the weighted sum of the components:
`y_pred = Σ_k w.<label> · y.<label>`.

## Example

VeloGP:

```
chr	pos	y_obs	y_obs_stderr	y_pred	p_pred
U00096.3	1499	1.03913	0.02119	1.02672	0.77583
U00096.3	2499	1.00152	0.02041	1.02649	0.77565
U00096.3	3499	1.04324	0.02134	1.02625	0.77547
```

VeloMix with three events (oriX fires, terD stalls, terA stalls; columns
abbreviated):

```
chr	pos	y_obs	y_obs_stderr	y_pred	p_pred	y.000	y.001	...	y.111	w.000	w.001	...	w.111
U00096.3	1499	1.03913	0.02119	1.02772	0.77704	1.02772	1.02772	...	1.02772	3.10734e-05	0.0127909	...	0.528144
U00096.3	2499	1.00152	0.02041	1.02749	0.77686	1.02749	1.02749	...	1.02749	3.10734e-05	0.0127909	...	0.528144
```
