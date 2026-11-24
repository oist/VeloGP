# VeloGP: deterministic model

VeloGP (`python -m velogp`) fits a deterministic model of replisome movement to marker-frequency (abundance) data.
- The stochastic extension VeloMix is described in [velomix.md](velomix.md).
- Usage examples are in the [README](../README.md#usage), and all file formats are listed in [the index](README.md).

## Input
- **Abundance table:** usually an
  [abundance file](https://github.com/oist/MFASeqPipeline/blob/master/docs/abundance_format.md) from
  MFASeqPipeline; the columns used are chosen with `--columns`. The default columns are `seq,midpoint,coverage_ratio,coverage_ratio_stderr,ratio_mappable`, i.e.:
  - chromosome;
  - window position;
  - abundance y_obs;
  - its Poisson std. error q;
  - validity.

  Windows are dropped if they have y ≤ 0, q ≤ 0 or validity 0.
- **Genome:** `--genome` FASTA (chromosome names and lengths). Chromosomes are circular unless listed in `--linear-chromosomes`.
- **Origins:** `--origins` takes a location (`[chr:]pos`, 1-based), a feature name, or a GFF feature type. The
  default is the type `origin_of_replication`. Origins found by type are named `<type>1..N`, ordered by
  (chromosome, position).
- **vx features:** the genomic intervals where replisomes may slow down. They are given by `--vx-feature-names` or
  `--vx-feature-types`, resolved in `--gff` / `--bed` files (see [features_format.md](features_format.md)). Each
  feature keeps its name and strand (+, −, or unstranded). Overlapping features are merged, and a feature that
  contains an origin is split there into parts named `<name>.1`, `<name>.2`, …, since the model requires that no
  origin lies strictly inside a feature.

## Model

### Velocity
- **Replisome velocity:** v(x, t) = v(t) / g(x), with v(t) = v0 · h(t).
- **h(t):**
  - piecewise constant on a grid `h_tgrid`, which runs from 0 to `--vt-tmax` doubling times (default 3) with spacing
    timescale / `--vt-resolution`;
  - log h(t) has a zero-mean GP prior with covariance `--vt-covariance` (se, m0, m1 or m2; default m1, i.e.
    Matérn-3/2), length scale `--vt-timescale` (default 0.2 doubling times) and std. dev. `--vt-magnitude`
    (default 0.1).
- **v0:** has a flat prior on [`--v0-min`, `--v0-max`]. On its own it is not identifiable from h, so VeloGP reports
  instead v0_proper = the geometric mean of v(t) over [0, C].
- **g(x):** g = 1 everywhere except on the vx features, where g_i = 1 + Δg_i ≥ 1 (a slow-down). Each Δg_i has an
  exponential prior with mean `delayscale · genome_length / feature_length`. With this prior the expected delay
  at one feature is about `--vx-delayscale` (default 0.01) times the time a single replisome would need for the
  whole genome. Δg_i is capped at 20 × its prior mean.
- **Origins:** every origin fires deterministically after a delay Δ_o ≥ 0, which has no prior. The delays are shifted
  so that the smallest delay in the genome is exactly 0.

### Replication time and abundance
- **Virtual time:** U = H(t) = ∫₀ᵗ h. In virtual time, replisomes move at the constant speed v0 / g(x).
- **Replication time of x:**

      T(x) = min over the surrounding origins o of [ H(Δ_o) + ∫_o^x g(u) du / v0 ],

  taken over the nearest origin on either side, with wrap-around on circular chromosomes. Replisomes stop when they
  meet. An origin that is replicated passively before it fires never fires.
- **Physical time:** t(x) = H⁻¹(T(x)).
- **Predicted abundance:** P(x) = exp(−λ · t(x)) for an exponentially growing population with growth rate λ
  (`--growthrate`, per hour; default 1, stored internally per second). The model can't determine λ, which only sets
  the time scale: times, velocities and delays all scale with 1/λ.
- **Prediction:** y_pred = scale · P(x), with `scale = mean(y_obs) / mean(P)`.

### Likelihood, error model and fit
- **Loss:** ½ Σ (y_obs − y_pred)² / s_obs², plus the negative log-priors of h and Δg (v0 and the delays have none).
- **Standard errors s_obs:** equal to q without `--error-model`. With `--error-model`, s_obs² = q² + d · y_obs²,
  where d ≥ 0 is a genome-wide overdispersion estimated by quasi-likelihood from the scatter between adjacent windows.
  `--plot-error-model` plots this scatter and the fitted error model.
- **Fit:** a gradient-informed stochastic search over v0, the delays, h and Δg. Perturbations start at 1.0 and are
  reduced by `--perturbation-reduction` (default 0.5^(1/4)) after `--perturbation-rejects` (default 5) rejections,
  until they fall below `--perturbation-min` (default 0.001). LBFGS then runs to convergence. Use `--seed` for
  reproducibility.
- **Posterior:** a Laplace approximation with covariance (JᵀWJ + prior precision)⁻¹. It is computed separately for
  `v0_log` and jointly for h and all Δg; the delays are not included. The std. err. of v(t) is propagated from the
  joint h/Δg covariance and excludes v0, which is well determined once h is fixed.

### Derived quantities
- **Replication time C:** the completion time, i.e. the first time after which no replisome is active.
- **Feature passage times:** time_s = ∫ g / v(t(x)) dx and delay_s = ∫ (g − 1) / v(t(x)) dx over the feature,
  summed over the replisomes that replicate it.
- **p_block:** the probability that a permanent block of the first-arriving replisome would give the same abundance
  just past the feature as the deterministic delay:
  `p = (1 − e^{−λ(t_x − t_1)}) / (1 − e^{−λ(t_2 − t_1)})`.
  - t_1 is the exit time without delay, t_x the delayed exit time, and t_2 the entry time of the opposite replisome.
  - p = 1 means the replisomes meet inside the feature only because of the delay (p = 1 is then a lower bound).
  - p is NaN if the replisomes would meet inside the feature even without the delay.

## Output files

All tables are tab-separated with one header line (gzip/bz2 if the filename ends in `.gz`/`.bz2`).

| Option | Content | Format |
|---|---|---|
| `--out-abundance` | observed and predicted abundance of each window | [fitted_abundance_format.md](fitted_abundance_format.md) |
| `--out-vt` | v(t) and the number of active replisomes | [vt_format.md](vt_format.md) |
| `--out-trajectories` | v(t) and the position and velocity of every replisome over time | [trajectories_format.md](trajectories_format.md) |
| `--out-vx-feature-delays` | g, delays and passing replisomes of each vx feature | [vx_feature_delays_format.md](vx_feature_delays_format.md) |
| `--out-objects`, `--out-model` | serialized model, for VeloMix | [objects_format.md](objects_format.md) |

### Plots and console output
- `--plot-abundance`: data (grey) and fit (blue), one panel per chromosome.
- `--plot-vt`: v(t) ± 2 std. err., in units of doubling times. Dotted line: v0_proper. Dashed line: one doubling
  time. Solid line: the time at which the last window is replicated.
- `--plot-error-model`: the neighbour scatter and the fitted error model.
- The console log reports the settings, d and the fit diagnostics of the error model, the progress of the fit, C
  and v0_proper.

## Notes for analysis
- **Weighted SSR:** compute it from the abundance table as Σ ((y_obs − y_pred) / y_obs_stderr)². With `--error-model`, SSR / (n − k_eff) should be close to 1, where k_eff is
  the trace of the hat matrix; values well above 1 indicate model misfit.
- **Delays at features:** compare features with `delay_s` (or `delay_rel`), not with `g`. g depends on the feature
  length, because a short feature needs a large g for the same delay.
- **Feature direction:** `direction` and the `origin_*` columns give the direction in which a feature is actually
  traversed. Ter sites act only when they are entered head-on (`-`).
