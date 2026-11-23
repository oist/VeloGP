# VeloGP: deterministic model and output formats

VeloGP (`python -m velogp`) fits a deterministic model of replisome movement to marker-frequency (abundance) data.
- The stochastic extension VeloMix is described in `mixture.md`.

## Input
- **Abundance table:** chosen with `--columns`. The default columns are `seq,midpoint,coverage_ratio,coverage_ratio_stderr,ratio_mappable`, i.e.:
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
  `--vx-feature-types`, resolved in `--gff` / `--bed` files. Each feature keeps its name and strand (+, −, or
  unstranded). No origin may lie strictly inside a feature.

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

All tables are tab-separated with one header line (gzip/bz2 if the filename ends in `.gz`/`.bz2`). Strands are
written as `+`, `-` and `*`. Feature coordinates are 0-based half-open [begin, end) as in BED; window positions are
as in the input.

### `--out-abundance`
One row per retained window.

| column | meaning |
|---|---|
| `chr`, `pos` | window chromosome and position |
| `y_obs` | observed abundance |
| `y_obs_stderr` | s_obs (includes the overdispersion term with `--error-model`) |
| `y_pred` | scale · P(x) |
| `p_pred` | unscaled P(x) = exp(−λ t(x)) |

### `--out-vt`
One row per point of `h_tgrid`.

| column | meaning |
|---|---|
| `t` | physical time [s] |
| `vt`, `vt_stderr` | v(t) [bp/s] and its posterior std. err. |
| `nforks` | number of active replisomes at t |
| `replicationtime` | C [s] (constant) |

### `--out-vx-feature-delays`
One row per vx feature, in model order (by chromosome, then position).

| column | meaning |
|---|---|
| `chr`, `begin`, `end`, `strand`, `length`, `name` | the feature |
| `delayscale` | prior scale of the feature |
| `g` | fitted slow-down factor (≥ 1) |
| `delay_bp` | (g − 1) · length: the extra distance in bp at speed v |
| `time_s`, `delay_s` | passage time and the part caused by g > 1 [s] |
| `delay_rel` | `delay_s` / C |
| `p_block` | equivalent blocking probability (see above) |
| `direction` | traversal direction of the first-entering replisome × feature strand: `+` co-directional, `-` head-on, `*` unstranded or simultaneous |
| `origin_forward`, `origin_reverse` | origin of the rightward (+) replisome from the left, and of the leftward (−) replisome from the right; `NA` if missing or never fires |
| `origin_*_enter`, `origin_*_exit` | physical entry and exit times [s] |

For the replisome that replicates the feature, the entry and exit times are actual. For the other replisome they are
counterfactual: where it would have arrived, assuming g = 1 beyond the meeting point. NaN if that replisome is missing.

### `--out-trajectories`
One row per point of `h_tgrid`.

| column | meaning |
|---|---|
| `t` | physical time [s] |
| `v0` | v0_proper [bp/s] (constant) |
| `vt`, `vt_stderr` | v(t) and its std. err. |
| `replicationtime` | C [s] (constant) |
| `<origin><d>` | state of each replisome: 0 not started, 1 active, 2 finished |
| `x.<origin><d>` | position |
| `dx.<origin><d>` | signed distance travelled |
| `v.<origin><d>` | current velocity v(t) / g(x); NaN unless active |
| `gx.<origin><d>` | g at the current position |

There is one set of the per-replisome columns for each origin (in model order: chromosomes in genome order, origins
by position) and each direction `d` (`-` then `+`).

**Origin delays:** the firing delay of an origin is the first `t` at which its replisomes are in state 1 (both
directions start together). The result is accurate to the grid spacing, timescale / `--vt-resolution` (by default
0.002 doubling times). A replisome that stays in state 0 throughout never fires: its origin is replicated passively
first.

### `--out-objects`, `--out-model`
Serialized internal objects (the fitted model, the observations and the posterior covariances), meant for re-loading
by VeloGP tools such as VeloMix. They are not meant for analysis; use the tables above.

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
