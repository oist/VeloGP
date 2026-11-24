# VeloMix: mixture model

VeloMix (`python -m velogp.velomix`) takes a finished VeloGP fit and refits it as a mixture of stochastic origin
firing and replisome stalling.

The deterministic VeloGP model is described in [velogp.md](velogp.md). Usage examples are in the
[README](../README.md#fitting-a-velomix-model), and all file formats are listed in [the index](README.md).

## Model definition

### Starting point
VeloMix reads the file written by VeloGP's `--out-objects`, which holds the fitted VeloGP model and the
observations; no other input files are needed. The observations (windows, y_obs, s_obs) are reused unchanged.

### Deterministic parts (shared by all components)
- **v(t)** = v0 · h(t), with the same GP prior as VeloGP, refitted.
- **g at every vx feature (g range) that is not a stall site**, refitted.
- **Chromosome delay D_c:** each chromosome has a single firing time D_c shared by all of its origins. It is fitted
  only when the genome has more than one chromosome. Its minimum over the chromosomes is clamped to 0.

### Stochastic events (independent Bernoulli variables)
1. **Origin firing.**
   - On each chromosome, every origin whose VeloGP delay equals that chromosome's smallest delay is *fixed* and always
     fires (usually the origin with delay 0, e.g. oriC).
   - Every other origin fires with probability `p_fire`. If it fires, it fires at D_c with no delay of its own. If it
     doesn't fire, it is absent and its region is replicated passively.
   - VeloGP's per-origin delays are therefore replaced by firing probabilities.
2. **Stalling at stall sites.**
   - The stall sites are given on the CLI as `--stall-sites name[:polarity],...` and must be names of vx features of
     the VeloGP fit.
   - A site stalls with probability `p_stall`, and then blocks replisomes indefinitely in the directions given by its
     polarity:
     - `both` (the default): both directions;
     - `co`: replisomes moving along the feature's strand;
     - `head`: replisomes moving against the feature's strand (the right choice for Tus/ter).
   - Replisomes moving in the other direction pass normally. Stall sites have no delay otherwise: g is fixed to 1.
   - In a component where a site stalls, a location is replicated by whichever replisome reaches it without crossing a
     blocked site. A location no replisome can reach has abundance 0, which is relevant only for windows strictly
     inside a `both` site; VeloMix warns about those.

### Event order and component labels
- **Event order** is fixed: first all non-fixed origins, sorted by (chromosome name, position); then all stall sites,
  sorted by (chromosome name, start).
- **Components:** there are 2^n components for n events. Each is labelled by an n-character bit string in event
  order: for an origin, `1` means it fires; for a stall site, `1` means it stalls. Example with events (oriZ, terB,
  terC): `111` = oriZ fires and terB and terC both stall.
- **Weights:** w_k = ∏ q_i^{b_ki} (1 − q_i)^{1 − b_ki}, where q is (p_fire, p_stall) in event order. The weights sum
  to 1.
- **Reference component:** the one in which every origin fires and nothing stalls (all origin bits 1, all stall bits
  0).

### Abundance and likelihood
- **Component abundance:** each component's abundance P_k(x) is computed by the unchanged VeloGP code:
  P_k(x) = exp(−λ · H⁻¹(T_k(x))), with a VeloGP model containing only that component's firing origins and with
  directional blocks at its stalling sites.
- **Mixture:** P(x) = Σ_k w_k P_k(x).
- **Prediction:** y_pred = scale · P(x), where `scale = mean(y_obs) / mean(P)`. This single scale applies to the
  mixture, not per component.
- **Loss:** ½ Σ (y_obs − y_pred)² / s_obs², plus the VeloGP priors on v0, h and g. The probabilities and D_c have
  no prior.
- **Error model:** `s_obs` is taken unchanged from the VeloGP fit, so VeloMix inherits whatever error model the VeloGP
  run used. The VeloGP runs for analysis use `--error-model`, in which case
  `s_obs² = q² + d · y_obs²`. Here q is the Poisson std. error from the abundance file, and d is a genome-wide
  overdispersion estimated from the scatter between adjacent windows. VeloMix doesn't
  re-estimate d.

### Fit
- **Starting values** (all parameters start from the VeloGP fit):
  - p_fire = exp(−λ · (delay − min delay));
  - p_stall = the VeloGP blocking probability that is equivalent to the site's delay, or 0.1 if that is undefined.
- **Procedure:** a stochastic search as in VeloGP, but with perturbations starting at 0.25 (default reduction
  0.5^(1/4) after 5 rejects, stopping below 0.001), followed by LBFGS to convergence.

### Standard errors
- **Method:** a Laplace approximation, with posterior covariance (JᵀWJ + prior precision)⁻¹.
- **Covariance over h, g and the events:** computed jointly, conditioned on the fixed entries. Fixed entries are
  stall-site g values and probabilities clamped at exactly 0 or 1; they have a NaN stderr.
- **Uninformative probabilities** (no effect on the prediction) have an infinite stderr.
- **v0** has its own covariance.

## Output files

All tables are tab-separated with one header line (gzip/bz2 if the filename ends in `.gz`/`.bz2`). Except for the
probabilities and the abundances, the outputs describe the *reference component* (all origins fire, nothing stalls).

| Option | Content | Format |
|---|---|---|
| `--out-probabilities` | firing probability of each origin and stalling probability of each stall site | [probabilities_format.md](probabilities_format.md) |
| `--out-abundance` | observed abundance, mixture prediction, and prediction and weight of each component | [fitted_abundance_format.md](fitted_abundance_format.md#velomix) |
| `--out-vt` | v(t) and the number of active replisomes | [vt_format.md](vt_format.md) |
| `--out-trajectories` | v(t) and the position and velocity of every replisome over time | [trajectories_format.md](trajectories_format.md) |
| `--out-vx-feature-delays` | g, delays and passing replisomes of each vx feature that is not a stall site | [vx_feature_delays_format.md](vx_feature_delays_format.md) |
| `--out-objects`, `--out-model` | serialized VeloGP and VeloMix objects | [objects_format.md](objects_format.md) |

### Plots and console output
- `--plot-abundance`: the data (grey), the mixture fit (blue) and the VeloGP fit (orange, dashed), one panel per
  chromosome.
- `--plot-vt`: v(t) ± 2 std. err.
- The console log reports the loss and the weighted SSR for both VeloMix and VeloGP, and each probability with its
  std. err.

## Notes for analysis
- **Comparing with VeloGP:** compare the weighted SSRs (both runs share the same `s_obs`), not the losses, whose prior
  terms differ.
  - With `--error-model`, the weights already include the overdispersion, so φ = SSR / (n − k_eff) should be close
    to 1. k_eff is the trace of the hat matrix.
  - If φ ≈ 1, ΔSSR / 2 is the log-likelihood ratio between the two fits.
  - If φ is clearly above 1 (model misfit, which the neighbour-scatter estimate doesn't capture), use ΔSSR / φ
    instead.
  - Report φ either way.
  - Without `--error-model` (Poisson errors only), φ was ~5 at 10 kb resolution.
- **Conditional probabilities:** p_stall for a site is conditional on the site being reached first by a replisome of
  the blocked direction. If an opposite fork always gets there first, p_stall is uninformative.
- **The reference component** is not the most probable component in general. For the per-window composition, use the
  `w.*` and `y.*` columns.
