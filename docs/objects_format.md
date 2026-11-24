# Objects file format

Produced by the `--out-objects` and `--out-model` options of VeloGP
(`python -m velogp`) and VeloMix (`python -m velogp.velomix`). In
[MFASeqPipeline](https://github.com/oist/MFASeqPipeline/blob/master/docs/mfa_pipeline.md#velogp)
these are `velogp_objects_file` (`{sample}.velogp.pkl`) and
`velomix_objects_file` (`{sample}.velomix.pkl`). The objects file of a VeloGP run
is the input of VeloMix.

These files hold VeloGP's internal Python objects. They are meant for VeloGP
tools such as VeloMix, and for custom analyses in Python that need more than the
tables. For everything else, use the tables described in the [index](README.md).

* `--out-objects`: a Python [pickle](https://docs.python.org/3/library/pickle.html)
  of a dictionary, gzip- or bz2-compressed if the file name ends in `.gz` or
  `.bz2`.
* `--out-model`: the fitted model alone, written with `torch.save`.
* Reading either file requires the `velogp` package to be importable (e.g. via
  `PYTHONPATH`). Files written by older versions of VeloGP can be read by newer
  versions, but not vice versa.
* Pickles can execute code when loaded; only load files from trusted sources.

## Contents

The dictionary written by VeloGP's `--out-objects`:

| Key | Type | Description |
|---|---|---|
| `model` | `velogp.VelocityModel` | The fitted model: v0 and h(t), and for each chromosome its origins, origin delays and vx features with their g |
| `loss` | `velogp.VelocityModelLoss` | The loss function, which holds the observations (chromosome, position, y_obs and s_obs of each window used) and the model |
| `algorithm` | `velogp.FitVelocityModel` | The fit algorithm in its final state |
| `cov_v0` | tensor | Posterior covariance of `temporal.v0_log` |
| `cov_v0.pmap` | dict | Maps the name of each parameter to the index of its first entry in `cov_v0` |
| `cov_hg` | tensor | Joint posterior covariance of h(t) and of the g of all vx features |
| `cov_hg.pmap` | dict | Maps the name of each parameter to the index of its first entry in `cov_hg` |

VeloMix's `--out-objects` writes the dictionary of the VeloGP run it started
from, with one more key, `mixture`, holding a dictionary with the VeloMix fit:

| Key | Type | Description |
|---|---|---|
| `model` | `velogp.velomix.MixtureModel` | The fitted mixture model |
| `loss` | | The mixture loss function |
| `algorithm` | `velogp.velomix.FitMixtureModel` | The fit algorithm in its final state |
| `cov_v0`, `cov_v0.pmap` | tensor, dict | Posterior covariance of v0, as for VeloGP |
| `cov_hg`, `cov_hg.pmap` | tensor, dict | Joint posterior covariance of h(t), g, the probabilities and the chromosome delays |
| `events` | list of str | Names of the stochastic events, in the order of the bits of the component labels |
| `components` | list of str | Labels of the components (see [velomix.md](velomix.md#event-order-and-component-labels)) |

## Example

```python
import pickle
with open("exp1.velogp.pkl", "rb") as f:
    objects = pickle.load(f)
model, loss = objects["model"], objects["loss"]
print(model.completion_time())                  # replication time C [s]
print([c.origin_names for c in model.chromosomes])
```
