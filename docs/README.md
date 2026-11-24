# VeloGP documentation

See the [main README](../README.md) for an overview, installation and usage
examples.

## Models

* [velogp.md](velogp.md): the deterministic VeloGP model, its options, and notes
  for analysis (`python -m velogp`).
* [velomix.md](velomix.md): the VeloMix mixture model of stochastic origin firing
  and replisome stalling (`python -m velogp.velomix`).

## File formats

### Input

* [Abundance file](https://github.com/oist/MFASeqPipeline/blob/master/docs/abundance_format.md)
  (MFASeqPipeline): the abundance profile fitted by VeloGP
* [Feature files](features_format.md): origins and vx features (`--gff`, `--bed`)
* [Growth rate file](https://github.com/oist/MFASeqPipeline/blob/master/docs/growthrates_format.md)
  (MFASeqPipeline): growth rates passed to `--growthrate` by the pipeline
* [Objects file](objects_format.md): a VeloGP fit, the input of VeloMix

### Output

| Format | Option | MFASeqPipeline file |
|---|---|---|
| [Fitted abundance file](fitted_abundance_format.md) | `--out-abundance` | `{sample}.velogp.abd.gz`, `{sample}.velomix.abd.gz` |
| [Trajectory file](trajectories_format.md) | `--out-trajectories` | `{sample}.velogp.vt.gz`, `{sample}.velomix.vt.gz` |
| [Velocity file](vt_format.md) | `--out-vt` | – |
| [Feature delay file](vx_feature_delays_format.md) | `--out-vx-feature-delays` | `{sample}.velogp.vx.gz`, `{sample}.velomix.vx.gz` |
| [Probability file](probabilities_format.md) | `--out-probabilities` (VeloMix only) | `{sample}.velomix.p.gz` |
| [Objects file](objects_format.md) | `--out-objects`, `--out-model` | `{sample}.velogp.pkl`, `{sample}.velomix.pkl` |

Plots are described in [velogp.md](velogp.md#plots-and-console-output) and
[velomix.md](velomix.md#plots-and-console-output).
