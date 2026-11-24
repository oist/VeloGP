# Feature delay file format

Produced by the `--out-vx-feature-delays` option of VeloGP (`python -m velogp`)
and VeloMix (`python -m velogp.velomix`). In
[MFASeqPipeline](https://github.com/oist/MFASeqPipeline/blob/master/docs/mfa_pipeline.md#velogp)
these are `velogp_vx_file` (`{sample}.velogp.vx.gz`) and `velomix_vx_file`
(`{sample}.velomix.vx.gz`). The file gives, for each vx feature (a region where
replisomes may slow down, selected with `--vx-feature-names` and
`--vx-feature-types`, see [features_format.md](features_format.md)), the fitted
slow-down, the resulting delay, and the replisomes that pass it.

* Tab-separated text, gzip- or bz2-compressed if the file name ends in `.gz` or
  `.bz2`.
* The first line is a header with the column names below.
* One row per vx feature, ordered by chromosome (in the order of the genome
  FASTA file) and position. Overlapping features are merged into one, named by
  joining the names with `,`. A feature that contains an origin is split there
  into parts named `<name>.1`, `<name>.2`, ….
* Coordinates are **0-based**, and `begin`/`end` are **half-open** `[begin, end)`
  as in BED. Times are in **seconds**.
* Strands and directions are written as `+`, `-` and `*` (unstranded).
* For VeloMix, the stall sites are left out, and all values refer to the
  *reference component*, in which every origin fires and no site stalls.

## Columns

| # | Column | Type | Description |
|---|---|---|---|
| 1 | `chr` | string | Chromosome name |
| 2 | `begin` | int | First base of the feature (0-based) |
| 3 | `end` | int | One past the last base of the feature |
| 4 | `strand` | `+`/`-`/`*` | Strand of the feature |
| 5 | `length` | int | `end - begin` |
| 6 | `name` | string | Feature name |
| 7 | `delayscale` | float | Prior scale `--vx-delayscale`: the expected delay as a fraction of the time a single replisome would need for the whole genome |
| 8 | `g` | float | Fitted slow-down factor g ≥ 1: replisomes move at v(t) / g inside the feature |
| 9 | `delay_bp` | float | `(g − 1) · length`: the extra distance, in bp at speed v(t), that the slow-down corresponds to |
| 10 | `delay_rel` | float | `delay_s` / C, where C is the replication time |
| 11 | `p_block` | float | Probability with which a permanent block of the first-arriving replisome would give the same abundance just past the feature as the fitted delay. 1 if the replisomes meet inside the feature only because of the delay (then a lower bound); `nan` if they would meet inside the feature even without delay |
| 12 | `time_s` | float | Time replisomes spend passing the feature, `∫ g / v(t(x)) dx` over the feature [s] |
| 13 | `delay_s` | float | Part of `time_s` caused by the slow-down, `∫ (g − 1) / v(t(x)) dx` [s] |
| 14 | `direction` | `+`/`-`/`*` | Direction in which the first replisome enters the feature, relative to its strand: `+` co-directional, `-` head-on, `*` unstranded feature or both replisomes enter simultaneously |
| 15 | `origin_forward` | string | Origin of the replisome entering the feature from the left, moving towards higher coordinates; `NA` if there is none or it never fires |
| 16 | `origin_forward_enter` | float | Time at which that replisome enters the feature [s] |
| 17 | `origin_forward_exit` | float | Time at which that replisome leaves the feature [s] |
| 18 | `origin_reverse` | string | Origin of the replisome entering the feature from the right, moving towards lower coordinates; `NA` if there is none or it never fires |
| 19 | `origin_reverse_enter` | float | Time at which that replisome enters the feature [s] |
| 20 | `origin_reverse_exit` | float | Time at which that replisome leaves the feature [s] |

The entry and exit times are actual for the replisome that replicates the
feature. For the other replisome they are counterfactual: when it would have
arrived had it not stopped at the meeting point, assuming g = 1 beyond it. They
are `nan` if the corresponding origin is `NA`.

To compare features, use `delay_s` or `delay_rel` rather than `g`, since a short
feature needs a large g for the same delay.

## Example

```
chr	begin	end	strand	length	name	delayscale	g	delay_bp	delay_rel	p_block	time_s	delay_s	direction	origin_forward	origin_forward_enter	origin_forward_exit	origin_reverse	origin_reverse_enter	origin_reverse_exit
U00096.3	223770	225312	+	1542	rna-b0201	0.01	9.58340	13235.60502	0.00480	0.00603261	59.50564	53.29640	+	oriC	6045.06423	6104.56988	oriX	17664.20735	17673.33783
U00096.3	225758	228662	+	2904	rna-b0204	0.01	2.00312	2913.07071	0.00105	0.00133531	23.40078	11.71864	+	oriC	6106.36475	6129.76554	oriX	17644.37220	17661.56649
```
