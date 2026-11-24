# Probability file format

Produced by the `--out-probabilities` option of VeloMix
(`python -m velogp.velomix`). In
[MFASeqPipeline](https://github.com/oist/MFASeqPipeline/blob/master/docs/mfa_pipeline.md#velomix)
this is `velomix_probabilities_file` (`{sample}.velomix.p.gz`). The file gives
the fitted firing probability of each origin and the stalling probability of
each stall site (see [velomix.md](velomix.md)).

* Tab-separated text, gzip- or bz2-compressed if the file name ends in `.gz` or
  `.bz2`.
* The first line is a header with the column names below.
* First one row per origin, including the fixed origins that always fire, then
  one row per stall site. Each part is ordered by chromosome name and position,
  as the events that label the mixture components (the fixed origins are not
  events and have no bit in the labels).
* Coordinates are **0-based**, and `begin`/`end` are **half-open** `[begin, end)`
  as in BED.

## Columns

| # | Column | Type | Description |
|---|---|---|---|
| 1 | `type` | `origin`/`stall` | Kind of event |
| 2 | `chr` | string | Chromosome name |
| 3 | `begin` | int | Origin: its position. Stall site: first base of the feature |
| 4 | `end` | int | Origin: `begin + 1`. Stall site: one past the last base of the feature |
| 5 | `strand` | `+`/`-`/`*` | Origins: `*`. Stall sites: strand of the feature |
| 6 | `name` | string | Origin or feature name |
| 7 | `polarity` | string | Origins: `NA`. Stall sites: the replisomes the site blocks when it stalls, `both`, `co` (moving along the feature's strand) or `head` (moving against it) |
| 8 | `probability` | float | Origins: firing probability p_fire (1 for fixed origins). Stall sites: stalling probability p_stall |
| 9 | `stderr` | float | Posterior standard error of `probability`. `nan` for fixed origins and probabilities at exactly 0 or 1; `inf` if the probability has no effect on the abundances |
| 10 | `fixed` | 0/1 | 1 for origins that always fire (on each chromosome, the origin(s) with the smallest VeloGP delay), else 0 |

p_stall is conditional on the site being reached first by a replisome of the
blocked direction. A large `stderr` indicates that the data contain little
information about the probability, e.g. because the opposite replisome usually
arrives first.

## Example

```
type	chr	begin	end	strand	name	polarity	probability	stderr	fixed
origin	U00096.3	2738124	2738125	*	oriX	NA	0.971834	0.006359	0
origin	U00096.3	3925859	3925860	*	oriC	NA	1.000000	nan	1
stall	U00096.3	1279526	1279549	+	terD	head	0.544771	26.388600	0
stall	U00096.3	1341744	1341767	+	terA	head	0.997577	0.094719	0
```
