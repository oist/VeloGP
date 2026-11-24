# Feature file format

An **input** of VeloGP. Feature files tell VeloGP where the replication origins
and the vx features (the regions where replisomes may slow down) lie. They are
given with `--gff` (GFF3) and `--bed` (BED); both options can be repeated, and
the files may be gzip- or bz2-compressed.

In [MFASeqPipeline](https://github.com/oist/MFASeqPipeline/blob/master/docs/mfa_pipeline.md#velogp)
the `velogp` step passes the genome annotation `gff_file` (`{base_dir}/genome.gff`)
with `--gff` and `origin_file` (`{base_dir}/origins.bed`) with `--bed`.

Features are looked up by:

* **name** (`--origins`, `--vx-feature-names`): GFF and BED. A name must match
  exactly one feature in all files together.
* **type** (`--origins`, `--vx-feature-types`): GFF only. A type matches every
  feature of that type. Origins found by type are named `<type>1`, …,
  `<type>N`, ordered by chromosome name and position.

Chromosome names must match the sequence names of the `--genome` FASTA file.
Features may lie on any chromosome, but every chromosome needs at least one
origin.

## GFF3

A [GFF3](https://github.com/The-Sequence-Ontology/Specifications/blob/master/gff3.md)
genome annotation, as provided by NCBI (e.g. from the
[refseq_pipeline](https://github.com/oist/MFASeqPipeline/blob/master/docs/refseq_pipeline.md)).
Sub-features (e.g. the exons of a gene) are included.

| GFF column / attribute | Use |
|---|---|
| `seqid` (1) | Chromosome |
| `type` (3) | Feature type, matched by `--origins` and `--vx-feature-types` (e.g. `origin_of_replication`, `rRNA`) |
| `start`, `end` (4, 5) | Location (1-based, inclusive, as always in GFF) |
| `strand` (7) | Strand of vx features (`+`, `-`; anything else is unstranded) |
| `ID`, `Name` attributes | Feature names, matched by `--origins` and `--vx-feature-names` |

## BED

A [BED](https://samtools.github.io/hts-specs/BEDv1.pdf) file with at least four
tab-separated columns. Coordinates are **0-based**, and `end` is
**exclusive**. Blank lines and lines starting with `#`, `track` or `browser` are
skipped. Features in BED files can only be found by name.

| # | Column | Type | Description |
|---|---|---|---|
| 1 | `chrom` | string | Chromosome name, as in the genome |
| 2 | `start` | int | First base of the feature (0-based) |
| 3 | `end` | int | One past the last base of the feature |
| 4 | `name` | string | Feature name, matched by `--origins` and `--vx-feature-names` |
| 5 | `score` | | Ignored |
| 6 | `strand` | `+`, `-`, `.` | Optional. Strand of vx features; `.` or absent means unstranded |

Further columns are ignored.

## How features are used

* **Origins** are placed at the midpoint `(start + end) / 2` (0-based) of their
  feature. The feature's strand is ignored.
* **vx features** cover their whole range. The strand determines whether a
  replisome passes it co-directionally or head-on (see the `direction` column of
  the [feature delay file](vx_feature_delays_format.md)) and which replisomes a
  VeloMix stall site with polarity `co` or `head` blocks.

## Example

An `origins.bed` for *E. coli* that holds the origins and the *ter* sites:

```
U00096.3	345086	345247	oriZ	*	.
U00096.3	1279526	1279549	terD	*	+
U00096.3	1341744	1341767	terA	*	+
U00096.3	1609156	1609179	terC	*	-
U00096.3	1684225	1684248	terB	*	-
U00096.3	2737912	2738336	oriX	*	.
U00096.3	3925743	3925975	oriC	*	.
```

With `--bed origins.bed --origins oriC,oriX --vx-feature-names terA,terB,terC,terD`,
oriC is placed at position 3925859 (0-based) and oriX at 2738124.
