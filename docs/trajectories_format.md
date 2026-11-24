# Trajectory file format

Produced by the `--out-trajectories` option of VeloGP (`python -m velogp`) and
VeloMix (`python -m velogp.velomix`). In
[MFASeqPipeline](https://github.com/oist/MFASeqPipeline/blob/master/docs/mfa_pipeline.md#velogp)
these are `velogp_vt_file` (`{sample}.velogp.vt.gz`) and `velomix_vt_file`
(`{sample}.velomix.vt.gz`); the pipeline names them `.vt` because they contain
the fork velocity v(t). The file gives the fitted fork velocity over time and the
state, position and velocity of every replisome over time. The simpler
[velocity file](vt_format.md) holds only v(t).

* Tab-separated text, gzip- or bz2-compressed if the file name ends in `.gz` or
  `.bz2`.
* The first line is a header with the column names below.
* One row per point of the time grid of h(t), from 0 to `--vt-tmax` doubling
  times (default 3) in steps of `--vt-timescale` / `--vt-resolution` doubling
  times (by default 0.002 doubling times). v(t) is constant between grid points.
* Time 0 is when the first origin fires. Times are in **seconds**, velocities in
  **bp/s**, and positions are **0-based**.
* For VeloMix, all columns refer to the *reference component*, in which every
  origin fires (without delay) and no stall site stalls.

## Columns

| # | Column | Type | Description |
|---|---|---|---|
| 1 | `t` | float | Time since the first origin fired [s] |
| 2 | `v0` | float | Baseline velocity v0: the geometric mean of v(t) over [0, C] [bp/s]; the same in all rows |
| 3 | `vt` | float | Fork velocity v(t) [bp/s], i.e. the velocity outside of vx features |
| 4 | `vt_stderr` | float | Posterior standard error of v(t), given v0 |
| 5 | `replicationtime` | float | Replication time C: the time after which no replisome is active [s]; the same in all rows |

These are followed by five groups of columns with one column per replisome. Each
origin has two replisomes, `<origin>-` moving towards lower and `<origin>+`
towards higher coordinates. Within each group, the origins are ordered by
chromosome (in the order of the genome FASTA file) and position, with `-`
before `+`. With origins oriX and oriC, each group thus has the columns
`oriX-`, `oriX+`, `oriC-`, `oriC+`.

| Group | Column | Type | Description |
|---|---|---|---|
| 1 | `<origin><d>` | 0/1/2 | State of the replisome: 0 not started yet, 1 active, 2 finished (met another replisome or the chromosome end) |
| 2 | `x.<origin><d>` | float | Position of the replisome. Before it starts, the origin's position; after it finishes, where it stopped |
| 3 | `dx.<origin><d>` | float | Signed distance travelled since the origin fired (negative for `-`) |
| 4 | `v.<origin><d>` | float | Current velocity v(t) / g(x) of the replisome; `nan` unless active |
| 5 | `gx.<origin><d>` | float | Slow-down factor g at the replisome's current position (1 outside of vx features) |

Origin names are those given with `--origins`, or `<type>1`, `<type>2`, … for
origins found by GFF feature type.

## Origin delays

The firing delay of an origin is the first `t` at which its replisomes are in
state 1 (both replisomes of an origin start together), accurate to the grid
spacing. An origin whose replisomes stay in state 0 throughout never fires,
because it is replicated passively first.

## Example

The first columns (groups 1, 2 and 5 abbreviated) of a fit with origins oriX and
oriC, in which oriX fires later than oriC:

```
t	v0	vt	vt_stderr	replicationtime	oriX-	oriX+	oriC-	oriC+	x.oriX-	x.oriX+	x.oriC-	x.oriC+	dx.oriX-	...
0.00000	180.00869	134.73442	9.93724	11108.22204	0	0	1	1	2738124.00000	2738124.00000	3925859.00000	3925859.00000	0.00000	...
27.88078	180.00869	134.28292	9.61492	11108.22204	0	0	1	1	2738124.00000	2738124.00000	3922102.49909	3929615.50091	0.00000	...
```
