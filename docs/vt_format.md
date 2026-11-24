# Velocity file format

Produced by the `--out-vt` option of VeloGP (`python -m velogp`) and VeloMix
(`python -m velogp.velomix`). MFASeqPipeline does not write this file; its
`{sample}.velogp.vt.gz` and `{sample}.velomix.vt.gz` files are
[trajectory files](trajectories_format.md), which contain the same velocity
columns plus the state and position of every replisome. The velocity file
gives the fitted fork velocity v(t) over time and the number of active
replisomes.

* Tab-separated text, gzip- or bz2-compressed if the file name ends in `.gz` or
  `.bz2`.
* The first line is a header with the column names below.
* One row per point of the time grid of h(t), from 0 to `--vt-tmax` doubling
  times (default 3) in steps of `--vt-timescale` / `--vt-resolution` doubling
  times (by default 0.002 doubling times). v(t) is constant between grid points.
* Time 0 is when the first origin fires. Times are in **seconds**, velocities in
  **bp/s**.
* For VeloMix, `nforks` and `replicationtime` refer to the *reference
  component*, in which every origin fires (without delay) and no stall site
  stalls.

## Columns

| # | Column | Type | Description |
|---|---|---|---|
| 1 | `t` | float | Time since the first origin fired [s] |
| 2 | `vt` | float | Fork velocity v(t) [bp/s], i.e. the velocity outside of vx features |
| 3 | `vt_stderr` | float | Posterior standard error of v(t), given v0 |
| 4 | `nforks` | int | Number of active replisomes at time t |
| 5 | `replicationtime` | float | Replication time C: the time after which no replisome is active [s]; the same in all rows |

The grid extends beyond C. There, v(t) is not determined by the data, which
shows as a large `vt_stderr`.

## Example

Rows of a fit with two origins, where the second origin fires at about 100 s
and two replisomes have finished at about 970 s:

```
t	vt	vt_stderr	nforks	replicationtime
0.00000	490.73926	24.43373	2	3134.33072
36.00765	476.94077	21.58083	2	3134.33072
...
108.02294	451.32840	18.06403	4	3134.33072
...
972.20644	822.81868	23.23416	2	3134.33072
```
