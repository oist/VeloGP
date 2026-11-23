#!/usr/bin/env python3
"""
VeloMix: re-fits a VeloGP fit as a mixture model of stochastic origin firing and replisome stalling

See `velogp.velomix.model` for the model. VeloMix reads the objects written by VeloGP's --out-objects, which contain
the fitted model and the observations, and needs no other input files.
"""
from typing import List, Tuple

import io
import sys
import math
import argparse
import pickle
import gzip
import bz2
import random
import numpy as np
import torch as pt
import matplotlib.pyplot as pp

from .. import model
from .model import MixtureModel, POLARITIES
from .fit import FitMixtureModel, make_loss, posterior_cov


class ArgumentError(Exception):
    pass


# Report invalid arguments (e.g. stall sites that cannot be resolved) like argparse does, i.e. as an error message
# without a traceback
def excepthook(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, ArgumentError):
        print(f"VeloMix: error: {exc_value}", file=sys.stderr)
        sys.exit(2)
    sys.__excepthook__(exc_type, exc_value, exc_traceback)
sys.excepthook = excepthook


def open_file(filename, mode):
    if filename.endswith(".gz"):
        return gzip.open(filename, mode)
    elif filename.endswith(".bz2"):
        return bz2.open(filename, mode)
    else:
        return open(filename, mode)


def write_table(filename: str, cols: list, fmt: tuple, tab: np.ndarray):
    """ Writes a structured array as a tab-separated table with a header line """
    with open_file(filename, "wt+") as f:
        f.write("\t".join([name for (name, type) in cols]) + "\n")
        np.savetxt(f, tab, delimiter="\t", fmt=fmt)


def parse_stall_sites(values: List[str]) -> List[Tuple[str, str]]:
    """ Parses stall sites specified as name[:polarity], possibly several separated by commas """
    sites = []
    for vs in values:
        for v in vs.split(","):
            if len(v) == 0:
                continue
            name, sep, polarity = v.rpartition(":")
            if sep and (polarity in POLARITIES):
                sites.append((name, polarity))
            elif sep and (polarity not in POLARITIES) and (polarity != ""):
                raise ArgumentError(f"invalid polarity {polarity} of stall site {name}, "
                                    f"must be one of {', '.join(POLARITIES)}")
            else:
                sites.append((v.rstrip(":"), "both"))
    return sites


# Symbols representing strands and directions in output files, as in VeloGP
STRAND_SYMBOLS = { 1: "+", -1: "-", 0: "*" }

# Make output unbuffered
sys.stdout = io.TextIOWrapper(open(sys.stdout.fileno(), 'wb', 0), write_through=True)

# Argument handling
parser = argparse.ArgumentParser(
    prog="VeloMix",
    description="Re-fit a VeloGP fit as a mixture model of stochastic origin firing and replisome stalling. On each "
                "chromosome, the origins with the smallest delay in the VeloGP fit always fire, all other origins "
                "fire with a fitted probability. The specified stall sites stall replisomes indefinitely with a "
                "fitted probability. v(t), g(x) at the other vx features and (for genomes with several "
                "chromosomes) the firing time of each chromosome's origins are re-fitted as well.")
parser.add_argument("objects",
                    help="objects written by VeloGP's --out-objects")
parser.add_argument("--stall-sites", default=[], metavar="feature[:polarity],...", action="append",
                    help="vx features of the VeloGP fit at which replisomes may stall. The polarity determines which "
                         "replisomes are stalled: 'both' (default) stalls replisomes moving in either direction, "
                         "'co' replisomes moving along the feature's strand, 'head' replisomes moving against it")
parser.add_argument("--out-objects", metavar="filename",
                    help="output file for the objects of the VeloGP fit plus the VeloMix loss function, fit "
                         "algorithm, fitted model and posterior covariances (under the key 'mixture')")
parser.add_argument("--out-model", metavar="filename",
                    help="output file for the fitted mixture model")
parser.add_argument("--out-probabilities", metavar="filename",
                    help="output file for the firing probability of each origin and the stalling probability of "
                         "each stall site")
parser.add_argument("--out-abundance", metavar="filename",
                    help="output file for the observed abundances, the abundances predicted by the mixture and by "
                         "each of its components, and the weights of the components")
parser.add_argument("--out-vt", metavar="filename",
                    help="output file for the velocity as a function of time as a tab-separated table")
parser.add_argument("--out-vx-feature-delays", metavar="filename",
                    help="output file for the delays incurred at each vx feature which is not a stall site, in the "
                         "component in which all origins fire and no site stalls")
parser.add_argument("--out-trajectories", metavar="filename",
                    help="output file for the trajectories of the individual replisomes in the component in which "
                         "all origins fire and no site stalls")
parser.add_argument("--plot-abundance", metavar="filename",
                    help="output file for a plot of the abundances and the fitted mixture")
parser.add_argument("--plot-vt", metavar="filename",
                    help="output file for a plot of the velocity as a function of time")
parser.add_argument('--verbose', default=False, action=argparse.BooleanOptionalAction,
                    help="output intermediate steps during parameter optimization")
parser.add_argument("--cores", type=int,
                    help="number of cores to use (default: autodetect)")
parser.add_argument("--seed", type=int,
                    help="random seed for reproducibility")
parser.add_argument("--perturbation-start", type=float, default=0.25,
                    help="initial perturbation strength (VeloGP starts with 1, VeloMix starts from the VeloGP fit)")
parser.add_argument("--perturbation-rejects", type=int, default=5,
                    help="number of rejects required to reduce perturbation strength")
parser.add_argument("--perturbation-reduction", type=float, default=0.5**(1.0/4),
                    help="factor by which perturbation strength is reduced")
parser.add_argument("--perturbation-min", type=float, default=0.001,
                    help="perturbation strength after which to stop")

ARGS = parser.parse_args()
STALL_SITES = parse_stall_sites(ARGS.stall_sites)

if ARGS.cores is not None:
    pt.set_num_threads(int(ARGS.cores))
if ARGS.seed is not None:
    pt.manual_seed(ARGS.seed)
    random.seed(ARGS.seed)

# Read the objects of the VeloGP fit
try:
    with open_file(ARGS.objects, "rb") as f:
        OBJECTS = pickle.load(f)
except Exception as e:
    raise ArgumentError(f"unable to read VeloGP objects from {ARGS.objects}: {e}")
if not isinstance(OBJECTS, dict) or ("model" not in OBJECTS) or ("loss" not in OBJECTS):
    raise ArgumentError(f"{ARGS.objects} does not contain the objects written by VeloGP's --out-objects")
ORIGINAL = OBJECTS["model"]
ORIGINAL_LOSS = OBJECTS["loss"]

# Create the mixture model and its loss
try:
    MODEL = MixtureModel(ORIGINAL, STALL_SITES)
except ValueError as e:
    raise ArgumentError(str(e))
LOSS = make_loss(MODEL, ORIGINAL_LOSS)
ALG = FitMixtureModel(MODEL, LOSS)
CHROMOSOME_NAMES = [c.name for c in MODEL.chromosomes]
DOUBLINGTIME = math.log(2) / MODEL.temporal.growthrate.item()


def all_origins() -> list:
    """ All origins (fixed and not) as (chromosome index, origin index) in event order (chromosome name, position) """
    origins = [(ci, i) for ci, c in enumerate(MODEL.chromosomes) for i in range(c.origins.numel())]
    return sorted(origins, key=lambda o: (CHROMOSOME_NAMES[o[0]], MODEL.chromosomes[o[0]].origins[o[1]].item()))


# Print summary of settings
print(f"*** SETTINGS")
print(f"VeloGP objects: {ARGS.objects}")
print(f"Random seed: {ARGS.seed}")
print(f"Number of cores: {pt.get_num_threads()}")
print(f"Perturbation: start at {ARGS.perturbation_start}, reduce by {ARGS.perturbation_reduction:.5f} after "
      f"{ARGS.perturbation_rejects} rejects, stop below {ARGS.perturbation_min}")
print(f"Genome configuration:")
for ci, c in enumerate(MODEL.chromosomes):
    print(f"  chromosome {c.name}: {c.length.item()}bp, {'circular' if c.is_circular else 'linear'}")
    for i in range(c.origins.numel()):
        fixed = i in MODEL.fixed_origins[ci]
        print(f"    origin {c.origin_names[i]} at {c.origins[i].item() + 1}: " +
              ("always fires" if fixed else "fires stochastically"))
    for e in [e for e in MODEL.stall_events if e.chromosome == ci]:
        print(f"    stall site {e.name} at {e.begin + 1}-{e.end}: polarity {e.polarity}, strand "
              f"{STRAND_SYMBOLS[e.strand]}, stalls replisomes moving in direction(s) "
              f"{', '.join(STRAND_SYMBOLS[d] for d in e.blocked_directions())}")
print(f"Mixture: {len(MODEL.events_list)} stochastic events, {len(MODEL.component_labels())} components")
print(f"Initial probabilities (from the VeloGP fit):")
for e, p in zip(MODEL.events_list, pt.cat((MODEL.events.p_fire, MODEL.events.p_stall)).tolist()):
    print(f"  {'firing' if e.kind == 'origin' else 'stalling'} probability of {e.name}: {p:.4f}")
print("Data:")
print(f"  data points: {len(LOSS.x)}")
n_inside = MODEL.windows_inside_blocked(LOSS.x, LOSS.chromosome)
if n_inside > 0:
    print(f"  WARNING: {n_inside} data points lie inside stall sites of polarity 'both', which are not replicated in "
          f"components in which these sites stall")

# Statistics of the VeloGP fit for comparison
def fit_statistics(loss: model.VelocityModelLoss) -> Tuple[float, float]:
    """ The loss (negative log-posterior) and the weighted sum of squared residuals """
    with pt.no_grad():
        ssr = pt.sum(loss.weights * (loss.y_obs - loss.y_pred()) ** 2).item()
        return loss().item(), ssr

ORIGINAL_STATS = fit_statistics(ORIGINAL_LOSS)

# Estimate parameters
print("*** FITTING MODEL")
ALG.optimize_parameters_stochastic(perturb_start=ARGS.perturbation_start,
                                   perturb_reduce=ARGS.perturbation_reduction,
                                   perturb_rejects=ARGS.perturbation_rejects,
                                   perturb_end=ARGS.perturbation_min,
                                   log_phases=True, log_rounds=True,
                                   log_steps=ARGS.verbose, log_params=ARGS.verbose)


# Compute posterior covariances of parameters
print("*** COMPUTING POSTERIOR DISTRIBUTIONS")
P_NAMES = [n for n in ["events.p_fire", "events.p_stall", "events.chromosome_delay_scaled"]
           if n in FitMixtureModel.parameter_names(MODEL)]
G_NAMES = [f"chromosomes.{i}.g_delta_scaled" for i in range(len(MODEL.chromosomes))]
FIXED = {**{f"chromosomes.{i}.g_delta_scaled": s for i, s in enumerate(MODEL.stall_ranges)},
         "events.p_fire": (MODEL.events.p_fire.detach() <= 0.0) | (MODEL.events.p_fire.detach() >= 1.0),
         "events.p_stall": (MODEL.events.p_stall.detach() <= 0.0) | (MODEL.events.p_stall.detach() >= 1.0)}
COV_V0, COV_V0_PMAP = posterior_cov(MODEL, LOSS, ["temporal.v0_log"], {})
# NOTE: As in VeloGP, v0 is excluded, see VelocityModel.v_t_posterior_stderr()
COV_HG, COV_HG_PMAP = posterior_cov(MODEL, LOSS, ["temporal.h_log_raw"] + G_NAMES + P_NAMES, FIXED)


def stderr(name: str) -> np.ndarray:
    """ Posterior standard errors of the entries of a parameter """
    i, n = COV_HG_PMAP[name], MODEL.get_parameter(name).numel()
    return pt.sqrt(pt.diagonal(COV_HG)[i:i + n]).numpy()


P_FIRE_STDERR = stderr("events.p_fire") if "events.p_fire" in COV_HG_PMAP else np.zeros(0)
P_STALL_STDERR = stderr("events.p_stall") if "events.p_stall" in COV_HG_PMAP else np.zeros(0)

with pt.no_grad():
    REPLICATION_TIME = MODEL.completion_time()
    V0 = MODEL.temporal.v0_proper(REPLICATION_TIME)
    MIXTURE_STATS = fit_statistics(LOSS)
print(f"Fit: loss {MIXTURE_STATS[0]:.1f} (VeloGP: {ORIGINAL_STATS[0]:.1f}), weighted sum of squared residuals "
      f"{MIXTURE_STATS[1]:.1f} (VeloGP: {ORIGINAL_STATS[1]:.1f}) for {len(LOSS.x)} data points")
print(f"Replication time C (all origins fire, no site stalls): {REPLICATION_TIME.item():.0f} [s], "
      f"baseline velocity v0 (log-mean of v(t) over [0, C]): {V0.item():.1f} [bp/s]")
for e, p, s in zip(MODEL.events_list, pt.cat((MODEL.events.p_fire, MODEL.events.p_stall)).tolist(),
                   np.concatenate((P_FIRE_STDERR, P_STALL_STDERR))):
    print(f"{'Firing' if e.kind == 'origin' else 'Stalling'} probability of {e.name}: {p:.4f} (std. err. {s:.4f})")
if len(MODEL.chromosomes) > 1:
    for c, d in zip(MODEL.chromosomes, MODEL.events.chromosome_delay_scaled):
        print(f"Firing time of the origins of chromosome {c.name}: "
              f"{(d / (10 * MODEL.temporal.growthrate)).item():.0f} [s]")

# Save results
print("*** SAVING RESULTS")

if ARGS.out_model:
    print(f"Saving fitted model to {ARGS.out_model}")
    try:
        pt.save(MODEL, ARGS.out_model)
    except Exception as e:
        print(f"ERROR: Failed to save {ARGS.out_model}: {e}")

if ARGS.out_objects:
    print(f"Saving VeloGP objects and VeloMix loss function, fit algorithm, fitted model and posterior covariances "
          f"to {ARGS.out_objects}")
    try:
        obj = { **OBJECTS,
                "mixture": {"model": MODEL, "loss": LOSS, "algorithm": ALG,
                            "cov_v0": COV_V0, "cov_v0.pmap": COV_V0_PMAP,
                            "cov_hg": COV_HG, "cov_hg.pmap": COV_HG_PMAP,
                            "events": MODEL.event_names(), "components": MODEL.component_labels()} }
        with open_file(ARGS.out_objects, "wb+") as f:
            pickle.dump(obj, f)
    except Exception as e:
        print(f"ERROR: Failed to save {ARGS.out_objects}: {e}")

if ARGS.out_probabilities:
    print(f"Saving firing and stalling probabilities to {ARGS.out_probabilities}")
    try:
        # All origins (including those which always fire), then the stall sites, see MixtureModel.events_list
        rows = []
        p_fire = dict(zip([(e.chromosome, e.index) for e in MODEL.origin_events],
                          zip(MODEL.events.p_fire.tolist(), P_FIRE_STDERR)))
        for ci, i in all_origins():
            c = MODEL.chromosomes[ci]
            fixed = i in MODEL.fixed_origins[ci]
            p, s = (1.0, math.nan) if fixed else p_fire[(ci, i)]
            pos = c.origins[i].item()
            rows.append(("origin", c.name, pos, pos + 1, "*", c.origin_names[i], "NA", p, s, int(fixed)))
        for e, p, s in zip(MODEL.stall_events, MODEL.events.p_stall.tolist(), P_STALL_STDERR):
            rows.append(("stall", e.chromosome_name, e.begin, e.end, STRAND_SYMBOLS[e.strand], e.name, e.polarity,
                         p, s, 0))
        slen = lambda k, header: max([len(header)] + [len(r[k]) for r in rows])
        cols = [('type', 'U6'), ('chr', f'U{slen(1, "chr")}'), ('begin', np.int64), ('end', np.int64),
                ('strand', 'U1'), ('name', f'U{slen(5, "name")}'), ('polarity', 'U4'),
                ('probability', np.float64), ('stderr', np.float64), ('fixed', np.int32)]
        fmt = ('%s', '%s', '%d', '%d', '%s', '%s', '%s', '%.6f', '%.6f', '%d')
        write_table(ARGS.out_probabilities, cols, fmt, np.array(rows, dtype=cols))
    except Exception as e:
        print(f"ERROR: Failed to save {ARGS.out_probabilities}: {e}")

if ARGS.out_abundance:
    print(f"Saving observed and predicted abundances to {ARGS.out_abundance}")
    try:
        with pt.no_grad():
            p_pred, scale = LOSS.abundance_and_scale()
            components = MODEL.component_abundances(LOSS.x_prep)
            weights = MODEL.component_weights()
        labels = MODEL.component_labels()
        slen = max(len(chr) for chr in CHROMOSOME_NAMES)
        cols = [ ('chr', f'U{slen}'), ('pos', np.int64), ('y_obs', np.float64), ('y_obs_stderr', np.float64),
                 ('y_pred', np.float64), ('p_pred', np.float64) ]
        cols += [ (f"y.{l}", np.float64) for l in labels ] + [ (f"w.{l}", np.float64) for l in labels ]
        fmt = ('%s', '%d', '%.5f', '%.5f', '%.5f', '%.5f') + ('%.5f',) * len(labels) + ('%.6g',) * len(labels)
        tab = np.zeros(LOSS.x.shape[0], dtype=cols)
        tab["chr"] = [CHROMOSOME_NAMES[i] for i in LOSS.chromosome.tolist()]
        tab["pos"] = LOSS.x.numpy()
        tab["y_obs"] = LOSS.y_obs.numpy()
        tab["y_obs_stderr"] = LOSS.s_obs.numpy()
        tab["y_pred"] = (p_pred * scale).numpy()
        tab["p_pred"] = p_pred.numpy()
        for k, l in enumerate(labels):
            tab[f"y.{l}"] = (components[:, k] * scale).numpy()
            tab[f"w.{l}"] = weights[k].item()
        write_table(ARGS.out_abundance, cols, fmt, tab)
    except Exception as e:
        print(f"ERROR: Failed to save {ARGS.out_abundance}: {e}")

if ARGS.out_vt:
    print(f"Saving v(t) to {ARGS.out_vt}")
    try:
        cols = [('t', np.float64), ('vt', np.float64), ('vt_stderr', np.float64), ('nforks', np.int32),
                ('replicationtime', np.float64)]
        fmt = ('%.5f', '%.5f', '%.5f', '%d', '%.5f')
        tab = np.zeros(MODEL.temporal.h_tgrid.shape[0], dtype=cols)
        with pt.no_grad():
            tab["t"] = MODEL.temporal.h_tgrid.numpy()
            tab["vt"] = MODEL.temporal.v_t(MODEL.temporal.h_tgrid).numpy()
            tab["vt_stderr"] = MODEL.v_t_posterior_stderr(MODEL.temporal.h_tgrid, posterior_cov=COV_HG,
                                                          pmap=COV_HG_PMAP).numpy()
            # Number of active replisomes and replication time in the component in which all origins fire and no
            # site stalls
            tab_state, _, _, _ = MODEL.trajectories(MODEL.temporal.h_tgrid)
            tab["nforks"] = pt.sum(tab_state == 1, dim=-1).numpy()
            tab["replicationtime"] = REPLICATION_TIME.item()
        write_table(ARGS.out_vt, cols, fmt, tab)
    except Exception as e:
        print(f"ERROR: Failed to save {ARGS.out_vt}: {e}")

if ARGS.out_vx_feature_delays:
    print(f"Saving v(x) to {ARGS.out_vx_feature_delays}")
    try:
        # All g ranges in model order, those which are not stall sites are written. Computed for the component in
        # which all origins fire and no site stalls, see the corresponding VeloGP output for the columns.
        ranges = [(ci, j) for ci, c in enumerate(MODEL.chromosomes) for j in range(c.g_xranges_l.numel())]
        keep = np.array([not MODEL.stall_ranges[ci][j].item() for ci, j in ranges], dtype=bool)
        with pt.no_grad():
            forks = MODEL.g_range_forks()
            time_s, delay_s = MODEL.g_range_passage_times()
            p_block = MODEL.g_range_block_probability()
            g = pt.cat([c.g for c in MODEL.chromosomes])
            delay_bp = pt.cat([c.g_delta * c.g_xranges_len for c in MODEL.chromosomes])
        names = [MODEL.chromosomes[ci].g_names[j] for ci, j in ranges]
        strands = [int(MODEL.chromosomes[ci].g_strands[j].item()) for ci, j in ranges]
        delayscale = [MODEL.chromosomes[ci].g_delayscale[j].item() if MODEL.chromosomes[ci].g_delayscale is not None
                      else math.nan for ci, j in ranges]
        origin_fwd = [o if o is not None else "NA" for o in forks.origin_fwd]
        origin_rev = [o if o is not None else "NA" for o in forks.origin_rev]
        enter_fwd, enter_rev = forks.enter_fwd.numpy(), forks.enter_rev.numpy()
        fwd_first = (enter_fwd < enter_rev) | (np.isnan(enter_rev) & ~np.isnan(enter_fwd))
        rev_first = (enter_rev < enter_fwd) | (np.isnan(enter_fwd) & ~np.isnan(enter_rev))
        traversal = fwd_first.astype(int) - rev_first.astype(int)
        chr_slen = max([len("chr")] + [len(n) for n in CHROMOSOME_NAMES])
        fn_slen = max([len("feature")] + [len(n) for n in names])
        on_slen = max([len("NA")] + [len(o) for o in origin_fwd + origin_rev])
        cols = [ ('chr', f'U{chr_slen}'), ('begin', np.int64), ('end', np.int64), ('strand', 'U1'),
                 ('length', np.int64),
                 ('name', f'U{fn_slen}'), ('delayscale', np.float64),
                 ('g', np.float64), ('delay_bp', np.float64), ('delay_rel', np.float64), ('p_block', np.float64),
                 ('time_s', np.float64), ('delay_s', np.float64), ('direction', 'U1'),
                 ('origin_forward', f'U{on_slen}'), ('origin_forward_enter', np.float64),
                 ('origin_forward_exit', np.float64),
                 ('origin_reverse', f'U{on_slen}'), ('origin_reverse_enter', np.float64),
                 ('origin_reverse_exit', np.float64) ]
        fmt = ('%s', '%d', '%d', '%s', '%d', '%s', '%.2f', '%.5f', '%.5f', '%.5f', '%.6g', '%.5f', '%.5f', '%s',
               '%s', '%.5f', '%.5f', '%s', '%.5f', '%.5f')
        tab = np.zeros(len(ranges), dtype=cols)
        tab["chr"] = [CHROMOSOME_NAMES[ci] for ci, j in ranges]
        tab["begin"] = [MODEL.chromosomes[ci].g_xranges_l[j].item() for ci, j in ranges]
        tab["end"] = [MODEL.chromosomes[ci].g_xranges_u[j].item() for ci, j in ranges]
        tab["strand"] = [STRAND_SYMBOLS[s] for s in strands]
        tab["length"] = tab["end"] - tab["begin"]
        tab["name"] = names
        tab["delayscale"] = delayscale
        tab["g"] = g.numpy()
        tab["delay_bp"] = delay_bp.numpy()
        tab["time_s"] = time_s.numpy()
        tab["delay_s"] = delay_s.numpy()
        tab["delay_rel"] = tab["delay_s"] / REPLICATION_TIME.item()
        tab["p_block"] = p_block.numpy()
        tab["direction"] = [STRAND_SYMBOLS[s * t] for s, t in zip(strands, traversal)]
        tab["origin_forward"] = origin_fwd
        tab["origin_forward_enter"] = enter_fwd
        tab["origin_forward_exit"] = forks.exit_fwd.numpy()
        tab["origin_reverse"] = origin_rev
        tab["origin_reverse_enter"] = enter_rev
        tab["origin_reverse_exit"] = forks.exit_rev.numpy()
        write_table(ARGS.out_vx_feature_delays, cols, fmt, tab[keep])
    except Exception as e:
        print(f"ERROR: Failed to save {ARGS.out_vx_feature_delays}: {e}")

if ARGS.out_trajectories:
    print(f"Saving velocity profile and replisome trajectories x(t) to {ARGS.out_trajectories}")
    try:
        # Component in which all origins fire and no site stalls. MODEL.trajectories() returns two replisomes per
        # origin, in the order of the model's origins.
        TRAJ_ORIGINS = [(c, name) for c in MODEL.chromosomes for name in c.origin_names]
        k = len(TRAJ_ORIGINS)
        cols = [('t', np.float64), ('v0', np.float64), ('vt', np.float64), ('vt_stderr', np.float64),
                ('replicationtime', np.float64) ]
        fmt = ('%.5f', '%.5f', '%.5f', '%.5f', '%.5f')
        for prefix, f in [("", '%d'), ("x.", '%.5f'), ("dx.", '%.5f'), ("v.", '%.5f'), ("gx.", '%.5f')]:
            cols += [ (f"{prefix}{name}{d}", np.float64) for c, name in TRAJ_ORIGINS for d in [ "-", "+" ]]
            fmt += (f,) * (2*k)
        tab = np.zeros(MODEL.temporal.h_tgrid.shape[0], dtype=cols)
        with pt.no_grad():
            tab["t"] = MODEL.temporal.h_tgrid.numpy()
            tab["v0"] = V0.numpy()
            tab["vt"] = MODEL.temporal.v_t(MODEL.temporal.h_tgrid).numpy()
            tab["vt_stderr"] = MODEL.v_t_posterior_stderr(MODEL.temporal.h_tgrid, posterior_cov=COV_HG,
                                                          pmap=COV_HG_PMAP).numpy()
            tab["replicationtime"] = REPLICATION_TIME.item()
            tab_state, tab_x, tab_dx, tab_v = MODEL.trajectories(MODEL.temporal.h_tgrid)
            for o_i, (c, name) in enumerate(TRAJ_ORIGINS):
                for d_i, d in enumerate([ "-", "+" ]):
                    r_i = 2*o_i + d_i
                    tab[f"{name}{d}"] = tab_state[:, r_i].numpy()
                    tab[f"x.{name}{d}"] = tab_x[:, r_i].numpy()
                    tab[f"dx.{name}{d}"] = tab_dx[:, r_i].numpy()
                    tab[f"v.{name}{d}"] = tab_v[:, r_i].numpy()
                    tab[f"gx.{name}{d}"] = c.g_fun(tab_x[:, r_i]).numpy()
        write_table(ARGS.out_trajectories, cols, fmt, tab)
    except Exception as e:
        print(f"ERROR: Failed to save {ARGS.out_trajectories}: {e}")

if ARGS.plot_vt:
    try:
        print(f"Plotting v(t) to {ARGS.plot_vt}")
        with pt.no_grad():
            t_fin = REPLICATION_TIME
            plot_t = MODEL.temporal.h_tgrid
            plot_vt = MODEL.temporal.v_t(plot_t)
            plot_vt_stderr = MODEL.v_t_posterior_stderr(plot_t, posterior_cov=COV_HG, pmap=COV_HG_PMAP)
            plot_yrange = [0.8 * pt.min(plot_vt[plot_t <= t_fin]).item(), pt.max(plot_vt[plot_t <= t_fin]).item() * 1.2]
        pp.clf()
        pp.figure(figsize=(2.5, 2.0))
        pp.ylim(plot_yrange)
        pp.axhline(y=V0.numpy(), linestyle="dotted", color="black", linewidth=0.5)
        pp.step(plot_t.numpy() / DOUBLINGTIME, plot_vt.numpy(), linewidth=0.5, color="navy")
        pp.fill_between(plot_t.numpy() / DOUBLINGTIME,
                        plot_vt.numpy() - 2 * plot_vt_stderr.numpy(), plot_vt.numpy() + 2 * plot_vt_stderr.numpy(),
                        color="lightblue")
        pp.axvline(x=1.0, linestyle="dashed", color="black", linewidth=0.5)
        pp.axvline(x=t_fin.numpy() / DOUBLINGTIME, linestyle="solid", color="black", linewidth=0.5)
        pp.tight_layout()
        pp.savefig(ARGS.plot_vt, dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"ERROR: Failed to save {ARGS.plot_vt}: {e}")

if ARGS.plot_abundance:
    try:
        print(f"Plotting data and fitted mixture to {ARGS.plot_abundance}")
        pp.clf()
        n = len(MODEL.chromosomes)
        fig, axes = pp.subplots(1, n, figsize=(2.5 * n, 2.0), squeeze=False, sharey=True)
        with pt.no_grad():
            plot_y_pred = LOSS.y_pred()
            plot_y_velogp = ORIGINAL_LOSS.y_pred()
        for i, c in enumerate(MODEL.chromosomes):
            sel = LOSS.chromosome == i
            ax = axes[0, i]
            ax.plot(LOSS.x[sel].numpy(), LOSS.y_obs[sel].numpy(), marker="x", markersize=0.25, linestyle="", color="grey")
            ax.plot(LOSS.x[sel].numpy(), plot_y_velogp[sel].numpy(), linewidth=0.5, color="darkorange", linestyle="dashed")
            ax.plot(LOSS.x[sel].numpy(), plot_y_pred[sel].numpy(), linewidth=0.5, color="navy")
            if n > 1:
                ax.set_title(c.name, fontsize="small")
        fig.tight_layout()
        pp.savefig(ARGS.plot_abundance, dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"ERROR: Failed to save {ARGS.plot_abundance}: {e}")
