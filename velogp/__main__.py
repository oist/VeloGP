#!/usr/bin/env python3
from typing import Dict, List, Set

import io
import sys
import math
import re
from dataclasses import dataclass
import argparse
import pickle
import gzip
import bz2
import random
import BCBio.GFF
import Bio.SeqIO
import numpy as np
import pandas as pd
import torch as pt
import matplotlib.pyplot as pp

from . import model
from . import fit

class ArgumentError(Exception):
    pass

class SplitAndAdd(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, str):
            values = [ values ]
        for vs in values:
            for v in vs.split(","):
                if len(v) > 0:
                    getattr(namespace, self.dest, set()).add(v)

def open_file(filename, mode):
    if filename.endswith(".gz"):
        return gzip.open(filename, mode)
    elif filename.endswith(".bz2"):
        return bz2.open(filename, mode)
    else:
        return open(filename, mode)

@dataclass
class Feature:
    chr: str
    begin: int
    end: int
    delayscale: float
    name: str

@dataclass
class Origin:
    chr: str
    pos: int
    name : str

# Make output unbuffered
sys.stdout = io.TextIOWrapper(open(sys.stdout.fileno(), 'wb', 0), write_through=True)

# Argument handling
parser = argparse.ArgumentParser(
    prog="VeloGP",
    description="Infer spatio-temporal replisome velocity from MFA data")

parser.add_argument("inputfile",
                    help="abundance table")
parser.add_argument("--out-model", metavar="filename",
                    help="output file for the fitted model")
parser.add_argument("--out-objects", metavar="filename",
                    help="output file for the loss function, fit algorithm and fitted model")
parser.add_argument("--out-abundance", metavar="filename",
                    help="output observed and predicted abundances")
parser.add_argument("--out-vt", metavar="filename",
                    help="output file for the velocity as a function of time as a tab-separated table")
parser.add_argument("--out-vx-feature-delays", metavar="filename",
                    help="output file for the delays incurred at each v(x) feature")
parser.add_argument("--out-trajectories", metavar="filename",
                    help="output file for the trajectories of the individual replisomes")
parser.add_argument("--out-stats", metavar="filename",
                    help="output file for fitting statistics")
parser.add_argument("--plot-abundance", metavar="filename",
                    help="output file for a plot of the input abundances and fitted model")
parser.add_argument("--plot-vt", metavar="filename",
                    help="output file for a plot of the velocity as a function of time")
parser.add_argument("--growthrate", default=1.0, type=float, metavar="lambda",
                    help="growthrate of sequenced population [1/hour]")
parser.add_argument("--origins", default={"oriC"}, metavar="origin,origin,...", action=SplitAndAdd,
                    help="names or locations of origins of replication")
parser.add_argument("--v0-min", default=10, type=float,
                    help="minimum baseline velocity")
parser.add_argument("--v0-max", default=10000, type=float,
                    help="maximal baseline velocity")
parser.add_argument("--vt-timescale", type=float, default=0.2, metavar="value",
                    help="timescale of temporal velocity fluctuations in units of doubling time (covariance length scale)")
parser.add_argument("--vt-magnitude", type=float, default=0.1, metavar="value",
                    help="magnitude of temporal velocity fluctuations (standard deviation of log-change)")
parser.add_argument("--vt-resolution", type=int, default=100, metavar="points",
                    help="number of points per timescale to use to represent v(t)")
parser.add_argument("--vt-tmax", type=float, default=3, metavar="tmax",
                    help="maximum time t for which to estimate v(t)")
parser.add_argument("--vx-delayscale", type=float, default=0.01, metavar="",
                    help="delay we can roughly expect to be incurred at a single vx feature as a fraction of total replication time")
parser.add_argument("--vx-feature-names", default=set(), metavar="feature,feature,...", action=SplitAndAdd,
                    help="add these features to list of sites at which the velocity is modulated")
parser.add_argument("--vx-feature-types", default=set(), metavar="type,type,...", action=SplitAndAdd,
                    help="add all features with these types to list of sites at which the velocity is modulated")
parser.add_argument("--gff", default=[], metavar="gff-file", action="append",
                    help="GFF file containing genome annotations (may be compressed)")
parser.add_argument("--bed", default=[], metavar="bed-file", action="append",
                    help="BED file containing genome annotations, must contain at least 4 columns chrom,start,end,name, " +
                    "(may be compressed)")
parser.add_argument("--genome", required=True, metavar="fasta-file",
                    help="FASTA file containing the reference genome (may be compressed)")
parser.add_argument("--columns", default="seq,midpoint,coverage_ratio,coverage_ratio_stderr,ratio_mappable",
                    help="comma-separated names of columns containing chromosome, position, abundance, std.error of abundance and validity")
parser.add_argument('--verbose', default=False, action=argparse.BooleanOptionalAction,
                    help="output intermediate steps during parameter optimization")
parser.add_argument("--cores", type=int,
                    help="number of cores to use (default: autodetect)")
parser.add_argument("--seed", type=int,
                    help="random seed for reproducibility")
parser.add_argument("--rounds", type=int, default=100,
                    help="number of optimization rounds")

# Parse arguments
ARGS = parser.parse_args()

# Post-process column names
ARGS.columns = ARGS.columns.split(",")
if len(ARGS.columns) != 5:
    raise ArgumentError(f"exactly five column names must be specified if --columns is used")
COLUMNS = {
    "seq": ARGS.columns[0],
    "x": ARGS.columns[1],
    "y_obs": ARGS.columns[2],
    "s_obs": ARGS.columns[3],
    "valid": ARGS.columns[4],
}

# Set number of cores to user
if ARGS.cores is not None:
    pt.set_num_threads(int(ARGS.cores))

# Set random seet
if ARGS.seed is not None:
    pt.manual_seed(ARGS.seed)
    random.seed(ARGS.seed)

# Read genome to find chromosome names and lengths
CHROMOSOME_LENGTHS : Dict[str, int] = {}
fasta_file = open_file(ARGS.genome, "rt")
try:
    for seq in Bio.SeqIO.parse(fasta_file, "fasta"):
        CHROMOSOME_LENGTHS[seq.id] = len(seq)
    SINGLE_CHROMOSOME = next(iter(CHROMOSOME_LENGTHS.keys())) if len(CHROMOSOME_LENGTHS) == 1 else None
finally:
    fasta_file.close()

# Origins of replication
# NOTE: Origins specified by name are resolved during GFF/BED file parsing
ORIGINS : List[Origin] = []
ORIGINS_NAMES : Set[str] = set()
for origin in ARGS.origins:
    m = re.match(r"^((.*):)?(\d+)$", origin)
    if m is None:
        ORIGINS_NAMES.add(origin)
        continue
    chr, pos = m.group(2), int(m.group(3))
    if chr is None:
        if SINGLE_CHROMOSOME is None:
            raise ArgumentError(f"origin {origin} does not specify chromosome and genome contains more than one")
        else:
            chr = SINGLE_CHROMOSOME
    if (origin < 1) or (origin > CHROMOSOME_LENGTHS[chr]):
        raise ArgumentError(f"origin {origin} lies outside the range of chromosome {chr} [1, {CHROMOSOME_LENGTHS[chr]}])")
    # NOTE: Locations on the command-line are one-based
    origin = Origin(chr=chr, pos=pos-1)
    ORIGINS.append(origin)
ORIGINS_NAMES_TO_BE_RESOLVED : Set[str] = ORIGINS_NAMES.copy()

# Genomic features at which replisomes might slow down
VX_FEATURES : List[Feature] = list()
VX_FEATURE_NAMES_TO_BE_RESOLVED : Set[str] = set(ARGS.vx_feature_names)

# Read GFF files
for gff_filename in ARGS.gff:
    gff_file = open_file(gff_filename, "rt")
    try:
        # Recursively iterate through features in the GFF file
        # (NOTE: It's silly that we have to do this since GFF files are actually
        # tabular. But it's apparently the only API that BCBio provides...)
        stack = list()
        for record in BCBio.GFF.parse(gff_file):
            # NOTE: The top-level objects are SeqRecord, which represents a sequence.
            # This of course has nothing to do with how a GFF file actually looks,
            # this is just more API sillyness. But anyway, we propagate that sequence
            # name down when we expand features and sub-features, since there seems to
            # be no way to get to it later otherwise...
            stack.extend([ (record.id, sf) for sf in record.features ])
            while len(stack) > 0:
                # Get next feature, push sub-features on stack to handle later
                # NOTE: The stack contains (seq, feature) pairs since we have to propagate
                # the sequence name downwards due to API sillyness, see above...
                seq, f = stack.pop()
                stack.extend([ (seq, sf) for sf in f.sub_features ])
                # Generate list of "names" of the feature. Oh, GFF...
                f_names = [ f.id ]
                if "Name" in f.qualifiers:
                    f_names.extend(f.qualifiers["Name"])
                # Resolve feature names, add all features of requested types
                f_matching_feature_names = [ fn for fn in f_names if fn in ARGS.vx_feature_names]
                if (f.type in ARGS.vx_feature_types) or len(f_matching_feature_names) > 0:
                    # Named should be unambiguous
                    for fn in [fn not in VX_FEATURE_NAMES_TO_BE_RESOLVED for fn in f_matching_feature_names]:
                        raise ArgumentError(f"feature name {fn} is ambiguous")
                    # Create feature
                    # NOTE: GFF positions are one-based and end is included, but BCBio.GFF already
                    # translates this into zero-based, non-inclusive ranges, so we use them as-is.
                    feature = Feature(chr=seq, begin=int(f.location.start), end=int(f.location.end),
                                      name=f_matching_feature_names[0] if len(f_matching_feature_names) > 0 else f_names[0],
                                      delayscale=ARGS.vx_delayscale)
                    # Make sure min < max
                    if feature.begin > feature.end:
                        feature.begin, feature.end = feature.end, feature.begin
                    elif feature.begin == feature.end:
                        feature.end += 1
                    # Add to feature list
                    VX_FEATURES.append(feature)
                    # Removed from to-be-resolved list
                    for fn in f_matching_feature_names:
                        VX_FEATURE_NAMES_TO_BE_RESOLVED.remove(fn)
                # Resolve unresolved origin names
                f_matching_origin_names = [ fn for fn in f_names if fn in ORIGINS_NAMES]
                if len(f_matching_origin_names) > 0:
                    # Named should be unambiguous
                    for fn in [fn not in ORIGINS_NAMES_TO_BE_RESOLVED for fn in f_matching_origin_names]:
                        raise ArgumentError(f"feature name {fn} is ambiguous")
                    origin = Origin(chr=f.id, pos=int((f.location.start + f.location.end)/2), name=f_matching_origin_names[0])
                    ORIGINS.append(origin)
                    for fn in f_matching_origin_names:
                        ORIGINS_NAMES_TO_BE_RESOLVED.remove(fn)
    finally:
        gff_file.close()

# Read BED files
for bed_filename in ARGS.bed:
    bed_file = open_file(bed_filename, "rt")
    for line in bed_file:
        fields = [ f.strip() for f in line.split("\t") ]
        if len(fields) < 4:
            raise ArgumentError(f"BED file must have at least 4 columns, found only {len(fields)} columns in line: {line}")
        chr, start, end, name = fields[0], int(fields[1]), int(fields[2]), fields[3]
        # Resolve feature names
        if name in ARGS.vx_feature_names:
            # Named should be unambiguous
            if name not in VX_FEATURE_NAMES_TO_BE_RESOLVED:
                raise ArgumentError(f"feature name {name} is ambiguous")
            # Create feature
            # NOTE: Position in BED files are zero-based, end is not included in the range
            feature = Feature(chr=chr, begin=start, end=end, name=name, delayscale=ARGS.vx_delayscale)
            # Make sure min < max
            if feature.begin > feature.end:
                feature.begin, feature.end = feature.end, feature.begin
            elif feature.begin == feature.end:
                feature.end += 1
            # Add to feature list
            VX_FEATURES.append(feature)
            # Removed from to-be-resolved list
            VX_FEATURE_NAMES_TO_BE_RESOLVED.remove(feature.name)
        # Resolve unresolved origin names:
        if name in ORIGINS_NAMES:
            if name not in ORIGINS_NAMES_TO_BE_RESOLVED:
                raise ArgumentError(f"origin name {name} is ambiguous")
            origin = Origin(chr=chr, pos=int((start + end) / 2), name=name)
            ORIGINS.append(origin)
            ORIGINS_NAMES_TO_BE_RESOLVED.remove(name)

# Complain if not all names were resolved
if len(VX_FEATURE_NAMES_TO_BE_RESOLVED) > 0:
    raise ArgumentError(f"unable to resolve feature names {', '.join(VX_FEATURE_NAMES_TO_BE_RESOLVED)}")
if len(ORIGINS_NAMES_TO_BE_RESOLVED) > 0:
    raise ArgumentError(f"unable to resolve origin names {', '.join(ORIGINS_NAMES_TO_BE_RESOLVED)}")

# Sort features and origins
VX_FEATURES.sort(key=lambda f: (f.chr, f.begin))
ORIGINS.sort(key=lambda o: (o.chr, o.pos))

# Check that features do not overlap
f_prev : Feature = None
for f in VX_FEATURES:
    if f_prev is not None and f_prev.chr == f.chr and f_prev.end > f.begin:
        raise ArgumentError(f"features {f_prev.name} and {f.name} overlap " +
                            f"({f_prev.chr}:{f_prev.begin+1}-{f_prev.end-1} and {f.chr}:{f.begin+1}-{f.end-1})")

# Check that every chromosome has an origin
for chr in CHROMOSOME_LENGTHS.keys():
    if len([o for o in ORIGINS if o.chr == chr]) == 0:
        raise ArgumentError(f"chromosome {chr} contains no origin")

# Multiple chromosomes are not yet supported
if not SINGLE_CHROMOSOME:
    raise ArgumentError("multiple chromosomes are not yet supported")

# Read abundance data
ABUNDANCE = pd.read_csv(ARGS.inputfile, delimiter="\t")
ABUNDANCE = ABUNDANCE.rename(columns={ fn: sn for sn, fn in COLUMNS.items() })
ABUNDANCE_ROWS_PREFILTER = len(ABUNDANCE)
ABUNDANCE = ABUNDANCE.loc[~pd.isna(ABUNDANCE.x) & ~pd.isna(ABUNDANCE.y_obs) & ~pd.isna(ABUNDANCE.s_obs) & (ABUNDANCE.valid > 0)]

# Create model
DOUBLINGTIME = 3600 * math.log(2) / ARGS.growthrate
MODEL = model.VelocityModel(
    genome_length=CHROMOSOME_LENGTHS[SINGLE_CHROMOSOME], is_circular=True,
    origins=pt.tensor([o.pos for o in ORIGINS if o.chr == SINGLE_CHROMOSOME], dtype=pt.int32),
    origin_names=[o.name for o in ORIGINS if o.chr == SINGLE_CHROMOSOME],
    h_tgrid=pt.arange(start=0, end=DOUBLINGTIME * ARGS.vt_tmax,
                         step=DOUBLINGTIME * ARGS.vt_timescale / ARGS.vt_resolution,
                         dtype=pt.float64),
    g_xranges_l=pt.tensor([f.begin for f in VX_FEATURES if f.chr == SINGLE_CHROMOSOME], dtype=pt.int32),
    g_xranges_u=pt.tensor([f.end for f in VX_FEATURES if f.chr == SINGLE_CHROMOSOME], dtype=pt.int32),
    g_delayscale=pt.tensor([f.delayscale for f in VX_FEATURES if f.chr == SINGLE_CHROMOSOME], dtype=pt.float64),
    v0_min=model.dtensor(ARGS.v0_min), v0_max=model.dtensor(ARGS.v0_max)
)
MODEL.growthrate[0] = ARGS.growthrate / 3600
MODEL.h_sigma[0] = ARGS.vt_magnitude
MODEL.h_timescale[0] = DOUBLINGTIME * ARGS.vt_timescale

# Create loss function
LOSS = model.VelocityModelLoss(
    x=pt.tensor(ABUNDANCE["x"].values, dtype=pt.float64),
    y_obs=pt.tensor(ABUNDANCE["y_obs"].values, dtype=pt.float64),
    s_obs=pt.tensor(ABUNDANCE["s_obs"].values, dtype=pt.float64),
    model=MODEL,
    params=["v0_log", "delay_scaled", "h_log_raw", "g_delta_scaled"])

# Create fit instance
ALG = fit.FitVelocityModel(MODEL, LOSS)

# Print summary of settings
print(f"*** SETTINGS")
print(f"Number of cores: {pt.get_num_threads()}")
print(f"Genome configuration:")
for chr, l in CHROMOSOME_LENGTHS.items():
    print(f"  chromosome {chr}: {l}bp")
    origins = [f"{o.pos+1} ({o.name})" for o in ORIGINS if o.chr == chr]
    print(f"    origins: {', '.join(origins)}")
    print(f"    allow for spatial velocity modulation at:")
    for f in [f for f in VX_FEATURES if f.chr == chr]:
        print(f"      {f.begin+1}-{f.end+1} ({f.name}): {f.end-f.begin}bp, delayscale {f.delayscale}")

print(f"Parameters:")
print(f"  growth rate: {ARGS.growthrate} [1/h]")
print(f"  v(t) timescale: {MODEL.h_timescale[0]:.0f} [s] (= {ARGS.vt_timescale} doublingtimes)")
print(f"  log v(t) std. err: {ARGS.vt_magnitude}")
print(f"  v(x) delay scale: {ARGS.vx_delayscale}")

print("Data:")
print(f"  data points: {len(ABUNDANCE)} ({100.0*len(ABUNDANCE)/ABUNDANCE_ROWS_PREFILTER:.0f}% of {ABUNDANCE_ROWS_PREFILTER} points pre-filtering)")

print("*** COMPILING MODELS")
MODEL.compile()
LOSS.compile()

# Estimate parameters
print("*** FITTING MODEL")
#ALG.optimize_parameters(log_phases=True, log_steps=ARGS.verbose)
ALG.optimize_parameters_stochastic(steps=100, log_phases=True, log_rounds=True, log_steps=ARGS.verbose)

# Compute posterior covariances of parameters
print("*** COMPUTING POSTERIOR DISTRIBUTIONS")
COV_V0, COV_V0_PMAP = MODEL.posterior_cov(LOSS, ["v0_log"])
COV_HG, COV_HG_PMAP = MODEL.posterior_cov(LOSS, ["h_log_raw", "g_delta_scaled"])

# Save results
print("*** SAVING RESULTS")

# Save fitted model
if ARGS.out_model is not None:
    print(f"Saving fitted model to {ARGS.out_model}")
    try:
        pt.save(MODEL, ARGS.out_model)
    except Exception as e:
        print(f"ERROR: Failed to save {ARGS.out_model}: {e}")

# Save all results
if ARGS.out_objects is not None:
    print(f"Saving fitting algorithm, loss function, fitted model and posterior covariances to {ARGS.out_objects}")
    f = None
    try:
        f = open_file(ARGS.out_objects, "wb+")
        obj = {"model": MODEL, "loss": LOSS, "algorithm": ALG,
               "cov_v0": COV_V0, "cov_v0.pmap": COV_V0_PMAP,
               "cov_hg": COV_HG, "cov_hg.pmap": COV_HG_PMAP }
        pickle.dump(obj, f)
    except Exception as e:
        print(f"ERROR: Failed to save {ARGS.out_objects}: {e}")
    finally:
        if f is not None:
            f.close()

# Save y_obs and y_pred
if ARGS.out_abundance is not None:
    print(f"Saving observed and predicted abundances to {ARGS.out_abundance}")
    f = None
    try:
        # Get model predictions
        p_pred, scale = LOSS.abundance_and_scale()
        # Build table
        slen = len(SINGLE_CHROMOSOME)
        cols = [ ('chr', f'U{slen}'), ('pos', np.int32), ('y_obs', np.float64), ('y_obs_stderr', np.float64),
                 ('y_pred', np.float64), ('p_pred', np.float64) ]
        fmt = ('%s', '%d', '%.5f', '%.5f', '%.5f', '%.5f' )
        tab = np.zeros(LOSS.x.shape[0], dtype=cols)
        tab["chr"] = SINGLE_CHROMOSOME
        tab["pos"] = LOSS.x
        tab["y_obs"] = ABUNDANCE["y_obs"]
        tab["y_obs_stderr"] = ABUNDANCE["s_obs"]
        tab["y_pred"] = p_pred.detach() * scale.detach()
        tab["p_pred"] = p_pred.detach()
        # Write tab-separated to output file
        f = open_file(ARGS.out_abundance, "wt+")
        f.write("\t".join([name for (name, type) in cols]) + "\n")
        np.savetxt(f, tab, delimiter="\t", fmt=fmt)
    except Exception as e:
        print(f"ERROR: Failed to save {ARGS.out_abundance}: {e}")
    finally:
        if f is not None:
            f.close()

# Save v(t)
if ARGS.out_vt is not None:
    print(f"Saving v(t) to {ARGS.out_vt}")
    f = None
    try:
        # Build table
        cols = [('t', np.float64), ('vt', np.float64), ('vt_stderr', np.float64)]
        fmt = ('%.5f', '%.5f', '%.5f')
        tab = np.zeros(MODEL.h_tgrid.shape[0], dtype=cols)
        tab["t"] = MODEL.h_tgrid
        tab["vt"] = MODEL.v_t(MODEL.h_tgrid).detach()
        tab["vt_stderr"] = MODEL.v_t_posterior_stderr(MODEL.h_tgrid, COV_HG, COV_HG_PMAP).detach()
        # Write tab-separated to output file
        f = open_file(ARGS.out_vt, "wt+")
        f.write("\t".join([name for (name, type) in cols]) + "\n")
        np.savetxt(f, tab, delimiter="\t", fmt=fmt)
    except Exception as e:
        print(f"ERROR: Failed to save {ARGS.out_vt}: {e}")
    finally:
        if f is not None:
            f.close()

# Save v(x), i.e. delays incurred at various genomic features
if ARGS.out_vx_feature_delays is not None:
    print(f"Saving v(x) to {ARGS.out_vx_feature_delays}")
    f = None
    try:
        # Check that we agree on the order of the genomic features...
        assert all(MODEL.g_xranges_l == pt.tensor(list([f.begin for f in VX_FEATURES if f.chr == SINGLE_CHROMOSOME]), dtype=pt.int32))
        # Build table
        chr_slen = max([len("chr")] + [ len(f.chr) for f in VX_FEATURES ])
        fn_slen = max([len("feature")] + [ len(f.name) for f in VX_FEATURES ])
        cols = [ ('chr', f'U{chr_slen}'), ('begin', np.int32), ('end', np.int32), ('length', np.int32),
                 ('name', f'U{fn_slen}'), ('delayscale', np.float64),
                 ('g', np.float64), ('v_bps', np.float64), ('delay_bp', np.float64), ('delay_rel', np.float64), ('delay_s', np.float64) ]
        fmt = ('%s', '%d', '%d', '%d', '%s', '%.2f', '%.5f', '%.5f', '%.5f', '%.5f', '%.5f')
        tab = np.zeros(MODEL.g_delta.shape[0], dtype=cols)
        features = list([f for f in VX_FEATURES if f.chr == SINGLE_CHROMOSOME])
        tab["chr"] = [f.chr for f in features ]
        tab["begin"] = [f.begin for f in features ]
        tab["end"] = [f.end for f in features ]
        tab["length"] = [f.end - f.begin for f in features ]
        tab["name"] = [f.name for f in features ]
        tab["delayscale"] = [f.delayscale for f in features ]
        tab["g"] = MODEL.g.detach().numpy()
        tab["v_bps"] = MODEL.v0.detach().numpy() / MODEL.g.detach().numpy()
        tab["delay_bp"] = MODEL.g_delta.detach().numpy() * MODEL.g_xranges_len.detach().numpy()
        tab["delay_rel"] = tab["delay_bp"] / MODEL.genome_length
        tab["delay_s"] = tab["delay_bp"] / MODEL.v0.detach().numpy()
        # Write tab-separated to output file
        f = open_file(ARGS.out_vx_feature_delays, "wt+")
        f.write("\t".join([name for (name, type) in cols]) + "\n")
        np.savetxt(f, tab, delimiter="\t", fmt=fmt)
    except Exception as e:
        print(f"ERROR: Failed to save {ARGS.out_vx_feature_delays}: {e}")
    finally:
        if f is not None:
            f.close()

if ARGS.out_trajectories is not None:
    print(f"Saving velocity profile and replisome trajectories x(t) to {ARGS.out_trajectories}")
    f = None
    try:
        k = sum([1 for o in ORIGINS if o.chr == SINGLE_CHROMOSOME])
        # Construct column list and format tuple
        cols = [('t', np.float64), ('vt', np.float64), ('vt_stderr', np.float64) ]
        fmt = ('%.5f', '%.5f', '%.5f')
        cols += [ (f"{o.name}{d}", np.float64) for o in ORIGINS if o.chr == SINGLE_CHROMOSOME for d in [ "-", "+" ]]
        fmt += ('%d',) * (2*k)
        cols += [ (f"x.{o.name}{d}", np.float64) for o in ORIGINS if o.chr == SINGLE_CHROMOSOME for d in [ "-", "+" ]]
        fmt += ('%.5f',) * (2*k)
        cols += [ (f"dx.{o.name}{d}", np.float64) for o in ORIGINS if o.chr == SINGLE_CHROMOSOME for d in [ "-", "+" ]]
        fmt += ('%.5f',) * (2*k)
        cols += [ (f"v.{o.name}{d}", np.float64) for o in ORIGINS if o.chr == SINGLE_CHROMOSOME for d in [ "-", "+" ]]
        fmt += ('%.5f',) * (2*k)
        cols += [ (f"gx.{o.name}{d}", np.float64) for o in ORIGINS if o.chr == SINGLE_CHROMOSOME for d in [ "-", "+" ] ]
        fmt += ('%.5f',) * (2*k)
        # Fill columns, v(t)=v0*h(t) first
        tab = np.zeros(MODEL.h_tgrid.shape[0], dtype=cols)
        tab["t"] = MODEL.h_tgrid
        tab["vt"] = MODEL.v_t(MODEL.h_tgrid).detach()
        tab["vt_stderr"] = MODEL.v_t_posterior_stderr(MODEL.h_tgrid, COV_HG, COV_HG_PMAP).detach()
        # Now fill in x(t), v_r(t)=v0*h(t) / g(x) and g(x) for all replisomes
        tab_state, tab_x, tab_dx, tab_v = MODEL.trajectories(MODEL.h_tgrid)
        for o_i, o in enumerate(ORIGINS):
            if o.chr != SINGLE_CHROMOSOME:
                continue
            for d_i, d in enumerate([ "-", "+" ]):
                r_i = 2*o_i + d_i
                tab[f"{o.name}{d}"] =  tab_state[:, r_i].detach().numpy()
                tab[f"x.{o.name}{d}"] =  tab_x[:, r_i].detach().numpy()
                tab[f"dx.{o.name}{d}"] =  tab_dx[:, r_i].detach().numpy()
                tab[f"v.{o.name}{d}"] = tab_v[:, r_i].detach().numpy()
                tab[f"gx.{o.name}{d}"] = MODEL.g_fun(tab_x[:, r_i]).detach().numpy()
        # Write tab-separated to output file
        f = open_file(ARGS.out_trajectories, "wt+")
        f.write("\t".join([name for (name, type) in cols]) + "\n")
        np.savetxt(f, tab, delimiter="\t", fmt=fmt)
    except Exception as e:
        print(f"ERROR: Failed to save {ARGS.out_trajectories}: {e}")
    finally:
        if f is not None:
            f.close()


# Save fitting statistics
if ARGS.out_stats is not None:
    raise ArgumentError("Saving stats is not yet implemented")

# Plot v(t)
if ARGS.plot_vt is not None:
    try:
        print(f"Plotting v(t) to {ARGS.plot_vt}")
        with pt.no_grad():
            x_min = LOSS.x[pt.argmin(LOSS.y_pred())]
            t_fin = MODEL.Hinv(MODEL.T(x_min))
            t_max = max(t_fin, DOUBLINGTIME) * 1.2

            plot_t = MODEL.h_tgrid
            plot_vt = MODEL.v_t(MODEL.h_tgrid).detach()
            plot_vt_stderr = MODEL.v_t_posterior_stderr(MODEL.h_tgrid, posterior_cov=COV_HG, pmap=COV_HG_PMAP).detach()
            plot_yrange = [0.8 * pt.min(plot_vt[plot_t <= t_fin]).item(), pt.max(plot_vt[plot_t <= t_fin]).item() * 1.2]

        pp.clf()
        pp.figure(figsize=(2.5, 2.0))
        pp.ylim(plot_yrange)
        pp.axhline(y=MODEL.v0.detach().numpy(), linestyle="dotted", color="black", linewidth=0.5)
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
    finally:
        pass

# Plot abundance data and fitted model
if ARGS.plot_abundance is not None:
    try:
        print(f"Plotting data and fitted model to {ARGS.plot_abundance}")
        pp.clf()
        pp.figure(figsize=(2.5, 2.0))
        pp.plot(LOSS.x.detach().numpy(), LOSS.y_obs.detach().numpy(), marker="x", markersize=0.25, linestyle="", color="grey")
        pp.plot(LOSS.x.detach().numpy(), LOSS.y_pred().detach().numpy(), linewidth=0.5, color="navy")
        pp.tight_layout()
        pp.savefig(ARGS.plot_abundance, dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"ERROR: Failed to save {ARGS.plot_abundance}: {e}")
    finally:
        pass
