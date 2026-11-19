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
import copy
import BCBio.GFF
import Bio.SeqIO
import numpy as np
import pandas as pd
import torch as pt
import matplotlib.pyplot as pp

from . import model
from . import fit
from . import errormodel

class ArgumentError(Exception):
    pass

# Report invalid arguments (e.g. origins or features that cannot be resolved) like argparse does,
# i.e. as an error message without a traceback
def excepthook(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, ArgumentError):
        print(f"VeloGP: error: {exc_value}", file=sys.stderr)
        sys.exit(2)
    sys.__excepthook__(exc_type, exc_value, exc_traceback)
sys.excepthook = excepthook

class SplitAndAdd(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, str):
            values = [ values ]
        for vs in values:
            for v in vs.split(","):
                if len(v) > 0:
                    if getattr(namespace, self.dest, None) is None:
                        setattr(namespace, self.dest, set())
                    getattr(namespace, self.dest).add(v)

def open_file(filename, mode):
    if filename.endswith(".gz"):
        return gzip.open(filename, mode)
    elif filename.endswith(".bz2"):
        return bz2.open(filename, mode)
    else:
        return open(filename, mode)

def window_index(seq: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """ Number windows consecutively in order of position within each chromosome """
    order = np.lexsort((np.asarray(pos), np.asarray(seq)))
    index = np.empty(len(order), dtype=np.int64)
    index[order] = np.arange(len(order))
    return index

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

# Origins used if none are specified with --origins. This is the GFF feature type of origins of replication
DEFAULT_ORIGINS = "origin_of_replication"

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
parser.add_argument("--plot-error-model", metavar="filename",
                    help="output file for a plot of the neighbour scatter of the abundances and the fitted error model "
                         "(the error model is fitted for this plot even without --error-model)")
parser.add_argument("--error-model", default=False, action="store_true",
                    help="account for overdispersion: estimate d in the error model stderr^2 = q^2 + d * abundance^2 "
                         "(q = std. error from the input) from the scatter between adjacent windows, and use the "
                         "resulting std. errors instead of q")
parser.add_argument("--growthrate", default=1.0, type=float, metavar="lambda",
                    help="growthrate of sequenced population [1/hour]")
parser.add_argument("--origins", default=None, metavar="origin,origin,...", action=SplitAndAdd,
                    help="origins of replication, specified by location ([chr:]pos), by name, or by GFF feature type. "
                         "A name must match exactly one feature, a type matches all features of that type, which are "
                         "named <type>1, ..., <type>N in order of chromosome name and position (even if they also "
                         "match a name). "
                         f"Default: {DEFAULT_ORIGINS}")
parser.add_argument("--v0-min", default=10, type=float,
                    help="minimum baseline velocity")
parser.add_argument("--v0-max", default=10000, type=float,
                    help="maximal baseline velocity")
parser.add_argument("--vt-covariance", choices=["se", "m0", "m1", "m2"], default="m1",
                    help="velocity covariance, determines smoothness degree d of v(t). For mK (=matern-K+1/2) d=K, for se (=squared-exp.) d=inf"),
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
parser.add_argument("--linear-chromosomes", default=set(), metavar="chromosome,chromosome,...", action=SplitAndAdd,
                    help="names of linear chromosomes, all other chromosomes are assumed to be circular")
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
parser.add_argument("--perturbation-rejects", type=int, default=5,
                    help="number of rejects required to reduce perturbation strength")
parser.add_argument("--perturbation-reduction", type=float, default=0.5**(1.0/4),
                    help="factor by which perturbation strength is reduced")
parser.add_argument("--perturbation-min", type=float, default=0.001,
                    help="perturbation strength after which to stop")


# Parse arguments
ARGS = parser.parse_args()

# Use the default origins only if no origins were specified
if not ARGS.origins:
    ARGS.origins = { DEFAULT_ORIGINS }

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

# Read genome to find chromosome names and lengths
CHROMOSOME_LENGTHS : Dict[str, int] = {}
fasta_file = open_file(ARGS.genome, "rt")
try:
    for seq in Bio.SeqIO.parse(fasta_file, "fasta"):
        CHROMOSOME_LENGTHS[seq.id] = len(seq)
    SINGLE_CHROMOSOME = next(iter(CHROMOSOME_LENGTHS.keys())) if len(CHROMOSOME_LENGTHS) == 1 else None
finally:
    fasta_file.close()
# Chromosomes in the order in which they appear in the genome, i.e. in the order used by the model
CHROMOSOME_NAMES : List[str] = list(CHROMOSOME_LENGTHS.keys())
for chr in ARGS.linear_chromosomes:
    if chr not in CHROMOSOME_LENGTHS:
        raise ArgumentError(f"linear chromosome {chr} does not exist in genome {ARGS.genome}")

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
    elif chr not in CHROMOSOME_LENGTHS:
        raise ArgumentError(f"origin {origin} lies on chromosome {chr} which does not exist in genome {ARGS.genome}")
    if (pos < 1) or (pos > CHROMOSOME_LENGTHS[chr]):
        raise ArgumentError(f"origin {origin} lies outside the range of chromosome {chr} [1, {CHROMOSOME_LENGTHS[chr]}]")
    # NOTE: Locations on the command-line are one-based
    ORIGINS.append(Origin(chr=chr, pos=pos-1, name=origin))
ORIGINS_NAMES_TO_BE_RESOLVED : Set[str] = ORIGINS_NAMES.copy()
# Origins found by GFF feature type (rather than by name) for each type, named once all GFF files have been read
ORIGINS_BY_TYPE : Dict[str, List[Origin]] = {}

# Genomic features at which replisomes might slow down
VX_FEATURES_TENTATIVE : List[Feature] = list()
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
            # this is just more API silliness. But anyway, we propagate that sequence
            # name down when we expand features and sub-features, since there seems to
            # be no way to get to it later otherwise...
            stack.extend([ (record.id, sf) for sf in record.features ])
            while len(stack) > 0:
                # Get next feature, push sub-features on stack to handle later
                # NOTE: The stack contains (seq, feature) pairs since we have to propagate
                # the sequence name downwards due to API silliness, see above...
                seq, f = stack.pop()
                stack.extend([ (seq, sf) for sf in f.sub_features ])
                # Generate list of "names" of the feature. Oh, GFF...
                f_names = [ f.id ]
                if "Name" in f.qualifiers:
                    f_names.extend(f.qualifiers["Name"])
                # The ID and Name(s) of a feature are often identical, but must count only once below
                f_names = list(dict.fromkeys(f_names))
                # Resolve feature names, add all features of requested types
                f_matching_feature_names = [ fn for fn in f_names if fn in ARGS.vx_feature_names]
                if (f.type in ARGS.vx_feature_types) or len(f_matching_feature_names) > 0:
                    # Named should be unambiguous
                    for fn in [fn for fn in f_matching_feature_names if fn not in VX_FEATURE_NAMES_TO_BE_RESOLVED]:
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
                    VX_FEATURES_TENTATIVE.append(feature)
                    # Removed from to-be-resolved list
                    for fn in f_matching_feature_names:
                        VX_FEATURE_NAMES_TO_BE_RESOLVED.remove(fn)
                # Resolve unresolved origin names, add all features of requested types
                f_matching_origin_names = [ fn for fn in f_names if fn in ORIGINS_NAMES]
                f_matching_origin_type = f.type in ORIGINS_NAMES
                if (len(f_matching_origin_names) > 0) or f_matching_origin_type:
                    # Named should be unambiguous
                    for fn in [fn for fn in f_matching_origin_names if fn not in ORIGINS_NAMES_TO_BE_RESOLVED]:
                        raise ArgumentError(f"feature name {fn} is ambiguous")
                    # Origins matched by type are named below, even if they are also matched by name. This keeps the
                    # numbering of the origins of a type independent of which of them are also requested by name.
                    origin = Origin(chr=seq, pos=int((f.location.start + f.location.end)/2),
                                    name=None if f_matching_origin_type else f_matching_origin_names[0])
                    ORIGINS.append(origin)
                    if f_matching_origin_type:
                        ORIGINS_BY_TYPE.setdefault(f.type, []).append(origin)
                    for fn in f_matching_origin_names:
                        ORIGINS_NAMES_TO_BE_RESOLVED.remove(fn)
                    # A type may match any number of features
                    ORIGINS_NAMES_TO_BE_RESOLVED.discard(f.type)
    finally:
        gff_file.close()

# Name origins found by type <type>1, ..., <type>N, ordered by chromosome name and position
for o_type, o_list in ORIGINS_BY_TYPE.items():
    for i, origin in enumerate(sorted(o_list, key=lambda o: (o.chr, o.pos))):
        origin.name = f"{o_type}{i + 1}"

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
            VX_FEATURES_TENTATIVE.append(feature)
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
    raise ArgumentError(f"unable to resolve origins {', '.join(sorted(ORIGINS_NAMES_TO_BE_RESOLVED))}: no feature in "
                        f"the GFF or BED files has this name or (GFF only) type")

# Sort features and origins
VX_FEATURES_TENTATIVE.sort(key=lambda f: (f.chr, f.begin))
ORIGINS.sort(key=lambda o: (o.chr, o.pos))

# Merge overlapping features
f_prev : Feature | None = None
for f in VX_FEATURES_TENTATIVE:
    if f_prev is not None and f_prev.chr == f.chr and f_prev.end > f.begin:
        # If f overlaps f_prev, extend f_prev and append name
        print(f"Merging vx feature {f.name} into previous feture {f_prev.name} due to overlap")
        f_prev.end = max(f_prev.end, f.end)
        f_prev.name = f"{f_prev.name},{f.name}"
    else:
        # Otherwise, add new feature
        f_prev = copy.copy(f)
        VX_FEATURES.append(f_prev)

# Check that every chromosome has an origin
for chr in CHROMOSOME_LENGTHS.keys():
    if len([o for o in ORIGINS if o.chr == chr]) == 0:
        raise ArgumentError(f"chromosome {chr} contains no origin")

# Origins and features must lie on chromosomes of the genome
for o in ORIGINS:
    if o.chr not in CHROMOSOME_LENGTHS:
        raise ArgumentError(f"origin {o.name} lies on sequence {o.chr} which does not exist in genome {ARGS.genome}")
for f in VX_FEATURES:
    if f.chr not in CHROMOSOME_LENGTHS:
        raise ArgumentError(f"vx feature {f.name} lies on sequence {f.chr} which does not exist in genome {ARGS.genome}")

# Read abundance data
# NOTE: Sequence names are read as strings even if they look like numbers, so that they match the genome
ABUNDANCE = pd.read_csv(ARGS.inputfile, delimiter="\t", dtype={COLUMNS["seq"]: str})
ABUNDANCE = ABUNDANCE.rename(columns={ fn: sn for sn, fn in COLUMNS.items() })
ABUNDANCE_ROWS_PREFILTER = len(ABUNDANCE)
# Number windows before filtering, so that the error model can tell which of the remaining windows are adjacent
ABUNDANCE["window"] = window_index(ABUNDANCE.seq.astype(str).values, ABUNDANCE.x.values)
# NOTE: A std. error of zero (e.g. for a window without any reads) would give the data point infinite weight
ABUNDANCE = ABUNDANCE.loc[~pd.isna(ABUNDANCE.x) & (ABUNDANCE.y_obs > 0) & (ABUNDANCE.s_obs > 0) & (ABUNDANCE.valid > 0)].copy()

# Fit the overdispersion error model on the remaining windows, using only pairs of adjacent windows that both remain
ERROR_MODEL = None
if ARGS.error_model or ARGS.plot_error_model:
    try:
        ERROR_MODEL = errormodel.fit(seq=ABUNDANCE.seq.astype(str).values, window=ABUNDANCE.window.values,
                                     y=ABUNDANCE.y_obs.values, q=ABUNDANCE.s_obs.values)
    except ValueError as e:
        raise ArgumentError(f"unable to fit error model: {e}")
    # Plot right away, since the plot only depends on the input data
    if ARGS.plot_error_model:
        print(f"Plotting error model to {ARGS.plot_error_model}")
        try:
            errormodel.plot(ARGS.plot_error_model, ERROR_MODEL, xlabel=COLUMNS["y_obs"], qlabel=COLUMNS["s_obs"])
        except Exception as e:
            print(f"ERROR: Failed to save {ARGS.plot_error_model}: {e}")
    if ARGS.error_model:
        ABUNDANCE["s_obs_poisson"] = ABUNDANCE.s_obs
        ABUNDANCE["s_obs"] = ERROR_MODEL.stderr(ABUNDANCE.y_obs.values, ABUNDANCE.s_obs.values)

# Determine the chromosome of each data point
ABUNDANCE_UNKNOWN_SEQ = ~ABUNDANCE.seq.isin(CHROMOSOME_NAMES)
if ABUNDANCE_UNKNOWN_SEQ.any():
    unknown = ", ".join(sorted(set(ABUNDANCE.seq[ABUNDANCE_UNKNOWN_SEQ].astype(str))))
    if SINGLE_CHROMOSOME is not None:
        print(f"WARNING: sequence(s) {unknown} in {ARGS.inputfile} do not exist in genome {ARGS.genome}, "
              f"assuming that all data points lie on chromosome {SINGLE_CHROMOSOME}")
        ABUNDANCE.loc[ABUNDANCE_UNKNOWN_SEQ, "seq"] = SINGLE_CHROMOSOME
    else:
        raise ArgumentError(f"sequence(s) {unknown} in {ARGS.inputfile} do not exist in genome {ARGS.genome}")
ABUNDANCE_CHROMOSOME = pt.tensor(ABUNDANCE.seq.map({ chr: i for i, chr in enumerate(CHROMOSOME_NAMES) }).values,
                                 dtype=pt.long)
for chr, l in CHROMOSOME_LENGTHS.items():
    outside = (ABUNDANCE.seq == chr) & ((ABUNDANCE.x < 0) | (ABUNDANCE.x > l))
    if outside.any():
        raise ArgumentError(f"{outside.sum()} data point(s) lie outside the range [0, {l}] of chromosome {chr}")

# Set random seet
if ARGS.seed is not None:
    pt.manual_seed(ARGS.seed)
    random.seed(ARGS.seed)

# Create model
DOUBLINGTIME = 3600 * math.log(2) / ARGS.growthrate
MODEL = model.VelocityModel(
    model.TemporalModel(
        h_tgrid=pt.arange(start=0, end=DOUBLINGTIME * ARGS.vt_tmax,
                          step=DOUBLINGTIME * ARGS.vt_timescale / ARGS.vt_resolution,
                          dtype=pt.float64),
        v0_min=model.dtensor(ARGS.v0_min), v0_max=model.dtensor(ARGS.v0_max)),
    [ model.ChromosomeModel(
        name=chr, length=CHROMOSOME_LENGTHS[chr], is_circular=chr not in ARGS.linear_chromosomes,
        origins=pt.tensor([o.pos for o in ORIGINS if o.chr == chr], dtype=pt.int32),
        origin_names=[o.name for o in ORIGINS if o.chr == chr],
        g_xranges_l=pt.tensor([f.begin for f in VX_FEATURES if f.chr == chr], dtype=pt.int32),
        g_xranges_u=pt.tensor([f.end for f in VX_FEATURES if f.chr == chr], dtype=pt.int32),
        g_delayscale=pt.tensor([f.delayscale for f in VX_FEATURES if f.chr == chr], dtype=pt.float64))
      for chr in CHROMOSOME_NAMES ])
MODEL.temporal.growthrate[0] = ARGS.growthrate / 3600
MODEL.temporal.h_cov_function = ARGS.vt_covariance
MODEL.temporal.h_sigma[0] = ARGS.vt_magnitude
MODEL.temporal.h_timescale[0] = DOUBLINGTIME * ARGS.vt_timescale

# Create loss function
LOSS = model.VelocityModelLoss(
    x=pt.tensor(ABUNDANCE["x"].values, dtype=pt.float64),
    y_obs=pt.tensor(ABUNDANCE["y_obs"].values, dtype=pt.float64),
    s_obs=pt.tensor(ABUNDANCE["s_obs"].values, dtype=pt.float64),
    model=MODEL,
    params=(["temporal.v0_log"] + [f"chromosomes.{i}.delay_scaled" for i in range(len(MODEL.chromosomes))] +
            ["temporal.h_log_raw"] + [f"chromosomes.{i}.g_delta_scaled" for i in range(len(MODEL.chromosomes))]),
    chromosome=ABUNDANCE_CHROMOSOME)

# Create fit instance
ALG = fit.FitVelocityModel(MODEL, LOSS)

# Print summary of settings
print(f"*** SETTINGS")
print(f"Random seed: {ARGS.seed}")
print(f"Number of cores: {pt.get_num_threads()}")
print(f"Perturbation: reduce by {ARGS.perturbation_reduction:.5f} after {ARGS.perturbation_rejects} rejects")
print(f"Genome configuration:")
for chr, l in CHROMOSOME_LENGTHS.items():
    print(f"  chromosome {chr}: {l}bp, {'linear' if chr in ARGS.linear_chromosomes else 'circular'}")
    origins = [f"{o.pos+1} ({o.name})" for o in ORIGINS if o.chr == chr]
    print(f"    origins: {', '.join(origins)}")
    print(f"    allow for spatial velocity modulation at:")
    for f in [f for f in VX_FEATURES if f.chr == chr]:
        print(f"      {f.begin+1}-{f.end+1} ({f.name}): {f.end-f.begin}bp, delayscale {f.delayscale}")

print(f"Parameters:")
print(f"  growth rate: {ARGS.growthrate} [1/h]")
print(f"  v(t) covariance: {ARGS.vt_covariance} "
      f"(makes v(t) { {'se': math.inf, 'm0': 0, 'm1': 1, 'm2': 2}[ARGS.vt_covariance] } times continuously differentiable)")
print(f"  v(t) timescale: {MODEL.temporal.h_timescale[0]:.0f} [s] (= {ARGS.vt_timescale} doublingtimes)")
print(f"  log v(t) std. err: {ARGS.vt_magnitude}")
print(f"  v(x) delay scale: {ARGS.vx_delayscale}")

print("Data:")
print(f"  data points: {len(ABUNDANCE)} ({100.0*len(ABUNDANCE)/ABUNDANCE_ROWS_PREFILTER:.0f}% of {ABUNDANCE_ROWS_PREFILTER} points pre-filtering)")
if len(CHROMOSOME_NAMES) > 1:
    for i, chr in enumerate(CHROMOSOME_NAMES):
        n = (LOSS.chromosome == i).sum().item()
        print(f"  data points on chromosome {chr}: {n}" + (" (WARNING: no data)" if n == 0 else ""))
print(f"  mean value: {LOSS.y_obs_mean.item():.3f}")
if ERROR_MODEL is not None:
    print(f"  error model stderr^2 = q^2 + d * abundance^2 {'(used)' if ARGS.error_model else '(NOT used, no --error-model)'}:")
    print(f"    fitted on {ERROR_MODEL.n} pairs of adjacent windows")
    print(f"    overdispersion d: {ERROR_MODEL.d:.4g} (relative std. error sqrt(d) = {math.sqrt(ERROR_MODEL.d):.4g})")
    print(f"    mean pair variance / q^2: {ERROR_MODEL.mean_ratio_poisson:.3f}, "
          f"/ (q^2 + d * abundance^2): {ERROR_MODEL.mean_ratio_model:.3f}")
print(f"  mean weight: {pt.mean(LOSS.weights).item():.4g}")

#print("*** COMPILING MODELS")
#MODEL.compile()
#LOSS.compile()

# Estimate parameters
print("*** FITTING MODEL")
#ALG.optimize_parameters(log_phases=True, log_steps=ARGS.verbose)
ALG.optimize_parameters_stochastic(perturb_reduce=ARGS.perturbation_reduction,
                                   perturb_rejects=ARGS.perturbation_rejects,
                                   perturb_end=ARGS.perturbation_min,
                                   log_phases=True, log_rounds=True,
                                   log_steps=ARGS.verbose, log_params=ARGS.verbose)

# Compute posterior covariances of parameters
print("*** COMPUTING POSTERIOR DISTRIBUTIONS")
COV_V0, COV_V0_PMAP = MODEL.posterior_cov(LOSS, ["temporal.v0_log"])
COV_HG, COV_HG_PMAP = MODEL.posterior_cov(LOSS, ["temporal.h_log_raw"] + [f"chromosomes.{i}.g_delta_scaled" for i in range(len(MODEL.chromosomes))])

# Save results
print("*** SAVING RESULTS")

# Save fitted model
if (ARGS.out_model is not None) and (ARGS.out_model != ""):
    print(f"Saving fitted model to {ARGS.out_model}")
    try:
        pt.save(MODEL, ARGS.out_model)
    except Exception as e:
        print(f"ERROR: Failed to save {ARGS.out_model}: {e}")

# Save all results
if (ARGS.out_objects is not None) and (ARGS.out_objects != ""):
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
if (ARGS.out_abundance is not None) and (ARGS.out_abundance != ""):
    print(f"Saving observed and predicted abundances to {ARGS.out_abundance}")
    f = None
    try:
        # Get model predictions
        p_pred, scale = LOSS.abundance_and_scale()
        # Build table
        slen = max(len(chr) for chr in CHROMOSOME_NAMES)
        cols = [ ('chr', f'U{slen}'), ('pos', np.int32), ('y_obs', np.float64), ('y_obs_stderr', np.float64),
                 ('y_pred', np.float64), ('p_pred', np.float64) ]
        fmt = ('%s', '%d', '%.5f', '%.5f', '%.5f', '%.5f' )
        tab = np.zeros(LOSS.x.shape[0], dtype=cols)
        tab["chr"] = ABUNDANCE["seq"].values
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
if (ARGS.out_vt is not None) and (ARGS.out_vt != ""):
    print(f"Saving v(t) to {ARGS.out_vt}")
    f = None
    try:
        # Build table
        cols = [('t', np.float64), ('vt', np.float64), ('vt_stderr', np.float64)]
        fmt = ('%.5f', '%.5f', '%.5f')
        tab = np.zeros(MODEL.temporal.h_tgrid.shape[0], dtype=cols)
        tab["t"] = MODEL.temporal.h_tgrid
        tab["vt"] = MODEL.temporal.v_t(MODEL.temporal.h_tgrid).detach()
        tab["vt_stderr"] = MODEL.v_t_posterior_stderr(MODEL.temporal.h_tgrid, posterior_cov=COV_HG, pmap=COV_HG_PMAP).detach()
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
if (ARGS.out_vx_feature_delays is not None) and (ARGS.out_vx_feature_delays != ""):
    print(f"Saving v(x) to {ARGS.out_vx_feature_delays}")
    f = None
    try:
        # Check that we agree on the order of the genomic features...
        features = [f for c in MODEL.chromosomes for f in VX_FEATURES if f.chr == c.name]
        for c in MODEL.chromosomes:
            assert all(c.g_xranges_l == pt.tensor([f.begin for f in VX_FEATURES if f.chr == c.name], dtype=pt.int32))
        # Build table
        chr_slen = max([len("chr")] + [ len(f.chr) for f in VX_FEATURES ])
        fn_slen = max([len("feature")] + [ len(f.name) for f in VX_FEATURES ])
        cols = [ ('chr', f'U{chr_slen}'), ('begin', np.int32), ('end', np.int32), ('length', np.int32),
                 ('name', f'U{fn_slen}'), ('delayscale', np.float64),
                 ('g', np.float64), ('delay_bp', np.float64), ('delay_rel', np.float64),
                 ('time_s', np.float64), ('delay_s', np.float64) ]
        fmt = ('%s', '%d', '%d', '%d', '%s', '%.2f', '%.5f', '%.5f', '%.5f', '%.5f', '%.5f')
        tab = np.zeros(len(features), dtype=cols)
        tab["chr"] = [f.chr for f in features ]
        tab["begin"] = [f.begin for f in features ]
        tab["end"] = [f.end for f in features ]
        tab["length"] = [f.end - f.begin for f in features ]
        tab["name"] = [f.name for f in features ]
        tab["delayscale"] = [f.delayscale for f in features ]
        tab["g"] = pt.cat([c.g for c in MODEL.chromosomes]).detach().numpy()
        tab["delay_bp"] = pt.cat([c.g_delta * c.g_xranges_len for c in MODEL.chromosomes]).detach().numpy()
        # Time replisomes spend passing each feature, int g(x) / v(t(x)) dx, and the part of it caused by
        # the feature stalling replisomes, int (g(x) - 1) / v(t(x)) dx, where t(x) is the replication time of x
        with pt.no_grad():
            time_s, delay_s = MODEL.g_range_passage_times()
            cycle_time = MODEL.completion_time()
        tab["time_s"] = time_s.numpy()
        tab["delay_s"] = delay_s.numpy()
        # Delays relative to the duration of the replication cycle
        tab["delay_rel"] = tab["delay_s"] / cycle_time.item()
        # Write tab-separated to output file
        f = open_file(ARGS.out_vx_feature_delays, "wt+")
        f.write("\t".join([name for (name, type) in cols]) + "\n")
        np.savetxt(f, tab, delimiter="\t", fmt=fmt)
    except Exception as e:
        print(f"ERROR: Failed to save {ARGS.out_vx_feature_delays}: {e}")
    finally:
        if f is not None:
            f.close()

if (ARGS.out_trajectories is not None) and (ARGS.out_trajectories != ""):
    print(f"Saving velocity profile and replisome trajectories x(t) to {ARGS.out_trajectories}")
    f = None
    try:
        # Origins in the order used by the model, MODEL.trajectories() returns two replisomes per origin
        TRAJ_ORIGINS = [(c, name) for c in MODEL.chromosomes for name in c.origin_names]
        k = len(TRAJ_ORIGINS)
        # Construct column list and format tuple
        cols = [('t', np.float64), ('v0', np.float64), ('vt', np.float64), ('vt_stderr', np.float64) ]
        fmt = ('%.5f', '%.5f', '%.5f', '%.5f')
        cols += [ (f"{name}{d}", np.float64) for c, name in TRAJ_ORIGINS for d in [ "-", "+" ]]
        fmt += ('%d',) * (2*k)
        cols += [ (f"x.{name}{d}", np.float64) for c, name in TRAJ_ORIGINS for d in [ "-", "+" ]]
        fmt += ('%.5f',) * (2*k)
        cols += [ (f"dx.{name}{d}", np.float64) for c, name in TRAJ_ORIGINS for d in [ "-", "+" ]]
        fmt += ('%.5f',) * (2*k)
        cols += [ (f"v.{name}{d}", np.float64) for c, name in TRAJ_ORIGINS for d in [ "-", "+" ]]
        fmt += ('%.5f',) * (2*k)
        cols += [ (f"gx.{name}{d}", np.float64) for c, name in TRAJ_ORIGINS for d in [ "-", "+" ] ]
        fmt += ('%.5f',) * (2*k)
        # Fill columns, v(t)=v0*h(t) first
        tab = np.zeros(MODEL.temporal.h_tgrid.shape[0], dtype=cols)
        tab["t"] = MODEL.temporal.h_tgrid.detach().numpy()
        tab["v0"] = MODEL.temporal.v0.detach().numpy()
        tab["vt"] = MODEL.temporal.v_t(MODEL.temporal.h_tgrid).detach().numpy()
        tab["vt_stderr"] = MODEL.v_t_posterior_stderr(MODEL.temporal.h_tgrid, posterior_cov=COV_HG, pmap=COV_HG_PMAP).detach()
        # Now fill in x(t), v_r(t)=v0*h(t) / g(x) and g(x) for all replisomes
        tab_state, tab_x, tab_dx, tab_v = MODEL.trajectories(MODEL.temporal.h_tgrid)
        for o_i, (c, name) in enumerate(TRAJ_ORIGINS):
            for d_i, d in enumerate([ "-", "+" ]):
                r_i = 2*o_i + d_i
                tab[f"{name}{d}"] =  tab_state[:, r_i].detach().numpy()
                tab[f"x.{name}{d}"] =  tab_x[:, r_i].detach().numpy()
                tab[f"dx.{name}{d}"] =  tab_dx[:, r_i].detach().numpy()
                tab[f"v.{name}{d}"] = tab_v[:, r_i].detach().numpy()
                tab[f"gx.{name}{d}"] = c.g_fun(tab_x[:, r_i]).detach().numpy()
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
if (ARGS.plot_vt is not None) and (ARGS.plot_vt != ""):
    try:
        print(f"Plotting v(t) to {ARGS.plot_vt}")
        with pt.no_grad():
            # Time at which the last observed position (i.e. the one with the lowest abundance) is replicated
            t_fin = MODEL.temporal.Hinv(MODEL.T(LOSS.x_prep[pt.argmin(LOSS.y_pred())]))
            t_max = max(t_fin, DOUBLINGTIME) * 1.2

            plot_t = MODEL.temporal.h_tgrid
            plot_vt = MODEL.temporal.v_t(MODEL.temporal.h_tgrid).detach()
            plot_vt_stderr = MODEL.v_t_posterior_stderr(MODEL.temporal.h_tgrid, posterior_cov=COV_HG, pmap=COV_HG_PMAP).detach()
            plot_yrange = [0.8 * pt.min(plot_vt[plot_t <= t_fin]).item(), pt.max(plot_vt[plot_t <= t_fin]).item() * 1.2]

        pp.clf()
        pp.figure(figsize=(2.5, 2.0))
        pp.ylim(plot_yrange)
        pp.axhline(y=MODEL.temporal.v0.detach().numpy(), linestyle="dotted", color="black", linewidth=0.5)
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
if (ARGS.plot_abundance is not None) and (ARGS.plot_abundance != ""):
    try:
        print(f"Plotting data and fitted model to {ARGS.plot_abundance}")
        # One panel per chromosome
        pp.clf()
        n = len(MODEL.chromosomes)
        fig, axes = pp.subplots(1, n, figsize=(2.5 * n, 2.0), squeeze=False, sharey=True)
        with pt.no_grad():
            plot_y_pred = LOSS.y_pred()
        for i, c in enumerate(MODEL.chromosomes):
            sel = LOSS.chromosome == i
            ax = axes[0, i]
            ax.plot(LOSS.x[sel].numpy(), LOSS.y_obs[sel].numpy(), marker="x", markersize=0.25, linestyle="", color="grey")
            ax.plot(LOSS.x[sel].numpy(), plot_y_pred[sel].numpy(), linewidth=0.5, color="navy")
            if n > 1:
                ax.set_title(c.name, fontsize="small")
        fig.tight_layout()
        pp.savefig(ARGS.plot_abundance, dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"ERROR: Failed to save {ARGS.plot_abundance}: {e}")
    finally:
        pass
