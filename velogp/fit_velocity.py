#!/usr/bin/env python3
from dataclasses import dataclass

import argparse
import gzip
import BCBio.GFF

import numpy as np
import torch
from torch import *
from builtins import float, int
import velocity_model
import estimate_velocity


class SplitAndAppend(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, getattr(namespace, self.dest, list()) + values.split(','))

@dataclass
class Feature:
    begin: int
    end: int
    delay: float
    name: int

# Argument handling
parser = argparse.ArgumentParser(
    prog="VeloGP",
    description="Infer spatio-temporal replisome velocity from MFA data")

parser.add_argument("inputfile",
                    help="abundance table")
parser.add_argument("--growthrate", default=1.0, type=float, metavar="lambda",
                    help="growthrate of sequenced population [1/hour]")
parser.add_argument("--origins", default=["oriC"], metavar="origin,origin,...", action=SplitAndAppend,
                    help="names or locations of origins of replication")
parser.add_argument("--vt-timescale", type=float, metavar="value",
                    help="timescale of temporal velocity fluctuations in units of doubling time (covariance length scale)")
parser.add_argument("--vt-magnitude", type=float, metavar="value",
                    help="magnitude of temporal velocity fluctuations (standard deviation of log-change)")
parser.add_argument("--vt-resolution", type=int, default=100, metavar="points",
                    help="number of points per timescale to use to represent v(t)")
parser.add_argument("--vt-tmax", type=float, default=3, metavar="tmax",
                    help="maximum time t for which to estimate v(t)")
parser.add_argument("--vx-magnitude", type=float, default=1.0, metavar="",
                    help="magnitude of temporal velocity fluctuations (???)")
parser.add_argument("--vx-delay", type=float, default=0.05, metavar="",
                    help="delay we can roughly expect to be incurred at a single vx feature")
parser.add_argument("--vx-magnitude", type=float, default=1.0, metavar="",
                    help="magnitude of deviations from expected delays at vx features (standard deviation of exponent)")
parser.add_argument("--vx-feature-names", default=[], nargs="+", metavar="feature,feature,...", action=SplitAndAppend,
                    help="add these features to list of sites at which the velocity is modulated")
parser.add_argument("--vx-feature-types", default=[], nargs="+", metavar="type,type,...", action=SplitAndAppend,
                    help="add all features with these types to list of sites at which the velocity is modulated")
parser.add_argument("--gff",  nargs="*", metavar="gff-file",
                    help="GFF file containing genome annotations")
parser.add_argument("--bed",  nargs="*", metavar="bed-file",
                    help="BED file containing genome annotations, must contain at least 4 columns (chrom,start,end,name)")
parser.add_argument("--columns", default="midpoint,coverage_ratio,coverage_ratio_stderr,ratio_mappable",
                    help="comma-separated names of columns containing position, abundance, std.error of abundance and validity")

# Parse arguments
ARGS = parser.parse_args()

# Origins of replication
# NOTE: Origins specified by name are resolved during GFF/BED file parsing
ORIGINS = [ int(origin) for origin in ARGS.origins if origin.isdigit() ]
ORIGINS_TO_BE_RESOLVED = [ origin for origin in ARGS.origins if not origin.isdigit() ]

# Genomic features at which replisomes might slow down
VX_FEATURES = list()

# Read GFF files
for gff_filename in ARGS.gff:
    if gff_filename.endswith(".gz"):
        gff_file = gzip.open(gff_filename, "rt")
    else:
        gff_file = open(gff_filename, "rt")
    # Recursively iterate through features in the GFF file
    # (NOTE: It's silly that we have to do this since GFF files are actually
    # tabular. But it's apparently the only API that BCBio provides...)
    stack = list()
    for record in BCBio.GFF.parse(gff_file):
        stack.extend(record.features)
        while len(stack) > 0:
            # Get next feature, push sub-features on stack to handle later
            feature = stack.pop()
            stack.extend(feature.sub_features)
            # Handle feature
            if (feature.type in ARGS.vx_feature_types) or (feature.name in ARGS.vx_feature_names):
                feature = Feature(begin=int(feature.location.start), end=int(feature.location.end),
                                  name=feature.name, delay=ARGS.vx_delay)
                # Make sure min < max
                if feature.min > feature.max:
                    feature.min, feature.max = feature.max, feature.min
                elif feature.min == feature.max:
                    feature.max += 1
                print(f"  spatial velocity modulation at {feature.name} ({feature.begin} - {feature.end})")
                VX_FEATURES.append(feature)
            if feature.name in ORIGINS_TO_BE_RESOLVED:
                position = int((feature.begin + feature.end)/2)
                ORIGINS.append(position)
                ORIGINS_TO_BE_RESOLVED.remove(feature.name)
                print(f"  origin {feature.name} ({position})")

# Read BED files
#for bed_filename in ARGS.bed:
#    raise RuntimeError("BED files are not yet supported")

# Read abundance data
ABUNDANCE = None
GENOME_LENGTH = 0

# Create model
DOUBLINGTIME =  3600 * math.log(2) / ARGS.growthrate
VX_FEATURES.sort(key=lambda f: f.begin)
MODEL = velocity_model.VelocityModel(
    genome_length=GENOME_LENGTH, is_circular=True,
    origins=tensor(sorted([ORIGINS]), dtype=torch.int32),
    h_tgrid=torch.arange(start=0, end=DOUBLINGTIME * ARGS.vt_tmax,
                         step=DOUBLINGTIME * ARGS.vt_timescale / ARGS.vt_resolution,
                         dtype=torch.float64),
    g_xranges_l=tensor([f.begin for f in VX_FEATURES], dtype=torch.int32),
    g_xranges_u=tensor([f.end for f in VX_FEATURES], dtype=torch.int32)
)
MODEL.growthrate[0] = ARGS.growthrate / 3600
MODEL.h_sigma[0] = ARGS.vt_magnitude
MODEL.g_sigma[0] = ARGS.vx_magnitude
MODEL.h_timescale[0] = DOUBLINGTIME * ARGS.vt_timescale

# Create loss function
LOSS = velocity_model.VelocityModelLoss(
    x=torch.tensor(ABUNDANCE["x"].values, dtype=torch.float64),
    y_obs=torch.tensor(ABUNDANCE["y_obs"].values, dtype=torch.float64),
    s_obs=torch.tensor(ABUNDANCE["s_obs"].values, dtype=torch.float64),
    model=MODEL)

# Create fit instance
FIT = estimate_velocity.FitVelocityModel(MODEL, LOSS)

# Estimate parameters
FIT.optimize_parameters()

# Compute posterior covariances of parameters
POSTERIOR = FIT.parameter_posterior_cov()
