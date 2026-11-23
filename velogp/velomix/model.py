"""
VeloMix: a mixture model of stochastic origin firing and replisome stalling built on a VeloGP fit

The deterministic VeloGP model (`velogp.model.VelocityModel`) fires every origin after a fitted delay and slows
replisomes down at genomic features (g ranges) by a fitted factor g. `MixtureModel` replaces two of these
deterministic effects by stochastic events:

1. Origins fail to fire. On each chromosome, the origins with the smallest delay in the original fit are fixed to
   fire. Every other origin fires with probability p_fire, independently of all other origins. All origins of a
   chromosome which fire do so at the same time, the chromosome delay D_c (D_c = 0 for genomes comprising a single
   chromosome).
2. Replisomes stall indefinitely at specified stall sites (a subset of the g ranges). Each stall site stalls with
   probability p_stall, independently of all other events. A stalling site blocks replisomes moving in the
   directions given by its polarity: "both" blocks both directions, "co" replisomes moving along the strand of the
   feature, "head" replisomes moving against it. Replisomes moving in the other direction pass the site normally.
   Stall sites otherwise cause no delay, i.e. their g is fixed to 1.

The predicted abundance is the mixture over all 2^n combinations (components) of the n events, each weighted by
the product of the probabilities of its events. v(t), the g of the remaining g ranges and the chromosome delays
are shared by all components.

Each component is evaluated by the existing VeloGP code: for each pattern of firing origins, `MixtureModel` holds a
`VelocityModel` whose chromosomes contain only the firing origins, and each component sets the directional block
masks (`ChromosomeModel.g_blocked_fwd`, `g_blocked_rev`) of its stalling sites on the model of its firing pattern.
These models share the parameters of the *master* model, i.e. of the `MixtureModel` itself, whose chromosomes
contain all origins and no blocks. The master model represents the reference component in which every origin fires
and no site stalls, so the methods inherited from `VelocityModel` (`T()`, `trajectories()`, `completion_time()`,
`g_range_*()`) refer to that component. `P()` and `forward()` return the mixture.
"""
from __future__ import annotations

import copy
import dataclasses
import itertools
import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn, Tensor

from ..model import VelocityModel, ChromosomeModel, TemporalModel, dtensor, STATE_VERSION

# Directions blocked by a stalling site, as a function of its polarity and its strand
POLARITIES = ("both", "co", "head")


def _assign(module: nn.Module, name: str, value: Tensor):
    """
    Assigns a tensor to an attribute of a module, whether or not the attribute currently holds an `nn.Parameter`

    `nn.Module` refuses to replace a registered parameter by a plain tensor, so the parameter is unregistered first.
    """
    if (name in module._parameters) and not isinstance(value, nn.Parameter):
        del module._parameters[name]
    setattr(module, name, value)


@dataclass(frozen=True)
class Event:
    """ A stochastic event of the mixture model, i.e. a non-fixed origin or a stall site """
    # "origin" or "stall"
    kind: str
    # Index of the chromosome within the model, and the chromosome's name
    chromosome: int
    chromosome_name: str
    # Index of the origin respectively g range within its chromosome
    index: int
    name: str
    # Location of the origin respectively range [begin, end) of the stall site
    begin: int
    end: int
    # Polarity ("both", "co", "head") and strand (+1, -1, 0) of a stall site, None for origins
    polarity: str | None = None
    strand: int | None = None

    def blocked_directions(self) -> Tuple[int, ...]:
        """ The directions (+1, -1) of the replisomes blocked by a stall site if it stalls """
        assert self.kind == "stall"
        if self.polarity == "both":
            return (+1, -1)
        elif self.polarity == "co":
            return (self.strand,)
        else:
            return (-self.strand,)


class MixtureEvents(nn.Module):
    """ The parameters specific to the mixture model, i.e. the event probabilities and the chromosome delays """

    def __init__(self, p_fire: Tensor, p_stall: Tensor, chromosome_delay_scaled: Tensor):
        super().__init__()
        with torch.no_grad():
            # Firing probabilities of the non-fixed origins, stalling probabilities of the stall sites, both in
            # event order (see `MixtureModel.events_list`)
            self.p_fire = p_fire.double().clone()
            self.p_stall = p_stall.double().clone()
            # Firing times of the origins of each chromosome, scaled like `ChromosomeModel.delay_scaled`
            self.chromosome_delay_scaled = chromosome_delay_scaled.double().clone()

    def owns(self, parameter: Tensor) -> bool:
        return (parameter is self.p_fire) or (parameter is self.p_stall) or (parameter is self.chromosome_delay_scaled)

    def clamp_parameters(self):
        """ Clamps probabilities to [0, 1] and chromosome delays to be non-negative with minimum zero """
        with torch.no_grad():
            for p in (self.p_fire, self.p_stall):
                assert torch.all(torch.isfinite(p))
                p.clamp_(0.0, 1.0)
            d = self.chromosome_delay_scaled
            assert torch.all(torch.isfinite(d))
            d.clamp_(0.0, math.inf)
            d -= torch.min(d)

    def prior(self, parameter, hyperparameter_dependence: bool):
        """ The event probabilities have no prior, the chromosome delays have none in VeloGP either """
        assert self.owns(parameter)
        return torch.zeros([], dtype=torch.double)

    def prior_precision(self, parameter) -> Tensor | None:
        assert self.owns(parameter)
        return None

    def sample_prior(self, parameter):
        """ Samples used to perturb parameters during the stochastic search, see `FitVelocityModel.run_stochastic()` """
        if (parameter is self.p_fire) or (parameter is self.p_stall):
            return torch.rand(parameter.shape, dtype=torch.double)
        elif parameter is self.chromosome_delay_scaled:
            # Same distribution as for ChromosomeModel.delay_scaled
            return torch.distributions.exponential.Exponential(dtensor(1.0)).sample(parameter.shape)
        raise RuntimeError("no prior distribution defined for this parameter")


class MixtureModel(VelocityModel):
    """ Mixture model of stochastic origin firing and replisome stalling, see the module documentation """

    @dataclass
    class X:
        """ Pre-processed input, i.e. the pre-processed input of the model of each firing pattern """
        shape: torch.Size
        parts: List[VelocityModel.X]

        def __getitem__(self, k) -> MixtureModel.X:
            parts = [p[k] for p in self.parts]
            return MixtureModel.X(parts[0].shape, parts)

    def __init__(self, original: VelocityModel, stall_sites: List[Tuple[str, str]]):
        """
        Creates a mixture model from a (fitted) VeloGP model

        :param original: The VeloGP model, it is copied and remains unchanged
        :param stall_sites: List of (name, polarity) pairs, where name is the name of a g range (see
        `ChromosomeModel.g_names`) and polarity one of "both", "co", "head"
        """
        original = copy.deepcopy(original)
        if any(c.g_names is None for c in original.chromosomes) and stall_sites:
            raise ValueError("the model predates names of g ranges (vx features), re-run VeloGP to use stall sites")
        temporal = original.temporal
        with torch.no_grad():
            # The blocking probability equivalent to the delay of each g range in the original model is used as
            # the initial stalling probability, see below
            p_block = original.g_range_block_probability()
            p_block_offsets = [0]
            for c in original.chromosomes:
                p_block_offsets.append(p_block_offsets[-1] + c.g_xranges_l.numel())
            # Origin delays of the original model, the chromosomes' delays are replaced by the chromosome delays below
            delays = [c.delay(temporal).detach().clone() for c in original.chromosomes]
            delays_scaled = [c.delay_scaled.detach().clone() for c in original.chromosomes]

        # Fixed origins: on each chromosome, all origins whose delay equals the smallest delay of the chromosome.
        # NOTE: VeloGP clamps delays such that they are exactly zero, comparing for equality is thus intended.
        fixed_origins = [torch.nonzero(d == torch.min(d)).reshape(-1).tolist() for d in delays_scaled]

        self._build(temporal, list(original.chromosomes), stall_sites, fixed_origins)

        # Initial parameters
        with torch.no_grad():
            # Firing probabilities from the delays relative to the fixed origins of the chromosome, p = exp(-lambda d)
            # is the fraction of cells in which the origin fired before being replicated passively, roughly
            lam = temporal.growthrate
            for k, e in enumerate(self.origin_events):
                d = delays[e.chromosome]
                self.events.p_fire[k] = torch.exp(-lam * (d[e.index] - torch.min(d))).reshape(())
            # Stalling probabilities from the equivalent blocking probabilities
            for k, e in enumerate(self.stall_events):
                p = p_block[p_block_offsets[e.chromosome] + e.index]
                self.events.p_stall[k] = 0.1 if torch.isnan(p) else p
            # Chromosome delays from the delays of the fixed origins
            self.events.chromosome_delay_scaled.copy_(torch.stack([torch.min(d) for d in delays_scaled]))
        self.clamp_parameters()

    def _build(self, temporal: TemporalModel, chromosomes: List[ChromosomeModel], stall_sites: List[Tuple[str, str]],
               fixed_origins: List[List[int]]):
        """
        Sets up the structure of the model, i.e. everything except the values of the event parameters

        Used by the constructor and when unpickling.

        :param temporal: The time-dependent part of the model
        :param chromosomes: The chromosomes of the master model
        :param stall_sites: The stall sites, see `__init__()`
        :param fixed_origins: For each chromosome, the indices of its fixed origins
        """
        VelocityModel.__init__(self, temporal, chromosomes)
        self.fixed_origins: List[List[int]] = [list(f) for f in fixed_origins]

        # Resolve stall sites
        sites: List[Event] = []
        seen = set()
        for name, polarity in stall_sites:
            if polarity not in POLARITIES:
                raise ValueError(f"invalid polarity {polarity} of stall site {name}, must be one of {', '.join(POLARITIES)}")
            matches = [(ci, j) for ci, c in enumerate(self.chromosomes) for j, n in enumerate(c.g_names) if n == name]
            if len(matches) == 0:
                raise ValueError(f"stall site {name} is not a feature of the model")
            if len(matches) > 1:
                raise ValueError(f"stall site {name} is ambiguous, it matches {len(matches)} features")
            if matches[0] in seen:
                raise ValueError(f"stall site {name} is specified more than once")
            seen.add(matches[0])
            ci, j = matches[0]
            c = self.chromosomes[ci]
            strand = int(c.g_strands[j].item())
            if (polarity != "both") and (strand == 0):
                raise ValueError(f"stall site {name} is unstranded, polarity {polarity} requires a strand")
            sites.append(Event("stall", ci, c.name, j, name, int(c.g_xranges_l[j].item()),
                               int(c.g_xranges_u[j].item()), polarity, strand))

        # Events in a deterministic order: origins first, then stall sites, each sorted by chromosome name and
        # position
        origins = [Event("origin", ci, c.name, i, c.origin_names[i], int(c.origins[i].item()), int(c.origins[i].item()))
                   for ci, c in enumerate(self.chromosomes) for i in range(c.origins.numel())
                   if i not in self.fixed_origins[ci]]
        origins.sort(key=lambda e: (e.chromosome_name, e.begin))
        sites.sort(key=lambda e: (e.chromosome_name, e.begin))
        self.origin_events: List[Event] = origins
        self.stall_events: List[Event] = sites
        self.events_list: List[Event] = origins + sites

        self.events = MixtureEvents(torch.zeros(len(origins), dtype=torch.double),
                                    torch.zeros(len(sites), dtype=torch.double),
                                    torch.zeros(len(self.chromosomes), dtype=torch.double))

        # Stall sites cause no delay of their own, they only stall replisomes
        self.stall_ranges: List[Tensor] = []
        for ci, c in enumerate(self.chromosomes):
            mask = torch.zeros(c.g_xranges_l.numel(), dtype=torch.bool)
            for e in sites:
                if e.chromosome == ci:
                    mask[e.index] = True
            self.stall_ranges.append(mask)
            # Origin delays are determined by the chromosome delays, see bind()
            _assign(c, "delay_scaled", c.delay_scaled.detach().clone())

        # Components: one for each combination of events, labelled by the event states in event order (for
        # origins 1 = fires, for stall sites 1 = stalls)
        n_o, n = len(origins), len(origins) + len(sites)
        bits = list(itertools.product((0, 1), repeat=n))
        self.component_bits = torch.tensor(bits, dtype=torch.bool).reshape(len(bits), n)
        # Firing patterns: one model per combination of firing origins, in which the chromosomes contain the fixed
        # origins and the firing non-fixed origins. These models are not sub-modules, they share the parameters of
        # this model, see bind().
        self.patterns: List[VelocityModel] = []
        self.pattern_origins: List[List[List[int]]] = []
        for fire in itertools.product((0, 1), repeat=n_o):
            firing = [sorted(self.fixed_origins[ci] + [e.index for e, f in zip(origins, fire) if f and e.chromosome == ci])
                      for ci in range(len(self.chromosomes))]
            self.pattern_origins.append(firing)
            self.patterns.append(VelocityModel(temporal, [self._pattern_chromosome(c, firing[ci])
                                                          for ci, c in enumerate(self.chromosomes)]))
        # Firing pattern and directional block masks of each component
        self.component_pattern: List[int] = []
        self.component_blocked_fwd: List[List[Tensor]] = []
        self.component_blocked_rev: List[List[Tensor]] = []
        for b in bits:
            self.component_pattern.append(int("".join(map(str, b[:n_o])) or "0", 2))
            fwd = [torch.zeros(c.g_xranges_l.numel(), dtype=torch.bool) for c in self.chromosomes]
            rev = [torch.zeros(c.g_xranges_l.numel(), dtype=torch.bool) for c in self.chromosomes]
            for e, s in zip(sites, b[n_o:]):
                if s:
                    if +1 in e.blocked_directions():
                        fwd[e.chromosome][e.index] = True
                    if -1 in e.blocked_directions():
                        rev[e.chromosome][e.index] = True
            self.component_blocked_fwd.append(fwd)
            self.component_blocked_rev.append(rev)


    def __getstate__(self) -> dict:
        """
        The state stored when pickling, see "Pickling" in velogp/model.py

        Comprises the master model's temporal model and chromosomes, the stall sites, the fixed origins and the event
        parameters. The firing pattern models are rebuilt when unpickling.
        """
        names = ["p_fire", "p_stall", "chromosome_delay_scaled"]
        return dict(version=STATE_VERSION, temporal=self.temporal, chromosomes=list(self.chromosomes),
                    stall_sites=[(e.name, e.polarity) for e in self.stall_events], fixed_origins=self.fixed_origins,
                    **{n: getattr(self.events, n).detach() for n in names},
                    parameters=[n for n in names if isinstance(getattr(self.events, n), nn.Parameter)],
                    training=self.training)

    def __setstate__(self, state: dict):
        """ Restores a pickled instance, see "Pickling" in velogp/model.py """
        if state["version"] > STATE_VERSION:
            raise RuntimeError(f"cannot unpickle MixtureModel: the pickle was written by a newer version of VeloGP")
        self._build(state["temporal"], state["chromosomes"], state["stall_sites"], state["fixed_origins"])
        with torch.no_grad():
            for n in ["p_fire", "p_stall", "chromosome_delay_scaled"]:
                v = state[n].detach().clone()
                setattr(self.events, n, nn.Parameter(v) if n in state["parameters"] else v)
            self.bind()
        self.training = state["training"]

    @staticmethod
    def _pattern_chromosome(c: ChromosomeModel, origins: List[int]) -> ChromosomeModel:
        """ A copy of the structure of chromosome c containing only the given origins """
        o = torch.tensor(origins, dtype=torch.long)
        return ChromosomeModel(c.name, int(c.length.item()), c.is_circular,
                               origins=c.origins[o].clone(), origin_names=[c.origin_names[i] for i in origins],
                               g_xranges_l=c.g_xranges_l.clone(), g_xranges_u=c.g_xranges_u.clone(),
                               # The names are only needed to resolve stall sites, which uses the master model
                               g_names=list(c.g_names) if c.g_names is not None else [""] * c.g_xranges_l.numel(),
                               g_strands=c.g_strands.clone(),
                               g_delayscale=c.g_delayscale.clone() if c.g_delayscale is not None else None)

    def bind(self):
        """ Makes the chromosomes of the firing pattern models (and the origin delays) use the current parameters """
        for ci, c in enumerate(self.chromosomes):
            d = self.events.chromosome_delay_scaled[ci]
            _assign(c, "delay_scaled", d * torch.ones(c.origins.numel(), dtype=torch.double))
            for vm in self.patterns:
                pc = vm.chromosomes[ci]
                _assign(pc, "g_delta_scaled", c.g_delta_scaled)
                _assign(pc, "delay_scaled", d * torch.ones(pc.origins.numel(), dtype=torch.double))

    def clamp_parameters(self):
        """ Clamps parameters to their domains, see `VelocityModel.clamp_parameters()` """
        with torch.no_grad():
            self.events.clamp_parameters()
            for c, stall in zip(self.chromosomes, self.stall_ranges):
                assert torch.all(torch.isfinite(c.g_delta_scaled))
                c.g_delta_scaled.clamp_(0.0, c.g_delta_scaled_max)
                # Stall sites have g = 1
                c.g_delta_scaled[stall] = 0.0
            self.temporal.clamp_parameters()
            self.bind()

    def owner(self, parameter: Tensor) -> TemporalModel | ChromosomeModel | MixtureEvents:
        if self.events.owns(parameter):
            return self.events
        return super().owner(parameter)

    def T(self, x: Tensor | VelocityModel.X) -> Tensor:
        """ The virtual replication time in the reference component (every origin fires, no site stalls) """
        x_prep = x if isinstance(x, VelocityModel.X) else VelocityModel.prepare_input(self, x)
        return super().T(x_prep)

    def prepare_input(self, x: Tensor, chromosome=None) -> MixtureModel.X:
        parts = [vm.prepare_input(x, chromosome) for vm in self.patterns]
        return MixtureModel.X(parts[0].shape, parts)

    def component_weights(self) -> Tensor:
        """ The weight of each component, i.e. the product of the probabilities of its events """
        q = torch.cat((self.events.p_fire, self.events.p_stall))
        return torch.prod(torch.where(self.component_bits, q, 1.0 - q), dim=1)

    def component_labels(self) -> List[str]:
        """ The label of each component, its event states in event order (origins: 1 = fires, stall sites: 1 = stalls) """
        return ["".join("1" if b else "0" for b in row.tolist()) for row in self.component_bits]

    def event_names(self) -> List[str]:
        return [e.name for e in self.events_list]

    def component_abundances(self, x: Tensor | X, chromosome=None) -> Tensor:
        """
        The abundances predicted by each component

        :param x: Genomic locations or a pre-processed input, see `prepare_input()`
        :param chromosome: The chromosomes of the locations if x is not pre-processed
        :return: Tensor of shape x.shape + (K,) where K is the number of components
        """
        x_prep = x if isinstance(x, MixtureModel.X) else self.prepare_input(x, chromosome)
        self.bind()
        result = []
        for k, f in enumerate(self.component_pattern):
            vm = self.patterns[f]
            for ci, pc in enumerate(vm.chromosomes):
                pc.g_blocked_fwd = self.component_blocked_fwd[k][ci]
                pc.g_blocked_rev = self.component_blocked_rev[k][ci]
            result.append(vm.P(x_prep.parts[f]))
        return torch.stack(result, dim=-1)

    def P(self, x: Tensor | X) -> Tensor:
        """ The abundance predicted by the mixture, i.e. the weighted sum of the components' abundances """
        return self.component_abundances(x) @ self.component_weights()

    def windows_inside_blocked(self, x: Tensor, chromosome: Tensor) -> int:
        """
        The number of locations which lie strictly inside a stall site of polarity "both"

        If such a site stalls, no replisome reaches these locations, see `ChromosomeModel.g_path_blocked()`.

        :param x: Genomic locations
        :param chromosome: Chromosome indices of the locations, see `VelocityModel.chromosome_indices()`
        """
        inside = torch.zeros(x.shape, dtype=torch.bool)
        for e in self.stall_events:
            if e.polarity == "both":
                inside |= (chromosome == e.chromosome) & (x > e.begin) & (x < e.end)
        return int(inside.sum().item())
