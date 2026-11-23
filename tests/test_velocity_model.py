"""
Unit tests for `velogp.VelocityModel`.

The abundance profile computed by `VelocityModel.forward()` is compared against an
independent, deliberately naive implementation of the model, which tracks each
replisome by solving the ODE

    dx/dt = +/- v0 * h(t) / g(x)

with a forward Euler scheme whose (variable) time step is chosen such that every
step advances the replisome by exactly one base pair, i.e.

    t_(k+1) = t_k + g(cell_k) / (v0 * h(t_k)),    x_(k+1) = x_k +/- 1,

where cell_k is the base traversed during the step. The replication time of each
position is the earliest arrival of a replisome, and the abundance is
P(x) = exp(-growthrate * t_rep(x)).

Like VelocityModel, the reference implementation assumes that a replisome stops
when it reaches the adjacent origin (or the end of a linear chromosome), i.e. each
position is replicated by one of the two origins surrounding it.

Run with `python -m pytest` from the repository root.
"""

from __future__ import annotations

import bisect
import math
import dataclasses
from dataclasses import dataclass, field

import numpy as np
import pytest
import torch
from torch import nn

from velogp import ChromosomeModel, TemporalModel, VelocityModel, VelocityModelLoss, dtensor


@dataclass
class Scenario:
    """ A set of model parameters for which to compare VelocityModel against the reference """
    name: str
    genome_length: int
    is_circular: bool
    origins: list[int]
    v0: float = 20.0
    # h(t) is piece-wise constant on h_tgrid, h(t) = h_values[i] for h_tgrid[i] <= t < h_tgrid[i+1]
    h_tgrid: list[float] = field(default_factory=lambda: [0.0])
    h_values: list[float] = field(default_factory=lambda: [1.0])
    # g(x) = 1 + g_delta[i] for g_xranges_l[i] <= x < g_xranges_u[i], and g(x) = 1 elsewhere
    g_xranges_l: list[int] = field(default_factory=list)
    g_xranges_u: list[int] = field(default_factory=list)
    g_delta: list[float] = field(default_factory=list)
    g_delayscale: list[float] | None = None
    # Origin firing delays (physical time), defaults to all zero
    delay: list[float] | None = None

    @property
    def delays(self) -> list[float]:
        return self.delay if self.delay is not None else [0.0] * len(self.origins)


def set_parameters(s: Scenario, temporal: TemporalModel, chromosome: ChromosomeModel):
    """ Sets the parameters of the scenario through the public setters """
    with torch.no_grad():
        temporal.v0 = dtensor(s.v0)
        # TemporalModel has no h_log setter; in the (default) canonical basis h_log_raw = log h(h_tgrid)
        assert temporal.h_basis == "canonical"
        temporal.h_log_raw = torch.log(dtensor(s.h_values))
        chromosome.g_delta = dtensor(s.g_delta)
        chromosome.set_delay(temporal, dtensor(s.delays))


def make_chromosome(s: Scenario, name: str) -> ChromosomeModel:
    """ Creates a ChromosomeModel for the given scenario """
    return ChromosomeModel(
        name, s.genome_length, s.is_circular,
        origins=torch.tensor(s.origins, dtype=torch.int32),
        origin_names=[f"{name}.ori{i}" for i in range(len(s.origins))],
        g_xranges_l=torch.tensor(s.g_xranges_l, dtype=torch.int32),
        g_xranges_u=torch.tensor(s.g_xranges_u, dtype=torch.int32),
        g_names=[f"site{i + 1}" for i in range(len(s.g_xranges_l))],
        g_strands=torch.zeros(len(s.g_xranges_l), dtype=torch.int32),
        g_delayscale=dtensor(s.g_delayscale) if s.g_delayscale is not None else None)


def make_model(s: Scenario) -> VelocityModel:
    """ Creates a single-chromosome VelocityModel for the given scenario """
    m = VelocityModel.single_chromosome(
        genome_length=s.genome_length, is_circular=s.is_circular,
        origins=torch.tensor(s.origins, dtype=torch.int32),
        origin_names=[f"ori{i}" for i in range(len(s.origins))],
        h_tgrid=dtensor(s.h_tgrid),
        g_xranges_l=torch.tensor(s.g_xranges_l, dtype=torch.int32),
        g_xranges_u=torch.tensor(s.g_xranges_u, dtype=torch.int32),
        g_names=[f"site{i + 1}" for i in range(len(s.g_xranges_l))],
        g_strands=torch.zeros(len(s.g_xranges_l), dtype=torch.int32),
        v0_min=dtensor([1.0]), v0_max=dtensor([1000.0]),
        g_delayscale=dtensor(s.g_delayscale) if s.g_delayscale is not None else None)
    set_parameters(s, m.temporal, m.chromosomes[0])
    return m


def model_positions(s: Scenario) -> torch.Tensor:
    """ All base-pair positions at which the profile is evaluated """
    # On circular genomes, position L is the same as position 0
    n = s.genome_length if s.is_circular else s.genome_length + 1
    return torch.arange(0, n, dtype=torch.int32)


def reference_replication_times(s: Scenario, return_owner: bool = False) -> np.ndarray | tuple[np.ndarray, list]:
    """
    Computes replication times by tracking every replisome with a forward Euler scheme

    :param return_owner: whether to also return the replisome replicating each position
    :return: array t_rep where t_rep[x] is the (physical) time at which position x is replicated, plus (if
    return_owner is true) a list owner where owner[x] is the pair (origin index, direction) of the replisome which
    replicates x
    """
    L = s.genome_length
    k = len(s.origins)
    # g for every base, base i spans the interval [i, i+1)
    g_bp = np.ones(L)
    for l, u, d in zip(s.g_xranges_l, s.g_xranges_u, s.g_delta):
        g_bp[l:u] = 1.0 + d

    def h(t: float) -> float:
        return s.h_values[bisect.bisect_right(s.h_tgrid, t) - 1]

    t_rep = np.full(L if s.is_circular else L + 1, np.inf)
    owner = [None] * len(t_rep)
    for i, (o, d) in enumerate(zip(s.origins, s.delays)):
        for direction in (+1, -1):
            # Number of base pairs the replisome travels before reaching the adjacent
            # origin, or the end of a linear chromosome
            if s.is_circular:
                o_next = s.origins[(i + direction) % k]
                steps = ((o_next - o) * direction) % L or L
            elif direction > 0:
                steps = (s.origins[i + 1] if i + 1 < k else L) - o
            else:
                steps = o - (s.origins[i - 1] if i > 0 else 0)
            # Forward Euler, each step advances the replisome by one base pair
            t, x = float(d), o
            p = x % L if s.is_circular else x
            if t < t_rep[p]:
                t_rep[p], owner[p] = t, (i, direction)
            for _ in range(steps):
                cell = x if direction > 0 else x - 1
                t += g_bp[cell % L] / (s.v0 * h(t))
                x += direction
                p = x % L if s.is_circular else x
                if t < t_rep[p]:
                    t_rep[p], owner[p] = t, (i, direction)
    assert np.all(np.isfinite(t_rep))
    return (t_rep, owner) if return_owner else t_rep


def reference_time_tolerance(s: Scenario) -> float:
    """
    Upper bound on the error of the Euler scheme for replication times

    Since g(x) is constant on each base pair, the scheme is exact except for steps during
    which h(t) jumps, each of which contributes an error of at most one step duration.
    Every replisome crosses each of the jumps of h at most once, and each replication time
    is determined by one replisome.
    """
    g_max = 1.0 + max(s.g_delta, default=0.0)
    step_max = g_max / (s.v0 * min(s.h_values))
    return (len(s.h_tgrid) + 1) * step_max


H_TGRID = [0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0]
H_VARYING = [1.0, 1.5, 2.0, 1.2, 0.8, 0.5, 0.7, 1.0, 1.3]

SCENARIOS = [
    # Circular chromosomes
    Scenario("circular, single origin", 20000, True, [5000]),
    Scenario("circular, constant v", 20000, True, [2000, 9000, 15000]),
    Scenario("circular, varying v", 20000, True, [2000, 9000, 15000],
             h_tgrid=H_TGRID, h_values=H_VARYING),
    Scenario("circular, g", 20000, True, [2000, 9000, 15000],
             g_xranges_l=[500, 4000, 12000], g_xranges_u=[1500, 6000, 12500], g_delta=[1.0, 0.5, 3.0]),
    Scenario("circular, g with delayscale", 20000, True, [2000, 9000, 15000],
             g_xranges_l=[500, 4000, 12000], g_xranges_u=[1500, 6000, 12500], g_delta=[1.0, 0.5, 3.0],
             g_delayscale=[0.01, 0.05, 0.02]),
    Scenario("circular, g ranges ending and starting at an origin", 20000, True, [2000, 9000, 15000],
             g_xranges_l=[8000, 9000], g_xranges_u=[9000, 10500], g_delta=[1.0, 2.0]),
    Scenario("circular, delays", 20000, True, [2000, 9000, 15000],
             delay=[0.0, 50.0, 180.0]),
    Scenario("circular, passively replicated origin", 20000, True, [2000, 9000, 11000],
             delay=[0.0, 0.0, 400.0]),
    Scenario("circular, all combined", 20000, True, [1000, 7000, 12000, 16000], v0=15.0,
             h_tgrid=H_TGRID, h_values=H_VARYING,
             g_xranges_l=[3000, 12000, 18000], g_xranges_u=[4000, 12600, 19500], g_delta=[1.5, 0.7, 2.5],
             delay=[30.0, 0.0, 120.0, 260.0]),
    # Linear chromosomes
    Scenario("linear, single origin", 20000, False, [6000],
             h_tgrid=H_TGRID, h_values=H_VARYING),
    Scenario("linear, constant v", 20000, False, [3000, 10000, 16000]),
    Scenario("linear, varying v", 20000, False, [3000, 10000, 16000],
             h_tgrid=H_TGRID, h_values=H_VARYING),
    Scenario("linear, g", 20000, False, [3000, 10000, 16000],
             g_xranges_l=[1000, 6000, 18000], g_xranges_u=[2000, 7500, 19500], g_delta=[2.0, 1.0, 0.5]),
    Scenario("linear, delays", 20000, False, [3000, 10000, 16000],
             delay=[100.0, 0.0, 40.0]),
    Scenario("linear, all combined", 20000, False, [2500, 8000, 11000, 17000], v0=15.0,
             h_tgrid=H_TGRID, h_values=H_VARYING,
             g_xranges_l=[500, 9000, 14000], g_xranges_u=[1800, 10500, 15000], g_delta=[1.0, 2.0, 0.3],
             delay=[0.0, 90.0, 350.0, 20.0]),
]


def assert_profile_matches_reference(s: Scenario, m: VelocityModel):
    """ Asserts that the profile computed by `m` matches the reference profile for scenario `s` """
    x = model_positions(s)
    with torch.no_grad():
        p_model = m.forward(x).numpy()
    t_ref = reference_replication_times(s)
    growthrate = m.temporal.growthrate.item()
    p_ref = np.exp(-growthrate * t_ref)
    # |dP/dt| = growthrate * P <= growthrate, so errors in t translate into errors in P of
    # at most growthrate times as large. Allow for a small absolute error on top.
    atol = growthrate * reference_time_tolerance(s) + 1e-9
    err = np.abs(p_model - p_ref)
    i = int(np.argmax(err))
    assert np.all(np.isfinite(p_model)), "model profile contains non-finite values"
    assert err[i] <= atol, (
        f"{s.name}: max |P_model - P_ref| = {err[i]:.3g} > {atol:.3g} at x={i} "
        f"(P_model={p_model[i]:.6f}, P_ref={p_ref[i]:.6f})")


@pytest.mark.parametrize("s", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_forward_matches_euler_reference(s: Scenario):
    assert_profile_matches_reference(s, make_model(s))


def test_prepared_input_matches_raw_input():
    s = SCENARIOS[-1]
    m = make_model(s)
    x = model_positions(s)
    with torch.no_grad():
        torch.testing.assert_close(m.forward(m.prepare_input(x)), m.forward(x))


BASE_SCENARIO = next(s for s in SCENARIOS if s.name == "circular, all combined")
PERTURBATIONS = {
    "h": dict(h_values=[v * (1.1 if i == 2 else 1.0) for i, v in enumerate(BASE_SCENARIO.h_values)]),
    "g": dict(g_delta=[d + 0.2 for d in BASE_SCENARIO.g_delta]),
    "delay": dict(delay=[d + (5.0 if i == 1 else 0.0) for i, d in enumerate(BASE_SCENARIO.delays)]),
    "v0": dict(v0=BASE_SCENARIO.v0 * 1.02),
}


@pytest.mark.parametrize("change", PERTURBATIONS.values(), ids=PERTURBATIONS.keys())
def test_reference_detects_differences(change: dict):
    """ Sanity check of the test itself: the comparison must fail if the parameters differ """
    other = Scenario(**{**BASE_SCENARIO.__dict__, **change})
    with pytest.raises(AssertionError):
        assert_profile_matches_reference(BASE_SCENARIO, make_model(other))


def test_submodel_attributes_cannot_be_assigned_on_velocity_model():
    """ Assigning e.g. `h_log_raw` or `g_delta` on the VelocityModel instead of a sub-model must fail loudly """
    m = make_model(BASE_SCENARIO)
    c = m.chromosomes[0]
    for name, value, owner in [("h_log_raw", torch.zeros_like(m.temporal.h_log_raw), "temporal"),
                               ("v0_log", nn.Parameter(m.temporal.v0_log.clone()), "temporal"),
                               ("growthrate", dtensor([1.0]), "temporal"),
                               ("delay_scaled", nn.Parameter(c.delay_scaled.clone()), "chromosome"),
                               ("g_delta", c.g_delta.clone(), "chromosome"),
                               ("origins", c.origins.clone(), "chromosome")]:
        with pytest.raises(AttributeError, match=owner):
            setattr(m, name, value)
    # Parameters of the sub-models are found under their qualified names, and nn.Module internals can still be
    # assigned
    c.delay_scaled = nn.Parameter(c.delay_scaled)
    m.temporal.v0_log = nn.Parameter(m.temporal.v0_log)
    m.train(False)
    assert [n for n, _ in m.named_parameters()] == ["temporal.v0_log", "chromosomes.0.delay_scaled"]


# Two chromosomes with different structure and parameters, but the same v(t). Chromosome names may contain dots.
GENOME_SCENARIOS = {
    "U00096.3": next(s for s in SCENARIOS if s.name == "circular, all combined"),
    "plasmid.1": next(s for s in SCENARIOS if s.name == "linear, all combined"),
}


def make_genome_model() -> VelocityModel:
    """ Creates a VelocityModel comprising the chromosomes in GENOME_SCENARIOS """
    scenarios = list(GENOME_SCENARIOS.values())
    assert all((s.v0, s.h_tgrid, s.h_values) == (scenarios[0].v0, scenarios[0].h_tgrid, scenarios[0].h_values)
               for s in scenarios)
    temporal = TemporalModel(dtensor(scenarios[0].h_tgrid), v0_min=dtensor([1.0]), v0_max=dtensor([1000.0]))
    chromosomes = [make_chromosome(s, name) for name, s in GENOME_SCENARIOS.items()]
    for s, c in zip(scenarios, chromosomes):
        set_parameters(s, temporal, c)
    return VelocityModel(temporal, chromosomes)


def genome_positions() -> tuple[torch.Tensor, torch.Tensor]:
    """ Positions on all chromosomes of GENOME_SCENARIOS in a random order, and the chromosome index of each """
    x = torch.cat([model_positions(s) for s in GENOME_SCENARIOS.values()])
    c = torch.cat([torch.full((len(model_positions(s)),), i) for i, s in enumerate(GENOME_SCENARIOS.values())])
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(0))
    return x[perm], c[perm]


def test_multi_chromosome_matches_single_chromosome_models():
    """ Each chromosome of a multi-chromosome model must produce the same profile as a single-chromosome model """
    m = make_genome_model()
    x, c = genome_positions()
    with torch.no_grad():
        p = m.forward(m.prepare_input(x, c))
        for i, s in enumerate(GENOME_SCENARIOS.values()):
            torch.testing.assert_close(p[c == i], make_model(s).forward(x[c == i]), rtol=0, atol=0)


@pytest.mark.parametrize("name", GENOME_SCENARIOS.keys())
def test_multi_chromosome_matches_reference(name: str):
    """ Positions given by chromosome name must match the Euler reference of that chromosome """
    m = make_genome_model()
    s = GENOME_SCENARIOS[name]
    x = model_positions(s)
    with torch.no_grad():
        p_model = m.forward(m.prepare_input(x, name))
    single = make_model(s)
    with torch.no_grad():
        torch.testing.assert_close(p_model, single.forward(x), rtol=0, atol=0)
    assert_profile_matches_reference(s, single)


def test_multi_chromosome_input_forms():
    """ Chromosomes can be given as names, indices or index tensors, and inputs can have arbitrary shape """
    m = make_genome_model()
    names = list(GENOME_SCENARIOS.keys())
    x, c = genome_positions()
    x, c = x[:120], c[:120]
    with torch.no_grad():
        p = m.forward(m.prepare_input(x, c))
        torch.testing.assert_close(m.forward(m.prepare_input(x, [names[i] for i in c.tolist()])), p, rtol=0, atol=0)
        torch.testing.assert_close(m.forward(m.prepare_input(x, c.tolist())), p, rtol=0, atol=0)
        p2 = m.forward(m.prepare_input(x.reshape(12, 10), c.reshape(12, 10)))
        torch.testing.assert_close(p2, p.reshape(12, 10), rtol=0, atol=0)
    with pytest.raises(ValueError):
        m.prepare_input(x)
    with pytest.raises(KeyError):
        m.prepare_input(x, "no such chromosome")


@pytest.mark.parametrize("k", [5, slice(10, 50, 3), torch.tensor([7, 3, 3, 90, 0])], ids=["int", "slice", "tensor"])
def test_multi_chromosome_prepared_input_subsets(k):
    """ Indexing prepared input must select the corresponding positions, as used by VelocityModelLoss.jacobian() """
    m = make_genome_model()
    x, c = genome_positions()
    x, c = x[:120], c[:120]
    x_prep = m.prepare_input(x, c)
    with torch.no_grad():
        expected = m.forward(x_prep)[k]
        # NOTE: For a single (0-d) position, P() returns a 1-element tensor since growthrate has shape (1,)
        torch.testing.assert_close(m.forward(x_prep[k]).reshape(expected.shape), expected, rtol=0, atol=0)
        torch.testing.assert_close(m.forward(x_prep[k]), m.forward(m.prepare_input(x[k], c[k])), rtol=0, atol=0)


def test_clamp_shifts_delays_genome_wide():
    """ Delays are relative to the first origin to fire in the whole genome, not on each chromosome """
    m = make_genome_model()
    c0, c1 = m.chromosomes
    with torch.no_grad():
        c0.delay_scaled.copy_(dtensor([5.0, 7.0, 6.0, 9.0]))
        c1.delay_scaled.copy_(dtensor([3.0, 4.0, -1.0, 8.0]))
    m.clamp_parameters()
    torch.testing.assert_close(c0.delay_scaled, dtensor([5.0, 7.0, 6.0, 9.0]))
    torch.testing.assert_close(c1.delay_scaled, dtensor([3.0, 4.0, 0.0, 8.0]))
    with torch.no_grad():
        c1.delay_scaled.copy_(dtensor([3.0, 4.0, 2.0, 8.0]))
    m.clamp_parameters()
    torch.testing.assert_close(c0.delay_scaled, dtensor([3.0, 5.0, 4.0, 7.0]))
    torch.testing.assert_close(c1.delay_scaled, dtensor([1.0, 2.0, 0.0, 6.0]))


def test_prior_dispatch():
    """ Priors of each parameter are provided by the sub-model the parameter belongs to """
    m = make_genome_model()
    c0, c1 = m.chromosomes
    with torch.no_grad():
        c0.g_delta_scaled.copy_(dtensor([1.0] * c0.g_delta_scaled.numel()))
        c1.g_delta_scaled.copy_(dtensor([2.0] * c1.g_delta_scaled.numel()))
    assert m.owner(m.temporal.h_log_raw) is m.temporal
    assert m.owner(c1.delay_scaled) is c1
    assert m.prior(c0.g_delta_scaled, False).item() == c0.g_delta_scaled.numel()
    assert m.prior(c1.g_delta_scaled, False).item() == 2 * c1.g_delta_scaled.numel()
    assert m.prior(c1.delay_scaled, False).item() == 0.0
    torch.testing.assert_close(m.prior(m.temporal.h_log_raw, True), m.temporal.prior_hlog(True))
    with pytest.raises(RuntimeError):
        m.owner(torch.zeros(1))


def test_trajectories_concatenate_chromosomes():
    m = make_genome_model()
    t = m.temporal.h_tgrid
    with torch.no_grad():
        combined = m.trajectories(t)
        separate = [c.trajectories(m.temporal, t) for c in m.chromosomes]
    for i, tab in enumerate(combined):
        torch.testing.assert_close(tab, torch.cat([s[i] for s in separate], dim=-1), rtol=0, atol=0, equal_nan=True)


def make_genome_loss(n: int = 80, permutation: torch.Tensor | None = None) -> tuple[VelocityModel, VelocityModelLoss]:
    """ Creates a two-chromosome model with all parameters trainable, and a loss for synthetic observations """
    m = make_genome_model()
    m.temporal.v0_log = nn.Parameter(m.temporal.v0_log)
    m.temporal.h_log_raw = nn.Parameter(m.temporal.h_log_raw)
    for c in m.chromosomes:
        c.delay_scaled = nn.Parameter(c.delay_scaled)
        c.g_delta_scaled = nn.Parameter(c.g_delta_scaled)
    x, c = genome_positions()
    x, c = x[:n].double(), c[:n]
    with torch.no_grad():
        p = m.forward(m.prepare_input(x, c))
    noise = 1 + 0.01 * torch.randn(n, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    y, s = 2.0 * p * noise, 0.01 * torch.ones(n, dtype=torch.float64)
    if permutation is not None:
        x, c, y, s = x[permutation], c[permutation], y[permutation], s[permutation]
    params = ["temporal.v0_log", "temporal.h_log_raw", "chromosomes.0.delay_scaled", "chromosomes.0.g_delta_scaled",
              "chromosomes.1.delay_scaled", "chromosomes.1.g_delta_scaled"]
    return m, VelocityModelLoss(x, y, s, m, params, chromosome=c)


def test_multi_chromosome_loss_value():
    """ The abundances on all chromosomes are scaled by a single factor """
    m, loss = make_genome_loss()
    with torch.no_grad():
        p = m.forward(m.prepare_input(loss.x, loss.chromosome))
        scale = torch.mean(loss.y_obs) / torch.mean(p)
        expected = 0.5 * torch.sum(loss.weights * (loss.y_obs - scale * p) ** 2)
        expected += sum(m.prior(pp, False).reshape(()) for _, pp in m.resolve_params(loss.params))
        torch.testing.assert_close(loss(), expected, rtol=1e-12, atol=0)
        # The loss does not depend on the order of the observations
        _, loss_perm = make_genome_loss(permutation=torch.randperm(80, generator=torch.Generator().manual_seed(2)))
        torch.testing.assert_close(loss_perm(), loss(), rtol=1e-12, atol=0)


def test_multi_chromosome_loss_requires_chromosomes():
    m, loss = make_genome_loss()
    with pytest.raises(ValueError):
        VelocityModelLoss(loss.x, loss.y_obs, loss.s_obs, m, loss.params)
    with pytest.raises(IndexError):
        VelocityModelLoss(loss.x, loss.y_obs, loss.s_obs, m, loss.params, chromosome=torch.full_like(loss.chromosome, 2))
    names = [m.chromosomes[i].name for i in loss.chromosome.tolist()]
    loss_names = VelocityModelLoss(loss.x, loss.y_obs, loss.s_obs, m, loss.params, chromosome=names)
    assert torch.equal(loss_names.chromosome, loss.chromosome)


def test_multi_chromosome_jacobian():
    """ The per-position Jacobian must match derivatives of the jointly computed prediction """
    m, loss = make_genome_loss(n=40)
    J, pmap = loss.jacobian(None)
    params = [p for _, p in m.resolve_params(loss.params)]
    y_pred = loss.y_pred()
    for k in range(len(loss.x)):
        g = torch.autograd.grad(y_pred[k], params, retain_graph=True, allow_unused=True)
        row = torch.cat([(gi if gi is not None else torch.zeros_like(p)).reshape(-1) for gi, p in zip(g, params)])
        torch.testing.assert_close(J[k], row, rtol=1e-10, atol=1e-14)
    # Through the common scale factor, predictions on one chromosome depend on the parameters of the other
    i = pmap["chromosomes.1.g_delta_scaled"]
    assert torch.any(J[loss.chromosome == 0, i:i + m.chromosomes[1].g_delta_scaled.numel()] != 0)


def test_multi_chromosome_posterior_includes_all_priors():
    """ The posterior precision matrix contains the prior precision blocks of h(t) and of g(x) of every chromosome """
    m, loss = make_genome_loss(n=40)
    u, pmap = m.posterior_cov_inv(loss, loss.params)
    u0, pmap0 = m.posterior_cov_inv(loss, loss.params, ignore_priors=True)
    assert pmap == pmap0
    expected = torch.zeros_like(u)
    blocks = [("temporal.h_log_raw", m.temporal.h_cov_inv(include_sigma=True))]
    blocks += [(f"chromosomes.{i}.g_delta_scaled", torch.diagflat(1.0 / c.g_delta_prefactor ** 2))
               for i, c in enumerate(m.chromosomes)]
    for name, block in blocks:
        i, l = pmap[name], block.shape[0]
        expected[i:i + l, i:i + l] = block
    torch.testing.assert_close(u - u0, expected, rtol=1e-10, atol=1e-10)


def test_g_prior_scaled_by_genome_length():
    """ Expected delays at g ranges are relative to the whole genome, and joining a genome preserves g_delta """
    # Give every g range of the genome scenarios an expected delay
    genome = {name: Scenario(**{**s.__dict__, "g_delayscale": [0.01 * (i + 1) for i in range(len(s.g_xranges_l))]})
              for name, s in GENOME_SCENARIOS.items()}
    scenarios = list(genome.values())
    chromosomes = [make_chromosome(s, name) for name, s in genome.items()]
    temporal = TemporalModel(dtensor(scenarios[0].h_tgrid), v0_min=dtensor([1.0]), v0_max=dtensor([1000.0]))
    for s, c in zip(scenarios, chromosomes):
        set_parameters(s, temporal, c)
        c.g_delta_scaled = nn.Parameter(c.g_delta_scaled)
    g_delta_before = [c.g_delta.detach().clone() for c in chromosomes]
    params_before = [c.g_delta_scaled for c in chromosomes]
    m = VelocityModel(temporal, chromosomes)
    total = sum(s.genome_length for s in scenarios)
    assert m.length.item() == total
    for s, c, g_delta, p in zip(scenarios, chromosomes, g_delta_before, params_before):
        assert c.reference_length.item() == total
        torch.testing.assert_close(c.g_delta_prefactor, dtensor(s.g_delayscale) * total / c.g_xranges_len.double())
        torch.testing.assert_close(c.g_delta, g_delta)
        assert c.g_delta_scaled is p
    torch.testing.assert_close(m.length_virtual, sum(c.length_virtual for c in chromosomes))


def test_g_prior_single_chromosome_unchanged():
    """ For a single chromosome, the genome is the chromosome, so the prior does not change """
    s = next(s for s in SCENARIOS if s.g_delayscale is not None)
    c = make_model(s).chromosomes[0]
    expected = dtensor(s.g_delayscale) * s.genome_length / c.g_xranges_len.double()
    assert torch.equal(c.g_delta_prefactor, expected)


# Scenarios for g_range_passage_times(), in addition to the scenarios in SCENARIOS that contain g ranges
PASSAGE_SCENARIOS = [s for s in SCENARIOS if len(s.g_xranges_l) > 0] + [
    Scenario("circular, replisomes meet inside g range, varying v", 20000, True, [2000, 9000, 15000],
             h_tgrid=H_TGRID, h_values=H_VARYING,
             g_xranges_l=[4000, 9500], g_xranges_u=[7000, 14000], g_delta=[1.5, 2.0]),
    Scenario("circular, g range starting at passively replicated origin, varying v", 20000, True,
             [2000, 9000, 11000], h_tgrid=H_TGRID, h_values=H_VARYING, delay=[0.0, 0.0, 400.0],
             g_xranges_l=[11000], g_xranges_u=[12000], g_delta=[2.0]),
    Scenario("circular, g range at chromosome end, varying v", 20000, True, [2000, 9000, 15000],
             h_tgrid=H_TGRID, h_values=H_VARYING,
             g_xranges_l=[0, 19000], g_xranges_u=[500, 20000], g_delta=[1.0, 3.0]),
    Scenario("linear, g ranges at chromosome ends, varying v", 20000, False, [3000, 10000, 16000],
             h_tgrid=H_TGRID, h_values=H_VARYING, delay=[0.0, 60.0, 10.0],
             g_xranges_l=[0, 6000, 17000], g_xranges_u=[2000, 7500, 20000], g_delta=[1.0, 2.5, 0.5]),
    # Two ranges replicated by the forward and one by the reverse fork. The counterfactual reverse fork crosses
    # its own range (with the fitted g) and one range replicated by the forward fork (with g = 1) before
    # reaching the other range replicated by the forward fork, and vice versa.
    Scenario("circular, counterfactual forks cross other ranges, varying v", 20000, True, [2000, 14000],
             h_tgrid=H_TGRID, h_values=H_VARYING,
             g_xranges_l=[3000, 5000, 12000], g_xranges_u=[4000, 6000, 13000], g_delta=[4.0, 3.0, 2.0]),
]


def reference_passage_times(s: Scenario, m: VelocityModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes int g(x) / v(t(x)) dx and int (g(x) - 1) / v(t(x)) dx over each g range by summing over base pairs

    The replication time of base [x, x+1) is approximated by the mean of the replication times of x and x+1,
    computed by the model. This is exact except for bases during whose replication h(t) jumps, at which
    replication times are discontinuous, or at which two replisomes meet.

    :return: Triple (time, delay, tolerance)
    """
    def h(t: float) -> float:
        return s.h_values[bisect.bisect_right(s.h_tgrid, t) - 1]

    time, delay, tol = [], [], []
    for l, u, d in zip(s.g_xranges_l, s.g_xranges_u, s.g_delta):
        x = torch.arange(l, u + 1, dtype=torch.int32)
        with torch.no_grad():
            t = m.temporal.Hinv(m.T(x)).numpy()
        t_mid = (t[:-1] + t[1:]) / 2.0
        v = s.v0 * np.array([h(ti) for ti in t_mid])
        time.append(np.sum((1.0 + d) / v))
        delay.append(np.sum(d / v))
        # Each base at which h(t) jumps (once per replisome), or at which t(x) is discontinuous, contributes an
        # error of at most the time it takes to replicate a single base
        n_bad = 2 * len(s.h_tgrid) + 4
        tol.append(n_bad * (1.0 + d) / (s.v0 * min(s.h_values)))
    return np.array(time), np.array(delay), np.array(tol)


@pytest.mark.parametrize("s", PASSAGE_SCENARIOS, ids=[s.name for s in PASSAGE_SCENARIOS])
def test_g_range_passage_times_match_reference(s: Scenario):
    m = make_model(s)
    with torch.no_grad():
        time, delay = (r.numpy() for r in m.g_range_passage_times())
    time_ref, delay_ref, tol = reference_passage_times(s, m)
    assert np.all(np.abs(time - time_ref) <= tol), f"time {time} != reference {time_ref} (tol {tol})"
    assert np.all(np.abs(delay - delay_ref) <= tol), f"delay {delay} != reference {delay_ref} (tol {tol})"
    # The delay is the fraction 1 - 1/g of the passage time
    g = 1.0 + np.array(s.g_delta)
    np.testing.assert_allclose(delay, time * (1.0 - 1.0 / g), rtol=1e-12)


def test_g_names_and_strands():
    s = Scenario("names", 20000, True, [2000, 15000],
                 g_xranges_l=[4000, 8000], g_xranges_u=[5000, 9000], g_delta=[1.0, 2.0])

    def make(names, strands):
        return ChromosomeModel("chr", s.genome_length, s.is_circular,
                               origins=torch.tensor(s.origins, dtype=torch.int32), origin_names=["a", "b"],
                               g_xranges_l=torch.tensor(s.g_xranges_l, dtype=torch.int32),
                               g_xranges_u=torch.tensor(s.g_xranges_u, dtype=torch.int32),
                               g_names=names, g_strands=torch.tensor(strands, dtype=torch.int32))

    c = make(["terA", "rrnB"], [1, -1])
    assert c.g_names == ["terA", "rrnB"]
    assert c.g_strands.tolist() == [1, -1]
    # One name and one strand per g range, strands must be +1, -1 or 0
    with pytest.raises(AssertionError):
        make(["terA"], [1, -1])
    with pytest.raises(AssertionError):
        make(["terA", "rrnB"], [1])
    with pytest.raises(AssertionError):
        make(["terA", "rrnB"], [1, 2])


@pytest.mark.parametrize("l, u", [(8000, 10500), (8999, 9001)])
def test_origin_inside_g_range_rejected(l: int, u: int):
    s = Scenario("origin inside g range", 20000, True, [2000, 9000, 15000],
                 g_xranges_l=[l], g_xranges_u=[u], g_delta=[1.0])
    with pytest.raises(ValueError, match="ori1"):
        make_chromosome(s, "chr")


@pytest.mark.parametrize("is_circular", [True, False])
def test_g_range_passage_times_constant_v(is_circular: bool):
    """ For constant v(t) = v0, the delay is g_delta * length / v0 if a single replisome passes the range """
    s = Scenario("constant v", 20000, is_circular, [3000, 10000, 16000],
                 g_xranges_l=[500, 4000, 12000], g_xranges_u=[1500, 6000, 12500], g_delta=[1.0, 0.5, 3.0])
    m = make_model(s)
    with torch.no_grad():
        time, delay = (r.numpy() for r in m.g_range_passage_times())
    length = np.array(s.g_xranges_u) - np.array(s.g_xranges_l)
    g_delta = np.array(s.g_delta)
    np.testing.assert_allclose(delay, g_delta * length / s.v0, rtol=1e-12)
    np.testing.assert_allclose(time, (1.0 + g_delta) * length / s.v0, rtol=1e-12)


def test_g_range_passage_times_no_delay_for_g_one():
    s = Scenario("g = 1", 20000, True, [2000, 9000, 15000], h_tgrid=H_TGRID, h_values=H_VARYING,
                 g_xranges_l=[4000, 9500], g_xranges_u=[7000, 14000], g_delta=[0.0, 0.0])
    m = make_model(s)
    with torch.no_grad():
        time, delay = m.g_range_passage_times()
    assert torch.all(delay == 0.0)
    assert torch.all(time > 0.0)


def test_g_range_passage_times_concatenate_chromosomes():
    scenarios = [s for s in SCENARIOS if s.name in ("circular, g", "linear, g")]
    temporal = TemporalModel(dtensor(scenarios[0].h_tgrid), v0_min=dtensor([1.0]), v0_max=dtensor([1000.0]))
    chromosomes = [make_chromosome(s, f"chr{i}") for i, s in enumerate(scenarios)]
    m = VelocityModel(temporal, chromosomes)
    for s, c in zip(scenarios, chromosomes):
        set_parameters(s, temporal, c)
    with torch.no_grad():
        time, delay = m.g_range_passage_times()
        parts = [c.g_range_passage_times(temporal) for c in chromosomes]
    assert torch.equal(time, torch.cat([p[0] for p in parts]))
    assert torch.equal(delay, torch.cat([p[1] for p in parts]))


@pytest.mark.parametrize("s", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_completion_time(s: Scenario):
    """ The completion time is the latest replication time, after which no replisome is active anymore """
    m = make_model(s)
    with torch.no_grad():
        c = m.completion_time().item()
        # The latest replication time of any base pair lies within one base pair of the completion time
        t_rep = m.temporal.Hinv(m.T(model_positions(s)))
        assert t_rep.max().item() <= c + 1e-9
        assert t_rep.max().item() >= c - reference_time_tolerance(s)
        # No replisome is active after the completion time, but some are active just before
        t = torch.cat((torch.linspace(0.0, c, 200, dtype=torch.float64)[:-1],
                       torch.linspace(c, c + 100.0, 50, dtype=torch.float64)[1:]))
        state, _, _, _ = m.trajectories(t)
    after = t > c
    assert torch.all(state[after] != 1)
    assert torch.any(state[~after][-1] == 1)
    # Replisomes that were started have all finished, those that never start stay in state 0
    assert torch.all(state[after] == state[after][0])


def test_completion_time_multi_chromosome():
    scenarios = [s for s in SCENARIOS if s.name in ("circular, constant v", "linear, delays")]
    temporal = TemporalModel(dtensor(scenarios[0].h_tgrid), v0_min=dtensor([1.0]), v0_max=dtensor([1000.0]))
    chromosomes = [make_chromosome(s, f"chr{i}") for i, s in enumerate(scenarios)]
    m = VelocityModel(temporal, chromosomes)
    for s, c in zip(scenarios, chromosomes):
        set_parameters(s, temporal, c)
    with torch.no_grad():
        assert m.completion_time().item() == max(c.completion_time(temporal).item() for c in chromosomes)


def reference_fork_times(s: Scenario, origin: int, direction: int, l: int, u: int,
                         owner: list | None = None) -> tuple[float, float]:
    """
    Computes the times at which the replisome of origin `origin` moving in direction `direction` enters and exits
    the range [l, u], ignoring all other replisomes, by tracking it with a forward Euler scheme (see
    `reference_replication_times()`). The forward replisome enters at l and exits at u, the reverse one vice versa.

    If `owner` is given (see `reference_replication_times()`), g = 1 is assumed for every base whose far end (in
    the direction of movement) is replicated by another replisome, i.e. g only applies to the replisome which
    actually replicates a base.
    """
    L = s.genome_length
    g_bp = np.ones(L)
    for gl, gu, d in zip(s.g_xranges_l, s.g_xranges_u, s.g_delta):
        g_bp[gl:gu] = 1.0 + d

    def h(t: float) -> float:
        return s.h_values[bisect.bisect_right(s.h_tgrid, t) - 1]

    o = s.origins[origin]
    entry = l if direction > 0 else u
    steps_entry = ((entry - o) * direction) % L if s.is_circular else (entry - o) * direction
    assert steps_entry >= 0
    t, x = float(s.delays[origin]), o
    times = []
    for steps in (steps_entry, u - l):
        for _ in range(steps):
            cell = x if direction > 0 else x - 1
            x += direction
            own = owner is None or owner[x % L if s.is_circular else x] == (origin, direction)
            t += (g_bp[cell % L] if own else 1.0) / (s.v0 * h(t))
        times.append(t)
    return times[0], times[1]


@pytest.mark.parametrize("s", PASSAGE_SCENARIOS, ids=[s.name for s in PASSAGE_SCENARIOS])
def test_g_range_forks_match_reference(s: Scenario):
    m = make_model(s)
    with torch.no_grad():
        forks = m.g_range_forks()
        time, _ = m.g_range_passage_times()
    # Closest origins to the left and right of each range
    fwd, rev = [], []
    for l, u in zip(s.g_xranges_l, s.g_xranges_u):
        left = [i for i, o in enumerate(s.origins) if o <= l]
        right = [i for i, o in enumerate(s.origins) if o >= u]
        if s.is_circular:
            left, right = left or [len(s.origins) - 1], right or [0]
        fwd.append(left[-1] if left else -1)
        rev.append(right[0] if right else -1)
    # Replisomes which never fire, i.e. which replicate nothing, count as missing
    t_rep, owner = reference_replication_times(s, return_owner=True)
    fwd = [i if i >= 0 and (i, +1) in owner else -1 for i in fwd]
    rev = [i if i >= 0 and (i, -1) in owner else -1 for i in rev]
    assert forks.oi_fwd.tolist() == fwd
    assert forks.oi_rev.tolist() == rev
    assert forks.origin_fwd == [f"ori{i}" if i >= 0 else None for i in fwd]
    assert forks.origin_rev == [f"ori{i}" if i >= 0 else None for i in rev]
    # The base at which the replisomes meet may be attributed to the wrong replisome, allow for one extra base
    g_max = 1.0 + max(s.g_delta, default=0.0)
    tol = reference_time_tolerance(s) + g_max / (s.v0 * min(s.h_values)) + 1e-9
    # Replication times of the first and last base inside each range, which are within the time it takes to
    # replicate a single base of the times at which the replisome replicating that base enters / exits the range.
    # Range ends may be origins which are replicated passively before they fire, see T().
    tol_rep = tol + g_max / (s.v0 * min(s.h_values))
    for i, (l, u) in enumerate(zip(s.g_xranges_l, s.g_xranges_u)):
        enter_fwd, exit_fwd = forks.enter_fwd[i].item(), forks.exit_fwd[i].item()
        enter_rev, exit_rev = forks.enter_rev[i].item(), forks.exit_rev[i].item()
        # Entry and exit times (actual or counterfactual) match the reference for a single replisome
        for oi, direction, enter, exit in ((fwd[i], +1, enter_fwd, exit_fwd), (rev[i], -1, enter_rev, exit_rev)):
            if oi < 0:
                assert np.isnan(enter) and np.isnan(exit)
                continue
            enter_ref, exit_ref = reference_fork_times(s, oi, direction, l, u, owner)
            assert abs(enter - enter_ref) <= tol, f"range {i}, direction {direction}: entry {enter} != {enter_ref}"
            assert abs(exit - exit_ref) <= tol, f"range {i}, direction {direction}: exit {exit} != {exit_ref}"
        # Actual times agree with the replication times. A replisome replicates the whole range if it exits
        # before the other one enters. Otherwise, the replisomes meet inside the range.
        if exit_fwd <= enter_rev or np.isnan(enter_rev):
            assert abs(enter_fwd - t_rep[l + 1]) <= tol_rep and abs(exit_fwd - t_rep[u - 1]) <= tol_rep
            np.testing.assert_allclose(time[i].item(), exit_fwd - enter_fwd, rtol=1e-9)
        elif exit_rev <= enter_fwd or np.isnan(enter_fwd):
            assert abs(enter_rev - t_rep[u - 1]) <= tol_rep and abs(exit_rev - t_rep[l + 1]) <= tol_rep
            np.testing.assert_allclose(time[i].item(), exit_rev - enter_rev, rtol=1e-9)
        else:
            assert abs(enter_fwd - t_rep[l + 1]) <= tol_rep and abs(enter_rev - t_rep[u - 1]) <= tol_rep
            assert time[i].item() < (exit_fwd - enter_fwd) + (exit_rev - enter_rev)


def test_g_range_forks_passive_origins():
    """ Replisomes which never fire count as missing, the other replisome of a passively replicated origin doesn't """
    # ori2 at 11000 is replicated passively from both sides and fires in neither direction
    s = next(s for s in PASSAGE_SCENARIOS if s.name == "linear, all combined")
    with torch.no_grad():
        forks = make_model(s).g_range_forks()
    assert s.g_xranges_l[1] == 9000 and s.g_xranges_u[1] == 10500
    assert forks.origin_fwd[1] == "ori1" and forks.origin_rev[1] is None and forks.oi_rev[1].item() == -1
    assert np.isnan(forks.enter_rev[1].item()) and np.isnan(forks.exit_rev[1].item())
    assert not np.isnan(forks.enter_fwd[1].item()) and not np.isnan(forks.exit_fwd[1].item())
    # ori2 at 11000 is replicated passively from the left, but still fires rightwards into [11000, 12000)
    s = next(s for s in PASSAGE_SCENARIOS if s.name.startswith("circular, g range starting at passively"))
    with torch.no_grad():
        forks = make_model(s).g_range_forks()
    assert forks.origin_fwd == ["ori2"] and forks.origin_rev == ["ori0"]


def test_g_range_forks_cases_covered():
    """ The scenarios contain ranges replicated by either replisome, ranges at which replisomes meet, and ranges
    without origin on one side """
    cases = set()
    for s in PASSAGE_SCENARIOS:
        with torch.no_grad():
            forks = make_model(s).g_range_forks()
        for i in range(len(s.g_xranges_l)):
            ef, xf, er, xr = (getattr(forks, n)[i].item() for n in ("enter_fwd", "exit_fwd", "enter_rev", "exit_rev"))
            if np.isnan(ef) or np.isnan(er):
                cases.add("missing")
            elif xf <= er:
                cases.add("forward")
            elif xr <= ef:
                cases.add("reverse")
            else:
                cases.add("meet")
    assert cases == {"missing", "forward", "reverse", "meet"}


def test_g_range_forks_concatenate_chromosomes():
    scenarios = [s for s in SCENARIOS if s.name in ("circular, g", "linear, g")]
    temporal = TemporalModel(dtensor(scenarios[0].h_tgrid), v0_min=dtensor([1.0]), v0_max=dtensor([1000.0]))
    chromosomes = [make_chromosome(s, f"chr{i}") for i, s in enumerate(scenarios)]
    m = VelocityModel(temporal, chromosomes)
    for s, c in zip(scenarios, chromosomes):
        set_parameters(s, temporal, c)
    with torch.no_grad():
        forks = m.g_range_forks()
        parts = [c.g_range_forks(temporal) for c in chromosomes]
    assert forks.origin_fwd == parts[0].origin_fwd + parts[1].origin_fwd
    assert forks.origin_rev == parts[0].origin_rev + parts[1].origin_rev
    assert forks.origin_fwd[0].startswith("chr0.") and forks.origin_fwd[-1].startswith("chr1.")
    for name in ("oi_fwd", "oi_rev", "enter_fwd", "exit_fwd", "enter_rev", "exit_rev"):
        assert torch.equal(getattr(forks, name).nan_to_num(-1.0),
                           torch.cat([getattr(p, name) for p in parts]).nan_to_num(-1.0))


def test_g_range_forks_without_g_ranges():
    s = next(s for s in SCENARIOS if s.name == "circular, constant v")
    with torch.no_grad():
        forks = make_model(s).g_range_forks()
    assert forks.origin_fwd == [] and forks.enter_fwd.numel() == 0


@pytest.mark.parametrize("s", PASSAGE_SCENARIOS, ids=[s.name for s in PASSAGE_SCENARIOS])
def test_g_range_block_probability_matches_mixture(s: Scenario):
    """ Blocking with probability p reproduces the abundance just past the range predicted by the delay """
    m = make_model(s)
    with torch.no_grad():
        forks = m.g_range_forks()
        p_block = m.g_range_block_probability().numpy()
    lam = m.temporal.growthrate.item()
    # The base at which the replisomes meet may be attributed to the wrong replisome, allow for one extra base
    g_max = 1.0 + max(s.g_delta, default=0.0)
    tol = reference_time_tolerance(s) + g_max / (s.v0 * min(s.h_values)) + 1e-9
    _, owner = reference_replication_times(s, return_owner=True)
    assert np.all(np.isnan(p_block) | ((p_block >= 0.0) & (p_block <= 1.0)))
    for i, (l, u) in enumerate(zip(s.g_xranges_l, s.g_xranges_u)):
        oi = {+1: forks.oi_fwd[i].item(), -1: forks.oi_rev[i].item()}
        ref = {d: reference_fork_times(s, oi[d], d, l, u, owner) if oi[d] >= 0 else (np.inf, np.inf)
               for d in (+1, -1)}
        # Determine the passing replisome (if any) from the reference
        passing = [d for d in (+1, -1) if ref[d][1] < ref[-d][0] - tol]
        if not passing:
            if all(ref[d][1] > ref[-d][0] + tol for d in (+1, -1)):
                # Replisomes meet inside. p = 1 if the replisome entering first would pass without delay, else NaN
                d = +1 if ref[+1][0] <= ref[-1][0] else -1
                s_nodelay = dataclasses.replace(s, g_delta=[0.0 if j == i else gd for j, gd in enumerate(s.g_delta)])
                t_1 = reference_fork_times(s_nodelay, oi[d], d, l, u, owner)[1]
                if t_1 < ref[-d][0] - tol:
                    assert p_block[i] == 1.0, f"range {i}: replisomes meet because of the delay, but p = {p_block[i]}"
                elif t_1 > ref[-d][0] + tol:
                    assert np.isnan(p_block[i]), f"range {i}: replisomes meet without delay, but p = {p_block[i]}"
            continue
        d = passing[0]
        assert not np.isnan(p_block[i])
        # Exit time with delay, exit time without delay (g = 1 on the range), entry time of the opposite replisome
        t_x = ref[d][1]
        s_nodelay = dataclasses.replace(s, g_delta=[0.0 if j == i else gd for j, gd in enumerate(s.g_delta)])
        t_1 = reference_fork_times(s_nodelay, oi[d], d, l, u, owner)[1]
        t_2 = ref[-d][0]
        a_mix = p_block[i] * np.exp(-lam * t_2) + (1.0 - p_block[i]) * np.exp(-lam * t_1)
        # Errors in the times translate into errors of at most lambda times as large in the abundances
        assert abs(a_mix - np.exp(-lam * t_x)) <= 3 * lam * tol, f"range {i}: mixture {a_mix} != {np.exp(-lam * t_x)}"
        if s.g_delta[i] == 0.0:
            assert p_block[i] == 0.0


def test_g_range_block_probability_cases():
    # No delay for g = 1 (each range is passed by a single replisome)
    s = Scenario("g = 1", 20000, True, [3000, 10000, 16000], h_tgrid=H_TGRID, h_values=H_VARYING,
                 g_xranges_l=[500, 4000, 12000], g_xranges_u=[1500, 6000, 12500], g_delta=[0.0, 0.0, 0.0])
    with torch.no_grad():
        assert torch.all(make_model(s).g_range_block_probability() == 0.0)
    # Replisomes meeting inside ranges even without delay
    s = next(s for s in PASSAGE_SCENARIOS if s.name.startswith("circular, replisomes meet inside"))
    s = dataclasses.replace(s, g_delta=[0.0] * len(s.g_delta))
    with torch.no_grad():
        assert torch.all(torch.isnan(make_model(s).g_range_block_probability()))
    # Replisomes meeting inside a short range only because of a large delay. Without delay, the forward replisome
    # from 2000 passes [6000, 6100) long before the reverse one from 15000 arrives.
    s = Scenario("delay causes meeting", 20000, True, [2000, 15000], h_tgrid=H_TGRID, h_values=H_VARYING,
                 g_xranges_l=[6000], g_xranges_u=[6100], g_delta=[400.0])
    with torch.no_grad():
        forks = make_model(s).g_range_forks()
        assert forks.enter_fwd[0] < forks.enter_rev[0] < forks.exit_fwd[0]
        assert make_model(s).g_range_block_probability()[0].item() == 1.0
    # Without opposite replisome, p = 1 - exp(-lambda delay). With constant v, delay = g_delta * length / v0
    s = next(s for s in PASSAGE_SCENARIOS if s.name.startswith("linear, g ranges at chromosome ends"))
    s = dataclasses.replace(s, h_tgrid=[0.0], h_values=[1.0])
    m = make_model(s)
    with torch.no_grad():
        p_block = m.g_range_block_probability()
    lam = m.temporal.growthrate.item()
    for i in (0, len(s.g_xranges_l) - 1):
        delay = s.g_delta[i] * (s.g_xranges_u[i] - s.g_xranges_l[i]) / s.v0
        np.testing.assert_allclose(p_block[i].item(), -np.expm1(-lam * delay), rtol=1e-9)


def test_g_range_block_probability_deleted_fork():
    """ A replisome which never fires counts as missing, i.e. p = 1 - exp(-lambda delay) for the other one """
    # The reverse replisome of range [9000, 10500) never fires since ori2 at 11000 is replicated passively
    s = next(s for s in PASSAGE_SCENARIOS if s.name == "linear, all combined")
    m = make_model(s)
    with torch.no_grad():
        p_block = m.g_range_block_probability()
    lam = m.temporal.growthrate.item()
    _, owner = reference_replication_times(s, return_owner=True)
    i, l, u = 1, s.g_xranges_l[1], s.g_xranges_u[1]
    t_x = reference_fork_times(s, 1, +1, l, u, owner)[1]
    s_nodelay = dataclasses.replace(s, g_delta=[0.0 if j == i else gd for j, gd in enumerate(s.g_delta)])
    t_1 = reference_fork_times(s_nodelay, 1, +1, l, u, owner)[1]
    tol = 2 * reference_time_tolerance(s)
    assert abs(p_block[i].item() + np.expm1(-lam * (t_x - t_1))) <= lam * tol


def test_g_range_block_probability_concatenate_chromosomes():
    scenarios = [s for s in SCENARIOS if s.name in ("circular, g", "linear, g")]
    temporal = TemporalModel(dtensor(scenarios[0].h_tgrid), v0_min=dtensor([1.0]), v0_max=dtensor([1000.0]))
    chromosomes = [make_chromosome(s, f"chr{i}") for i, s in enumerate(scenarios)]
    m = VelocityModel(temporal, chromosomes)
    for s, c in zip(scenarios, chromosomes):
        set_parameters(s, temporal, c)
    with torch.no_grad():
        p = m.g_range_block_probability()
        parts = [c.g_range_block_probability(temporal) for c in chromosomes]
    assert torch.equal(p.nan_to_num(-1.0), torch.cat(parts).nan_to_num(-1.0))


@pytest.mark.parametrize("tmax", [5.0, 10.0, 25.0, 37.5, 100.0])
def test_v0_proper_is_log_mean_of_v(tmax: float):
    """ v0_proper(tmax) is the geometric mean of v(t) = v0 * h(t) over [0, tmax], checked by numerical integration """
    temporal = TemporalModel(dtensor([0.0, 10.0, 20.0, 40.0]), v0_min=dtensor([1.0]), v0_max=dtensor([1000.0]))
    with torch.no_grad():
        temporal.v0 = dtensor(50.0)
        temporal.h_log_raw = torch.log(dtensor([0.5, 2.0, 1.5, 0.8]))
    # N is chosen such that the jumps of h(t) at 10, 20, 40 fall on sample boundaries for every tmax
    t = (torch.arange(240000, dtype=torch.float64) + 0.5) * (tmax / 240000)
    expected = torch.exp(torch.mean(torch.log(temporal.v_t(t))))
    assert temporal.v0_proper(dtensor(tmax)).item() == pytest.approx(expected.item(), rel=1e-9)


def test_v0_proper_invariant_under_v0_h_rescaling():
    """ Moving a constant factor between v0 and h(t) leaves v(t), and hence v0_proper, unchanged """
    temporal = TemporalModel(dtensor([0.0, 10.0, 20.0, 40.0]), v0_min=dtensor([1.0]), v0_max=dtensor([1000.0]))
    h_log = torch.log(dtensor([0.5, 2.0, 1.5, 0.8]))
    with torch.no_grad():
        temporal.v0, temporal.h_log_raw = dtensor(50.0), h_log.clone()
        a = temporal.v0_proper(dtensor(30.0)).item()
        temporal.v0, temporal.h_log_raw = dtensor(200.0), h_log - math.log(4.0)
        b = temporal.v0_proper(dtensor(30.0)).item()
    assert a == pytest.approx(b, rel=1e-12)
