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
        v0_min=dtensor([1.0]), v0_max=dtensor([1000.0]),
        g_delayscale=dtensor(s.g_delayscale) if s.g_delayscale is not None else None)
    set_parameters(s, m.temporal, m.chromosomes[0])
    return m


def model_positions(s: Scenario) -> torch.Tensor:
    """ All base-pair positions at which the profile is evaluated """
    # On circular genomes, position L is the same as position 0
    n = s.genome_length if s.is_circular else s.genome_length + 1
    return torch.arange(0, n, dtype=torch.int32)


def reference_replication_times(s: Scenario) -> np.ndarray:
    """
    Computes replication times by tracking every replisome with a forward Euler scheme

    :return: array t_rep where t_rep[x] is the (physical) time at which position x is replicated
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
            t_rep[x % L if s.is_circular else x] = min(t_rep[x % L if s.is_circular else x], t)
            for _ in range(steps):
                cell = x if direction > 0 else x - 1
                t += g_bp[cell % L] / (s.v0 * h(t))
                x += direction
                p = x % L if s.is_circular else x
                if t < t_rep[p]:
                    t_rep[p] = t
    assert np.all(np.isfinite(t_rep))
    return t_rep


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
    Scenario("circular, g range containing an origin", 20000, True, [2000, 9000, 15000],
             g_xranges_l=[8000], g_xranges_u=[10500], g_delta=[2.0]),
    Scenario("circular, delays", 20000, True, [2000, 9000, 15000],
             delay=[0.0, 50.0, 180.0]),
    Scenario("circular, passively replicated origin", 20000, True, [2000, 9000, 11000],
             delay=[0.0, 0.0, 400.0]),
    Scenario("circular, all combined", 20000, True, [1000, 7000, 12000, 16000], v0=15.0,
             h_tgrid=H_TGRID, h_values=H_VARYING,
             g_xranges_l=[3000, 11500, 18000], g_xranges_u=[4000, 12600, 19500], g_delta=[1.5, 0.7, 2.5],
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
