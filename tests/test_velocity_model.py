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

from velogp import VelocityModel, dtensor


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


def make_model(s: Scenario) -> VelocityModel:
    """ Creates a VelocityModel for the given scenario, setting parameters through the public setters """
    m = VelocityModel(
        genome_length=s.genome_length, is_circular=s.is_circular,
        origins=torch.tensor(s.origins, dtype=torch.int32),
        origin_names=[f"ori{i}" for i in range(len(s.origins))],
        h_tgrid=dtensor(s.h_tgrid),
        g_xranges_l=torch.tensor(s.g_xranges_l, dtype=torch.int32),
        g_xranges_u=torch.tensor(s.g_xranges_u, dtype=torch.int32),
        v0_min=dtensor([1.0]), v0_max=dtensor([1000.0]),
        g_delayscale=dtensor(s.g_delayscale) if s.g_delayscale is not None else None)
    with torch.no_grad():
        m.temporal.v0 = dtensor(s.v0)
        # VelocityModel has no h_log setter; in the (default) canonical basis h_log_raw = log h(h_tgrid)
        assert m.temporal.h_basis == "canonical"
        m.temporal.h_log_raw = torch.log(dtensor(s.h_values))
        m.g_delta = dtensor(s.g_delta)
        m.delay = dtensor(s.delays)
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


def test_temporal_attributes_cannot_be_assigned_on_velocity_model():
    """ Assigning e.g. `h_log_raw` on the VelocityModel instead of its TemporalModel must fail loudly """
    m = make_model(BASE_SCENARIO)
    for name, value in [("h_log_raw", torch.zeros_like(m.temporal.h_log_raw)),
                        ("v0_log", nn.Parameter(m.temporal.v0_log.clone())),
                        ("growthrate", dtensor([1.0]))]:
        with pytest.raises((AttributeError, KeyError), match="temporal"):
            setattr(m, name, value)
    # Attributes owned by VelocityModel itself, and nn.Module internals, can still be assigned
    m.delay_scaled = nn.Parameter(m.delay_scaled)
    m.train(False)
    assert [n for n, _ in m.named_parameters()] == ["delay_scaled"]
