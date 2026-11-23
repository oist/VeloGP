"""
Unit tests for `velogp.velomix.MixtureModel`.

The abundance profile of each mixture component is compared against the naive Euler reference implementation of
`test_velocity_model.py`, applied to a scenario which contains the component's firing origins and blocks the
component's stalling sites in the directions given by their polarities.

Run with `python -m pytest` from the repository root.
"""

from __future__ import annotations

import copy
import itertools
import pickle

import numpy as np
import pytest
import torch
from torch import nn

from velogp import ChromosomeModel, TemporalModel, VelocityModel, VelocityModelLoss, dtensor
from velogp.velomix import MixtureModel

from test_velocity_model import (Scenario, H_TGRID, H_VARYING, make_model, make_chromosome, set_parameters,
                                 model_positions, reference_replication_times, reference_time_tolerance)


# Scenarios with stall sites and their polarities, as (scenario, strands of the g ranges, stall sites)
MIXTURE_SCENARIOS = {
    "circular": (
        Scenario("circular", 20000, True, [2000, 9000, 15000],
                 h_tgrid=H_TGRID, h_values=H_VARYING,
                 g_xranges_l=[500, 4000, 7000, 12000], g_xranges_u=[1500, 6000, 7500, 12500],
                 g_delta=[1.0, 0.5, 0.8, 3.0], delay=[0.0, 50.0, 120.0]),
        [-1, 1, 0, 1],
        [("site1", "co"), ("site2", "both"), ("site4", "head")]),
    "linear": (
        Scenario("linear", 20000, False, [3000, 10000, 16000], v0=15.0,
                 g_xranges_l=[1000, 6000, 12000, 18000], g_xranges_u=[2000, 7500, 13000, 19500],
                 g_delta=[2.0, 1.0, 0.7, 0.5], delay=[100.0, 0.0, 40.0]),
        [1, -1, 1, 0],
        [("site2", "co"), ("site3", "head")]),
}


def make_mixture_original(name: str) -> VelocityModel:
    s, strands, _ = MIXTURE_SCENARIOS[name]
    m = make_model(s)
    m.chromosomes[0].g_strands = torch.tensor(strands, dtype=torch.int32)
    return m


def make_mixture(name: str) -> MixtureModel:
    return MixtureModel(make_mixture_original(name), MIXTURE_SCENARIOS[name][2])


def component_scenario(s: Scenario, m: MixtureModel, k: int, chromosome: int = 0, delay: float = 0.0,
                       strands: list[int] | None = None, sites: list[tuple[str, str]] = ()) -> Scenario:
    """
    The scenario representing component k of mixture m (built from scenario s) on the given chromosome

    Only the component's label and the event names are taken from the model. Which origins fire and which ranges
    are blocked in which direction is derived independently from the scenario, the strands and the stall sites.
    """
    c = m.chromosomes[chromosome]
    bits = dict(zip(m.event_names(), m.component_labels()[k]))
    # Origins with the smallest delay always fire, the others if their bit is set
    fixed = [i for i, d in enumerate(s.delays) if d == min(s.delays)]
    pattern = [i for i, n in enumerate(c.origin_names) if (i in fixed) or (bits[n] == "1")]
    strands = strands if strands is not None else [0] * len(s.g_xranges_l)
    polarity = dict(sites)
    g_delta, blocked = [], []
    for j, name in enumerate(c.g_names):
        if name in polarity:
            g_delta.append(0.0)
            if bits[name] == "0":
                blocked.append("none")
            elif polarity[name] == "both":
                blocked.append("both")
            else:
                # Direction of the blocked replisomes: along the strand for "co", against it for "head"
                direction = strands[j] if polarity[name] == "co" else -strands[j]
                blocked.append("fwd" if direction > 0 else "rev")
        else:
            g_delta.append(s.g_delta[j])
            blocked.append("none")
    return Scenario(**{**s.__dict__, "name": f"{s.name}, component {k}",
                       "origins": [s.origins[i] for i in pattern], "delay": [delay] * len(pattern),
                       "g_delta": g_delta, "g_blocked": blocked})


def assert_close_to_reference(s: Scenario, p_model: np.ndarray, growthrate: float):
    t_ref = reference_replication_times(s)
    p_ref = np.exp(-growthrate * t_ref)
    atol = growthrate * reference_time_tolerance(s) + 1e-9
    err = np.abs(p_model - p_ref)
    i = int(np.argmax(err))
    assert err[i] <= atol, (f"{s.name}: max |P_model - P_ref| = {err[i]:.3g} > {atol:.3g} at x={i} "
                            f"(P_model={p_model[i]:.6f}, P_ref={p_ref[i]:.6f})")


@pytest.mark.parametrize("name", MIXTURE_SCENARIOS.keys())
def test_components_match_euler_reference(name: str):
    s, strands, sites = MIXTURE_SCENARIOS[name]
    m = make_mixture(name)
    x = model_positions(s)
    with torch.no_grad():
        a = m.component_abundances(x).numpy()
    assert a.shape == (len(x), 2 ** len(m.events_list))
    for k in range(a.shape[1]):
        assert_close_to_reference(component_scenario(s, m, k, strands=strands, sites=sites), a[:, k],
                                  m.temporal.growthrate.item())


@pytest.mark.parametrize("name", MIXTURE_SCENARIOS.keys())
def test_all_fire_none_stall_matches_velocity_model(name: str):
    """ If all origins fire and no site stalls, the mixture is the VeloGP model with zero delays and g = 1 at stall sites """
    s = MIXTURE_SCENARIOS[name][0]
    m = make_mixture(name)
    with torch.no_grad():
        m.events.p_fire.fill_(1.0)
        m.events.p_stall.fill_(0.0)
    stall = {e.index for e in m.stall_events}
    ref = make_model(Scenario(**{**s.__dict__, "delay": None,
                                 "g_delta": [0.0 if j in stall else d for j, d in enumerate(s.g_delta)]}))
    x = model_positions(s)
    with torch.no_grad():
        torch.testing.assert_close(m.forward(x), ref.forward(x))
        # The master model represents this component
        torch.testing.assert_close(VelocityModel.P(m, x), ref.forward(x))


def test_mixture_is_weighted_sum():
    m = make_mixture("circular")
    with torch.no_grad():
        m.events.p_fire.copy_(dtensor([0.3, 0.8]))
        m.events.p_stall.copy_(dtensor([0.6, 0.1, 0.5]))
        x = model_positions(MIXTURE_SCENARIOS["circular"][0])
        a = m.component_abundances(x)
        w = m.component_weights()
        torch.testing.assert_close(m.forward(x), a @ w)
    # Explicit product of the event probabilities
    q = [0.3, 0.8, 0.6, 0.1, 0.5]
    for k, label in enumerate(m.component_labels()):
        expected = np.prod([qi if b == "1" else 1.0 - qi for qi, b in zip(q, label)])
        assert w[k].item() == pytest.approx(expected, rel=1e-12)
    assert torch.sum(w).item() == pytest.approx(1.0, rel=1e-12)


def test_events_and_labels():
    m = make_mixture("circular")
    # Origin 0 has the smallest delay and is fixed, the other origins and the stall sites are events
    assert m.fixed_origins == [[0]]
    assert m.event_names() == ["ori1", "ori2", "site1", "site2", "site4"]
    assert m.component_labels()[0] == "00000" and m.component_labels()[-1] == "11111"
    assert len(set(m.component_labels())) == 32
    # Directions blocked by the stall sites: site1 lies on the reverse strand, site4 on the forward strand
    blocked = {e.name: e.blocked_directions() for e in m.stall_events}
    assert blocked == {"site1": (-1,), "site2": (+1, -1), "site4": (-1,)}
    # Initial probabilities
    lam = m.temporal.growthrate.item()
    torch.testing.assert_close(m.events.p_fire, dtensor([np.exp(-lam * 50.0), np.exp(-lam * 120.0)]))
    assert torch.all((m.events.p_stall >= 0.0) & (m.events.p_stall <= 1.0))


def two_chromosome_original(names: tuple[str, str], delays: tuple[list[float], list[float]]) -> VelocityModel:
    s1 = Scenario("chr1", 20000, True, [2000, 9000, 15000],
                  g_xranges_l=[4000, 12000], g_xranges_u=[4500, 12500], g_delta=[0.5, 1.0], delay=delays[0])
    s2 = Scenario("chr2", 10000, False, [2000, 6000],
                  g_xranges_l=[3000, 8000], g_xranges_u=[3500, 9000], g_delta=[1.0, 0.2], delay=delays[1])
    temporal = TemporalModel(dtensor([0.0]), v0_min=dtensor([1.0]), v0_max=dtensor([1000.0]))
    chromosomes = [make_chromosome(s, n) for s, n in zip((s1, s2), names)]
    for s, c in zip((s1, s2), chromosomes):
        set_parameters(s, temporal, c)
    return VelocityModel(temporal, chromosomes), (s1, s2)


def test_event_order_across_chromosomes():
    # The model lists chromosome "chrB" first, events are nevertheless ordered by chromosome name
    original, _ = two_chromosome_original(("chrB", "chrA"), ([0.0, 30.0, 0.0], [50.0, 60.0]))
    m = MixtureModel(original, [])
    assert m.fixed_origins == [[0, 2], [0]]
    assert [(e.chromosome_name, e.name) for e in m.events_list] == [("chrA", "chrA.ori1"), ("chrB", "chrB.ori1")]
    # Ties in the smallest delay fix all origins sharing it, also for a non-zero smallest delay
    original, _ = two_chromosome_original(("chrB", "chrA"), ([0.0, 30.0, 40.0], [50.0, 50.0]))
    m = MixtureModel(original, [])
    assert m.fixed_origins == [[0], [0, 1]]
    assert m.event_names() == ["chrB.ori1", "chrB.ori2"]
    assert m.events.p_fire.numel() == 2


def test_stall_site_order_across_chromosomes():
    original, _ = two_chromosome_original(("chrB", "chrA"), ([0.0, 30.0, 40.0], [0.0, 60.0]))
    for c in original.chromosomes:
        c.g_names = [f"{c.name}.site{j + 1}" for j in range(len(c.g_names))]
    m = MixtureModel(original, [("chrB.site1", "both"), ("chrA.site2", "both"), ("chrA.site1", "both")])
    assert m.event_names() == ["chrA.ori1", "chrB.ori1", "chrB.ori2", "chrA.site1", "chrA.site2", "chrB.site1"]


def test_chromosome_delays():
    original, scenarios = two_chromosome_original(("chr1", "chr2"), ([0.0, 30.0, 40.0], [80.0, 90.0]))
    m = MixtureModel(original, [])
    lam = m.temporal.growthrate.item()
    # The chromosome delays are the delays of the fixed origins, the delays of the other origins are relative to it
    torch.testing.assert_close(m.chromosomes[1].delay(m.temporal), dtensor([80.0, 80.0]))
    torch.testing.assert_close(m.events.p_fire, dtensor([np.exp(-lam * 30.0), np.exp(-lam * 40.0),
                                                         np.exp(-lam * 10.0)]))
    # All origins of a chromosome fire at its chromosome delay
    x = torch.cat([model_positions(s) for s in scenarios])
    c = torch.cat([torch.full((len(model_positions(s)),), i) for i, s in enumerate(scenarios)])
    with torch.no_grad():
        a = m.component_abundances(x, c).numpy()
    for k in range(a.shape[1]):
        for ci, s in enumerate(scenarios):
            d = m.events.chromosome_delay_scaled[ci].item() / (10 * lam)
            cs = component_scenario(s, m, k, chromosome=ci, delay=d)
            assert_close_to_reference(cs, a[(c == ci).numpy(), k], lam)


@pytest.mark.parametrize("sites, message", [
    ([("nosuchsite", "both")], "not a feature"),
    ([("site1", "both"), ("site1", "co")], "more than once"),
    ([("site3", "co")], "unstranded"),
    ([("site1", "sideways")], "invalid polarity"),
])
def test_stall_site_errors(sites, message):
    with pytest.raises(ValueError, match=message):
        MixtureModel(make_mixture_original("circular"), sites)


def test_ambiguous_stall_site():
    original, _ = two_chromosome_original(("chr1", "chr2"), ([0.0, 30.0, 40.0], [0.0, 60.0]))
    with pytest.raises(ValueError, match="ambiguous"):
        MixtureModel(original, [("site1", "both")])


def weights_gradient(q: list[float], labels: list[str]) -> np.ndarray:
    """ d w_k / d q_j for w_k = prod_i (q_i if bit i of k is set else 1 - q_i) """
    g = np.zeros((len(labels), len(q)))
    for k, label in enumerate(labels):
        for j in range(len(q)):
            others = np.prod([qi if b == "1" else 1.0 - qi for i, (qi, b) in enumerate(zip(q, label)) if i != j])
            g[k, j] = others if label[j] == "1" else -others
    return g


@pytest.mark.parametrize("q", [[0.0, 1.0, 0.0, 1.0, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5], [1.0, 1.0, 0.0, 0.0, 0.0]])
def test_gradients(q):
    m = make_mixture("circular")
    s = MIXTURE_SCENARIOS["circular"][0]
    with torch.no_grad():
        m.events.p_fire.copy_(dtensor(q[:2]))
        m.events.p_stall.copy_(dtensor(q[2:]))
    m.events.p_fire.requires_grad_(True)
    m.events.p_stall.requires_grad_(True)
    m.temporal.v0_log.requires_grad_(True)
    m.chromosomes[0].g_delta_scaled.requires_grad_(True)
    x = model_positions(s)[::50]
    p = m.forward(x)
    torch.sum(p).backward()
    for t in (m.events.p_fire, m.events.p_stall, m.temporal.v0_log, m.chromosomes[0].g_delta_scaled):
        assert torch.all(torch.isfinite(t.grad))
    # dP/dq_j = sum_k A_k dw_k/dq_j
    with torch.no_grad():
        a = m.component_abundances(x).numpy()
    expected = np.sum(a @ weights_gradient(q, m.component_labels()), axis=0)
    grad = torch.cat((m.events.p_fire.grad, m.events.p_stall.grad)).numpy()
    np.testing.assert_allclose(grad, expected, rtol=1e-9, atol=1e-12)


def test_prepared_input_subsets():
    m = make_mixture("linear")
    x = model_positions(MIXTURE_SCENARIOS["linear"][0])
    x_prep = m.prepare_input(x)
    with torch.no_grad():
        p = m.forward(x)
        torch.testing.assert_close(m.forward(x_prep), p, rtol=0, atol=0)
        torch.testing.assert_close(m.forward(x_prep[100:200]), p[100:200], rtol=0, atol=0)
        assert m.forward(x_prep[1234]).item() == p[1234].item()


def test_loss_jacobian_and_posterior():
    m = make_mixture("circular")
    s = MIXTURE_SCENARIOS["circular"][0]
    x = model_positions(s)[::400].clone()
    m.events.p_fire = nn.Parameter(m.events.p_fire)
    m.events.p_stall = nn.Parameter(m.events.p_stall)
    m.temporal.v0_log = nn.Parameter(m.temporal.v0_log)
    with torch.no_grad():
        y = 2.0 * m.forward(x)
    params = ["temporal.v0_log", "events.p_fire", "events.p_stall"]
    loss = VelocityModelLoss(x, y.clone(), torch.full_like(y, 0.01), m, params)
    assert loss().item() == pytest.approx(0.0, abs=1e-12)
    J, pmap = loss.jacobian(params)
    assert J.shape == (len(x), 1 + 2 + 3)
    assert torch.all(torch.isfinite(J))
    u, pmap = m.posterior_cov_inv(loss, params)
    assert torch.all(torch.isfinite(u))
    assert set(pmap.keys()) == set(params)


def test_clamp_parameters():
    original, _ = two_chromosome_original(("chr1", "chr2"), ([0.0, 30.0, 40.0], [80.0, 90.0]))
    m = MixtureModel(original, [])
    with torch.no_grad():
        m.events.p_fire.copy_(dtensor([-0.5, 1.5, 0.3]))
        m.events.chromosome_delay_scaled.copy_(dtensor([0.7, 0.2]))
    m.clamp_parameters()
    torch.testing.assert_close(m.events.p_fire, dtensor([0.0, 1.0, 0.3]))
    torch.testing.assert_close(m.events.chromosome_delay_scaled, dtensor([0.5, 0.0]))
    # Origin delays of all models follow the chromosome delays
    torch.testing.assert_close(m.chromosomes[0].delay_scaled, dtensor([0.5, 0.5, 0.5]))
    for vm in m.patterns:
        torch.testing.assert_close(vm.chromosomes[0].delay_scaled,
                                   torch.full((vm.chromosomes[0].origins.numel(),), 0.5, dtype=torch.double))
    # Stall sites have g = 1
    m = make_mixture("circular")
    with torch.no_grad():
        m.chromosomes[0].g_delta_scaled.fill_(0.5)
    m.clamp_parameters()
    torch.testing.assert_close(m.chromosomes[0].g_delta_scaled, dtensor([0.0, 0.0, 0.5, 0.0]))


def test_original_unchanged_and_pickle():
    original = make_mixture_original("circular")
    before = copy.deepcopy(original.state_dict(keep_vars=False))
    delays_before = original.chromosomes[0].delay_scaled.clone()
    g_before = original.chromosomes[0].g_delta_scaled.clone()
    m = MixtureModel(original, MIXTURE_SCENARIOS["circular"][2])
    with torch.no_grad():
        m.chromosomes[0].g_delta_scaled.fill_(2.0)
        m.temporal.v0_log.fill_(3.0)
    m.clamp_parameters()
    assert torch.equal(original.chromosomes[0].delay_scaled, delays_before)
    assert torch.equal(original.chromosomes[0].g_delta_scaled, g_before)
    assert m.temporal is not original.temporal
    x = model_positions(MIXTURE_SCENARIOS["circular"][0])[::100]
    m2 = pickle.loads(pickle.dumps(m))
    with torch.no_grad():
        torch.testing.assert_close(m2.forward(x), m.forward(x), rtol=0, atol=0)
    # The firing pattern models of the unpickled model share its parameters
    with torch.no_grad():
        m2.temporal.v0_log.fill_(3.5)
        m2.chromosomes[0].g_delta_scaled[2] = 1.0
        assert not torch.equal(m2.forward(x), m.forward(x))
        assert all(vm.temporal is m2.temporal for vm in m2.patterns)


def test_windows_inside_blocked():
    m = make_mixture("circular")
    x = torch.tensor([3999, 4000, 4001, 5999, 6000, 7200, 12100], dtype=torch.int32)
    # Only site2 (4000-6000) has polarity "both"
    assert m.windows_inside_blocked(x, torch.zeros_like(x, dtype=torch.long)) == 2


def test_model_without_feature_names():
    """ Models unpickled from files written before g ranges had names cannot be given stall sites """
    original = make_mixture_original("circular")
    original.chromosomes[0].g_names = None
    MixtureModel(original, [])
    with pytest.raises(ValueError, match="re-run VeloGP"):
        MixtureModel(original, [("site1", "both")])
