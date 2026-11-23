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
import math
import pathlib
import pickle

import numpy as np
import pytest
import torch
from torch import nn

from velogp import ChromosomeModel, TemporalModel, VelocityModel, VelocityModelLoss, FitVelocityModel, dtensor
from velogp.velomix import MixtureModel, FitMixtureModel, make_loss, posterior_cov

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


# A small scenario in which both events clearly affect the abundance, for fitting
FIT_SCENARIO = (
    Scenario("fit", 20000, True, [2000, 11000], h_tgrid=H_TGRID, h_values=[1.0] * len(H_TGRID),
             g_xranges_l=[5000, 14000], g_xranges_u=[5500, 15000], g_delta=[0.5, 1.0], delay=[0.0, 50.0]),
    [0, 1],
    [("site1", "both")])


def make_fit_problem(p_fire: float, p_stall: float):
    s, strands, sites = FIT_SCENARIO
    original = make_model(s)
    original.chromosomes[0].g_strands = torch.tensor(strands, dtype=torch.int32)
    truth = MixtureModel(original, sites)
    with torch.no_grad():
        truth.events.p_fire.fill_(p_fire)
        truth.events.p_stall.fill_(p_stall)
        x = model_positions(s)[::100].clone()
        y = 1.7 * truth.forward(x)
    original_loss = VelocityModelLoss(x, y.clone(), 0.002 * y, original, [])
    m = MixtureModel(original, sites)
    return m, FitMixtureModel(m, make_loss(m, original_loss))


def test_fit_parameters():
    m, alg = make_fit_problem(0.4, 0.6)
    assert FitMixtureModel.parameter_names(m) == ["temporal.v0_log", "temporal.h_log_raw",
                                                  "chromosomes.0.g_delta_scaled", "events.p_fire", "events.p_stall"]
    # Turning the fitted tensors into parameters keeps their values, i.e. h(t) and g are not reset. (The loss
    # changes, since it only includes the priors of registered parameters.)
    with torch.no_grad():
        before = alg.loss.y_pred().clone()
        g_before = m.chromosomes[0].g_delta_scaled.clone()
    params = alg.make_parameters()
    assert all(isinstance(p, nn.Parameter) for p in params) and len(params) == 5
    with torch.no_grad():
        torch.testing.assert_close(alg.loss.y_pred(), before, rtol=1e-12, atol=0)
        torch.testing.assert_close(m.chromosomes[0].g_delta_scaled, g_before)
    # Gradients reach all parameters through the firing pattern models
    alg.evaluate_model_for_parameters(params)
    assert all((p.grad is not None) and torch.all(torch.isfinite(p.grad)) for p in params)
    assert m.events.p_fire.grad.abs().item() > 0.0 and m.events.p_stall.grad.abs().item() > 0.0
    with pytest.raises(NotImplementedError):
        alg.optimize_parameters()


def test_fit_recovers_probabilities():
    torch.manual_seed(0)
    m, alg = make_fit_problem(0.4, 0.6)
    alg.optimize_parameters_stochastic(perturb_rejects=2, perturb_end=0.1)
    assert m.events.p_fire.item() == pytest.approx(0.4, abs=0.02)
    assert m.events.p_stall.item() == pytest.approx(0.6, abs=0.02)
    # g of the remaining range is recovered, the stall site keeps g = 1
    torch.testing.assert_close(m.chromosomes[0].g, dtensor([1.0, 2.0]), rtol=0, atol=0.05)


def test_pickle_fitted_mixture():
    """ A mixture with parameters, its loss and fit algorithm survive pickling, as written to --out-objects """
    m, alg = make_fit_problem(0.4, 0.6)
    params = alg.make_parameters()
    with torch.no_grad():
        m.events.p_fire.fill_(0.3)
        m.chromosomes[0].g_delta_scaled[1] = 0.05
    o = pickle.loads(pickle.dumps({"model": m, "loss": alg.loss, "algorithm": alg}))
    m2, loss2, alg2 = o["model"], o["loss"], o["algorithm"]
    assert (loss2.model is m2) and (alg2.model is m2) and (alg2.loss is loss2)
    assert [n for n, _ in m2.named_parameters()] == [n for n, _ in m.named_parameters()]
    assert (m2.event_names(), m2.fixed_origins, m2.component_labels()) == \
           (m.event_names(), m.fixed_origins, m.component_labels())
    assert m2.events.p_fire.item() == 0.3
    with torch.no_grad():
        assert loss2().item() == pytest.approx(alg.loss().item(), rel=1e-12)
    # The firing pattern models of the unpickled mixture use its parameters, so gradients reach them
    params2 = alg2.make_parameters()
    alg2.evaluate_model_for_parameters(params2)
    assert all(torch.all(torch.isfinite(p.grad)) for p in params2)
    assert m2.events.p_fire.grad.abs().item() > 0.0 and m2.chromosomes[0].g_delta_scaled.grad[1].abs().item() > 0.0


def test_model_without_feature_names():
    """ Models unpickled from files written before g ranges had names cannot be given stall sites """
    original = make_mixture_original("circular")
    original.chromosomes[0].g_names = None
    MixtureModel(original, [])
    with pytest.raises(ValueError, match="re-run VeloGP"):
        MixtureModel(original, [("site1", "both")])


def write_velogp_objects(path):
    """ Writes the objects of a (simulated) VeloGP fit as VeloGP's --out-objects does """
    s, strands, sites = FIT_SCENARIO
    original = make_model(s)
    original.chromosomes[0].g_strands = torch.tensor(strands, dtype=torch.int32)
    truth = MixtureModel(original, sites)
    with torch.no_grad():
        truth.events.p_fire.fill_(0.4)
        truth.events.p_stall.fill_(0.6)
        x = model_positions(s)[::100].clone()
        y = 1.7 * truth.forward(x)
    loss = VelocityModelLoss(x.double(), y.clone(), 0.002 * y, original, ["temporal.v0_log"])
    import gzip
    with gzip.open(path, "wb") as f:
        pickle.dump({"model": original, "loss": loss, "algorithm": FitVelocityModel(original, loss)}, f)


def run_velomix(*args):
    import os, subprocess, sys
    root = pathlib.Path(__file__).parent.parent
    env = {**os.environ, "PYTHONPATH": str(root), "MPLBACKEND": "Agg"}
    return subprocess.run([sys.executable, "-m", "velogp.velomix", *map(str, args)], cwd=root, env=env,
                          capture_output=True, text=True)


def test_cli(tmp_path):
    import pandas as pd
    objects = tmp_path / "velogp.objs.pkl.gz"
    write_velogp_objects(objects)
    out = {k: tmp_path / f"velomix.{k}" for k in ["objs.pkl", "model.pt", "prob", "abd", "vt", "vx", "traj",
                                                   "abd.pdf", "vt.pdf"]}
    r = run_velomix(objects, "--stall-sites", "site1:both", "--seed", "1", "--perturbation-rejects", "2",
                    "--perturbation-min", "0.1",
                    "--out-objects", out["objs.pkl"], "--out-model", out["model.pt"], "--out-probabilities", out["prob"],
                    "--out-abundance", out["abd"], "--out-vt", out["vt"], "--out-vx-feature-delays", out["vx"],
                    "--out-trajectories", out["traj"], "--plot-abundance", out["abd.pdf"], "--plot-vt", out["vt.pdf"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ERROR" not in r.stdout, r.stdout
    assert all(p.exists() for p in out.values())
    # Probabilities: all origins (ori0 always fires), then the stall sites
    prob = pd.read_csv(out["prob"], sep="\t")
    assert prob.name.tolist() == ["ori0", "ori1", "site1"]
    assert prob.type.tolist() == ["origin", "origin", "stall"]
    assert prob.fixed.tolist() == [1, 0, 0] and prob.probability[0] == 1.0
    assert prob.probability[1] == pytest.approx(0.4, abs=0.02) and prob.probability[2] == pytest.approx(0.6, abs=0.02)
    assert np.isnan(prob.stderr[0]) and np.all(prob.stderr[1:] > 0.0)
    assert prob.begin.tolist() == [2000, 11000, 5000] and prob.polarity[2] == "both"
    # Abundances of the mixture and of each component, and the component weights
    abd = pd.read_csv(out["abd"], sep="\t")
    labels = ["00", "01", "10", "11"]
    assert [c for c in abd.columns if c[:2] in ("y.", "w.")] == [f"y.{l}" for l in labels] + [f"w.{l}" for l in labels]
    w = np.array([abd[f"w.{l}"][0] for l in labels])
    assert w.sum() == pytest.approx(1.0, abs=1e-5)
    mix = sum(abd[f"w.{l}"] * abd[f"y.{l}"] for l in labels)
    np.testing.assert_allclose(mix, abd.y_pred, atol=1e-4)
    np.testing.assert_allclose(abd.y_pred, abd.y_obs, rtol=0.02)
    # v(x) delays only for the features which are not stall sites
    vx = pd.read_csv(out["vx"], sep="\t")
    assert vx.name.tolist() == ["site2"]
    assert vx.g[0] == pytest.approx(2.0, abs=0.05)
    # The objects of the VeloGP fit plus the mixture
    with open(out["objs.pkl"], "rb") as f:
        o = pickle.load(f)
    assert {"model", "loss", "algorithm", "mixture"} <= set(o.keys())
    mix = o["mixture"]
    assert isinstance(mix["model"], MixtureModel) and (mix["loss"].model is mix["model"])
    assert mix["events"] == ["ori1", "site1"] and mix["components"] == labels
    assert mix["model"].events.p_fire.item() == pytest.approx(prob.probability[1], abs=1e-6)
    m = torch.load(out["model.pt"], weights_only=False)
    assert m.event_names() == ["ori1", "site1"]


@pytest.mark.parametrize("sites, message", [("nosuchsite", "not a feature"), ("site1:sideways", "invalid polarity"),
                                            ("site1:co", "unstranded")])
def test_cli_invalid_stall_sites(tmp_path, sites, message):
    objects = tmp_path / "velogp.objs.pkl.gz"
    write_velogp_objects(objects)
    r = run_velomix(objects, "--stall-sites", sites)
    assert r.returncode == 2
    assert message in r.stderr and "Traceback" not in r.stderr


def test_posterior_cov_fixed_and_uninformative():
    """ Fixed entries have NaN variance, entries without any effect on the abundance infinite variance """
    # Single origin at 2000: the replisomes meet at 12000. site1 blocks only replisomes moving in direction -1
    # ("head" on the forward strand), but the replisome moving leftwards never reaches it. site2 is passed first
    # by that replisome and blocks both directions.
    s = Scenario("posterior", 20000, True, [2000], g_xranges_l=[5000, 14000, 17000], g_xranges_u=[5500, 14500, 18000],
                 g_delta=[0.0, 0.0, 1.0])
    original = make_model(s)
    original.chromosomes[0].g_strands = torch.tensor([1, 0, 0], dtype=torch.int32)
    m = MixtureModel(original, [("site1", "head"), ("site2", "both")])
    with torch.no_grad():
        m.events.p_stall.copy_(dtensor([0.3, 0.6]))
        x = model_positions(s)[::100].clone()
        y = m.forward(x)
    alg = FitMixtureModel(m, make_loss(m, VelocityModelLoss(x.double(), y.clone(), 0.01 * y, original, [])))
    alg.make_parameters()
    fixed = {"chromosomes.0.g_delta_scaled": m.stall_ranges[0]}
    cov, pmap = posterior_cov(m, alg.loss, ["chromosomes.0.g_delta_scaled", "events.p_stall"], fixed)
    var = torch.diagonal(cov)
    g, p = pmap["chromosomes.0.g_delta_scaled"], pmap["events.p_stall"]
    # g of the stall sites is fixed, g of site3 is free
    assert torch.isnan(var[g]) and torch.isnan(var[g + 1]) and torch.isfinite(var[g + 2]) and var[g + 2] > 0
    # site1 has no effect, site2 does
    assert var[p].item() == math.inf
    assert torch.isfinite(var[p + 1]) and var[p + 1] > 0
