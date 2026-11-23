"""
Tests for pickling and unpickling the model classes, the loss and the fit algorithm, see "Pickling" in velogp/model.py

The legacy pickles in tests/data were written by older versions of VeloGP (see tests/data/make_legacy_pickles.py),
together with the values these versions computed for the pickled model.

Run with `python -m pytest` from the repository root.
"""

from __future__ import annotations

import copy
import io
import pathlib
import pickle

import pytest
import torch
from torch import nn

from velogp import ChromosomeModel, TemporalModel, VelocityModel, VelocityModelLoss, FitVelocityModel, dtensor
from velogp import model as model_module

from test_velocity_model import (SCENARIOS, BLOCKED_SCENARIOS, make_model, model_positions, make_genome_model,
                                 genome_positions)

DATA = pathlib.Path(__file__).parent / "data"
# Commits which wrote the legacy pickles. 6368479 predates g_names, g_strands and blocked g ranges.
LEGACY = ["6368479", "d3a818e"]
PARAMETER_NAMES = ["temporal.v0_log", "temporal.h_log_raw", "chromosomes.0.delay_scaled", "chromosomes.0.g_delta_scaled",
                   "chromosomes.1.delay_scaled", "chromosomes.1.g_delta_scaled"]


def round_trip(obj):
    return pickle.loads(pickle.dumps(obj))


def load_legacy(commit: str) -> dict:
    with open(DATA / f"legacy-{commit}.pkl", "rb") as f:
        return pickle.load(f)


@pytest.mark.parametrize("commit", LEGACY)
def test_legacy_pickle_reproduces_values(commit: str):
    o = load_legacy(commit)
    m, loss, alg, expected = o["model"], o["loss"], o["algorithm"], o["expected"]
    with torch.no_grad():
        assert loss().item() == pytest.approx(expected["loss"], rel=1e-12)
        torch.testing.assert_close(loss.y_pred(), expected["y_pred"], rtol=1e-12, atol=0)
        assert m.completion_time().item() == pytest.approx(expected["completion_time"], rel=1e-12)
        torch.testing.assert_close(m.trajectories()[1], expected["trajectories_x"], rtol=1e-12, atol=1e-9)
        torch.testing.assert_close(m.g_range_passage_times()[0], expected["passage_times"], rtol=1e-12, atol=0)
        torch.testing.assert_close(m.temporal.h_log("canonical"), expected["h_canonical"], rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("commit", LEGACY)
def test_legacy_pickle_structure(commit: str):
    o = load_legacy(commit)
    m, loss, alg = o["model"], o["loss"], o["algorithm"]
    # Shared references and parameters are preserved
    assert (loss.model is m) and (alg.model is m) and (alg.loss is loss)
    assert [n for n, _ in m.named_parameters()] == PARAMETER_NAMES
    assert m.temporal.h_basis == "eigenbasis_scaled"
    # Members which did not exist when the pickle was written receive their defaults
    for c in m.chromosomes:
        assert not torch.any(c.g_blocked_fwd) and not torch.any(c.g_blocked_rev)
    if commit == "6368479":
        assert all(c.g_names is None for c in m.chromosomes)
        assert all(torch.all(c.g_strands == 0) for c in m.chromosomes)
    else:
        assert m.chromosomes[0].g_names == ["chrA.site1", "chrA.site2"]
        assert m.chromosomes[0].g_strands.tolist() == [1, -1]
    # The model is fully functional, including the posterior covariance (which needs the h(t) caches)
    cov, pmap = m.posterior_cov(loss, ["temporal.h_log_raw"])
    assert torch.all(torch.isfinite(cov))
    alg.run_lbfgs([p for _, p in m.named_parameters()], iterations=5, steps=1)


@pytest.mark.parametrize("commit", LEGACY)
def test_legacy_pickle_rewritten_in_current_format(commit: str):
    o = load_legacy(commit)
    o2 = round_trip(o)
    with torch.no_grad():
        # h(t) is stored in the canonical basis, converting it back to the eigenbasis is exact up to rounding
        assert o2["loss"]().item() == pytest.approx(o["loss"]().item(), rel=1e-12)
    assert o2["loss"].model is o2["model"]
    assert [n for n, _ in o2["model"].named_parameters()] == PARAMETER_NAMES


@pytest.mark.parametrize("s", SCENARIOS + BLOCKED_SCENARIOS, ids=[s.name for s in SCENARIOS + BLOCKED_SCENARIOS])
def test_round_trip_reproduces_model(s):
    m = make_model(s)
    c = m.chromosomes[0]
    c.g_names = [f"feature{j}" for j in range(len(c.g_names))]
    c.g_strands = torch.tensor([(-1) ** j for j in range(len(c.g_names))], dtype=torch.int32)
    m2 = round_trip(m)
    c2 = m2.chromosomes[0]
    x = model_positions(s)
    with torch.no_grad():
        torch.testing.assert_close(m2.forward(x), m.forward(x), rtol=0, atol=0)
    assert (c2.name, c2.is_circular, c2.origin_names, c2.g_names) == (c.name, c.is_circular, c.origin_names, c.g_names)
    for n in ["length", "origins", "g_xranges_l", "g_xranges_u", "g_strands", "delay_scaled", "g_delta_scaled",
              "g_delta_prefactor", "reference_length", "g_blocked_fwd", "g_blocked_rev"]:
        assert torch.equal(getattr(c2, n), getattr(c, n)), n
    for n in ["h_tgrid", "growthrate", "v0_log", "h_log_raw", "h_sigma", "h_timescale"]:
        assert torch.equal(getattr(m2.temporal, n), getattr(m.temporal, n)), n


def test_round_trip_multi_chromosome_objects():
    """ The objects written by --out-objects: model, loss, fit algorithm and posterior covariances """
    m = make_genome_model()
    x, chrom = genome_positions()
    x, chrom = x[::200], chrom[::200]
    m.temporal.h_switch_basis("eigenbasis_scaled")
    m.temporal.h_log_raw = nn.Parameter(m.temporal.h_log_raw)
    m.chromosomes[1].g_delta_scaled = nn.Parameter(m.chromosomes[1].g_delta_scaled)
    with torch.no_grad():
        y = m.forward(m.prepare_input(x, chrom))
    loss = VelocityModelLoss(x.double(), y.clone(), 0.01 * y, m, ["temporal.h_log_raw", "chromosomes.1.g_delta_scaled"],
                             chromosome=chrom)
    alg = FitVelocityModel(m, loss)
    cov, pmap = m.posterior_cov(loss, ["temporal.h_log_raw"])
    o = round_trip({"model": m, "loss": loss, "algorithm": alg, "cov_hg": cov, "cov_hg.pmap": pmap})
    m2, loss2, alg2 = o["model"], o["loss"], o["algorithm"]
    assert (loss2.model is m2) and (alg2.model is m2) and (alg2.loss is loss2)
    assert [n for n, _ in m2.named_parameters()] == [n for n, _ in m.named_parameters()]
    assert [c.name for c in m2.chromosomes] == ["U00096.3", "plasmid.1"]
    with torch.no_grad():
        # h(t) is stored in the canonical basis, converting it back to the eigenbasis is exact up to rounding
        assert loss2().item() == pytest.approx(loss().item(), rel=1e-12)
        torch.testing.assert_close(loss2.y_pred(), loss.y_pred(), rtol=1e-12, atol=0)
        torch.testing.assert_close(m2.temporal.h_log_raw, m.temporal.h_log_raw, rtol=0, atol=1e-12)
    assert torch.equal(loss2.chromosome, loss.chromosome)
    torch.testing.assert_close(o["cov_hg"], cov, rtol=0, atol=0)


def test_loss_with_parameter_objects():
    """ A loss given parameter objects instead of names refers to the unpickled model's parameters """
    m = make_model(SCENARIOS[-1])
    m.temporal.v0_log = nn.Parameter(m.temporal.v0_log)
    m.chromosomes[0].g_delta_scaled = nn.Parameter(m.chromosomes[0].g_delta_scaled)
    x = model_positions(SCENARIOS[-1])[::50].double()
    with torch.no_grad():
        y = m.forward(x.int())
    loss = VelocityModelLoss(x, y.clone(), 0.01 * y, m, [m.temporal.v0_log, m.chromosomes[0].g_delta_scaled])
    loss2 = round_trip(loss)
    assert loss2.params == ["temporal.v0_log", "chromosomes.0.g_delta_scaled"]
    with torch.no_grad():
        assert loss2().item() == pytest.approx(loss().item(), rel=1e-12)


def test_derived_members_are_not_stored():
    """ Only structure and parameters are pickled, derived members and caches are recomputed when unpickling """
    m = make_genome_model()
    x, chrom = genome_positions()
    m.temporal.h_switch_basis("eigenbasis_scaled")
    with torch.no_grad():
        y = m.forward(m.prepare_input(x, chrom))
    loss = VelocityModelLoss(x.double(), y.clone(), 0.01 * y, m, [], chromosome=chrom)
    data = pickle.dumps(loss)
    for name in [b"g_xgrid", b"g_delta_prefactor", b"origins_augmented", b"h_cov_eig", b"h_cov_chol", b"h_tgrid_len",
                 b"x_prep", b"weights", b"_forward_hooks"]:
        assert name not in data, name


def test_members_added_later_are_initialized(monkeypatch):
    """ Members added to a class after a pickle was written are initialized by the constructor when unpickling """
    data = pickle.dumps(make_genome_model())
    original_init = ChromosomeModel.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.member_added_later = "default"

    monkeypatch.setattr(ChromosomeModel, "__init__", init)
    m = pickle.loads(data)
    assert all(c.member_added_later == "default" for c in m.chromosomes)


def test_deepcopy():
    m = make_model(SCENARIOS[-1])
    m.temporal.v0_log = nn.Parameter(m.temporal.v0_log)
    m2 = copy.deepcopy(m)
    assert isinstance(m2.temporal.v0_log, nn.Parameter) and (m2.temporal.v0_log is not m.temporal.v0_log)
    with torch.no_grad():
        m2.temporal.v0_log += 0.1
        m2.chromosomes[0].g_delta_scaled += 0.1
    assert not torch.equal(m2.temporal.v0_log, m.temporal.v0_log)
    assert not torch.equal(m2.chromosomes[0].g_delta_scaled, m.chromosomes[0].g_delta_scaled)


def test_torch_save_and_load():
    """ --out-model writes the model with torch.save() """
    m = make_genome_model()
    buffer = io.BytesIO()
    torch.save(m, buffer)
    buffer.seek(0)
    m2 = torch.load(buffer, weights_only=False)
    x, chrom = genome_positions()
    with torch.no_grad():
        torch.testing.assert_close(m2.forward(m2.prepare_input(x, chrom)), m.forward(m.prepare_input(x, chrom)),
                                   rtol=0, atol=0)


def test_newer_state_version_rejected(monkeypatch):
    data = pickle.dumps(make_genome_model())
    monkeypatch.setattr(model_module, "STATE_VERSION", model_module.STATE_VERSION - 1)
    with pytest.raises(RuntimeError, match="newer version"):
        pickle.loads(data)


def test_single_chromosome_layout_rejected():
    """ Pickles written before VelocityModel supported multiple chromosomes cannot be converted """
    m = VelocityModel.__new__(VelocityModel)
    with pytest.raises(RuntimeError, match="multiple chromosomes"):
        m.__setstate__({"_modules": {"temporal": make_genome_model().temporal}, "training": True})
