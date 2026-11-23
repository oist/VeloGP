"""
Creates the legacy pickles used by tests/test_pickle.py

The pickles contain a model, a loss and a fit algorithm as written to --out-objects, plus the values which the code
that wrote them computed for the model, so that the tests can check that unpickling with the current code reproduces
them. They must be written by the version of VeloGP they represent, i.e. run this script with an old checkout on
the Python path, e.g.

    git worktree add /tmp/velogp-old <commit>
    PYTHONPATH=/tmp/velogp-old python tests/data/make_legacy_pickles.py tests/data/legacy-<commit>.pkl
    git worktree remove /tmp/velogp-old
"""
import inspect
import pickle
import sys

import torch as pt
from torch import nn

from velogp import model, fit

pt.manual_seed(0)
H_TGRID = pt.arange(0, 3000, 30, dtype=pt.float64)


def chromosome(name, length, circular, origins, ranges, strands):
    kwargs = dict(origins=pt.tensor(origins, dtype=pt.int32), origin_names=[f"{name}.ori{i}" for i in range(len(origins))],
                  g_xranges_l=pt.tensor([l for l, u in ranges], dtype=pt.int32),
                  g_xranges_u=pt.tensor([u for l, u in ranges], dtype=pt.int32),
                  g_delayscale=pt.tensor([0.01] * len(ranges), dtype=pt.float64))
    # g_names and g_strands only exist since commit 6c0d640
    if "g_names" in inspect.signature(model.ChromosomeModel.__init__).parameters:
        kwargs.update(g_names=[f"{name}.site{j + 1}" for j in range(len(ranges))],
                      g_strands=pt.tensor(strands, dtype=pt.int32))
    return model.ChromosomeModel(name, length, circular, **kwargs)


m = model.VelocityModel(
    model.TemporalModel(H_TGRID.clone(), v0_min=model.dtensor(1.0), v0_max=model.dtensor(1000.0)),
    [chromosome("chrA", 100000, True, [10000, 60000], [(30000, 31000), (80000, 82000)], [1, -1]),
     chromosome("chrB", 40000, False, [20000], [(5000, 6000)], [0])])
m.temporal.growthrate[0] = 1.0 / 1800
m.temporal.h_cov_function = "m1"
m.temporal.h_sigma[0] = 0.1
m.temporal.h_timescale[0] = 300.0
with pt.no_grad():
    m.temporal.v0 = model.dtensor(60.0)
    m.chromosomes[0].set_delay(m.temporal, model.dtensor([0.0, 200.0]))
    m.chromosomes[1].set_delay(m.temporal, model.dtensor([120.0]))
    for c in m.chromosomes:
        c.g = pt.linspace(1.5, 3.0, c.g_xranges_l.numel(), dtype=pt.float64)
# Parameters as they are after fitting with FitVelocityModel.optimize_parameters_stochastic()
m.temporal.h_switch_basis("eigenbasis_scaled")
m.temporal.h_log_raw = nn.Parameter(0.3 * pt.randn(m.temporal.h_log_raw.shape, dtype=pt.float64))
m.temporal.v0_log = nn.Parameter(m.temporal.v0_log)
for c in m.chromosomes:
    c.delay_scaled = nn.Parameter(c.delay_scaled)
    c.g_delta_scaled = nn.Parameter(c.g_delta_scaled)
m.clamp_parameters()

x = pt.cat([pt.arange(500, 100000, 1000), pt.arange(500, 40000, 1000)]).double()
chrom = pt.cat([pt.zeros(100, dtype=pt.long), pt.ones(40, dtype=pt.long)])
with pt.no_grad():
    y = 1.3 * m.P(m.prepare_input(x, chrom))
y_obs = y * (1.0 + 0.01 * pt.randn(y.shape, dtype=pt.float64))
params = (["temporal.v0_log"] + [f"chromosomes.{i}.delay_scaled" for i in range(2)] +
          ["temporal.h_log_raw"] + [f"chromosomes.{i}.g_delta_scaled" for i in range(2)])
loss = model.VelocityModelLoss(x=x, y_obs=y_obs, s_obs=0.01 * y_obs, model=m, params=params, chromosome=chrom)
alg = fit.FitVelocityModel(m, loss)

with pt.no_grad():
    expected = dict(loss=loss().item(), y_pred=loss.y_pred().clone(), completion_time=m.completion_time().item(),
                    trajectories_x=m.trajectories()[1].clone(), passage_times=m.g_range_passage_times()[0].clone(),
                    h_canonical=m.temporal.h_log("canonical").clone())
with open(sys.argv[1], "wb") as f:
    pickle.dump({"model": m, "loss": loss, "algorithm": alg, "expected": expected}, f)
print(f"wrote {sys.argv[1]}")
