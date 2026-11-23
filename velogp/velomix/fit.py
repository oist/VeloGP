"""
Fitting of the VeloMix mixture model, see `velogp.velomix.model.MixtureModel`
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
from torch import nn, Tensor

from ..fit import FitVelocityModel
from ..model import VelocityModelLoss
from .model import MixtureModel


def make_loss(model: MixtureModel, original_loss: VelocityModelLoss) -> VelocityModelLoss:
    """
    Creates the loss function of the mixture model for the observations of the original VeloGP fit

    :param model: The mixture model
    :param original_loss: The loss function of the original VeloGP fit, which contains the observations
    :return: A `VelocityModelLoss` instance which includes the priors of all parameters fitted by `FitMixtureModel`
    """
    return VelocityModelLoss(x=original_loss.x.clone(), y_obs=original_loss.y_obs.clone(),
                             s_obs=original_loss.s_obs.clone(), model=model,
                             params=FitMixtureModel.parameter_names(model),
                             chromosome=original_loss.chromosome.clone())


class FitMixtureModel(FitVelocityModel):
    """
    Fits a `MixtureModel`, starting from the parameters of the VeloGP fit it was created from

    Fits v(t), the g of the g ranges which are not stall sites, the event probabilities and (for genomes comprising
    more than one chromosome) the chromosome delays. All other methods are inherited from `FitVelocityModel`.
    """

    def __init__(self, model: MixtureModel, loss: VelocityModelLoss):
        super().__init__(model, loss)

    @staticmethod
    def parameter_names(model: MixtureModel) -> List[str]:
        """ The names of the parameters which are fitted """
        names = ["temporal.v0_log", "temporal.h_log_raw"]
        names += [f"chromosomes.{i}.g_delta_scaled" for i in range(len(model.chromosomes))]
        if model.events.p_fire.numel() > 0:
            names.append("events.p_fire")
        if model.events.p_stall.numel() > 0:
            names.append("events.p_stall")
        if len(model.chromosomes) > 1:
            names.append("events.chromosome_delay_scaled")
        return names

    def make_parameters(self) -> List[nn.Parameter]:
        """ Turns the fitted tensors into `nn.Parameter`s (keeping their values) and returns them """
        m: MixtureModel = self.model
        # The prior on h(t) is simplest in the scaled eigenbasis, see FitVelocityModel.optimize_parameters()
        m.temporal.h_switch_basis("eigenbasis_scaled")
        for module, name in [(m.temporal, "v0_log"), (m.temporal, "h_log_raw"),
                             *[(c, "g_delta_scaled") for c in m.chromosomes],
                             (m.events, "p_fire"), (m.events, "p_stall"), (m.events, "chromosome_delay_scaled")]:
            if not isinstance(getattr(module, name), nn.Parameter):
                setattr(module, name, nn.Parameter(getattr(module, name).detach().clone()))
        # Bind the firing pattern models to the new parameter objects
        m.clamp_parameters()
        return [p for n, p in m.resolve_params(self.parameter_names(m))]

    def optimize_parameters_stochastic(self, perturb_rejects: int = 10, perturb_reduce: float = 0.5,
                                       perturb_end: float = 0.01, perturb_start: float = 0.25,
                                       log_phases=False, log_rounds=False, log_steps=False, log_iterations=False,
                                       log_params=False):
        """
        Optimize all parameters using gradient-informed stochastic searching, starting from their current values

        Unlike `FitVelocityModel.optimize_parameters_stochastic()`, h(t) and g are not reset, and the search starts
        with smaller perturbations (`perturb_start`) since the parameters of the VeloGP fit are a good starting point.
        """
        params = self.make_parameters()
        if log_phases:
            print("Stochastic optimization of v0, h(t), g(x), firing and stalling probabilities" +
                  (" and chromosome delays" if len(self.model.chromosomes) > 1 else ""))
        self.run_stochastic(params, perturb_rejects=perturb_rejects, perturb_reduce=perturb_reduce,
                            perturb_start=perturb_start, perturb_end=perturb_end,
                            log_rounds=log_rounds, log_steps=log_steps, log_iterations=log_iterations,
                            log_params=log_params)
        if log_phases:
            print("Finalizing all parameters until the objective cannot be further improved")
        self.run_lbfgs(params, lr_start=1.0, lr_end=1.0, iterations=100, steps=None, retry=30,
                       tolerance=1e-6, line_search_fn="strong_wolfe",
                       log_steps=log_steps, log_iterations=log_iterations, log_params=log_params)

    def optimize_parameters(self, *args, **kwargs):
        raise NotImplementedError("VeloMix only supports optimize_parameters_stochastic()")


def posterior_cov(model: MixtureModel, loss: VelocityModelLoss, params: List[str],
                  fixed: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, int]]:
    """
    The posterior covariance of the given parameters, conditional on the entries listed in `fixed`

    Entries which are fixed (stall sites' g) or lie on the boundary of their domain (probabilities 0 or 1) are not
    free parameters, their rows and columns are removed from the posterior precision matrix before inverting it,
    and are NaN in the result. Entries which carry no information at all (e.g. the probability of an event which
    does not affect the predicted abundances, and has no prior) have a zero row in the precision matrix (up to
    rounding errors, entries whose diagonal element is below 1e-12 times the largest one are considered
    uninformative). They are removed as well, their variance is +inf.

    :param model: The mixture model
    :param loss: Its loss function
    :param params: Parameter names
    :param fixed: Maps parameter names to boolean masks of the fixed entries
    """
    u, pmap = model.posterior_cov_inv(loss, params)
    keep = torch.ones(u.shape[0], dtype=torch.bool)
    for n, mask in fixed.items():
        if n in pmap:
            keep[pmap[n]:pmap[n] + mask.numel()] = ~mask
    # Rows of uninformative entries are zero up to rounding errors
    d = torch.diagonal(u)
    uninformative = keep & (d <= 1e-12 * torch.max(torch.abs(d)))
    keep &= ~uninformative
    cov = torch.full_like(u, math.nan)
    k = torch.nonzero(keep).reshape(-1)
    cov[k[:, None], k[None, :]] = torch.linalg.inv(u[k][:, k])
    i = torch.nonzero(uninformative).reshape(-1)
    cov[i, i] = math.inf
    return cov, pmap
