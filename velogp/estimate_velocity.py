from typing import Any, Dict, Optional, Union, Tuple
import operator
from functools import reduce
from velocity_model import VelocityModel, VelocityModelLoss, dtensor
import math
import numpy as np
import torch
from torch import nn, optim
from torch import *
from builtins import float, int, range

class FitVelocityModel:
    def __init__(self, model: VelocityModel, loss: VelocityModelLoss):
        self.model = model
        self.loss = loss

    def parameter_loss_function(self, optimizer, params):
        def s():
            # If any parameter is nan, return nan
            if not reduce(operator.and_, [ all(isfinite(p)) for p in params ]):
                return torch.tensor([math.nan])
            # Clamp parameters to their domain before each iteration
            self.model.clamp_parameters()
            # Evaluate loss and gradients
            with no_grad():
                self.model.update_h_cov_eig()
            optimizer.zero_grad()
            l = self.loss() + reduce(operator.add, [self.model.prior(p, False) for p in params])
            l.backward()
            return l
        return s

    def parameter_posterior_cov_inv(self):
        params = [self.model.v0, self.model.g_log, self.model.h_log_raw]
        dM, pmap = self.loss.jacobian(params)
        # Compute dM^T diag(1/s_1^2, ..., 1/s_N^2) dM
        u = einsum("ji,j,jk->ik", dM, self.loss.weights, dM).clone()
        # Add Sigma_h^-1 for h_log_raw, which has a GP prior
        # We do this by querying the parameter map to find the
        # sub-matrix of u that corresponds to the parameter h_log_raw,
        # and adding the inverse of its prior covariance matrix
        i, l = pmap["h_log_raw"], numel(self.model.h_log_raw)
        u[i:(i+l), i:(i+l)] += self.model.h_cov_inv()
        # Add prior for g_log
        i, l = pmap["g_log"], numel(self.model.g_log)
        u[i:(i+l), i:(i+l)] += 1.0 / self.model.g_sigma ** 2
        return u, pmap

    def parameter_posterior_cov(self):
        u, pmap = self.parameter_posterior_cov_inv()
        eb = VelocityModel.Eigenbasis.make(u)
        uinv = einsum("ij,j,kj->ik", eb.eigenvectors, tensor([1.0]) / eb.eigenvalues, eb.eigenvectors)
        return uinv, pmap

    def h_sigma_lengthscale_loss_function(self, optimizer):
        def s():
            raise "Implement me"

    def local_fit(self, origin : Tensor, neighbourhood : int):
        # We use that A(d) = A(o) * exp(- k * |d|) where k = lambda/v, o is
        # the origin and d is the distance from the origin. It follows that
        # log A(d) = log A(o) - k * |d|, i.e. we can estimate A(o) and k using
        # ordinary least squares.
        s = (abs(self.loss.x - origin) <= neighbourhood)
        x = self.loss.x[s]
        log_a = log(self.loss.y_obs[s]).detach().numpy()
        d = abs(x - origin).detach().numpy()
        r = np.linalg.lstsq(np.vstack([d, np.ones_like(d)]).T, log_a, rcond=None)[0]
        k = - r[0]
        log_a0 = r[1]
        if k <= 0:
            raise RuntimeError(f"no local maximum around origin at {origin}")
        return dtensor(k), dtensor(log_a0)

    def optimize_parameters(self):
        self.model.h_switch_basis("canonical")
        # I. Fit A(d) = A(o) * exp(- k * |d|) where k = lambda/v locally. Then
        # use the global relationship A(o) = A_max * exp(- lambda * delta_o)
        # between the abundances A(o) at different. origins o and the firing
        # delay delta_o of those origins to estimate delta_o.
        n = self.model.origins.size()[0]
        origin_v = dtensor([nan] * n)
        origin_loga0 = dtensor([nan] * n)
        for i in range(0, n):
            k, log_a0 = self.local_fit(self.model.origins[i], 200000)
            origin_v[i] = self.model.growthrate / k
            origin_loga0[i] = log_a0
        self.model.delay[:] = (max(origin_loga0) - origin_loga0) / self.model.growthrate
        self.model.v0[0] = mean(origin_v)
        # II. Optimize g()
        self.model.v0 = nn.Parameter(self.model.v0)
        self.model.g_log = nn.Parameter(self.model.g_log)
        v0_g = [ self.model.v0, self.model.g_log ]
        optimize_v0_g = optim.LBFGS(v0_g, max_iter=1000, line_search_fn="strong_wolfe", lr=0.01)
        optimize_v0_g.step(self.parameter_loss_function(optimize_v0_g, params=v0_g))
        # Make sure the final values lie within the parameter's domains
        self.model.clamp_parameters()
        # III. Optimize h()
        self.model.h_switch_basis("eigenbasis")
        self.model.h_log_raw = nn.Parameter(self.model.h_log_raw)
        v0_g_h = v0_g + [ self.model.h_log_raw ]
        optimize_v0_g_h = optim.LBFGS(v0_g_h, max_iter=500, line_search_fn="strong_wolfe", lr=0.01)
        optimize_v0_g_h.step(self.parameter_loss_function(optimize_v0_g_h, params=v0_g_h))
        # Make sure the final values lie within the parameter's domains
        self.model.clamp_parameters()
        # Normalize estimated parameters
