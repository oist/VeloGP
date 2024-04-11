from typing import Any, Dict, Optional, Union, Tuple
import operator
from functools import reduce
from velocity_model import VelocityModel, VelocityModelLoss
import math
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

    def parameter_posterior_cov(self):
        params = [self.model.v0, self.model.delay, self.model.g_log, self.model.h_log_raw]
        dM, pmap = self.loss.jacobian(params)
        # Compute dM^T diag(1/s_1^2, ..., 1/s_N^2) dM
        u = einsum("ij,j,jk->ik", t(dM), self.loss.weights, dM)
        # Add Sigma_h^-1 for h_log_raw, which has a GP prior
        # We do this by querying the parameter map to find the
        # sub-matrix of u that corresponds to the parameter h_log_raw,
        # and adding the inverse of its prior covariance matrix
        i, l = pmap["h_log_raw"], numel(self.model.h_log_raw)
        u[i:(i+l), i:(i+l)] += self.model.h_cov_inv()
        # The covariance matrix is the inverse of u
        return cholesky_inverse(u), pmap

    def h_sigma_lengthscale_loss_function(self, optimizer):
        def s():
            raise "Implement me"


    def optimize_parameters(self):
        self.model.h_switch_basis("eigenbasis")
        # I. Optimize v0 and origin delays
        self.model.v0 = nn.Parameter(self.model.v0)
        self.model.delay = nn.Parameter(self.model.delay)
        v0_delay = [self.model.v0, self.model.delay ]
        optimize_v0_delay = optim.LBFGS(v0_delay, max_iter=500, line_search_fn="strong_wolfe", lr=0.01)
        optimize_v0_delay.step(self.parameter_loss_function(optimize_v0_delay, params=v0_delay))
        # Make sure the final values lie within the parameter's domains
        self.model.clamp_parameters()
        # Normalize estimated parameters
        self.model.normalize_parameters()
        # II. Optimize g()
        self.model.g_log = nn.Parameter(self.model.g_log)
        v0_delay_g = v0_delay + [ self.model.g_log ]
        optimize_v0_delay_g = optim.LBFGS(v0_delay_g, max_iter=500, history_size=100, line_search_fn="strong_wolfe", lr=0.01)
        optimize_v0_delay_g.step(self.parameter_loss_function(optimize_v0_delay_g, params=v0_delay_g))
        # Make sure the final values lie within the parameter's domains
        self.model.clamp_parameters()
        # Normalize estimated parameters
        self.model.normalize_parameters()
        # III. Optimize h()
        self.model.h_log_raw = nn.Parameter(self.model.h_log_raw)
        v0_delay_g_h = v0_delay_g + [ self.model.h_log_raw ]
        optimize_v0_delay_g_h = optim.LBFGS(v0_delay_g_h, max_iter=500, line_search_fn="strong_wolfe", lr=0.01)
        optimize_v0_delay_g_h.step(self.parameter_loss_function(optimize_v0_delay_g_h, params=v0_delay_g_h))
        # Make sure the final values lie within the parameter's domains
        self.model.clamp_parameters()
        # Normalize estimated parameters
        self.model.normalize_parameters()
        #self.model.h_switch_basis("canonical")
