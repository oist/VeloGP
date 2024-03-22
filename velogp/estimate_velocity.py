from typing import Any, Dict, Optional, Union, Tuple
from velocity_model import VelocityModel, VelocityModelLoss
import torch
from torch import nn, optim
from torch import *
from builtins import float, int, range

class FitVelocityModel:
    def __init__(self, model: VelocityModel, loss: VelocityModelLoss):
        self.model = model
        self.loss = loss

    def parameter_loss_function(self, optimizer):
        def s():
            # Clamp parameters to their domain before each iteration
            self.model.clamp_parameters()
            # Evaluate loss and gradients
            optimizer.zero_grad()
            l = self.loss() + self.model.prior_hlog()
            l.backward()
            return l
        return s

    def h_sigma_lengthscale_loss_function(self, optimizer):
        def s():
            raise "Implement me"


    def optimize_parameters(self):
        self.model.h_switch_basis("eigenbasis")
        # First, optimize v0 and origin delays only
        self.model.v0 = nn.Parameter(self.model.v0)
        self.model.delay = nn.Parameter(self.model.delay)
        optimize_v0_delay = optim.LBFGS([self.model.v0, self.model.delay],
                                        max_iter=500, line_search_fn="strong_wolfe")
        optimize_v0_delay.step(self.parameter_loss_function(optimize_v0_delay))
        # Make sure the final values lie within the parameter's domains
        self.model.clamp_parameters()
        # Normalize estimated parameters
        self.model.normalize_parameters()
        # Now, also optimize h() and g()
        self.model.h_log_raw = nn.Parameter(self.model.h_log_raw)
        self.model.g_log = nn.Parameter(self.model.g_log)
        optimize_params = optim.LBFGS([self.model.v0, self.model.delay, self.model.h_log_raw, self.model.g_log],
                                      max_iter=500, line_search_fn="strong_wolfe")
        optimize_params.step(self.parameter_loss_function(optimize_params))
        # Make sure the final values lie within the parameter's domains
        self.model.clamp_parameters()
        # Normalize estimated parameters
        self.model.normalize_parameters()
        #self.model.h_switch_basis("canonical")
