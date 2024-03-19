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

    def loss_function(self, optimizer):
        def s():
            self.model.clamp_parameters()
            optimizer.zero_grad()
            l = self.loss()
            l.backward()
            return l
        return s

    def optimize(self):
        self.model.v0 = nn.Parameter(self.model.v0)
        self.model.delay = nn.Parameter(self.model.delay)
        optimize_v0_delay = optim.LBFGS([self.model.v0, self.model.delay],
                                        max_iter=500, line_search_fn="strong_wolfe")
        optimize_v0_delay.step(self.loss_function(optimize_v0_delay))
        self.model.h_log = nn.Parameter(self.model.h_log)
        self.model.g_log = nn.Parameter(self.model.g_log)
        optimize_params = optim.LBFGS([self.model.v0, self.model.delay, self.model.h_log, self.model.g_log],
                                      max_iter=500, line_search_fn="strong_wolfe")
        optimize_params.step(self.loss_function(optimize_params))
