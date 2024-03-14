from typing import Any, Dict, Optional, Union, Tuple
import torch
from jinja2 import optimizer
from torch import nn
from torch import *
from builtins import float, int

class FitModel:
    def __init__(self, model: nn.Module, loss: nn.Module, optimizer: torch.optim):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer

    def optimize(self, iterations: int):
        for i in range(iterations):
            l = self.loss.forward()
            # backward pass
            optimizer.zero_grad()
            l.backward()
            # update weights
            optimizer.step()
