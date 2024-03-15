from typing import Any, Dict, Optional, Union, Tuple
import torch
from torch import nn, optim
from torch import *
from builtins import float, int, range

class FitModel:
    def __init__(self, model: nn.Module, loss: nn.Module):
        self.model = model
        self.loss = loss

    def optimize(self, iterations: int):
        self.optimizer = optim.SGD(self.model.parameters())

        for i in range(0, iterations):
            # Compute loss
            l = self.loss()
            # backward pass
            self.optimizer.zero_grad()
            l.backward(retain_graph=True)
            # update weights
            self.optimizer.step()
