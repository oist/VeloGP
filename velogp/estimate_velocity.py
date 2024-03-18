from typing import Any, Dict, Optional, Union, Tuple
import torch
from torch import nn, optim
from torch import *
from builtins import float, int, range

class FitModel:
    def __init__(self, model: nn.Module, loss: nn.Module):
        self.model = model
        self.loss = loss

    def optimize(self):
        self.optimizer = optim.LBFGS(self.model.parameters(), max_iter=50)
        def s():
            l = self.loss()
            self.optimizer.zero_grad()
            l.backward()
            return l
        self.optimizer.step(s)

#        for i in range(0, iterations):
#            # Compute loss
#            l = self.loss()
#            # backward pass
#            self.optimizer.zero_grad()
#            l.backward(retain_graph=True)
#            # update weights
#            self.optimizer.step()
