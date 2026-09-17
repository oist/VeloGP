from typing import Any, Dict, Optional, Union, Tuple, List
import operator
from functools import reduce

import numpy as np

import torch
from torch import nn, linalg, optim
from torch import *

import builtins
import math
import sys
from builtins import float, int, range
from .model import VelocityModel, VelocityModelLoss, dtensor

class FitVelocityModel:
    def __init__(self, model: VelocityModel, loss: VelocityModelLoss):
        self.model = model
        self.loss = loss
        self.loss_prefactor = dtensor(1.0)
        self.loss_prefactor.requires_grad = False

    def parameter_loss_function(self, params: List[nn.Parameter], log_params_default=False, log_loss_default=False):
        def s(update_gradients=True, log_params=log_params_default, log_loss=log_loss_default):
            l = self.evaluate_model_for_parameters(params, update_gradients=update_gradients)
            if log_params:
                for n, p in self.model.named_parameters():
                    if builtins.any([ id(p) == id(pp) for pp in params ]):
                        print(f"      {n}: {p.detach().numpy()}")
                        if update_gradients:
                            print(f"      grad {n}: {p.grad.detach().numpy()}")
            if log_loss:
                    print(f"      loss: {l.item() * self.loss_prefactor.item():.3e} = {l.item():.3e} * {self.loss_prefactor.item():.3e}")
            return l
        return s

    def evaluate_model_for_parameters(self, params: List[nn.Parameter], update_gradients=True):
        # If any parameter is nan, return nan
        if not reduce(operator.and_, [all(isfinite(p)) for p in params]):
            return torch.tensor([math.nan])
        # Clamp parameters to their domain before each iteration
        self.model.clamp_parameters()
        # Evaluate loss and gradients
        # Zero gradients
        if update_gradients:
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()
        # Compute loss
        try:
            l = self.loss() / self.loss_prefactor
        except Exception as e:
            print("ERROR: Model evaluation failed, current parameter values are:", file=sys.stderr)
            for n, p in self.model.named_parameters():
                print(f"Parameter {n}: {p.detach().numpy()}", file=sys.stderr)
            raise e
        # Compute gradients
        if update_gradients:
            l.backward()
        return l

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

    def run_lbfgs(self, params: List[nn.Parameter], lr_start=0.1, lr_end=0.1, iterations=100,
                  steps: int | None = 10, retry=5, tolerance=0.0, log_steps=False, log_iterations=False,
                  log_params=False, **kwargs):
        lossfn = self.parameter_loss_function(params, log_loss_default=log_iterations,
                                              log_params_default=log_params and log_iterations)
        max_iter = int(iterations / steps) if (steps is not None) else iterations
        # Update loss scale so that scaled loss is 1.0
        with no_grad():
            if log_steps:
                print("  Initially:")
            l = lossfn(update_gradients=False, log_loss=log_steps, log_params=log_params and log_steps)
            self.loss_prefactor *= l
        i = 0
        while (steps is None) or i < steps:
            i += 1
            # Determine current learning rate
            lr = lr_start + (lr_end - lr_start) * (i-1) / (steps - 1) if (steps is not None) and (steps > 1) else lr_end
            # Log step
            if log_steps:
                print(f"  LBFGS Step {i}{f'/{steps}' if steps is not None else ''}, learning rate {lr}")
            # Save parameter values
            fm = self.model.flatmap(params)
            p_before = self.model.pvector(params, fm)
            # Perform step, retry with reduced learning rate if we fail
            j = 0
            while True:
                # Increase attempt count
                j += 1
                # Create and run optimizer
                opt = optim.LBFGS(params, lr=lr, max_iter=max_iter, max_eval=10 * iterations, **kwargs)
                opt.step(lossfn)
                # If the parameter are still finite, the step was successfull
                if all(isfinite(self.model.pvector(params, fm))) and all(isfinite(lossfn(update_gradients=False))):
                    break
                # Otherwise, reset parameters
                with no_grad():
                    self.model.pvector_assign(params, p_before, fm)
                # and complain if the number of retries is exceeded
                if j >= retry:
                    raise RuntimeError("optimization failed")
                # otherwise and half the learning rate, then try again
                lr = lr / 2
                if log_steps:
                    print(f"  LBFGS Step {i}{f'/{steps}' if steps is not None else ''} failed, " +
                          f"will make attempt {j+1} with learning rate reduced to {lr}")
            # Log number of interations done
            if log_steps:
                print(f"    LBFGS iterations: {opt.state_dict()['state'][0]['n_iter']}")
                print(f"    After step:")
            # Evaluate once more to see improvement
            l = lossfn(update_gradients=False, log_loss=log_steps, log_params=log_params and log_steps)
            gradients = self.model.pgradient(params, fm)
            with no_grad():
                self.loss_prefactor *= l
            # If number of steps is not fixed, stop once there's no improvement
            if log_steps:
                print(f"    Final gradients: pos={sum(gradients > 0.0)}, neg={sum(gradients < 0)}, zero={sum(gradients == 0.0)}, " +
                      f"max={max(gradients).item()}, min={min(gradients).item()}, " +
                      f"log-avg. magnitude={exp(mean(log(abs(gradients[gradients != 0.0])))).item()}")
            if steps is None and (1.0 - l <= tolerance):
                if log_steps:
                    print(f"    Relative improvement {1.0 - l.item()}  does not exceed tolerance {tolerance}, stopping")
                break
            elif log_steps:
                print(f"    Relative improvement {1.0 - l.item()}")
        # Make sure the final values lie within the parameter's domains
        self.model.clamp_parameters()
        return opt

    def run_sgd(self, params: List[nn.Parameter], lr_start=0.001, lr_end=0.001, iterations=1000,
                log_steps=False, log_iterations=False, log_params=False, **kwargs):
        lossfn = self.parameter_loss_function(params, log_loss_default=log_iterations,
                                              log_params_default=log_params and log_iterations)
        for i in range(iterations):
            opt = optim.SGD(params, lr=lr_start + (lr_end - lr_start) * i / (iterations-1), **kwargs)
            lossfn()
            opt.step()

    def run_adam(self, params: List[nn.Parameter], lr_start=0.001, lr_end=0.001, iterations=1000,
                 log_iterations=False, log_params=False, **kwargs):
        lossfn = self.parameter_loss_function(params, log_loss_default=log_iterations,
                                              log_params_default=log_params and log_iterations)
        for i in range(iterations):
            opt = optim.Adam(params, lr=lr_start + (lr_end - lr_start) * i / (iterations-1), **kwargs)
            lossfn()
            opt.step()
    
    def run_newton(self, lr: float=0.1, iterations: int=10):
#        basis = self.model.h_basis
#        self.model.h_switch_basis("canonical")
        p = [self.model.h_log_raw]
        for i in range(iterations):
            u = self.loss.newton_step(p)
            with no_grad():
                self.model.pvector_assign(p, self.model.pvector(p)*(1-lr) + u*lr)            
                self.model.clamp_parameters()
#        self.model.h_switch_basis(basis)

    def initialize_v0_delays(self):
        # Fit A(d) = A(o) * exp(- k * |d|) where k = lambda/v locally. Then
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
        with no_grad():
            self.model.delay[:] = (max(origin_loga0) - origin_loga0) / self.model.growthrate
            self.model.v0 = mean(origin_v)

    def run_stochastic(self, params: List[nn.Parameter], steps, perturb_start=1.0, perturb_end=0.01,
                       log_rounds=True, log_steps=False, log_iterations=False, log_params=False):
        for s in range(0, steps):
            # Compute perturbation strength for the current step
            perturb = perturb_start * (perturb_end / perturb_start) ** (s/(steps-1)) if steps > 1 else perturb_start
            if log_rounds:
                print(f"Round {s}: perturb={perturb}")
            # Evaluate loss & log
            loss_before = self.parameter_loss_function(params)(update_gradients=False) * self.loss_prefactor
            # Perturb parameters
            with no_grad():
                pv_before = self.model.pvector(params)
            for n, p in self.model.resolve_params(self.loss.params):
                with no_grad():
                    pp = self.model.sample_prior(p)
                    p.copy_(p*(1 - perturb) + pp*perturb)
            # Optimize                    
            self.run_lbfgs(params, lr_start=0.1, lr_end=1.0, iterations=100, steps=10,
                           tolerance_change=-math.inf, line_search_fn="strong_wolfe",
                           log_steps=log_steps, log_iterations=log_iterations, log_params=log_params)
            # Evaluate loss again
            loss = self.parameter_loss_function(params)(update_gradients=False) * self.loss_prefactor
            # Accept if the loss improved
            if loss > loss_before:
                if log_rounds:
                    print(f"Round {s} (reject): loss={loss_before} -> {loss}, perturb={perturb}")
                with no_grad():
                    self.model.pvector_assign(params, pv_before)
            elif log_rounds:
                 print(f"Round {s} (accept): loss={loss_before} -> {loss}, perturb={perturb}")
                

    def optimize_parameters_stochastic(self, steps: int, log_phases=False, log_rounds=False, log_steps=False, log_iterations=False, log_params=False):
        # I. Find origin delays
        if log_phases:
            print("Determining initial estimates for v0 and delay")
        self.initialize_v0_delays()
        # II. Stochastic optimization
        self.model.h_switch_basis("eigenbasis_scaled")
        self.model.v0_log = nn.Parameter(self.model.v0_log)
        self.model.h_log_raw = nn.Parameter(self.model.h_log_raw)
        with no_grad():
            self.model.h_log_raw.zero_()
        self.model.g_delta_scaled = nn.Parameter(self.model.g_delta_scaled)
        with no_grad():
            self.model.g_delta_scaled.zero_()
        if log_phases:
            print("Stochastic optimization of v0, h(t) and g(x)")
        self.run_stochastic(steps=steps,
                            perturb_start=1.0, perturb_end=0.1,
                            params=[ self.model.v0_log, self.model.g_delta_scaled, self.model.h_log_raw ],
                            log_rounds=log_rounds, log_steps=log_steps, log_iterations=log_iterations, log_params=log_params)
    
    def optimize_parameters(self, log_phases=False, log_steps=False, log_iterations=False, log_params=False):
        # I. Find origin delays
        if log_phases:
            print("Determining initial estimates for v0 and delay")
        self.initialize_v0_delays()
        # II. Optimize g()
        # We first optimize v0 and the spatial delays since these are easier to fit.
        # We start with a small learning rate because L-BGFS is otherwise prone to making
        # too large steps in the beginning, which causes numerical issues (infinite values e.g.)
        # when the model is evaluated. Once the solution improves, the L-BGFS is usually able
        # to find a good step size on its own, so we slowly increase the learning rate towards
        # it's default of 1.0, which means the step size suggested by L-BGFS's line search.
        # We only do 100 iterations because we don't need a perfect solution, just a good
        # starting point for the next step.
        if log_phases:
            print("Optimizing v0 and g()")
        self.model.v0_log = nn.Parameter(self.model.v0_log)
        self.model.g_delta_scaled = nn.Parameter(self.model.g_delta_scaled)
        self.run_lbfgs([self.model.v0_log, self.model.g_delta_scaled],
                       lr_start=0.1, lr_end=1.0, iterations=100, steps=10,
                       tolerance_change=-math.inf, line_search_fn="strong_wolfe",
                       log_steps=log_steps, log_iterations=log_iterations, log_params=log_params)
        # III. Optimize h()
        # Next we make a first optimization pass for h() were we keep v0 and g()
        # fixed, and find a rough fit for h(). We do this in the scaled eigenbasis
        # where the prior on h() simply amounts to the sum-of-squares of coefficients.
        # The advantage of the scaled basis over the normalized basis (in which the
        # basis vectors have length one) is that it makes the magnitudes of the gradients
        # more uniform. In the normalized basis, the basis vectors with small eigenvectors
        # (i.e. small variances) produce large gradients, because even small changes
        # drastically change the prior on h(). We again start with a small learning rate, for
        # the same reason as above. This might not be necessary here since we should start with
        # a reasonably good solution, but it's safer and doesn't hurt much. We again do 100
        # iterations because we don't need a perfect solution, just a good starting point for the
        # final step.
        if log_phases:
            print("Optimizing h()")
        self.model.h_switch_basis("eigenbasis_scaled")
        self.model.h_log_raw = nn.Parameter(self.model.h_log_raw)
        self.run_lbfgs([ self.model.v0_log, self.model.h_log_raw ],
                       lr_start=0.1, lr_end=1.0, iterations=100, steps=10,
                       tolerance_change=-math.inf, line_search_fn="strong_wolfe",
                       log_steps=log_steps, log_iterations=log_iterations, log_params=log_params)
        # IV. Optimize all parameters
        # Now that we should have a good starting point, we optimize all parameters
        # together. Starting with a small learning rate is probably unnecessary here, but
        # again seems safer. We now do 1000 iterations which is hopefully enough to converge.
        # Note that for lack of a good test, we currently don't check for convergence...
        if log_phases:
            print("Optimizing v0, g() and h() simultaneously")
        self.run_lbfgs([ self.model.v0_log, self.model.g_delta_scaled, self.model.h_log_raw ],
                       lr_start=0.1, lr_end=1.0, iterations=1000, steps=10,
                       tolerance_change=-math.inf, line_search_fn="strong_wolfe",
                       log_steps=log_steps, log_iterations=log_iterations, log_params=log_params)
        if log_phases:
            print("Finalizing v0, g() and h() until the object cannot be further improved")
        self.run_lbfgs([ self.model.v0_log, self.model.g_delta_scaled, self.model.h_log_raw ],
                       lr_start=1.0, lr_end=1.0, iterations=100, steps=None, retry=30,
                       tolerance=1e-6, line_search_fn="strong_wolfe",
                       log_steps=log_steps, log_iterations=log_iterations, log_params=log_params)
