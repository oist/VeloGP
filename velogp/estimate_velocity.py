from typing import Any, Dict, Optional, Union, Tuple
import operator
from functools import reduce
from velocity_model import VelocityModel, VelocityModelLoss, dtensor
import sys
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
        self.loss_prefactor = dtensor([1.0])
        self.loss_prefactor.requires_grad = False

    def parameter_loss_function(self, params, log_params_default=False, log_loss_default=False):
        def s(update_gradients=True, log_params=log_params_default, log_loss=log_loss_default):
            l = self.evaluate_model_for_parameters(params, update_gradients=update_gradients)
            if log_params:
                for n, p in self.model.named_parameters():
                    print(f"{n}: {p}")
                    print(f"grad {n}: {p.grad}")
            if log_loss:
                    print(f"loss_prefactor: {self.loss_prefactor}")
                    print(f"loss (scaled): {l}")
            return l
        return s

    def evaluate_model_for_parameters(self, params, update_gradients=True):
        # If any parameter is nan, return nan
        if not reduce(operator.and_, [all(isfinite(p)) for p in params]):
            return torch.tensor([math.nan])
        # Clamp parameters to their domain before each iteration
        self.model.clamp_parameters()
        # Evaluate loss and gradients
        # Update eigenbasis for current hyperparameters in a no-grad context,
        # this avoids doing the update in an autograd context later which is
        # unnecessary and could cause problems
        with no_grad():
            self.model.update_h_cov_eig()
        # Zero gradients
        if update_gradients:
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()
        # Compute loss
        l = (self.loss() + reduce(operator.add, [self.model.prior(p, False) for p in params])) / self.loss_prefactor
        # Compute gradients
        if update_gradients:
            l.backward()
        return l

    def parameter_posterior_cov_inv(self):
        params = [self.model.v0_scaled, self.model.g_log, self.model.h_log_raw]
        _, pmap = self.model.flatmap(params)
        dM = self.loss.jacobian(params)
        # Compute dM^T diag(1/s_1^2, ..., 1/s_N^2) dM
        u = einsum("ji,j,jk->ik", dM, self.loss.weights, dM).clone()
        # Add Sigma_h^-1 for h_log_raw, which has a GP prior
        # We do this by querying the parameter map to find the
        # sub-matrix of u that corresponds to the parameter h_log_raw,
        # and adding the inverse of its prior covariance matrix
        i, l = pmap["h_log_raw"], numel(self.model.h_log_raw)
        u[i:(i+l), i:(i+l)] += self.model.h_cov_inv(include_sigma=True)
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

    def optimizer_loop(self, loss_function, optimizer, steps=10000):
        for i in range(steps):
            loss_function()
            optimizer.step()

    def run_lbfgs(self, params, lr_start=0.1, lr_end=0.1, iterations=100,
                  log_steps=False, log_iterations=False, log_params=False, **kwargs):
        K = 10
        RETRY = 5
        lossfn = self.parameter_loss_function(params, log_loss_default=log_iterations,
                                              log_params_default=log_params and log_iterations)
        for i in range(K):
            # Determine current learning rate
            lr = lr_start + (lr_end - lr_start) * i / (K - 1)
            # Log step
            if log_steps:
                print(f"Step {i+1} of {K}, learning rate {lr}")
            # Update loss scale
            with no_grad():
                if log_steps:
                    print("Before step:")
                self.loss_prefactor[0] *= lossfn(update_gradients=False, log_loss=log_steps, log_params=log_params and log_steps)[0]
            # Save parameter values
            fm = self.model.flatmap(params)
            p_before = self.model.pvector(params, fm)
            # Perform step, retry with reduced learning rate if we fail
            j = 0
            while True:
                # Increase attempt count
                j += 1
                # Create and run optimizer
                opt = optim.LBFGS(params, lr=lr, max_iter=int(iterations / K), max_eval=10 * iterations, **kwargs)
                opt.step(lossfn)
                # If the parameter are still finite, the step was successfull
                if all(isfinite(self.model.pvector(params, fm))):
                    break
                # Otherwise, reset parameters
                with no_grad():
                    self.model.pvector_assign(params, p_before, fm)
                # and complain if the number of retries is exceeded
                if j >= RETRY:
                    raise RuntimeError("optimization failed")
                # otherwise and half the learning rate, then try again
                lr = lr / 2
                if log_steps:
                    print(f"Step {i+1} of {K} failed, will make attempt {j+1} with learning rate reduced to {lr}")
            # Log number of interations done
            if log_steps:
                print(f"Iterations: {opt.state_dict()['state'][0]['n_iter']}")
        # Make sure the final values lie within the parameter's domains
        self.model.clamp_parameters()
        return opt

    def run_sgd(self, params, lr_start=0.1, lr_end=0.1, iterations=100, **kwargs):
        opt = optim.SGD(params, lr=lr_start, **kwargs)
        loss = self.parameter_loss_function(params, log=True)
        for i in range(iterations):
            opt.param_groups[0]["lr"] = lr_start + (lr_end - lr_start) * i / (iterations-1)
            print(opt.param_groups[0]["lr"])
            loss()
            opt.step()


    def optimize_parameters(self, log_phases=False, log_steps=False, log_iterations=False, log_params=False):
        self.model.h_switch_basis("canonical")
        # I. Find origin delays
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
            self.model.v0_prefactor[0] = mean(origin_v)
            self.model.v0_scaled[0] = 1.0
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
        self.model.v0_scaled = nn.Parameter(self.model.v0_scaled)
        self.model.g_log = nn.Parameter(self.model.g_log)
        self.run_lbfgs([self.model.v0_scaled, self.model.g_log],
                       lr_start=0.1, lr_end=1.0, iterations=100,
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
        self.run_lbfgs([ self.model.v0_scaled, self.model.h_log_raw ],
                       lr_start=0.1, lr_end=1.0, iterations=100,
                       tolerance_change=-math.inf, line_search_fn="strong_wolfe",
                       log_steps=log_steps, log_iterations=log_iterations, log_params=log_params)
        # IV. Optimize all parameters
        # Now that we should have a good starting point, we optimize all parameters
        # together. Starting with a small learning rate is probably unnecessary here, but
        # again seems safer. We now do 1000 iterations which is hopefully enough to converge.
        # Note that for lack of a good test, we currently don't check for convergence...
        if log_phases:
            print("Optimizing v0, g() and h() simultaneously")
        self.run_lbfgs([ self.model.v0_scaled, self.model.g_log, self.model.h_log_raw ],
                       lr_start=0.1, lr_end=1.0, iterations=1000,
                       tolerance_change=-math.inf, line_search_fn="strong_wolfe",
                       log_steps=log_steps, log_iterations=log_iterations, log_params=log_params)
