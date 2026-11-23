from __future__ import annotations
from typing import Any, Dict, Optional, Union, Tuple, List
import dataclasses
from dataclasses import dataclass

import numpy as np

import torch
import torch._functorch.config
from torch import nn, linalg
from torch import *


import builtins
import math
from builtins import float, int

# Required so that retain_graph=True in jacobian() doesn't
# error out for compiled models. Unclear why.
torch._functorch.config.donated_buffer=False

def dtensor(v, **kw_args) -> Tensor:
    """ Creates a 64-bit floating point tensor """
    return torch.tensor(v, dtype=torch.double, **kw_args)

def grid_index(grid: Tensor, x: Tensor) -> Tensor:
    """
    Returns for each x the largest i such that grid[i] <= x (or -1 if there is none)

    :param grid: 1-d tensor of grid points in ascending order
    :param x: tensor of arbitrary shape
    :return: tensor of indices with the same shape as x
    """
    with no_grad():
        g = grid.detach()
        v = x.detach().to(g.dtype).reshape(-1)
        return (torch.searchsorted(g, v, right=True) - 1).reshape(x.shape)


class TemporalModel(nn.Module):
    """
    The time-dependent part v(t) = v0 * h(t) of the replication velocity v(x,t) = v(t) / g(x)

    Holds the baseline velocity v0, the function h(t) with its Gaussian process prior, and the population
    growth rate. These are shared by all chromosomes of a genome.
    """

    H_EIGENFUNCTION_VARIANCE_THRESHOLD : float = 1e-6

    @dataclass
    class Eigenbasis:
        """ Represents a basis the diagonalizes the covariance matrix of h(t) """

        eigenvalues: Tensor
        eigenvectors: Tensor
        n: int
        k: int
        log_variance_ignored: Tensor

        @staticmethod
        def make(M: Tensor):
            e = linalg.eigh(M)
            # Remove the first i eigenvalues (the smallest) where is chosen such that no
            # more than a fraction of self.H_EIGENFUNCTION_VARIANCE_THRESHOLD of the total
            # variance (i.e. sum of eigenvalues) is removed. This essentially removes
            # components for which even small changes would have very drastic effects on
            # the prior likelihood, which poses a challenge for the optimizer.
            variance_fraction = cumsum(e.eigenvalues, dim=0) / sum(e.eigenvalues)
            i = argmax((variance_fraction >= dtensor([TemporalModel.H_EIGENFUNCTION_VARIANCE_THRESHOLD])).view(torch.int8))
            log_variance_ignored = log(sum(e.eigenvalues[:(i - 1)])) if i > 0 else dtensor([0.0])
            n = e.eigenvalues.size()[0]
            return TemporalModel.Eigenbasis(e.eigenvalues[i:], e.eigenvectors[:, i:], n, n - i, log_variance_ignored)

    def __init__(self, h_tgrid: DoubleTensor, v0_min: DoubleTensor, v0_max: DoubleTensor):
        super().__init__()
        with no_grad():
            # Growth rate
            self.growthrate = dtensor([1.0/3600.0], requires_grad=False)
            # Baseline velocity
            self.v0_min = v0_min
            self.v0_max = v0_max
            self.v0_log_min = log(self.v0_min)
            self.v0_log_max = log(self.v0_max)
            self.v0_log = 0.5*self.v0_log_min + 0.5*self.v0_log_max
            # Gaussian process hyperparameters
            self.h_cov_function = "se"
            self.h_timescale = tensor([1e2], requires_grad=False)
            self.h_sigma = tensor([1.0], requires_grad=False)
            # Grid on which h(t) is defined as a piece-wise constant (right-continuous) function
            self.h_tgrid = h_tgrid
            self.h_tgrid.requires_grad = False
            # Check that h_tgrid is a 1-d tensor with entries in ascending order
            assert (self.h_tgrid.dtype == torch.double)
            assert (self.h_tgrid.dim() == 1)
            assert all(self.h_tgrid < cat((self.h_tgrid[1:], dtensor([float('inf')])))).detach().item()
            self.h_tgrid_len = cat((self.h_tgrid[1:], dtensor([float('inf')]))) - self.h_tgrid
            self.h_tgrid_len.requires_grad = False
            self.h_log_raw = zeros(h_tgrid.size(), dtype=torch.double)
            self.h_basis = "canonical"
            # Cholesky factorization of covariance matrix of H
            self.h_cov_chol = None
            self.h_cov_chol_timescale = None
            # Eigenbasis of the covariance matrix of H, see update_h_cov_eig()
            self.h_cov_eig = None
            self.h_cov_eig_timescale = None

    def clamp_parameters(self):
        """ Clamp parameters to their respective domains, see `VelocityModel.clamp_parameters()` """
        with no_grad():
            assert all(isfinite(self.v0_log))
            self.v0_log.clamp_(self.v0_log_min, self.v0_log_max)
            assert all(isfinite(self.h_timescale))
            self.h_timescale.clamp_(0.0, math.inf)
            assert all(isfinite(self.h_sigma))
            self.h_sigma.clamp_(0.0, math.inf)
            if self.h_basis == "canonical":
                h_log_raw_scale = self.h_sigma
            elif self.h_basis == "cov":
                with no_grad():
                    h_log_raw_scale = self.h_transform_basis("cov", "canonical", self.h_sigma * ones_like(self.h_log_raw))
            elif self.h_basis == "col":
                with no_grad():
                    h_log_raw_scale = dtensor([1.0])
            elif self.h_basis == "eigenbasis":
                with no_grad():
                    h_log_raw_scale = self.h_sigma * sqrt(self.h_eigenvalues())
            elif self.h_basis == "eigenbasis_scaled":
                with no_grad():
                    h_log_raw_scale = dtensor([1.0])
            else:
                raise RuntimeError("unknown basis " + self.h_basis)
            assert all(isfinite(self.h_log_raw))
            self.h_log_raw.clamp_(-10 * h_log_raw_scale, 10 * h_log_raw_scale)

    def v_t(self, t: Tensor) -> Tensor:
        """Evaluates v_t(t) = v0 * h(t) at the specified points t

        :param t: Times t as a tensor of arbitrary shape
        :return: v_t(t) as a tensor of the same shape as t
        """
        return self.v0 * self.h(t)

    @property
    def v0(self) -> Tensor:
        """ The current value of v0 """
        return exp(self.v0_log)

    @v0.setter
    def v0(self, value: Tensor) -> Tensor:
        """ Sets the current value of v0 """
        assert((value.dim() == 0))
        self.v0_log.copy_(log(value))

    def v0_proper(self, tmax: Tensor) -> Tensor:
        """The log-mean of v(t) over [0, tmax], i.e. exp( (1/tmax) int_0^tmax log v(t) dt )

        The split of v(t) = v0 * h(t) into v0 and h(t) is not identified by the data, only by the zero-mean GP prior
        on log h(t). The raw parameter v0 is therefore a GLS-weighted (not a uniform) mean of v(t), and is not suitable
        for reporting. This method instead returns the geometric mean of v(t) over the interval [0, tmax], typically
        the replication cycle [0, C] with C = `VelocityModel.completion_time()`. The parameters v0, h(t) are unchanged.

        :param tmax: End of the averaging interval (0-dim tensor), must be positive
        :return: v0_proper as a 0-dim tensor
        """
        assert (tmax.numel() == 1) and (tmax.item() > 0.0)
        tmax = tmax.reshape(())
        # Since h(t) is piece-wise constant on h_tgrid, the integral of log h(t) is the sum over grid cells
        # [h_tgrid[i], h_tgrid[i+1]) of log h[i] times the length of the cell's intersection with [0, tmax]
        with no_grad():
            c = clamp(minimum(tmax - self.h_tgrid, self.h_tgrid_len), min=0.0)
        return self.v0 * exp(sum(c * self.h_log("canonical")) / tmax)

    def h_indices(self, t: Tensor) -> Tensor:
        """
        Map times `t` to their corresponding indices in `h_tgrid`
        :param t: times to map to indices
        :return: the indices in `h_tgrid` that correspond to the given times
        """

        assert all(t >= self.h_tgrid[0])
        # Find i where h_tgrid[i] <= t but h_tgrid[i+1] > t
        return grid_index(self.h_tgrid, t)

    def h(self, t: Tensor, interpolate=True) -> Tensor:
        """Evaluates h(t) at the specified points t

        h(t) is right-continuous and piece-wise constant on the (fixed) grid h_tgrid and obeys h(h_tgrid[i]) = h[i].

        :param t: Times t as a tensor of arbitrary shape
        :return: h(t) as a tensor of the same shape as t
        """
        return exp(self.h_log("canonical")[self.h_indices(t)])

    def h_interpolate(self, t: Tensor) -> Tensor:
        s = zeros(numel(self.h_tgrid), numel(self.h_tgrid))
        s += self.h_sigma**2 * self.h_cov_reduced_canonical()
        s += inverse()

    def h_posterior_stderr(self, times: Tensor, h_log_raw_posterior_cov: Tensor) -> Tensor:
        """
        Computes the posterior standard deviation of h(t) from the posterior covariance matrix of the model parameters

        NOTE: The covariance matrix should not include the parameter v0, see discussion in `VelocityModel.v_t_posterior_stderr()`

        :param times: Times for which to compute the posterior standard deviation of h(t)
        :param h_log_raw_posterior_cov: Posterior covariance matrix of model parameters (excluding
        :return: posterior standard deviation of h(t)
        """
        assert ((h_log_raw_posterior_cov.dim() == 2) and
                (h_log_raw_posterior_cov.size() == Size([self.h_log_raw.size()[0], self.h_log_raw.size()[0]])))
        # We transform the posterior covariance matrix of the parameters into a covariance matrix
        # for h_log in the canonical basis, then extract the diagonal and take the square root to
        # get the standard deviations. Note that the covariances between h() at different times t
        # is lost. We use that if the covariance matrix of y is S, then the covariance matrix of
        # Ay is A S A^T
        idx = self.h_indices(times)
        if self.h_basis == "canonical":
            # The posterior covariance matrix is already expressed in the canonical basis
            d = diagonal(h_log_raw_posterior_cov)
        elif self.h_basis == "cov":
            # The posterior covariance matrix S applies to h_log_raw = Sigma'^{-1} h_log, where
            # Sigma' = Sigma / h_sigma^2, see h_switch_basis(). Therefore, the covariance of
            # h_log = Sigma' h_log_raw is Sigma' S Sigma'^T. Since Sigma is symmetric, so is
            # Sigma' and thus Sigma'^T = Sigma'.
            h = self.h_cov_reduced_canonical() # = Sigma'
            d = diagonal(matmul(matmul(h, h_log_raw_posterior_cov), h)) 
        elif self.h_basis == "col":
            # The posterior covariance matrix S applies to h_log_raw = C^-1 h_log / h_sigma,
            # where Sigma = CC^T h_sigma^2, see h_switch_basis(). Therefore, the covariance of
            # h_log = h_sigma C h_log_raw is h_sigma^2 C S C^T.
            c = self.h_cov_reduced_cholesky() # = C
            d = (self.h_sigma**2) * diagonal(matmul(matmul(c, h_log_raw_posterior_cov), t(c))) 
        elif self.h_basis == "eigenbasis":
            # The posterior covariance matrix S applies to h_log_raw = M^T h_log, see h_switch_basis().
            # The covariance of h_log = M h_log_raw is therefore M S M^T.
            d = diagonal(matmul(self.h_eigenbasis(), matmul(h_log_raw_posterior_cov, t(self.h_eigenbasis()))))
        elif self.h_basis == "eigenbasis_scaled":
            # The posterior covariance matrix S applies to h_log_raw = (1/h_sigma) L^(-1/2) M^T h_log,
            # see h_switch_basis() for the definitions of M, L. The covariance of h_log = h_sigma M L^(1/2) h_log_raw
            # is therefore h_sigma^2 M L^(1/2) * S * L^(1/2) * M^T
            s = self.h_sigma * sqrt(self.h_eigenvalues())
            cv = einsum("i,ij,j->ij", s, h_log_raw_posterior_cov, s)
            d = diagonal(matmul(self.h_eigenbasis(), matmul(cv, t(self.h_eigenbasis()))))
        else:
            raise ValueError("invalid basis " + self.h_basis)
        return exp(self.h_log("canonical")[idx]) * sqrt(d[idx])

    def H(self, t: Tensor) -> Tensor:
        """Evaluates H(t) = int_0^t h(u) du at the specified points t

        h(t) is right-continuous and piece-wise constant on the (fixed) grid h_tgrid and obeys h(h_tgrid[i]) = h[i].

        :param t: Times t as a tensor of arbitrary shape
        :return: H(t) as a tensor of the same shape as t
        """
        # Tensor dist_ij = t[j] - h_tgrid[i] (j can comprise multiple indices)
        dist = t[None,...] - self.h_tgrid[(...,) + (None,) * t.dim()]
        with no_grad():
            # Tensors with the same shape as dist which contain 0 respectively h_tgrid[i+1] - h_tgrid[i].
            zeros = zeros_like(dist)
            length = ones_like(dist) * self.h_tgrid_len[(...,) + (None,) * t.dim()]
        # Tensor containing the length of the intersection of [0, t[j]] and [h_tgrid[i], h_tgrid[i+1]]
        # NOTE: This must be computed outside of no_grad(), otherwise H(t) is not differentiable with
        #       respect to t (dH/dt = h(t)), and origin delays, which enter T(x) as H(delay), receive
        #       no gradient.
        c = maximum(zeros, minimum(dist, length))
        # H(t[j]) = sum_i h[i] * c[i,j]
        return sum(c * exp(self.h_log("canonical")[(...,) + (None,) * t.dim()]), dim=0)

    def Hinv(self, u: Tensor) -> Tensor:
        """Evaluates H^-1(u) at the specified points u

        :param u: Points at which to evaluate H^-1
        :return: H^-1(u) at the specified points u
        """
        assert all(u >= self.h_tgrid[0])
        # Compute H(t) on the (fixed) grid h_tgrid (we could call H here, but this is faster)
        # NOTE: We required H to contain +inf in its last element, since that guarantees that
        #       the lookup below always finds at least one grid cell where u < H. In principle, simply
        #       using the full h_tgrid_len and h_log vectors would also achieve that. This involves a
        #       multiplication of the form c * inf, however, which the back-propagation code cannot
        #       handle correctly.
        h_log = self.h_log("canonical")
        H = cat((dtensor([0.0]), torch.cumsum(self.h_tgrid_len[:-1] * exp(h_log[:-1]), dim=0),
                 dtensor([float('inf')])))
        # Find t_i where H(t_i) <= u but H(t_(i+1)) > u
        idx = grid_index(H, u)
        # Within [t_i, t_(i+1)] the function H is piece-wise linear, i.e.
        #   H(t) = H(t_i) + (t - t_i) * exp(h[i])
        # and its inverse is thus
        #   H^-1(u) = t_i + (u - H(t_i)) / exp(h[i])
        return self.h_tgrid[idx] + (u - H[idx]) / exp(h_log[idx])

    def h_switch_basis(self, newbasis: str):
        """Switch the basis in which h_log is represented

        The represented h_log stays the same, i.e. h_log("canonical") does not change (except for numerical error).

        The different bases are defined in the following way. Let Sigma be the covariance matrix of h_log,
        i.e. of the vector [ log(h(h_tgrid[1])), ..., log(h(h_tgrid[k])) ] where we assume that by construction
        diag(Sigma) = (h_sigma^2, ..., h_sigma^2). Let further M be an orthonormal matrix and L=diag(l[1],... , l[n])
        be a diagonal matrix such that Sigma = h_sigma^2 M L M^T and thus Sigma^-1 = M L^-1 M^T / h_sigma^2.
        The possible bases are then:

        (1) 'canonical'. log(h(t)) is represented by the vector h_log_raw = h_log. The vector h_log_raw = h_log has
        distribution N(0, Sigma) in this case.

        (2) 'cov'. log(h(t)) is represented by the vector h_log_raw = (Sigma / h_sigma^2)^{-1} h_log. 
        h_log_raw has distribution N(0, Sigma^{-1} * h_sigma^4) in this case.

        (3) 'col'. log(h(t)) is represented by the vector h_log_raw = C^-1 h_log / h_sigma where
        CC^T = Sigma / h_sigma^2. The vector h_log_raw has distribution N(0, I) in this case.

        (4) 'eigenbasis'. log(h(t)) is represented by the vector h_log_raw = M^T h_log. The vector h_log_raw has
        distribution N(0, h_sigma^2 L) in this case.

        (5) 'eigenbasis_scaled'. log(h(t)) is represented by the vector h_log_raw = L^(-1/2) M^T h_log / h_sigma.
        The vector h_log_raw has distribution N(0, I) in this case. Here, 'I' denotes the identity matrix, and
        L^(-1/2) = diag(1/sqrt(l[1]),...,1/sqrt(l[n])) denotes the inverse of the square root of the (diagonal!)
        matrix L.

        :param newbasis: The basis to switch to ('canonical', 'cov', 'eigenbasis' or 'eigenbasis_scaled')
        """
        if self.h_basis != newbasis:
            with no_grad():
                if isinstance(self.h_log_raw, nn.Parameter):
                    self.h_log_raw = nn.Parameter(self.h_log(newbasis))
                else:
                    self.h_log_raw = self.h_log(newbasis)
            self.h_basis = newbasis

    def h_transform_basis(self, basis: str, current_basis: str, current_h_log: Tensor) -> Tensor:
        """Transforms the vector current_h_log from current_basis to basis.

        See `h_switch_basis` for a discussion of the possible bases.

        :param basis: The basis to transform to
        :param current_basis: The current basis
        :param current_h_log: The current h_log_raw        
        :return: The vector log(h(h_tgrid)) in the specified basis
        """
        if (current_basis == "canonical") and (basis == "cov"):
            return cholesky_solve(current_h_log.reshape(-1, 1), self.h_cov_reduced_cholesky()).reshape(-1)
        if (current_basis == "canonical") and (basis == "col"):
            return linalg.solve_triangular(self.h_cov_reduced_cholesky(), current_h_log.reshape(-1, 1), upper=False).reshape(-1) / self.h_sigma
        elif (current_basis == "canonical") and (basis == "eigenbasis"):
            return matmul(current_h_log, self.h_eigenbasis())
        elif (current_basis == "canonical") and (basis == "eigenbasis_scaled"):
            return matmul(current_h_log, self.h_eigenbasis()) / (self.h_sigma * sqrt(self.h_eigenvalues()))
        elif (current_basis == "cov") and (basis == "canonical"):
            return matmul(self.h_cov_reduced_canonical(), current_h_log)
        elif (current_basis == "col") and (basis == "canonical"):
            return matmul(self.h_cov_reduced_cholesky(), current_h_log * self.h_sigma)
        elif (current_basis == "eigenbasis") and (basis == "canonical"):
            return matmul(self.h_eigenbasis(), current_h_log)
        elif (current_basis == "eigenbasis") and (basis == "eigenbasis_scaled"):
            return current_h_log / (self.h_sigma * sqrt(self.h_eigenvalues()))
        elif (current_basis == "eigenbasis_scaled") and (basis == "canonical"):
            return matmul(self.h_eigenbasis(), current_h_log * self.h_sigma * sqrt(self.h_eigenvalues()))
        elif (current_basis == "eigenbasis_scaled") and (basis == "eigenbasis"):
            return current_h_log * self.h_sigma * sqrt(self.h_eigenvalues())
        elif current_basis == basis:
            return current_h_log
        elif (current_basis != "canonical") and (basis != "canonical"):
            return self.h_transform_basis(basis, "canonical", self.h_transform_basis("canonical", current_basis, current_h_log))
        else:
            raise RuntimeError("Cannot transform from " + current_basis + " to " + basis)

    def h_log(self, basis: str) -> Tensor:
        """Returns the vector log(h(h_tgrid)) in the requested basis.

        See `h_switch_basis` for a discussion of the possible bases.

        :param basis: The basis to return log(h(h_tgrid)) in, either 'canonical' or 'eigenbasis'
        :return: The vector log(h(h_tgrid)) in the specified basis
        """
        return self.h_transform_basis(basis, self.h_basis, self.h_log_raw)
    
    def h_cov_reduced_canonical(self) -> Tensor:
        """Returns Sigma / h_sigma^2, where Sigma is the covariance matrix of h_log for canonical coordinates"""
        d = abs(self.h_tgrid[None, :] - self.h_tgrid[:, None]) / self.h_timescale
        if self.h_cov_function == "se":            
            # Squared-exp
            return exp(- (d ** 2) / 2)
        elif self.h_cov_function == "m0":
            # Matern-1/2
            return exp(-d)
        elif self.h_cov_function == "m1":
            # Matern-3/2
            dp = math.sqrt(3) * d
            return (1 + dp) * exp(-dp)
        elif self.h_cov_function == "m2":
            # Matern-5/2
            dp = math.sqrt(5) * d 
            return (1 + dp + (5/3) * (d**2)) * exp(-dp)
        else:
            raise ValueError("invalid covariance function " + self.h_cov_function)

    def update_h_cov_chol(self):
        """Recompute cholesky factorization of self.h_cov_reduced_canonical()"""
        if (self.h_cov_chol is None) or (self.h_cov_chol_timescale != self.h_timescale):     
            # Changing the time-scale changes the cholesky factorization, so to keep h_log the same
            # in the canonical basis (which is the representation we ultimately care about)
            # we switch to the canonical basis before changing h_cov_cholesky.
            basis = self.h_basis
            self.h_switch_basis("canonical")
            # Find cholesky factorization of the covariance matrix of h_log for canonical coordinates.
            # We don't include the pre-factor h_sigma^2 here, this avoids having to recompute things
            # if the prefactor changes.
            self.h_cov_chol = linalg.cholesky(self.h_cov_reduced_canonical())
            self.h_cov_chol_timescale = self.h_timescale
            # Update the non-h_log-dependent prefactor of the multivariate normal density of h_log
            # We thus compute h_prior_prefactor = -log(sqrt[(2pi)^k * |Sigma|]) where
            # k is the dimension of h_log and Sigma the covariance matrix. We use the cholesky
            # factorization L such that LL^T = Sigma / h_sigma^2 computed above to compute |Sigma|,
            # using that log |Sigma| = 2k log(h_sigma) + 2 sum_i log(L_ii).
            self.h_prior_prefactor = (
                dtensor([0.5 * math.log(2*math.pi) * len(self.h_tgrid)])
                + log(self.h_sigma) * len(self.h_tgrid)
                + sum(log(diagonal(self.h_cov_chol)))
            )
            # Switch back to the original basis
            self.h_switch_basis(basis)            

    def h_cov_reduced_cholesky(self) -> Tensor:
        self.update_h_cov_chol()
        return self.h_cov_chol
    
    def h_cov(self, include_sigma=True) -> Tensor:
        """The covariance matrix of (h_log_raw[1], ..., h_log_raw[k])

        :param  include_sigma: Whether to Don't include the pre-factor h_sigma ** 2
        :return: the covariance matrix
        """
        if self.h_basis == "canonical":
            s = self.h_cov_reduced_canonical()
            return (self.h_sigma ** 2) * s if include_sigma else s
        elif self.h_basis == "col":
            s = diagflat(ones_like(self.h_log_raw))
            return s / (self.h_sigma ** 2) if not include_sigma else s
        elif self.h_basis == "eigenbasis":
            # The basis is orthonormal here, and h_sigma is not included in the eigenvalues
            s = diagflat(self.h_eigenvalues())
            return (self.h_sigma ** 2) * s if include_sigma else s
        elif self.h_basis == "eigenbasis_scaled":
            # The basis is orthogonal but not normalized here, h_sigma and eigenvalues are absorbed into the basis
            s = diagflat(ones_like(self.h_eigenvalues()))
            return s  / (self.h_sigma ** 2) if not include_sigma else s
        else:
            raise ValueError("unsupported basis " + self.h_basis)

    def h_cov_inv(self, include_sigma=True) -> Tensor:
        """The inverse of the prior covariance matrix of (h_log_raw[1], ..., h_log_raw[k])

        :param  include_sigma: Whether to include the pre-factor 1 / h_sigma ** 2
        :return: the covariance matrix
        """
        if self.h_basis == "canonical":
            s = einsum("ij,j,kj->ik", self.h_eigenbasis(), dtensor([1.0]) / self.h_eigenvalues(), self.h_eigenbasis())
            return s / (self.h_sigma ** 2) if include_sigma else s
        elif self.h_basis == "cov":
            # h_log_raw has distribution N(0, Sigma^{-1} * h_sigma^4), see h_switch_basis()
            # So in terms of Sigma' = Sigma / h_sigma^2, the inverse covariance matrix is
            # Sigma / h_sigma^4 = Sigma' / h_sigma^2
            s = self.h_cov_reduced_canonical() # Sigma'
            return s / (self.h_sigma ** 2) if include_sigma else s
        elif self.h_basis == "col":
            s = diagflat(ones_like(self.h_log_raw))
            return s * (self.h_sigma ** 2) if not include_sigma else s
        elif self.h_basis == "eigenbasis":
            # The basis is orthonormal here, and h_sigma is not included in the eigenvalues
            s = diagflat(dtensor([1.0]) / self.h_eigenvalues())
            return s / (self.h_sigma ** 2) if include_sigma else s
        elif self.h_basis == "eigenbasis_scaled":
            # The basis is orthogonal but not normalized here, h_sigma and eigenvalues are absorbed into the basis
            s = diagflat(ones_like(self.h_eigenvalues()))
            return s * (self.h_sigma ** 2) if not include_sigma else s
        else:
            raise ValueError("unsupported basis " + self.h_basis)

    def update_h_cov_eig(self):
        """Recompute the eigenvalues and eigenvectors of self.h_cov()"""
        if (self.h_cov_eig is None) or (self.h_cov_eig_timescale != self.h_timescale):
            # Changing the time-scale changes the eigenbasis, so to keep h_log the same
            # in the canonical basis (which is the representation we ultimately care about)
            # we switch to the canonical basis before changing h_cov_eig.
            basis = self.h_basis
            self.h_switch_basis("canonical")
            # Find eigenvalues and eigenvectors of the symmetric matrix h_cov(). We don't
            # include the pre-factor h_sigma^2 here, this avoids having to recompute things
            # if the prefactor changes.
            self.h_cov_eig = TemporalModel.Eigenbasis.make(self.h_cov_reduced_canonical())
            self.h_cov_eig_timescale = self.h_timescale
            # Update the non-h_log-dependent prefactor of the multivariate normal density of h_log
            # We thus compute h_prior_prefactor = -log(sqrt[(2pi)^k * |Sigma|]) where
            # k is the dimension of h_log and Sigma the covariance matrix. Since we have
            # already computed the eigenbasis of Sigma / h_sigma^2 above,
            # we use that |Sigma| = h_sigma^(2k) * lambda_1 * ... * lambda_n where
            # lambda_1, ..., lambda_n are the eigenvalues we computed above. We cannot, however,
            # simply ignore eigenvalues less than the variance threshold here, because
            # that would lead to discontinuities if the number of these eigenvalues changes. We
            # thus replace those eigenvalues by the threshold, thus essentially bounding the
            # eigenvalues from below.
            self.h_prior_prefactor = (
                dtensor([0.5 * math.log(2*math.pi) * len(self.h_tgrid)])
                + log(self.h_sigma) * len(self.h_tgrid)
                + 0.5 * sum(log(self.h_cov_eig.eigenvalues))
                + 0.5 * self.h_cov_eig.log_variance_ignored
            )
            # Switch back to the original basis
            self.h_switch_basis(basis)

    def h_eigenbasis(self):
        """ The eigenvectors of self.h_cov() whose eigenvalues exceed the variance threshold"""
        self.update_h_cov_eig()
        return self.h_cov_eig.eigenvectors

    def h_eigenvalues(self):
        """ The eigenvalues of self.h_cov() that exceed the variance threshold"""
        self.update_h_cov_eig()
        return self.h_cov_eig.eigenvalues

    def prior_hlog(self, hyperparameter_dependence: bool):
        """The prior log-probability of self.h_log as a realization of a Gaussian process

        Returns the negative log of the probability (or, rather, density) of self.h_log as a sample
        of a multivariate Gaussian distribution with mean zero and covariance matrix
        self.h_sigma^2 * self.h_cov_reduced_canonical().

        :param hyperparameter_dependence: Whether to return the full density, or only the h_log-dependent terms
        """
        # In the eigenbasis, the covariance matrix diagonalizes, so we just sum the squared entries of h_log
        # (since the expected value is zero) scaled by the diagonal element (the eigenvalue). Have to manually
        # add the h_sigma pre-factor since we left that out when computing the covariance matrix.
        # NOTE: The h_prior_prefactor also includes terms for those eigenvalues which were included
        # from the Eigenbasis because they lie below the variance threshold. Here, we don't explicitly account
        # for these components, which amounts to treating them as being zero (and thus they don't contribute
        # to the squared-distance from the mean)
        # NOTE: Must not update r in-place, since r may be the cached tensor self.h_prior_prefactor
        if (self.h_basis == "cov"):
            self.update_h_cov_chol()
            r = self.h_prior_prefactor if hyperparameter_dependence else zeros(1)
            return r + 0.5 * sum(self.h_log_raw * matmul(self.h_cov_reduced_canonical(), self.h_log_raw))
        elif (self.h_basis == "col"):
            self.update_h_cov_chol()
            r = self.h_prior_prefactor if hyperparameter_dependence else zeros(1)
            return r + 0.5 * sum(self.h_log("col") ** 2)
        else:
            self.update_h_cov_eig()
            r = self.h_prior_prefactor if hyperparameter_dependence else zeros(1)
            return r + 0.5 * sum(self.h_log("eigenbasis_scaled") ** 2)

    def sample_hlog(self, basis : str, sample_shape : torch.Size | None = None):
        """ Generates a random vector log(h(t_grid[0])), ..., log(h(t_grid[n]))

        :param basis:basis to return the vector in
        :param sample_shape: one vector is created for every element that a tensor with the given shape has
        :return: A vector with shape (k, sample_shape) where k is the number of basis vectors in the selected basis
        """

        self.update_h_cov_eig()
        # Generate as many standard-normal random variables for each sample as the covariance matrix of h
        # has eigenvectors that are significantly larger than zero. This is a representation of a random
        # h in eigenbasis_scaled.
        shape = (self.h_cov_eig.eigenvalues.size()[0],) + (sample_shape if sample_shape is not None else ())
        z = torch.distributions.normal.Normal(dtensor(0.0), dtensor(1.0)).sample(shape)
        if (basis == "eigenbasis_scaled") or (basis == "col"):
            return z
        z_scaled = (self.h_sigma**2) * z * sqrt(self.h_eigenvalues())[:, (None,) * len(sample_shape)]
        if basis == "eigenbasis":
            return z_scaled
        elif basis == "canonical":
            return matmul(self.h_eigenbasis(), z_scaled)
        else:
            raise ValueError("Invalid basis " + basis)

    def owns(self, parameter: Tensor) -> bool:
        """ Whether `parameter` is one of the parameters of the temporal model """
        return (parameter is self.v0_log) or (parameter is self.h_log_raw)

    def prior(self, parameter, hyperparameter_dependence: bool):
        if parameter is self.v0_log:
            return zeros([])
        elif parameter is self.h_log_raw:
            return self.prior_hlog(hyperparameter_dependence=hyperparameter_dependence)
        else:
            raise RuntimeError("no prior defined for this parameter")

    def prior_precision(self, parameter) -> Tensor | None:
        """ The inverse covariance matrix of the (approximately) Gaussian prior of `parameter`, or None if there is none """
        if parameter is self.v0_log:
            return None
        elif parameter is self.h_log_raw:
            return self.h_cov_inv(include_sigma=True)
        else:
            raise RuntimeError("no prior defined for this parameter")

    def sample_prior(self, parameter):
        if parameter is self.v0_log:
            return torch.distributions.uniform.Uniform(low=self.v0_log_min, high=self.v0_log_max).sample(parameter.shape).squeeze()
        elif parameter is self.h_log_raw:
            return self.sample_hlog(self.h_basis)
        else:
            raise RuntimeError("no prior distribution defined for this parameter")

class ChromosomeModel(nn.Module):
    """
    The position-dependent part of the model of replisome movement on a single chromosome

    Holds the chromosome's structure (length, circularity, origins of replication, intervals on which g(x) may
    differ from 1) and the chromosome-specific parameters, i.e. the origin firing delays and g(x). The
    time-dependent part v(t) of the velocity v(x,t) = v(t) / g(x) is shared between chromosomes and is passed
    to the methods which need it as a `TemporalModel` instance.
    """

    int32_min : int = -2**31
    int32_max : int = 2**31 - 1
    # Upper bound of g_delta_scaled, i.e. g_delta is at most this multiple of its prior mean g_delta_prefactor
    g_delta_scaled_max : float = 20.0

    @dataclass
    class X:
        """
        Represents a set of genomic positions, pre-processed for faster model evaluation

        All fields only depend on the positions and on the fixed model structure (chromosome length, circularity,
        origins, g_xranges), never on fitted parameters, and all have the same shape as x.
        """
        x: Tensor
        # Indices of the closest origin to the left and right, see `ChromosomeModel.surrounding_origins()`
        origin_index_left: Tensor
        origin_index_right: Tensor
        # Index i of the cell of g_xgrid containing x, and the offset x - g_xgrid[i]
        g_index: Tensor
        g_offset: Tensor
        # Whether the path from the left origin (in direction +1) respectively the right origin (in direction -1)
        # to x wraps around the end of a circular genome
        wrap_left: Tensor
        wrap_right: Tensor
        # Whether x actually has an origin to its left respectively right (always true on circular genomes)
        has_left: Tensor
        has_right: Tensor
        # Whether x coincides with an origin
        at_origin: Tensor

        def __getitem__(self, k) -> ChromosomeModel.X:
            """ Returns the pre-processed input for the subset x[k] of positions """
            return ChromosomeModel.X(*(getattr(self, f.name)[k] for f in dataclasses.fields(self)))

    def __init__(self, name: str, length: int, is_circular: bool,
                 origins: IntTensor, origin_names: list[str],
                 g_xranges_l: IntTensor, g_xranges_u: IntTensor,
                 g_names: list[str], g_strands: IntTensor,
                 g_delayscale: DoubleTensor | None = None):
        super().__init__()
        with no_grad():
            # Chromosome
            self.name = name
            self.length = tensor([length], dtype=torch.int32, requires_grad=False)
            self.is_circular = is_circular
            # Origins of replication
            self.origins = origins
            self.origin_names = origin_names
            self.origins.requires_grad = False
            # Check that origins is a 1-d tensor with entries in ascending order
            assert (self.origins.dtype == torch.int32)
            assert (self.origins.dim() == 1)
            assert all(self.origins < cat((self.origins[1:], dtensor([float('inf')])))).detach().item()
            # Origins of replication, augmented to account for wrap-around if the genome is circular
            if self.is_circular:
                self.origins_augmented = cat((self.origins[-1] - dtensor([length]), self.origins, dtensor([length]) + self.origins[0]))
                self.origins_augmented.requires_grad = False
            else:
                self.origins_augmented = self.origins
                self.origins_augmented.requires_grad = False
            # Replication delays
            self.delay_scaled = zeros_like(self.origins, dtype=torch.double)
            # Intervals on which g(x) differs from 1.0
            self.g_xranges_l = g_xranges_l
            self.g_xranges_u = g_xranges_u
            self.g_xranges_l.requires_grad = False
            self.g_xranges_u.requires_grad = False
            # Check that g_xranges_l, g_xranges_u are a 1-d tensors with the same length
            assert (self.g_xranges_l.dtype == torch.int32) and (self.g_xranges_u.dtype == torch.int32)
            assert (self.g_xranges_l.dim() == 1) and (self.g_xranges_u.dim() == 1)
            assert self.g_xranges_l.size() == self.g_xranges_u.size()
            # Check that g_xranges_l[i] < g_xranges_u[i] <= g_xranges_l[i+1]
            assert all(self.g_xranges_l < self.g_xranges_u).detach().item()
            assert all(self.g_xranges_u <= cat((self.g_xranges_l[1:], dtensor([float('inf')])))).detach().item()
            # Origins must not lie strictly inside a g range (they may coincide with a range's ends). Each g range
            # thus lies between two adjacent origins and is replicated by at most two replisomes, one coming from
            # either side, see `g_range_passage_times()`.
            inside = (self.g_xranges_l[:, None] < self.origins[None, :]) & (self.origins[None, :] < self.g_xranges_u[:, None])
            if any(inside).item():
                i, j = (int(k) for k in nonzero(inside)[0])
                raise ValueError(f"origin {origin_names[j]} at {self.origins[j].item()} lies inside g range "
                                 f"[{self.g_xranges_l[i].item()}, {self.g_xranges_u[i].item()})")
            self.g_xranges_len = self.g_xranges_u - self.g_xranges_l
            # Names of the g ranges and the strands they lie on (+1 forward, -1 reverse, 0 unstranded)
            assert len(g_names) == numel(self.g_xranges_l), "g_names must contain one name per g range"
            self.g_names = list(g_names)
            assert (g_strands.dim() == 1) and (numel(g_strands) == numel(self.g_xranges_l)), \
                "g_strands must contain one strand per g range"
            assert all((g_strands == 1) | (g_strands == -1) | (g_strands == 0)).item(), "strands must be +1, -1 or 0"
            self.g_strands = g_strands.to(torch.int32)
            self.g_strands.requires_grad = False
            # Grid on which g(x) is defined, i.e. the grid
            # [ l_1, u_1, l_2, u_2, .... ] where l = from g_xranges_l, u = g_xranges_u
            self.g_xgrid = cat((tensor([0]), flatten(stack((self.g_xranges_l, self.g_xranges_u), dim=1))))
            self.g_xgrid.requires_grad = False
            # Floating-point copy of g_xgrid used for lookups with torch.searchsorted()
            self.g_xgrid_double = self.g_xgrid.double()
            self.g_xgrid_len = cat((self.g_xgrid[1:], dtensor([length]))) - self.g_xgrid
            self.g_xgrid_len.requires_grad = False
            self.g_delta_scaled = zeros(self.g_xranges_l.size(), dtype=torch.float64)
            # If provided, g_delayscale specifies for each g range, the magnitudes of the expected delays,
            # measured relative to the time a single replisome would take to replicate the *genome*, i.e.
            # a stretch of length `reference_length`. Until the chromosome becomes part of a `VelocityModel`, which
            # sets the reference length to the total length of all its chromosomes, this is the chromosome itself.
            if g_delayscale is not None:
                # Check that we have as many delayscales as g ranges
                assert (g_delayscale.dim() == 1) and (g_delayscale.size() == self.g_xranges_l.size())
            self.g_delayscale = g_delayscale
            self.reference_length = self.length
            self.g_delta_prefactor = self.compute_g_delta_prefactor(self.reference_length)
            self.g_delta_scaled.requires_grad = False
            # Blocked g ranges cannot be passed by replisomes moving in the blocked direction, i.e. their delay is
            # infinite for these replisomes, see `g_path_blocked()`. g_blocked_fwd blocks replisomes moving in
            # direction +1 (towards increasing coordinates), g_blocked_rev those moving in direction -1. These are
            # not parameters, they are part of the structure of the model.
            self.g_blocked_fwd = zeros(self.g_xranges_l.size(), dtype=torch.bool)
            self.g_blocked_rev = zeros(self.g_xranges_l.size(), dtype=torch.bool)

    def compute_g_delta_prefactor(self, reference_length: Tensor) -> Tensor:
        """ The scale of g_delta for each g range, i.e. the expected value of g_delta under its prior """
        if self.g_delayscale is None:
            return ones_like(self.g_delta_scaled)
        return self.g_delayscale.double() * reference_length.double() / self.g_xranges_len.double()

    def set_reference_length(self, reference_length: int | Tensor):
        """
        Sets the length relative to which `g_delayscale` is measured, see `__init__()`

        Changes the prior on g_delta, but preserves the current value of g_delta.
        """
        reference_length = tensor([reference_length], dtype=torch.int32) \
            if not isinstance(reference_length, Tensor) else reference_length.reshape(1).to(torch.int32)
        if torch.equal(reference_length, self.reference_length):
            return
        with no_grad():
            g_delta = self.g_delta
            self.reference_length = reference_length
            self.g_delta_prefactor = self.compute_g_delta_prefactor(reference_length)
            # Update in-place to preserve the identity of the parameter object
            self.g_delta_scaled.copy_(g_delta / self.g_delta_prefactor)

    def clamp_parameters(self):
        """ Clamp parameters to their respective domains, see `VelocityModel.clamp_parameters()`

        NOTE: Delays are only clamped to be non-negative here. The delays are relative to the time at which the
        first origin *in the genome* fires, so shifting them such that the smallest delay is zero must be done
        across all chromosomes, see `VelocityModel.clamp_parameters()`.
        """
        with no_grad():
            assert all(isfinite(self.delay_scaled))
            self.delay_scaled.clamp_(0.0, math.inf)
            assert all(isfinite(self.g_delta_scaled))
            self.g_delta_scaled.clamp_(0.0, self.g_delta_scaled_max)

    def owns(self, parameter: Tensor) -> bool:
        """ Whether `parameter` is one of the parameters of this chromosome """
        return (parameter is self.delay_scaled) or (parameter is self.g_delta_scaled)

    def delay(self, temporal: TemporalModel) -> Tensor:
        """ The current value of the origin delays """
        return self.delay_scaled / (10 * temporal.growthrate)

    def set_delay(self, temporal: TemporalModel, value: Tensor):
        """ Sets the current value of the origin delays """
        assert(value.shape == self.delay_scaled.shape)
        self.delay_scaled.copy_(value * 10.0 * temporal.growthrate)

    @property
    def g_delta(self) -> Tensor:
        """ The current value of g_delta ( = g - 1.0 ) """
        return self.g_delta_scaled * self.g_delta_prefactor

    @g_delta.setter
    def g_delta(self, value: Tensor):
        """ Set g_delta ( = g - 1.0 ) to the specified value """
        assert ((value.dim() == 1) and (value.size(0) == self.g_delta_prefactor.size(0)))
        assert all(value >= dtensor([0.0]))
        self.g_delta_scaled = value / self.g_delta_prefactor

    @property
    def g(self) -> Tensor:
        """ The current value of g ( = g_delta + 1.0) """
        return (dtensor([1.0]) + self.g_delta)

    @g.setter
    def g(self, value: Tensor):
        """ Set g ( = g_delta + 1.0) to the specified value """
        assert((value.dim() == 1) and (value.size(0) == self.g_delta_prefactor.size(0)))
        assert all(value >= dtensor([1.0]))
        self.g_delta_scaled = (value - dtensor([1.0])) / self.g_delta_prefactor

    @property
    def length_virtual(self):
        """ Returns the length of the chromosome in virtual coordinates """
        return self.length.double() + sum(self.g_delta * self.g_xranges_len)

    def replisome_position(self, temporal: TemporalModel, t: Tensor, origin: Tensor, dir: Tensor, delay: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # Compute distance travelled in virtual coordinates
        # TODO: We ignore diffusion here atm
        z = self.G_0(origin) + dir * temporal.v0 * maximum(temporal.H(t) - temporal.H(delay), dtensor([0.0]))
        # Translate distance travelled to position (plus wrap-around count)
        if self.is_circular:
            z_wrap = div(z, self.length_virtual, rounding_mode="floor")
            z = remainder(z, self.length_virtual)
        else:
            z_wrap = zeros_like(z, dtype=bool)
            z = max(dtensor([0.0]), min(z, self.length_virtual))
        # Translate position to physical coordinates
        x = self.G_0_inv(z)
        # Translate position and wrap-around count to distance travelled in physical coordinates
        dx = x - origin
        dx += self.length * z_wrap
        # Compute instantaneous velocities
        v = temporal.v_t(t) / self.g_fun(x)
        return x, dx, v

    @dataclass
    class Segment:
        """ The part of a chromosome between two adjacent origins (or an origin and the end of a linear chromosome) """
        # Indices of the replisomes moving into the segment from the left and from the right (see `trajectories()`)
        ri_left: int
        ri_right: int
        # Whether there is an origin delimiting the segment on the left respectively right
        has_left: bool
        has_right: bool
        # Left and right ends of the segment (origins or chromosome ends) and the delays of the origins
        o_left: Tensor
        o_right: Tensor
        d_left: Tensor | None
        d_right: Tensor | None
        # Whether the replisome moving into the segment from the left respectively right is ever started
        left_fires: bool
        right_fires: bool
        # The (physical) time at which the segment is fully replicated
        t_meet: Tensor
        # The virtual time at which the segment is fully replicated, i.e. H(t_meet), and the (physical) location
        # at which this happens, i.e. where the two replisomes meet. If only one replisome fires, this is the
        # opposite end of the segment.
        u_meet: Tensor
        x_meet: Tensor

    def segments(self, temporal: TemporalModel) -> List[Segment]:
        """
        Returns the segments between adjacent origins, and the times at which they are fully replicated

        :param temporal: The time-dependent part of the model
        :return: list of segments
        """
        self.assert_unblocked("segments()")
        delay = self.delay(temporal)
        k = 2*numel(self.origins)
        result = []
        # Iterate over pairs of replisomes moving towards one another
        # For the first and last origin, there's a pair that wraps around on circular
        # chromosomes. On linear chromosomes, these chromosomes hit the ends instead of
        # meeting another chromosome
        for o_i in builtins.range(0, numel(self.origins) + 1):
            oi_left = o_i - 1
            ri_left = 2*oi_left + 1
            oi_right = o_i
            ri_right = 2*oi_right
            # Consider wrap-around for circular chromosomes
            if self.is_circular:
                if oi_right >= numel(self.origins):
                    # On a circular chromosomes this is the same as o_left=-1, o_right=0
                    break
                ri_left = ri_left % k
                oi_left = oi_left % numel(self.origins)
            # On linear chromosomes, the chromosome ends delimit the first and last segment. There is
            # no replisome coming from the chromosome end, the replisome simply stops once it reaches it.
            has_left = oi_left >= 0
            has_right = oi_right < numel(self.origins)
            # Get left and right origins (or chromosome ends) and the delays of the origins
            o_left = self.origins[oi_left] if has_left else dtensor([0.0])
            d_left = delay[oi_left] if has_left else None
            o_right = self.origins[oi_right] if has_right else self.length
            d_right = delay[oi_right] if has_right else None
            # Virtual time a single replisome needs to traverse the whole segment between the origins
            # NOTE: Must be careful on a circular genome with a single origin since G(x, x) == 0, the
            #       segment then comprises the whole chromosome.
            # TODO: This has to be adapted if G(x,o,d) ever becomes direction-dependent.
            if self.is_circular and (o_left == o_right).item():
                u_segment = self.G_0(self.length) / temporal.v0
            else:
                u_segment = self.G(o_right, o_left, d=dtensor([1.0])) / temporal.v0
            # Virtual firing times of the origins
            u_left = temporal.H(d_left) if has_left else None
            u_right = temporal.H(d_right) if has_right else None
            # If one replisome reaches the opposite origin before that origin fires, the replisome
            # of the opposite origin never starts and the first replisome stops at the opposite
            # origin. This mirrors T(), which only considers the two origins surrounding a location.
            left_fires = has_left and (not has_right or (u_left - u_right < u_segment).item())
            right_fires = has_right and (not has_left or (u_right - u_left < u_segment).item())
            # Compute time at which the segment is fully replicated
            # TODO: We ignore diffusion here atm
            if left_fires and right_fires:
                u_meet = (u_segment + u_left + u_right) / 2.0
                u_meet_left = u_meet - u_left
            elif left_fires:
                u_meet = u_left + u_segment
                u_meet_left = u_segment
            else:
                u_meet = u_right + u_segment
                u_meet_left = zeros_like(u_segment)
            t_meet = temporal.Hinv(u_meet)
            # Location of the meeting point. The replisome coming from the left travels the virtual distance
            # u_meet_left * v0 before meeting the other replisome (or reaching the end of the segment).
            z_meet = self.G_0(o_left) + u_meet_left * temporal.v0
            if self.is_circular:
                z_meet = where(z_meet >= self.G_0(self.length), z_meet - self.G_0(self.length), z_meet)
            x_meet = self.G_0_inv(z_meet)
            result.append(ChromosomeModel.Segment(ri_left, ri_right, has_left, has_right, o_left, o_right,
                                                  d_left, d_right, left_fires, right_fires, t_meet,
                                                  u_meet.reshape(()), x_meet.reshape(())))
        return result

    def completion_time(self, temporal: TemporalModel) -> Tensor:
        """
        Returns the (physical) time at which the chromosome is fully replicated

        This is the first time after which no replisome is active, i.e. after which all replisomes are in state 0
        (never started) or 2 (finished), see `trajectories()`.
        """
        return max(stack([seg.t_meet.reshape(()) for seg in self.segments(temporal)]))

    def trajectories(self, temporal: TemporalModel, t: Tensor = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        self.assert_unblocked("trajectories()")
        # Use grid on which h() is defined if no output times are specified
        if t is None:
            t = temporal.h_tgrid
        # Physical origin firing delays
        delay = self.delay(temporal)
        # Create output tensors, one for replisome positions, or for velocities
        k = 2*numel(self.origins)
        tab_state = ones(t.shape + (k,), dtype=torch.int32)
        tab_x = zeros(t.shape + (k,), dtype=torch.float64)
        tab_dx = zeros(t.shape + (k,), dtype=torch.float64)
        tab_v = zeros(t.shape + (k,), dtype=torch.float64)
        for o_i in builtins.range(0, numel(self.origins)):
            for d_i, d in enumerate([-1, 1]):
                x, dx, v = self.replisome_position(temporal, t, origin=self.origins[o_i], dir=d, delay=delay[o_i])
                tab_x[:, 2*o_i + d_i] = x
                tab_dx[:, 2*o_i + d_i] = dx
                tab_v[:, 2*o_i + d_i] = v
        # Determine chromosome meeting points, see segments()
        for seg in self.segments(temporal):
            ri_left, ri_right = seg.ri_left, seg.ri_right
            has_left, has_right = seg.has_left, seg.has_right
            o_left, o_right = seg.o_left, seg.o_right
            d_left, d_right = seg.d_left, seg.d_right
            left_fires, right_fires = seg.left_fires, seg.right_fires
            t_meet = seg.t_meet
            # Before replisome creation (or forever, if it is never created),
            # state = 0, position = origin, distance = 0.0, v = nan
            if has_left:
                w = (t < d_left) if left_fires else ones_like(t, dtype=torch.bool)
                tab_state[w, ri_left] = 0
                tab_x[w, ri_left] = o_left.double()
                tab_dx[w, ri_left] = 0.0
                tab_v[w, ri_left] = nan
            if has_right:
                w = (t < d_right) if right_fires else ones_like(t, dtype=torch.bool)
                tab_state[w, ri_right] = 0
                tab_x[w, ri_right] = o_right.double()
                tab_dx[w, ri_right] = 0.0
                tab_v[w, ri_right] = nan
            # After the segment is replicated, state = 2, position = meeting point, distance = dist. travelled, v = nan
            w = t > t_meet
            if left_fires:
                x_meet_left, dx_meet_left, _ = self.replisome_position(temporal, t_meet, origin=o_left, dir=dtensor([+1.0]), delay=d_left)
                tab_state[w, ri_left] = 2
                tab_x[w, ri_left] = x_meet_left
                tab_dx[w, ri_left] = dx_meet_left
                tab_v[w, ri_left] = nan
            if right_fires:
                x_meet_right, dx_meet_right, _ = self.replisome_position(temporal, t_meet, origin=o_right, dir=dtensor([-1.0]), delay=d_right)
                tab_state[w, ri_right] = 2
                tab_x[w, ri_right] = x_meet_right
                tab_dx[w, ri_right] = dx_meet_right
                tab_v[w, ri_right] = nan
        # Remove first row which ri_right to the entry t=0 we added at the top
        return tab_state, tab_x, tab_dx, tab_v

    def G_0(self, x: Tensor) -> Tensor:
        """
        Evaluates G_0(x) = int_0^x g(u) du

        g(x) is piece-wise constant and = 1 except on the intervals [g_xranges_l[i], g_xranges_u[i]] where g(x) = g[i].

        :param x:  Locations as a tensor of arbitrary shape
        :return: G(x,o) as a tensor of the same shape as x
        """
        assert all(logical_and(x >= tensor([0]), x <= tensor([self.length])))
        g, G = self.g_grid_values()
        idx, offset = self.g_grid_locate(x)
        return G[idx] + offset * g[idx]

    def g_grid_values(self) -> Tuple[Tensor, Tensor]:
        """
        Returns the values of g(x) and G_0(x) on the grid g_xgrid

        :return: Pair (g, G) where g[i] is the value of g(x) on [g_xgrid[i], g_xgrid[i+1]) and G[i] = G_0(g_xgrid[i]).
        G contains one more element than g, the last element is G_0(L), where L is the length of the genome.
        """
        g = cat((dtensor([1.0]), flatten(stack((self.g, ones_like(self.g_delta_scaled)), dim=1))))
        G = cat((dtensor([0.0]), torch.cumsum(self.g_xgrid_len * g, dim=0)))
        return g, G

    def g_grid_locate(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Locates positions x on the grid g_xgrid

        :param x: Locations as a tensor of arbitrary shape
        :return: Pair (i, d) of tensors with the same shape as x where i is the index of the grid cell containing x,
        i.e. g_xgrid[i] <= x < g_xgrid[i+1], and d = x - g_xgrid[i] the offset of x within that cell
        """
        with no_grad():
            idx = grid_index(self.g_xgrid_double, x)
            return idx, x.double() - self.g_xgrid_double[idx]

    def G_0_inv(self, z: Tensor):
        """
        Evaluates G_0^-1(z) for the specified points z

        :param u: Points at which to evaluate G_0^-1
        :return: G_0^-1(u) at the specified points u
        """

        assert all(z >= 0)
        # Values of g(x) and G_0(x) at the grid points g_xgrid
        # Note: We add an extra point at the end to account for numerical
        # inaccuracies in z. Other than x, which is an integer tensor, z
        # is a floating-point tensor!
        g, Gp = self.g_grid_values()
        G = cat((Gp, Gp[-1, None]+1), dim=0)
        assert all(z < G[-1])
        # Find x_i where G(x_i) <= z but G(x_(i+1)) > z
        with no_grad():
            idx = grid_index(G, z)
            # z = G_0(L) (i.e. the chromosome end) lies in the extra cell added above, map it to the last
            # real grid cell. This happens for replisomes that reached the end of a linear chromosome.
            idx = clamp(idx, max=numel(self.g_xgrid) - 1)
        # Within [z_i, z_(i+1)] the function G is linear, i.e.
        #   G(t) = G(z_i) + (z - z_i) * g[i]
        # and its inverse is thus
        #   G^-1(z) = z_i + (z - G(z_i)) / g[i]
        return self.g_xgrid[idx] + (z - G[idx]) / g[idx]

    @property
    def any_blocked(self) -> bool:
        """ Whether any g range is blocked in any direction """
        return (any(self.g_blocked_fwd) or any(self.g_blocked_rev)).item()

    def assert_unblocked(self, method: str):
        """ Raises an error if any g range is blocked, for methods which do not support blocked g ranges """
        if self.any_blocked:
            raise RuntimeError(f"{method} does not support blocked g ranges")

    def g_blocked_counts(self, mask: Tensor, idx: Tensor, offset: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Counts the g ranges [l, u) selected by `mask` with l < x respectively u <= x for locations x

        :param mask: Boolean tensor selecting g ranges, e.g. `g_blocked_fwd`
        :param idx: Indices of the grid cells containing the locations, see `g_grid_locate()`
        :param offset: Offsets of the locations within their grid cells, see `g_grid_locate()`
        :return: Pair (lt, le) of integer tensors with the same shape as idx
        """
        with no_grad():
            # Grid cell 2*i+1 is g range i, see g_xgrid
            blocked = cat((tensor([False]), flatten(stack((mask, zeros_like(mask)), dim=1))))
            counts = cat((tensor([0]), torch.cumsum(blocked.long(), dim=0)))
            le = counts[idx]
            lt = le + (blocked[idx] & (offset > 0)).long()
        return lt, le

    def g_path_blocked(self, x_idx: Tensor, x_offset: Tensor, o_idx: Tensor, o_offset: Tensor,
                       d: Tensor, wrap: Tensor) -> Tensor:
        """
        Whether the path from o to x in direction d enters a g range blocked in that direction

        Paths in direction +1 are blocked by the ranges in `g_blocked_fwd`, paths in direction -1 by those in
        `g_blocked_rev`. A path enters the range [l, u) if it intersects its interior (l, u), i.e. a replisome coming
        from the left still reaches l, and one coming from the right still reaches u, but if the range is blocked in
        its direction, it gets no further. Replisomes moving in the other direction pass the range normally. Locations
        are passed as grid indices and offsets, see `g_grid_locate()`.

        :param x_idx: Grid cell indices of the end points x
        :param x_offset: Offsets of the end points x within their grid cells
        :param o_idx: Grid cell indices of the starting points o
        :param o_offset: Offsets of the starting points o within their grid cells
        :param d: Directions of the paths (+1 or -1)
        :param wrap: Whether the paths wrap around the end of a circular chromosome, see `G()`
        :return: Boolean tensor of the broadcast shape of the inputs
        """
        with no_grad():
            # Number of blocked ranges intersecting the open interval (a, b) is #{l < b} - #{u <= a}. A path that
            # wraps around covers (a, L) and [0, b), which adds all blocked ranges.
            x_lt, _ = self.g_blocked_counts(self.g_blocked_fwd, x_idx, x_offset)
            _, o_le = self.g_blocked_counts(self.g_blocked_fwd, o_idx, o_offset)
            n_fwd = x_lt - o_le + where(wrap, torch.sum(self.g_blocked_fwd.long()), 0)
            _, x_le = self.g_blocked_counts(self.g_blocked_rev, x_idx, x_offset)
            o_lt, _ = self.g_blocked_counts(self.g_blocked_rev, o_idx, o_offset)
            n_rev = o_lt - x_le + where(wrap, torch.sum(self.g_blocked_rev.long()), 0)
            return where(d > 0, n_fwd, n_rev) > 0

    def G(self, x: Tensor, o: Tensor, d: Tensor) -> Tensor:
        """
        Evaluates G(x,o) = | int_p g(u) du | for the triples (x, o, d) where p is a path from o to x in direction d.

        g(x) is piece-wise constant and = 1 except on the intervals [g_xranges_l[i], g_xranges_u[i]] where g(x) = g[i].

        If the path enters a g range blocked in direction d (see `g_path_blocked()`), G(x,o) is +inf.

        :param x:  Genomic locations as a tensor of arbitrary shape
        :param o: Origins as tensor of the same shape as x
        :return: G(x,o) as a tensor of the same shape as x
        """
        result = (self.G_0(x) - self.G_0(o)) * sign(d)
        wrap = zeros_like(result, dtype=torch.bool)
        if self.is_circular:
            wrap = logical_or(logical_and(x < o, d > tensor([0])), logical_and(o < x, d < tensor([0])))
            result = where(wrap, result + self.G_0(self.length), result)
        if self.any_blocked:
            x_idx, x_offset = self.g_grid_locate(x)
            o_idx, o_offset = self.g_grid_locate(o)
            blocked = self.g_path_blocked(x_idx, x_offset, o_idx, o_offset, d, wrap)
            result = where(blocked, dtensor(float('inf')), result)
        return result

    def g_fun(self, x: Tensor):
        # Find i where x_grid[i] <= x but x_grid[i+1] > x and return g[i]
        g, _ = self.g_grid_values()
        idx, _ = self.g_grid_locate(x)
        return g[idx]


    def surrounding_origins(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Returns a pair (l, r) of tensors of the same shape as x containing the closest left and right origin location

        :param x: tensor containing genomic locations
        :return: pair (l, r) of tensors containg the closest left and right origin location
        """
        assert all(logical_and(x >= tensor([0]), x <= tensor([self.length])))
        # Compute distance between locations x and origins
        # Note: For origins o_1,o_2,...,o_k on a circular genome of length L,
        #       origins_augmented contains o_0 = -(L - o_k), o_1, ..., o_k, o_(k+1) = L + o_1
        dist = x[None,...] - self.origins_augmented[(...,) + (None,) * x.dim()]
        k = self.origins.size()[0]
        # Find index of closest origin to the left of each x
        l_idx = argmin(where(dist >= tensor([0]), dist, tensor([self.int32_max])), dim=0)
        if self.is_circular:
            # If the closest origin is o_0, it's actually o_k and the path wraps around the circular genome
            l_idx = where(l_idx == tensor([0]), tensor([k]), l_idx) - tensor([1])
        # Find index of closest origin to the right of each x
        r_idx = argmax(where(dist <= tensor([0]), dist, tensor([self.int32_min])), dim=0)
        if self.is_circular:
            # If the closest origin is o_(k+1), it's actually o_1 and the path wraps around the circular genome
            r_idx = where(r_idx == tensor([k+1]), tensor([1]), r_idx) - tensor([1])
        # Return origin positions
        return l_idx, r_idx

    def prepare_input(self, x: Tensor):
        """ Performs parameter-independent computations on the model input (i.e. tensor containing genomic locations),
        and returns a `ChromosomeModel.X` instance that represents the processed input. Passing this to `self.T()`
        speeds up the computation.

        :param x: tensor containing genomic locations
        :return: `ChromosomeModel.X` instance containing pre-processed input
        """
        assert all((x >= tensor([0], dtype=torch.int32)) & (x <= tensor([self.length], dtype=torch.int32)))
        with no_grad():
            (i_left, i_right) = self.surrounding_origins(x)
            o_left, o_right = self.origins[i_left], self.origins[i_right]
            g_index, g_offset = self.g_grid_locate(x)
            if self.is_circular:
                # See G(): the path from o in direction d to x wraps around if x < o for d = +1 or o < x for d = -1
                wrap_left, wrap_right = x < o_left, o_right < x
                has_left = has_right = ones_like(x, dtype=torch.bool)
            else:
                wrap_left = wrap_right = zeros_like(x, dtype=torch.bool)
                # surrounding_origins() returns origin index 0 if there is no origin to the left or right
                has_left, has_right = x >= self.origins[0], x <= self.origins[-1]
            at_origin = x == o_left
        return ChromosomeModel.X(x, i_left, i_right, g_index, g_offset, wrap_left, wrap_right,
                               has_left, has_right, at_origin)

    def T(self, x: Tensor | X, temporal: TemporalModel) -> Tensor:
        """Returns the virtual replication time T(x)

        T(x) = min { H(Delta_l) + int_(o_l)^x g(u)/v0 du, H(Delta_r) + int_x^(o_r) g(u)/v0 du }, which is the virtual
        time at which the location x, surrounded by origins o_l on the left and o_r on the right is replicated.
        Delta_l, Delta_r are the (physical) delays with which these origins fire, see Eq. (S1.6). On linear
        chromosomes, locations left of the first or right of the last origin have only one surrounding origin.

        A location x = o_i that coincides with an origin is its own left and right surrounding origin. Such a
        location is replicated either when o_i fires, or when the replisome of one of the adjacent origins
        o_(i-1), o_(i+1) reaches o_i, whichever happens first. Replisomes still stop at the adjacent origin, i.e.
        locations beyond o_i are never replicated by the replisomes of o_(i-1) respectively o_(i+1).

        Replisomes cannot pass g ranges blocked in their direction of movement (see `g_path_blocked()`). Locations
        which no replisome reaches have T(x) = +inf.

        :param x: Genomic locations as a tensor of arbitrary shape or as a prepapred input returned by self.prepare_input
        :param temporal: The time-dependent part of the model
        :return: Virtual replication times as a tensor of the same shape
        """
        x_prep = x if isinstance(x, ChromosomeModel.X) else self.prepare_input(x)
        delay_virtual = temporal.H(self.delay(temporal))
        t_left, t_right = self.fork_arrival_times(x_prep, temporal, delay_virtual)
        # Virtual times at which the replisomes of the adjacent origins o_(i-1) and o_(i+1) reach origin o_i
        origins_prev, delay_virtual_prev = roll(self.origins, 1), roll(delay_virtual, 1)
        origins_next, delay_virtual_next = roll(self.origins, -1), roll(delay_virtual, -1)
        def over_v0(G: Tensor) -> Tensor:
            # G is +inf for paths entering a blocked g range. Avoid dividing inf by v0, whose gradient is NaN even
            # if the result is masked out later.
            finite = isfinite(G)
            return where(finite, where(finite, G, 0.0) / temporal.v0, dtensor(float('inf')))
        arrival_from_prev = delay_virtual_prev + over_v0(self.G(self.origins, origins_prev, dtensor([+1.0])))
        arrival_from_next = delay_virtual_next + over_v0(self.G(self.origins, origins_next, dtensor([-1.0])))
        if not self.is_circular:
            # The first origin has no left neighbour and the last origin no right neighbour on a linear chromosome
            is_first = arange(numel(self.origins)) == 0
            is_last = arange(numel(self.origins)) == numel(self.origins) - 1
            arrival_from_prev = where(is_first, dtensor(float('inf')), arrival_from_prev)
            arrival_from_next = where(is_last, dtensor(float('inf')), arrival_from_next)
        arrival_from_adjacent = minimum(arrival_from_prev, arrival_from_next)
        # Locations that coincide with an origin are also replicated by the replisomes of the adjacent origins
        t = minimum(t_left, t_right)
        return where(x_prep.at_origin, minimum(t, arrival_from_adjacent[x_prep.origin_index_left]), t)

    def fork_arrival_times(self, x_prep: X, temporal: TemporalModel,
                           delay_virtual: Tensor | None = None) -> Tuple[Tensor, Tensor]:
        """Returns the virtual times at which the replisomes of the surrounding origins reach x

        These are H(Delta_l) + int_(o_l)^x g(u)/v0 du for the replisome moving rightwards from the left origin o_l,
        and H(Delta_r) + int_x^(o_r) g(u)/v0 du for the replisome moving leftwards from the right origin o_r, see
        `T()`. On linear chromosomes, the time is +inf if there is no origin to the left respectively right of x.

        :param x_prep: Pre-processed genomic locations, see `prepare_input()`
        :param temporal: The time-dependent part of the model
        :param delay_virtual: The virtual origin firing times H(Delta), computed if not provided
        :return: Pair (t_left, t_right) of virtual arrival times with the same shape as x
        """
        if delay_virtual is None:
            delay_virtual = temporal.H(self.delay(temporal))
        # Evaluate G(x, o_l, +1) and G(x, o_r, -1), see G(), using the pre-processed input
        g, G = self.g_grid_values()
        G_0_x = G[x_prep.g_index] + x_prep.g_offset * g[x_prep.g_index]
        G_0_origins = self.G_0(self.origins)
        G_0_L = G[-1]
        G_left = where(x_prep.wrap_left, G_0_L, 0.0) + (G_0_x - G_0_origins[x_prep.origin_index_left])
        G_right = where(x_prep.wrap_right, G_0_L, 0.0) - (G_0_x - G_0_origins[x_prep.origin_index_right])
        t_left = delay_virtual[x_prep.origin_index_left] + G_left / temporal.v0
        t_right = delay_virtual[x_prep.origin_index_right] + G_right / temporal.v0
        if not self.is_circular:
            # surrounding_origins() returns origin index 0 if there is no origin to the left or right on a
            # linear chromosome. Exclude these, otherwise G() is negative and T(x) precedes the origin firing time.
            t_left = where(x_prep.has_left, t_left, dtensor(float('inf')))
            t_right = where(x_prep.has_right, t_right, dtensor(float('inf')))
        if self.any_blocked:
            # Replisomes cannot pass g ranges blocked in their direction of movement, see g_path_blocked()
            o_idx, o_offset = self.g_grid_locate(self.origins)
            il, ir = x_prep.origin_index_left, x_prep.origin_index_right
            blocked_left = self.g_path_blocked(x_prep.g_index, x_prep.g_offset, o_idx[il], o_offset[il],
                                               dtensor(+1.0), x_prep.wrap_left)
            blocked_right = self.g_path_blocked(x_prep.g_index, x_prep.g_offset, o_idx[ir], o_offset[ir],
                                                dtensor(-1.0), x_prep.wrap_right)
            t_left = where(blocked_left, dtensor(float('inf')), t_left)
            t_right = where(blocked_right, dtensor(float('inf')), t_right)
        return t_left, t_right

    def g_range_passage_times(self, temporal: TemporalModel) -> Tuple[Tensor, Tensor]:
        """Returns the (physical) time replisomes spend in each g range, and the part of it caused by g > 1

        For g range i = [x_0, x_1) these are
            time_i  = int_(x_0)^(x_1) g(x) / v(t(x)) dx,
            delay_i = int_(x_0)^(x_1) (g(x) - 1) / v(t(x)) dx,
        where v(t) = v0 * h(t) and t(x) is the (physical) time at which x is replicated. If multiple replisomes
        replicate parts of the range, their times are summed up.

        Since g(x) = g_i is constant on the range, dt/dx = g_i / v(t) along each replisome trajectory, so time_i is
        the sum of the times each replisome spends replicating its part of the range, and
        delay_i = (1 - 1/g_i) * time_i. Since no origin lies strictly inside a g range [p,q] (see `__init__()`),
        each of its locations is replicated by the replisome of the origin to the left or right of the range, see
        `T()`. The virtual arrival times of both replisomes are linear with slope +/- g_i / v0 within the range,
        the replisomes therefore meet at the point m in [p,q] where these lines intersect. The left replisome
        replicates [p,m], the right one [m,q].

        :param temporal: The time-dependent part of the model
        :return: Pair (time, delay) of tensors containing one entry per g range
        """
        if numel(self.g_xranges_l) == 0:
            return zeros(0, dtype=torch.float64), zeros(0, dtype=torch.float64)
        p, q = self.g_xranges_l, self.g_xranges_u
        u_left_p, u_right_q, _, _ = self.g_range_arrivals(temporal)
        # Virtual time per base pair within the range
        c = self.g / temporal.v0
        # Meeting point m of the two replisomes, clamped to [p,q]. A missing replisome has arrival time +inf,
        # which moves m to the end of the range at which that replisome would have entered.
        p_d, q_d = p.double(), q.double()
        m = minimum(maximum((p_d + q_d) / 2.0 + (u_right_q - u_left_p) / (2.0 * c), p_d), q_d)
        len_left, len_right = m - p_d, q_d - m
        # Physical time each replisome spends in the range. Replace arrival times of replisomes that don't enter
        # the range by a finite placeholder to avoid inf - inf.
        u_safe = temporal.h_tgrid[0]
        u_l = where(len_left > 0, u_left_p, u_safe)
        u_r = where(len_right > 0, u_right_q, u_safe)
        time_left = where(len_left > 0, temporal.Hinv(u_l + c * len_left) - temporal.Hinv(u_l), 0.0)
        time_right = where(len_right > 0, temporal.Hinv(u_r + c * len_right) - temporal.Hinv(u_r), 0.0)
        time = time_left + time_right
        delay = time * self.g_delta / self.g
        return time, delay

    def g_range_arrivals(self, temporal: TemporalModel) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Returns the virtual times at which the replisomes surrounding each g range arrive at the range

        For g range [p,q] these are the virtual arrival times of the replisome moving rightwards from the closest
        origin to the left of the range at p, and of the replisome moving leftwards from the closest origin to the
        right of the range at q. Each arrival time ignores the respective other replisome, i.e. a replisome may in
        fact meet the other replisome before it arrives. For p (respectively q) the origin to the left (right) is
        the same as for the interior of [p,q] since there are no origins strictly between p and q (see
        `__init__()`), and an origin at p (q) is its own surrounding origin.

        :param temporal: The time-dependent part of the model
        :return: Tuple (u_left_p, u_right_q, oi_left, oi_right) where u_left_p, u_right_q are the virtual arrival
        times and oi_left, oi_right the indices of the origins. On linear chromosomes, there may be no origin to
        the left or right of a range, the arrival time is then +inf and the origin index -1.
        """
        self.assert_unblocked("g_range_arrivals()")
        delay_virtual = temporal.H(self.delay(temporal))
        p_prep, q_prep = self.prepare_input(self.g_xranges_l), self.prepare_input(self.g_xranges_u)
        u_left_p, _ = self.fork_arrival_times(p_prep, temporal, delay_virtual)
        _, u_right_q = self.fork_arrival_times(q_prep, temporal, delay_virtual)
        with no_grad():
            oi_left = where(p_prep.has_left, p_prep.origin_index_left, -1)
            oi_right = where(q_prep.has_right, q_prep.origin_index_right, -1)
        return u_left_p, u_right_q, oi_left, oi_right

    @dataclass
    class GRangeForks:
        """
        The two replisomes surrounding each g range, see `ChromosomeModel.g_range_forks()`

        The forward replisome is the one moving rightwards (direction +1) from the closest origin to the left of the
        range, the reverse replisome the one moving leftwards (direction -1) from the closest origin to the right.
        All fields contain one entry per g range.
        """
        # Indices and names of the origins of the forward and reverse replisome. On linear chromosomes, there may
        # be no origin to the left or right of a range. A replisome also counts as missing if it never fires
        # because its origin is replicated passively before (see `segments()`). The index is then -1 and the name
        # None.
        oi_fwd: Tensor
        oi_rev: Tensor
        origin_fwd: list[str | None]
        origin_rev: list[str | None]
        # (Physical) times at which the replisomes enter and exit the range if they didn't stop when meeting the
        # respective other replisome. The forward replisome enters at the left end and exits at the right end, the
        # reverse replisome vice versa. For the replisome which replicates the whole range these are the actual
        # times. For the other replisome, which then never reaches the range, they are counterfactual, i.e. the
        # times at which that replisome would reach the range next. If the replisomes meet inside the range, entry
        # times are actual and exit times counterfactual for both. Since the delay caused by a region may depend on
        # the direction in which it is replicated, counterfactual times assume g = 1 beyond the point where the
        # replisome would have met the other replisome, i.e. on all locations actually replicated by the other
        # replisome. NaN if the replisome is missing.
        enter_fwd: Tensor
        exit_fwd: Tensor
        enter_rev: Tensor
        exit_rev: Tensor

        @staticmethod
        def cat(parts: list[ChromosomeModel.GRangeForks]) -> ChromosomeModel.GRangeForks:
            """ Concatenates the g ranges of multiple chromosomes """
            return ChromosomeModel.GRangeForks(*(
                builtins.sum((getattr(p, f.name) for p in parts), []) if isinstance(getattr(parts[0], f.name), list)
                else cat([getattr(p, f.name) for p in parts])
                for f in dataclasses.fields(ChromosomeModel.GRangeForks)))

    def g_range_forks(self, temporal: TemporalModel) -> GRangeForks:
        """Returns the replisomes surrounding each g range, and when they (would) enter and exit the range

        Entry and exit times ignore that replisomes stop when they meet, see `GRangeForks`. A range [p,q] lies
        within the segment between its forward and reverse origin (see `segments()`), in which the replisomes of
        these origins meet at virtual time u_meet at location x_meet (see `Segment`). Up to x_meet, a replisome's
        (virtual) arrival times are the actual ones, see `g_range_arrivals()`. Since g(x) = g_i is constant on the range, a
        replisome arriving at one end of the range at virtual time u arrives at the other end at virtual time
        u + g_i * (q - p) / v0. Beyond x_meet, the replisome continues counterfactually with g = 1, i.e. arrives at
        location x at virtual time u_meet + |x - x_meet| / v0. A replisome reaches x_meet at virtual time u_meet,
        locations x it reaches before (after) that are thus those with actual arrival times <= u_meet (> u_meet).
        A replisome which never fires (because its origin is replicated passively first) is treated as missing.

        :param temporal: The time-dependent part of the model
        :return: A `GRangeForks` instance
        """
        u_left_p, u_right_q, oi_fwd, oi_rev = self.g_range_arrivals(temporal)
        p_d, q_d = self.g_xranges_l.double(), self.g_xranges_u.double()
        L = self.length.double()
        # Meeting time and location within the segment containing each range. The segment between origins
        # i and i+1 is segment i+1 (with wrap-around on circular chromosomes), see `segments()`.
        segments = self.segments(temporal)
        seg_index = oi_fwd + 1
        if self.is_circular:
            seg_index = remainder(seg_index, numel(self.origins))
        u_meet = stack([sg.u_meet for sg in segments])[seg_index]
        x_meet = stack([sg.x_meet for sg in segments])[seg_index]
        # Replisomes which never fire count as missing
        left_fires = tensor([sg.left_fires for sg in segments], dtype=torch.bool)[seg_index]
        right_fires = tensor([sg.right_fires for sg in segments], dtype=torch.bool)[seg_index]
        oi_fwd = where(left_fires, oi_fwd, -1)
        oi_rev = where(right_fires, oi_rev, -1)
        has_fwd, has_rev = oi_fwd >= 0, oi_rev >= 0

        def wrap(d: Tensor) -> Tensor:
            """ Maps a (directed) distance between two locations of a circular chromosome to [0, L) """
            if self.is_circular:
                d = where(d < -L / 2, d + L, d)
            return clamp(d, min=0.0)

        # Replace arrival times of missing replisomes (+inf) by a finite placeholder to avoid inf - inf
        u_safe = temporal.h_tgrid[0]
        u_l_p = where(has_fwd, u_left_p, u_safe)
        u_r_q = where(has_rev, u_right_q, u_safe)
        # Actual virtual arrival times at both ends of the range
        u_range = self.g * (q_d - p_d) / temporal.v0
        u_l_q, u_r_p = u_l_p + u_range, u_r_q + u_range
        # Counterfactual virtual arrival times beyond the meeting point
        u_l_p = where(u_l_p <= u_meet, u_l_p, u_meet + wrap(p_d - x_meet) / temporal.v0)
        u_l_q = where(u_l_q <= u_meet, u_l_q, u_meet + wrap(q_d - x_meet) / temporal.v0)
        u_r_q = where(u_r_q <= u_meet, u_r_q, u_meet + wrap(x_meet - q_d) / temporal.v0)
        u_r_p = where(u_r_p <= u_meet, u_r_p, u_meet + wrap(x_meet - p_d) / temporal.v0)
        nan_ = dtensor(nan)
        return ChromosomeModel.GRangeForks(
            oi_fwd=oi_fwd, oi_rev=oi_rev,
            origin_fwd=[self.origin_names[i] if i >= 0 else None for i in oi_fwd.tolist()],
            origin_rev=[self.origin_names[i] if i >= 0 else None for i in oi_rev.tolist()],
            enter_fwd=where(has_fwd, temporal.Hinv(u_l_p), nan_),
            exit_fwd=where(has_fwd, temporal.Hinv(u_l_q), nan_),
            enter_rev=where(has_rev, temporal.Hinv(u_r_q), nan_),
            exit_rev=where(has_rev, temporal.Hinv(u_r_p), nan_))

    def g_range_block_probability(self, temporal: TemporalModel) -> Tensor:
        """Returns for each g range the probability p of permanently blocking replisomes equivalent to its delay

        Instead of delaying the replisome passing a g range deterministically, consider a replisome which is blocked
        permanently at the range with probability p, and passes it without delay with probability 1 - p. A blocked
        replisome leaves the location just past the range to be replicated by the next replisome coming from the
        opposite direction (see `GRangeForks`), giving the abundance A = p exp(-lambda t_2) + (1-p) exp(-lambda t_1)
        at that location. Here t_1 is the exit time without delay, t_2 the entry time of the opposite replisome,
        and lambda the growth rate. p is chosen such that A equals the abundance exp(-lambda t_x) predicted by the
        deterministic model, where t_x is the (delayed) exit time, i.e.
            p = (1 - exp(-lambda (t_x - t_1))) / (1 - exp(-lambda (t_2 - t_1))).
        If there is no opposite replisome, t_2 = +inf.

        p refers to the replisome which enters the range first. If it passes the range, t_1 <= t_x <= t_2 and p lies
        in [0, 1]. If the replisomes meet inside the range only because of the delay, i.e. t_1 < t_2 < t_x, no
        replisome passes the range, which corresponds to complete blocking, and p = 1. Note that this is only a lower
        bound, since the data cannot distinguish complete blocking from blocking with a probability close to 1. If the
        replisomes would meet inside the range even without delay (t_2 <= t_1), the range does not block anything
        and p is NaN. p = 0 if the range causes no delay (g = 1).

        :param temporal: The time-dependent part of the model
        :return: Tensor containing one probability per g range
        """
        forks = self.g_range_forks(temporal)
        lam = temporal.growthrate
        has_fwd, has_rev = forks.oi_fwd >= 0, forks.oi_rev >= 0
        inf_ = dtensor(float('inf'))
        enter_fwd, exit_fwd = where(has_fwd, forks.enter_fwd, inf_), where(has_fwd, forks.exit_fwd, inf_)
        enter_rev, exit_rev = where(has_rev, forks.enter_rev, inf_), where(has_rev, forks.exit_rev, inf_)
        # The replisome which enters the range first. If any replisome passes the range, it is this one.
        first_fwd = has_fwd & (enter_fwd <= enter_rev)
        # Entry and (delayed) exit time of that replisome, entry time of the opposite one
        has_any = has_fwd | has_rev
        t_enter = where(has_any, where(first_fwd, enter_fwd, enter_rev), 0.0)
        t_x = where(first_fwd, exit_fwd, exit_rev)
        t_2 = where(first_fwd, enter_rev, enter_fwd)
        # Exit time without delay, i.e. with g = 1 on the range
        t_1 = temporal.Hinv(temporal.H(t_enter) + self.g_xranges_len.double() / temporal.v0)
        # p = expm1(-lambda (t_x - t_1)) / expm1(-lambda (t_2 - t_1)), expm1 avoids cancellation for small delays.
        # expm1(-inf) = -1 covers the case of a missing opposite replisome. If the replisomes meet inside the range
        # because of the delay, t_x > t_2 and thus p > 1, which is clamped to 1 (complete blocking).
        delay = clamp(t_x - t_1, min=0.0)
        p = clamp(expm1(-lam * delay) / expm1(-lam * (t_2 - t_1)), max=1.0)
        p = where(delay > 0.0, p, 0.0)
        # If the replisomes would meet inside the range even without delay, the range blocks nothing
        return where(has_any & (t_1 < t_2), p, dtensor(nan))

    def prior_gdelta(self, hyperparameter_dependence: bool):
        """The prior log-probability of self.g_delta as a realization of independent Gaussians

        Returns the negative log-probability (or, rather, log-density) of self.g_delta as samples of independent
        Exponential distribution with mean self.g_delta_prefactor. Since g_delta = g_delta_scaled * g_delta_prefactor, if
        g_delta follows an exp. dist. with mean self.g_delta_prefactor, g_delta_scaled follows an exp. dist with mean 1.0.

        :param hdependent_only: Whether to return the full density, or only the g_delta-dependent terms

        TODO: Is there really no hyperparameter-dependent term?
        """
        return sum(abs(self.g_delta_scaled))

    def prior(self, parameter, hyperparameter_dependence: bool):
        if parameter is self.delay_scaled:
            return zeros([])
        elif parameter is self.g_delta_scaled:
            return self.prior_gdelta(hyperparameter_dependence=hyperparameter_dependence)
        else:
            raise RuntimeError("no prior defined for this parameter")

    def prior_precision(self, parameter) -> Tensor | None:
        """ The inverse covariance matrix of the (approximately) Gaussian prior of `parameter`, or None if there is none """
        if parameter is self.delay_scaled:
            return None
        elif parameter is self.g_delta_scaled:
            # NOTE: This is only a very rough approximation since the prior on g_delta IS NOT Gaussian.
            return diagflat(dtensor([1.0]) / self.g_delta_prefactor ** 2)
        else:
            raise RuntimeError("no Gaussian prior defined for this parameter")

    def sample_prior(self, parameter):
        if parameter is self.delay_scaled:
            return torch.distributions.exponential.Exponential(dtensor(1.0)).sample(parameter.shape).squeeze()
        elif parameter is self.g_delta_scaled:
            return torch.distributions.exponential.Exponential(dtensor(1.0)).sample(parameter.shape).squeeze()
        else:
            raise RuntimeError("no prior distribution defined for this parameter")

class VelocityModel(nn.Module):
    """
    Model of replisome movement on a genome comprising one or more chromosomes

    The replication velocity v(x,t) = v(t) / g(x) is the product of a time-dependent part v(t), represented by the
    `TemporalModel` instance `self.temporal` and shared by all chromosomes, and a position-dependent part g(x).
    The latter, together with the origins of replication and their firing delays, is specific to each chromosome
    and represented by the `ChromosomeModel` instances in `self.chromosomes`.
    """

    @dataclass
    class X:
        """
        Represents a set of genomic positions on one or more chromosomes, pre-processed for faster model evaluation

        The positions are grouped by chromosome: `parts[i]` holds the positions on chromosome i as a 1-d
        `ChromosomeModel.X`, and `indices[i]` their indices within the flattened input. `inverse` maps the
        concatenation of the per-chromosome parts back to the order of the flattened input, i.e. the position
        with index j in the flattened input is element `inverse[j]` of the concatenation. `part[j]` and `offset[j]`
        are the part containing the j-th position of the flattened input and its index within that part.
        `shape` is the shape of the input.
        """
        shape: torch.Size
        parts: list[ChromosomeModel.X]
        indices: list[Tensor]
        inverse: Tensor
        part: Tensor
        offset: Tensor

        @staticmethod
        def make(shape: torch.Size, parts: list[ChromosomeModel.X], indices: list[Tensor]) -> VelocityModel.X:
            """ Creates an instance from the per-chromosome parts and their indices within the flattened input """
            with no_grad():
                order = cat(indices)
                inverse = empty_like(order)
                inverse[order] = arange(numel(order))
                part = cat([full((numel(ix),), i, dtype=torch.long) for i, ix in enumerate(indices)])[inverse]
                offset = cat([arange(numel(ix)) for ix in indices])[inverse]
            return VelocityModel.X(shape, parts, indices, inverse, part, offset)

        def __getitem__(self, k) -> VelocityModel.X:
            """ Returns the pre-processed input for the subset x[k] of positions """
            with no_grad():
                # Indices (into the flattened input) of the selected positions, in the order of x[k]
                selected = arange(numel(self.inverse)).reshape(self.shape)[k]
                s = selected.reshape(-1)
                s_part, s_offset = self.part[s], self.offset[s]
                parts, indices = [], []
                for i, p in enumerate(self.parts):
                    j = nonzero(s_part == i).reshape(-1)
                    parts.append(p[s_offset[j]])
                    indices.append(j)
            return VelocityModel.X.make(selected.shape, parts, indices)

    def __init__(self, temporal: TemporalModel, chromosomes: List[ChromosomeModel]):
        """
        Creates a model for a genome with the given chromosomes

        :param temporal: The time-dependent part of the model, shared by all chromosomes
        :param chromosomes: The chromosomes of the genome, the chromosome names must be unique
        """
        super().__init__()
        assert len(chromosomes) > 0
        assert len(set(c.name for c in chromosomes)) == len(chromosomes), "chromosome names must be unique"
        self.temporal = temporal
        # NOTE: We use a list instead of a nn.ModuleDict because chromosome names may contain '.', which
        # nn.ModuleDict does not allow. Parameter names are therefore 'chromosomes.<index>.<parameter>'.
        self.chromosomes = nn.ModuleList(chromosomes)
        # The expected delays at g ranges (see `ChromosomeModel.__init__()`) are relative to the replication time
        # of the whole genome
        for c in self.chromosomes:
            c.set_reference_length(self.length)

    @property
    def length(self) -> Tensor:
        """ The total length of all chromosomes """
        return sum(cat([c.length for c in self.chromosomes])).reshape(1)

    @property
    def length_virtual(self) -> Tensor:
        """ The total length of all chromosomes in virtual coordinates """
        return sum(cat([c.length_virtual.reshape(1) for c in self.chromosomes])).reshape(1)

    @classmethod
    def single_chromosome(cls, genome_length: int, is_circular: bool,
                          origins: IntTensor, origin_names: list[str],
                          h_tgrid: DoubleTensor,
                          g_xranges_l: IntTensor, g_xranges_u: IntTensor,
                          g_names: list[str], g_strands: IntTensor,
                          v0_min: DoubleTensor, v0_max: DoubleTensor,
                          g_delayscale: DoubleTensor | None = None,
                          name: str = "chromosome") -> VelocityModel:
        """ Creates a model for a genome comprising a single chromosome """
        temporal = TemporalModel(h_tgrid, v0_min, v0_max)
        chromosome = ChromosomeModel(name, genome_length, is_circular, origins, origin_names,
                                     g_xranges_l, g_xranges_u, g_names, g_strands, g_delayscale)
        return cls(temporal, [chromosome])

    def __setattr__(self, name: str, value: Any):
        # Catch assignments like `model.h_log_raw = ...` or `model.g_delta = ...` that should go to the temporal
        # model or to a chromosome. Without this check, they would silently create a new attribute that the model
        # never uses.
        modules = self.__dict__.get("_modules", {})
        if (not hasattr(type(self), name) and (name not in self.__dict__) and (name not in self._parameters)
                and (name not in self._buffers) and (name not in modules)):
            temporal, chromosomes = modules.get("temporal"), modules.get("chromosomes")
            if (temporal is not None) and hasattr(temporal, name):
                raise AttributeError(f"'{name}' is an attribute of the temporal model, use `model.temporal.{name}`")
            if (chromosomes is not None) and builtins.any(hasattr(c, name) for c in chromosomes):
                raise AttributeError(f"'{name}' is an attribute of the chromosome models, use "
                                     f"`model.chromosomes[i].{name}` or `model.chromosome(name).{name}`")
        super().__setattr__(name, value)

    def chromosome_index(self, chromosome: str | int) -> int:
        """ Returns the index of the chromosome with the given name (or the index itself if passed an index) """
        if isinstance(chromosome, builtins.int):
            if not (0 <= chromosome < len(self.chromosomes)):
                raise IndexError(f"chromosome index {chromosome} out of range")
            return chromosome
        for i, c in enumerate(self.chromosomes):
            if c.name == chromosome:
                return i
        raise KeyError(f"unknown chromosome {chromosome}")

    def chromosome(self, chromosome: str | int) -> ChromosomeModel:
        """ Returns the chromosome with the given name or index """
        return self.chromosomes[self.chromosome_index(chromosome)]

    def clamp_parameters(self):
        """ Clamp parameters to their respective domains """

        # We must update parameters in-place since optimizes identify parameters
        # by the *identity* of the corresponding nn.Parameter object. In-place
        # updates are forbidding while auto-grad is active, but that's OK, this
        # function should be called *before* the model is evaluated.
        with no_grad():
            for c in self.chromosomes:
                c.clamp_parameters()
            # Delays are measured relative to the time at which the first origin in the *genome* fires
            delay_min = min(cat([c.delay_scaled for c in self.chromosomes]))
            for c in self.chromosomes:
                c.delay_scaled[:] -= delay_min
            self.temporal.clamp_parameters()

    def owner(self, parameter: Tensor) -> TemporalModel | ChromosomeModel:
        """ Returns the sub-model (the temporal model or one of the chromosomes) that `parameter` belongs to """
        if self.temporal.owns(parameter):
            return self.temporal
        for c in self.chromosomes:
            if c.owns(parameter):
                return c
        raise RuntimeError("parameter does not belong to this model")

    def resolve_params(self, params: List[nn.Parameter | str] | List[Tuple[str, nn.Parameter]], strict=True) -> List[Tuple[str, nn.Parameter]]:
        """ Converts a list of either `nn.Parameter` instances *or* parameter names into a list of (name, `nn.Parameter`) tuples """

        # Return unmodified parameter list if it is already resolved
        if (len(params) == 0) or isinstance(params[0], tuple):
            return params
        # Create name map
        nmap = {}
        pmap = {}
        for name, param in self.named_parameters():
            nmap[id(param)] = name
            pmap[name] = param
        # Return list of (`str`, `nn.Parameter`) tuples
        r = []
        for p in params:
            if isinstance(p, nn.Parameter) and id(p) in nmap:
                r.append((nmap[id(p)], p))
            elif p in pmap:
                r.append((p, pmap[p]))
            elif strict:
                raise RuntimeError("Parameter " + p + " does not exist")
        return r

    def flatmap(self, params: List[nn.Parameter | str] | List[Tuple[str, nn.Parameter]]) -> Tuple[List[int], Dict[str, int]]:
        """
        Creates a mapping of the given parameters into a flattened vector
        :param params: List of `nn.Parameter` instances, parameter names, or (name, nn.Parameter) tuples
        :return: Tuple (c, pmap) where c[i] is the starting index of the i-th parameter, and pmap[n] the starting
        index of the parameter named n.
        """

        params = self.resolve_params(params)
        # The i-th parameter tensor corresponds to columns c_i,...,c_i + n_i
        c = [0] + np.cumsum([ numel(p) for i, (n, p) in enumerate(params) ]).tolist()
        pmap = { ni: ci for (ci, (ni, pi)) in zip(c[:-1], params) }
        return c, pmap

    def pvector(self, params: List[nn.Parameter | str] | List[Tuple[str, nn.Parameter]], flatmap: Tuple[List[int], Dict[str, int]] | None = None) -> Tensor:
        """
        Returns a flat vector containing the current values of the specified parameters
        :param params: List of nn.Parameter instances, parameter names, or (name, nn.Parameter) tuples
        :param fm: A mapping (c, pmap) as created by `self.flatmap`, if `None` a new mapping is created
        :return: A 1-D tensor containing the concatenated values of the given parameters
        """

        # Obtain parameter map
        params = self.resolve_params(params)
        if flatmap is None:
            c, pmap = self.flatmap(params)
        else:
            c, pmap = flatmap

        # Create output vector and assign values
        r = zeros(Size([c[-1]]), dtype=torch.float64)
        for (ci, (ni, pi)) in zip(c[:-1], params):
            ni = numel(pi)
            r[ci:(ci + ni)] = pi
        return r

    def pvector_assign(self, params: List[nn.Parameter | str] | List[Tuple[str, nn.Parameter]], values : Tensor, flatmap = None):
        """
        Assigns values to the specified parameters taken from the flat parameter vector `values`
        :param params: List of nn.Parameter instances, parameter names, or (name, nn.Parameter) tuples
        :param fm: A mapping (c, pmap) as created by `self.flatmap`, if `None` a new mapping is created
        """

        # Obtain parameter map
        params = self.resolve_params(params)
        if flatmap is None:
            c, pmap = self.flatmap(params)
        else:
            c, pmap = flatmap

        # Assign values from input vector
        assert(values.dim() == 1 and values.size(0) == c[-1])
        for (ci, (ni, pi)) in zip(c[:-1], params):
            ni = numel(pi)
            v = values[ci:(ci + ni)]
            pi.copy_(v.squeeze() if pi.shape + (1,) == v.shape else v)

    def pgradient(self, params: List[nn.Parameter], flatmap = None) -> Tensor:
        """
        Returns a flat vector containing the gradients of the specified parameters
        :param params: List of nn.Parameter instances, parameter names, or (name, nn.Parameter) tuples
        :param fm: A mapping (c, pmap) as created by `self.flatmap`, if `None` a new mapping is created
        :return: A 1-D tensor containing the concatenated values of the given parameters
        """

        # Obtain parameter map
        params = self.resolve_params(params)
        if flatmap is None:
            c, pmap = self.flatmap(params)
        else:
            c, pmap = flatmap

        # Create output vector and assign gradients
        g = zeros(Size([c[-1]]), dtype=torch.float64)
        for (ci, (ni, pi)) in zip(c[:-1], params):
            ni = numel(pi)
            if pi.grad is not None:
                g[ci:(ci + ni)] = pi.grad
        return g

    def chromosome_indices(self, chromosome: str | int | List[str | int] | Tensor | None, shape: torch.Size) -> Tensor:
        """
        Returns the index of the chromosome containing each of a set of locations

        :param chromosome: the chromosome(s) containing the locations. Either a single chromosome name or index which
        applies to all locations, a list of names or indices or an integer tensor of indices with one entry per
        location, or None if the genome comprises only a single chromosome.
        :param shape: the shape of the tensor containing the locations
        :return: an integer tensor of the given shape containing the chromosome indices
        """
        with no_grad():
            if chromosome is None:
                if len(self.chromosomes) != 1:
                    raise ValueError("chromosome must be specified if the genome comprises more than one chromosome")
                c_idx = zeros(shape, dtype=torch.long)
            elif isinstance(chromosome, (str, builtins.int)):
                c_idx = full(shape, self.chromosome_index(chromosome), dtype=torch.long)
            elif isinstance(chromosome, Tensor):
                if chromosome.dtype.is_floating_point or chromosome.dtype == torch.bool:
                    raise ValueError("chromosome must be an integer tensor of chromosome indices")
                c_idx = chromosome.long()
                if (numel(c_idx) > 0) and ((min(c_idx) < 0) or (max(c_idx) >= len(self.chromosomes))):
                    raise IndexError("chromosome index out of range")
            else:
                c_idx = tensor([self.chromosome_index(c) for c in chromosome], dtype=torch.long)
            if c_idx.shape != shape:
                raise ValueError("chromosome must have the same shape as x")
        return c_idx

    def prepare_input(self, x: Tensor, chromosome: str | int | List[str | int] | Tensor | None = None) -> VelocityModel.X:
        """
        Performs parameter-independent computations on the model input (i.e. tensor containing genomic locations),
        and returns a `VelocityModel.X` instance that represents the processed input. Passing this to `self.forward()`
        speeds up the computation.

        :param x: tensor containing genomic locations
        :param chromosome: the chromosome(s) containing the locations, see `chromosome_indices()`
        :return: `VelocityModel.X` instance containing pre-processed input
        """
        with no_grad():
            # Determine the index of the chromosome containing each location
            c_idx = self.chromosome_indices(chromosome, x.shape)
            # Pre-process the locations on each chromosome
            x_flat, c_flat = x.reshape(-1), c_idx.reshape(-1)
            parts, indices = [], []
            for i, c in enumerate(self.chromosomes):
                ix = nonzero(c_flat == i).reshape(-1)
                parts.append(c.prepare_input(x_flat[ix]))
                indices.append(ix)
        return VelocityModel.X.make(x.shape, parts, indices)

    def T(self, x: Tensor | X) -> Tensor:
        """Returns the virtual replication time T(x), see `ChromosomeModel.T()`

        :param x: Genomic locations as a tensor of arbitrary shape or as a prepared input returned by
        self.prepare_input. Passing a tensor is only possible if the genome comprises a single chromosome.
        :return: Virtual replication times as a tensor of the same shape
        """
        x_prep = x if isinstance(x, VelocityModel.X) else self.prepare_input(x)
        t = [c.T(p, self.temporal) for c, p in zip(self.chromosomes, x_prep.parts) if numel(p.x) > 0]
        if len(t) == 0:
            return zeros(x_prep.shape, dtype=torch.float64)
        return cat(t)[x_prep.inverse].reshape(x_prep.shape)

    def P(self, x: Tensor | X) -> Tensor:
        """Returns the predicted DNA abundance P(x)

        :param x: Genomic locations as a tensor of arbitrary shape or as a prepared input returned by
        self.prepare_input. Passing a tensor is only possible if the genome comprises a single chromosome.
        :return: Predicted DNA abundance
        """
        t = self.T(x)
        # Locations which no replisome reaches (because of blocked g ranges) have T(x) = +inf and abundance 0.
        # Replace their T(x) by a finite placeholder, otherwise Hinv() and its gradient produce NaNs.
        reached = isfinite(t)
        t_safe = where(reached, t, self.temporal.h_tgrid[0])
        return where(reached, exp(- self.temporal.growthrate * self.temporal.Hinv(t_safe)), 0.0)

    def forward(self, x: Tensor | X):
        return self.P(x)

    def trajectories(self, t: Tensor = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Returns the replisome trajectories for all chromosomes, see `ChromosomeModel.trajectories()`

        The last dimension of the returned tensors enumerates the replisomes, the replisomes of chromosome
        `self.chromosomes[i]` follow those of chromosome `self.chromosomes[i-1]`.
        """
        r = [c.trajectories(self.temporal, t) for c in self.chromosomes]
        return tuple(cat(parts, dim=-1) for parts in zip(*r))

    def prior(self, parameter, hyperparameter_dependence: bool):
        return self.owner(parameter).prior(parameter, hyperparameter_dependence)

    def sample_prior(self, parameter):
        return self.owner(parameter).sample_prior(parameter)

    def g_range_passage_times(self) -> Tuple[Tensor, Tensor]:
        """
        Returns the passage times and delays of all g ranges, see `ChromosomeModel.g_range_passage_times()`

        The g ranges of chromosome `self.chromosomes[i]` follow those of chromosome `self.chromosomes[i-1]`.
        """
        r = [c.g_range_passage_times(self.temporal) for c in self.chromosomes]
        return tuple(cat(parts) for parts in zip(*r))

    def g_range_forks(self) -> ChromosomeModel.GRangeForks:
        """
        Returns the replisomes surrounding each g range, see `ChromosomeModel.g_range_forks()`

        The g ranges of chromosome `self.chromosomes[i]` follow those of chromosome `self.chromosomes[i-1]`. Origin
        indices refer to the origins of the chromosome containing the respective g range.
        """
        return ChromosomeModel.GRangeForks.cat([c.g_range_forks(self.temporal) for c in self.chromosomes])

    def g_range_block_probability(self) -> Tensor:
        """
        Returns the blocking probability equivalent to the delay at each g range, see
        `ChromosomeModel.g_range_block_probability()`

        The g ranges of chromosome `self.chromosomes[i]` follow those of chromosome `self.chromosomes[i-1]`.
        """
        return cat([c.g_range_block_probability(self.temporal) for c in self.chromosomes])

    def completion_time(self) -> Tensor:
        """
        Returns the (physical) time at which the whole genome is replicated, i.e. the duration of the replication cycle

        Times are measured relative to the firing of the first origin of the genome (see `clamp_parameters()`), so
        this is the latest time at which a chromosome is fully replicated, see `ChromosomeModel.completion_time()`.
        """
        return max(stack([c.completion_time(self.temporal) for c in self.chromosomes]))

    def v_t_posterior_stderr(self, t: Tensor,
                             loss: VelocityModelLoss | None = None,
                             posterior_cov: Tensor | None = None,
                             pmap: dict[str, int] | None = None) -> Tensor:
        """
        The standard deviation of the posterior distribution of v(t)

        NOTE: the posterior covariance matrix posterior_cov should *not*
        include the parameter temporal.v0_log. In our model, the only part of
        the posterior likelihood that discriminates between two solutions
        v0, h(t) and v0', h' where v0 = a * v0' and h'(t) = a * h(t) is the
        prior on log h -- it will prefer the solution which reduces the overall
        square-deviation of log h(t) from zero. Therefore, if we include v0
        in the posterior covariance matrix, the credible interval for h(t)
        is not, in fact determined by the data -- it is solely determined
        by the prior on h(t). While this is in principle correct, it inflates
        the credible interval of v0 * h(t) unless the anti-correlation of these
        two parameters is correctly taken into account, which we currently do
        not do. We therefore instead warn if the posterior covariance matrix
        includes v0. In principle, this could be handled properly by taking the
        correlation between v0 and h(t) into account when computing the credible
        interval for v0 * h(t). However, the benefit is small, since once h(t) is
        fixed, v0 is defined by the data very precisely, and thus adds little to
        the uncertainty about v0 * h(t).

        :param t: The times t at which to evaluate the standard deviation of v(t)
        :param loss: The loss object
        :param posterior_cov: The posterior covariance matrix of the raw model parameters
        :param pmap: Parameter map for `posterior_cov` that maps parameter names to their
        starting indices in `posterior_cov`
        :return: A tensor containing the posterior standard deviation of v(t) at times `t`
        """

        if (posterior_cov is None) and (pmap is None) and (loss is not None):
            posterior_cov, pmap = self.posterior_cov(loss, ["temporal.h_log_raw"])
        elif (posterior_cov is not None) and (pmap is not None) and (loss is None):
            pass
        else:
            raise RuntimeError("Either posterior_cov and pmap, or loss must be specififed (but not both)")
        if "temporal.v0_log" in pmap:
            print("WARNING: Parameter temporal.v0_log should be excluded from the covariance " +
                  "matrix used with v_t_posterior_stderr(), otherwise the posterior std. " +
                  "err. is inflated due to the anti-correlation of v0 and h(t).")

        # Extract covariances of h_log_raw
        assert "temporal.h_log_raw" in pmap
        i, l = pmap["temporal.h_log_raw"], numel(self.temporal.h_log_raw)
        pc = posterior_cov[i:(i + l), i:(i + l)]

        # Return scaled posterior standard deviation of h(t)
        return self.temporal.v0 * self.temporal.h_posterior_stderr(t, pc)

    def posterior_cov_inv(self, loss,
                          params: List[nn.Parameter | str] | List[Tuple[str, nn.Parameter]] | None,
                          ignore_priors=False) -> Tuple[Tensor, Dict[str, int]]:
        # Get parameters
        if params is None:
            params = [*self.parameters()]
        # Obtain jacobian for parameters
        dM, pmap = loss.jacobian(params=params)
        # Compute dM^T diag(1/s_1^2, ..., 1/s_N^2) dM
        u = einsum("ji,j,jk->ik", dM, loss.weights, dM).clone()
        if not ignore_priors:
            # Add the inverse prior covariance matrix for each parameter that has a (approximately) Gaussian
            # prior. We do this by querying the parameter map to find the sub-matrix of u that corresponds
            # to the parameter, and adding the inverse of its prior covariance matrix.
            for name, i in pmap.items():
                p = self.get_parameter(name)
                prior_precision = self.owner(p).prior_precision(p)
                if prior_precision is not None:
                    l = numel(p)
                    u[i:(i+l), i:(i+l)] += prior_precision
        return u, pmap

    def posterior_cov(self, loss,
                      params: List[nn.Parameter | str] | List[Tuple[str, nn.Parameter]] | None,
                      ignore_priors=False) -> Tuple[Tensor, Dict[str, int]]:
        # Obtain inverse posterior cov matrix and invert
        u, pmap = self.posterior_cov_inv(loss, params, ignore_priors)
        return inverse(u), pmap
        

class VelocityModelLoss(nn.Module):
    """
    The negative log-posterior of the model parameters given observed abundances y_obs (with standard errors s_obs)

    The observations may lie on several chromosomes. Since they stem from a single sample, the predicted abundances
    on all chromosomes are scaled by the same factor, which is chosen such that the mean predicted abundance matches
    the mean observed abundance. This factor depends on all parameters, so the predicted abundance at a position on
    one chromosome depends (weakly) on the parameters of all other chromosomes as well.
    """

    def __init__(self, x: Tensor, y_obs: Tensor, s_obs: Tensor, model: VelocityModel, params: List[nn.Parameter],
                 chromosome: str | int | List[str | int] | Tensor | None = None):
        """
        :param x: 1-d tensor containing the genomic positions of the observations
        :param y_obs: observed abundances, the first dimension must match x
        :param s_obs: standard errors of the observed abundances, must have the same shape as y_obs
        :param model: the model
        :param params: the parameters which are being fitted, their priors are included in the loss
        :param chromosome: the chromosome(s) containing the positions x, see `VelocityModel.chromosome_indices()`.
        May be omitted if the genome comprises only a single chromosome.
        """
        super().__init__()
        self.model = model
        self.params = params
        self.x = x
        self.x.requires_grad = False
        assert self.x.dim() == 1
        # Index of the chromosome containing each position, i.e. x[i] lies on model.chromosomes[chromosome[i]]
        self.chromosome = self.model.chromosome_indices(chromosome, self.x.shape)
        self.x_prep = self.model.prepare_input(self.x, self.chromosome)
        self.y_obs = y_obs
        self.y_obs.requires_grad = False
        if any(isnan(self.y_obs)):
            raise RuntimeError("y_obs contains NaNs")
        self.s_obs = s_obs
        self.s_obs.requires_grad = False
        if any(isnan(self.s_obs)):
            raise RuntimeError("s_obs contains NaNs")
        assert self.y_obs.dim() in [1, 2]
        assert self.y_obs.size()[0] == self.x.size()[0]
        assert self.s_obs.size() == self.y_obs.size()
        with no_grad():
            self.y_obs_mean = mean(self.y_obs, dim=0, keepdim=True)
            self.weights = tensor([1.0]) / (self.s_obs ** 2)

    def abundance_and_scale(self) -> Tuple[Tensor, Tensor]:
        abd = self.model(self.x_prep)
        scale = self.y_obs_mean / mean(abd)
        assert scale.dim() == self.y_obs.dim()
        assert scale.size()[0] == 1
        return (abd, scale)

    def y_pred(self) -> Tensor:
        abd, scale = self.abundance_and_scale()
        return scale * abd

    def forward(self) -> Tensor:
        yp = self.y_pred()
        if any(isnan(yp)):
            print(f"WARNING: model prediction is NaN for windows {','.join(map(str, nonzero(isnan(yp)).squeeze(1).tolist()))}")
        l = 0.5 * sum(self.weights * ((self.y_obs - yp) ** 2))
        if any(isnan(l)):
            print("WARNING: log-likelihood is NaN")
        for n, p in self.model.resolve_params(self.params, strict=False):
            v = self.model.prior(p, False).reshape(())
            if any(isnan(v)):
                print(f"WARNING: prior for parameter {n} is NaN")
            l += v
        return l
    
    def jacobian(self, params: List[nn.Parameter | str] | List[Tuple[str, nn.Parameter]] | None) -> [Tensor, Dict[str, int]]:
        # Get parameters
        if params is None:
            params = self.params
        # Resolve parameters
        params = self.model.resolve_params(params)
        # The i-th parameter tensor corresponds to columns c_i,...,c_i + n_i of the
        # jacobian, where n_i is the number of elements of the i-th parameter tensor
        c, pmap = self.model.flatmap(params)
        # Evaluate model for all locations to find scaling factor
        _, scale = self.abundance_and_scale()
        # Prepare output matrix
        n =  self.x.size()[0]
        J = zeros(Size([n, c[-1]]), dtype=torch.float64)
        # Evaluate at locations x_1,..., x_N individually to build jacobian
        for k in builtins.range(0,n):
            # Zero gradients
            for n, p in params:
                if p.grad is not None:
                    p.grad.zero_()
            # Evaluate model at position x_i and compute gradients
            # NOTE: retain_graph=True is necessary because scale depends on the parameters (it is
            #       computed from the model prediction at all locations), and its graph is re-used
            #       for every k. This is intended, since the gradient of M(x) in Eq. (S1.30) includes
            #       the dependence of the scale factor on the parameters. Computing the gradient
            #       of scale once and applying the product rule would allow us to re-enable donated buffers.
            y_pred = scale * self.model(self.x_prep[k])
            y_pred.backward(inputs=[p for n, p in params], retain_graph=True)
            J[k, :] = self.model.pgradient(params, (c, pmap))
        return J, pmap

    def newton_step(self, params: List[nn.Parameter | str]) -> Tensor:
        # NOTE: This is based on the observation that we expect 
        #   tilde-Sigma_h dM^T Sigma_y^-1 (y - m) - tilde-Pi_h Theta
        # to vanish when Theta is a MAP estimate of the parameters.
        # However, the updates this suggests for Theta do not seem to be reasonable,
        # it's unclear whether the problem is numerical or principal issue.
        # Code left in for now because it may be useful in the future
        # The i-th parameter tensor corresponds to columns c_i,...,c_i + n_i of the
        # jacobian, where n_i is the number of elements of the i-th parameter tensor
        c, pmap = self.model.flatmap(params)
        # Compute jacobian dM
        dM = self.jacobian(params)
        # Construct covariance matrix tilde-Sigma_h = Sigma_h ^ -1
        tS_h = zeros(Size([c[-1], c[-1]]), dtype=torch.float64)
        i, l = pmap["temporal.h_log_raw"], numel(self.model.temporal.h_log_raw)
        tS_h[i:(i+l), i:(i+l)] = self.model.temporal.h_cov(include_sigma=True)
        # Compute tSigma_h * dM^T * Sigma_y^-1 * (y - m)
        r1 = self.weights * ((self.y_obs - self.y_pred()) ** 2)
        r2 = t(dM) @ r1
        r3 = tS_h @ r2
        return r3
        
        
