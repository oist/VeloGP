from typing import Any, Dict, Optional, Union, Tuple, List
import builtins
import functools
from dataclasses import dataclass
from collections import namedtuple
import math
import numpy as np
import torch
from torch import nn, linalg
from torch import *
from builtins import float, int

def dtensor(v, **kw_args) -> DoubleTensor:
    return torch.tensor(v, dtype=torch.double, **kw_args)

class VelocityModel(nn.Module):
    int32_min : int = -2**31
    int32_max : int = 2**31 - 1

    H_EIGENFUNCTION_VARIANCE_THRESHOLD : float = 1e-12

    @dataclass
    class Eigenbasis:
        eigenvalues: Tensor
        eigenvectors: Tensor
        k: int
        threshold: Tensor

    @dataclass
    class X:
        x: Tensor
        origin_index_left: Tensor
        origin_index_right: Tensor

    def __init__(self, genome_length: int, is_circular: bool, origins: IntTensor, h_tgrid: DoubleTensor, g_xranges_l: IntTensor, g_xranges_u: IntTensor):
        super().__init__()
        with no_grad():
            # Genome
            self.genome_length = tensor([genome_length], dtype=torch.int32, requires_grad=False)
            self.is_circular = is_circular
            # Growth rate
            self.growthrate = dtensor([1/3600], requires_grad=False)
            # Baseline velocity
            self.v0 = dtensor([100.0])
            # Gaussian process hyperparameters
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
            # Eigenbasis of the covariance matrix of H, see update_h_cov_eig()
            self.h_cov_eig = None
            self.h_cov_eig_timescale = None
            # Origins of replication
            self.origins = origins
            self.origins.requires_grad = False
            # Check that origins is a 1-d tensor with entries in ascending order
            assert (self.origins.dtype == torch.int32)
            assert (self.origins.dim() == 1)
            assert all(self.origins < cat((self.origins[1:], dtensor([float('inf')])))).detach().item()
            # Origins of replication, augmented to account for wrap-around if the genome is circular
            if self.is_circular:
                self.origins_augmented = cat((self.origins[-1] - dtensor([genome_length]), self.origins, dtensor([genome_length]) + self.origins[0]))
                self.origins_augmented.requires_grad = False
            else:
                self.origins_augmented = self.origins
                self.origins_augmented.requires_grad = False
            # Replication delayys
            self.delay = zeros_like(self.origins, dtype=torch.double)
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
            # Grid on which g(x) is defined, derived from g_xranges_l, g_xranges_u
            self.g_xgrid = cat((tensor([0]), flatten(stack((self.g_xranges_l, self.g_xranges_u), dim=1))))
            self.g_xgrid.requires_grad = False
            self.g_xgrid_len = cat((self.g_xgrid[1:], dtensor([genome_length]))) - self.g_xgrid
            self.g_xgrid_len.requires_grad = False
            self.g_log = zeros(self.g_xranges_l.size(), dtype=torch.double)

    def clamp_parameters(self):
        # We must update parameters in-place since optimizes identify parameters
        # by the *identity* of the corresponding nn.Parameter object. In-place
        # updates are forbidding while auto-grad is active, but that's OK, this
        # function should be called *before* the model is evaluated.
        with no_grad():
            assert all(isfinite(self.v0))
            self.v0.clamp_(0.0, math.inf)
            assert all(isfinite(self.delay))
            self.delay.clamp_(0.0, math.inf)
            assert all(isfinite(self.g_log))
            self.g_log.clamp_(0.0, 10)
            assert all(isfinite(self.h_timescale))
            self.h_timescale.clamp_(0.0, math.inf)
            assert all(isfinite(self.h_sigma))
            self.h_sigma.clamp_(0.0, math.inf)
            if self.h_basis == "canonical":
                h_log_raw_scale = 1
            elif self.h_basis == "eigenbasis":
                with no_grad():
                    h_log_raw_scale = (self.h_sigma ** 2) * self.h_eigenvalues()
            assert all(isfinite(self.h_log_raw))
            self.h_log_raw.clamp_(-5 * h_log_raw_scale, 5 * h_log_raw_scale)

    def normalize_parameters(self):
        with no_grad():
            # The delay parameter actually only has k-1 degrees of freedom,
            # where k is the number of origins. Actually parameterizing with
            # k-1 parameters, however, is difficult without either breaking
            # symmetry or interfering with gradient descent. We therefore
            # enforce the constraint that one delay must be here, which takes
            # effect *after* the optimizer has finished.
            self.delay.sub_(min(self.delay))

    def v_t(self, t: Tensor) -> Tensor:
        """Evaluates v_t(t) = v0 * h(t) at the specified points t

        :param t: Times t as a tensor of arbitrary shape
        :return: v_t(t) as a tensor of the same shape as t
        """
        return self.v0 * self.h(t)

    def v_t_posterior_stderr(self, t: Tensor, posterior_cov: Tensor, pmap: Dict[int, int] = None) -> Tensor:
        if pmap is None:
            pc = posterior_cov
        else:
            assert "h_log_raw" in pmap
            i, l = pmap["h_log_raw"], numel(self.h_log_raw)
            pc = posterior_cov[i:(i+l), i:(i+l)]

        return self.v0 * self.h_posterior_stderr(t, pc)


    def h_indices(self, t: Tensor) -> Tensor:
        assert all(t >= self.h_tgrid[0])
        # Compute H(t) on the (fixed) grid h_tgrid (we could call H here, but this is faster)
        # NOTE: We required h_tgrid to contain +inf in its last element, since that guarantees that
        #       the lookup below always finds at least one grid cell where u < H.
        h_tgrid_augmented = cat((self.h_tgrid, dtensor([float('inf')])))
        # Find i where h_tgrid[i] <= u but h_tgrid[i+1] > u
        with no_grad():
            lt = t[None, ...] < h_tgrid_augmented[(...,) + (None,) * t.dim()]
            return argmax(lt.view(torch.int8), dim=0) - tensor([1])

    def h(self, t: Tensor) -> Tensor:
        """Evaluates h(t) at the specified points t

        h(t) is right-continuous and piece-wise constant on the (fixed) grid h_tgrid and obeys h(h_tgrid[i]) = h[i].

        :param t: Times t as a tensor of arbitrary shape
        :return: h(t) as a tensor of the same shape as t
        """
        return exp(self.h_log("canonical")[self.h_indices(t)])

    def h_posterior_stderr(self, times: Tensor, h_log_raw_posterior_cov: Tensor) -> Tensor:
        assert ((h_log_raw_posterior_cov.dim() == 2) and
                (h_log_raw_posterior_cov.size() == Size([self.h_log_raw.size()[0], self.h_log_raw.size()[0]])))
        idx = self.h_indices(times)
        if self.h_basis == "canonical":
            d = sqrt(diagonal(h_log_raw_posterior_cov)[idx])
            return exp(self.h_log_raw[idx]) * d
        elif self.h_basis == "eigenbasis":
            d = sqrt(diagonal(matmul(self.h_eigenbasis(),
                                     matmul(h_log_raw_posterior_cov,
                                            t(self.h_eigenbasis()))))[idx])
            return exp(self.h_log("canonical")[idx]) * d
        else:
            raise ValueError("invalid basis " + self.h_basis)

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
            c = maximum(zeros, minimum(dist, length))
        # H(t[j]) = sum_i h[i] * c[i,j]
        return sum(c * exp(self.h_log()[(...,) + (None,) * t.dim()]), dim=0)

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
        with no_grad():
            lt = u[None, ...] < H[(...,) + (None,) * u.dim()]
            idx = argmax(lt.view(torch.int8), dim=0) - tensor([1])
        # Within [t_i, t_(i+1)] the function H is piece-wise linear, i.e.
        #   H(t) = H(t_i) + (t - t_i) * exp(h[i])
        # and its inverse is thus
        #   H^-1(u) = t_i + (u - H(t_i)) / exp(h[i])
        return self.h_tgrid[idx] + (u - H[idx]) / exp(h_log[idx])

    def G_0(self, x: Tensor) -> Tensor:
        """
        Evaluates G_0(x) = int_0^x du

        g(x) is piece-wise constant and = 1 except on the intervals [g_xranges_l[i], g_xranges_u[i]] where g(x) = g[i].

        :param x:  Locations as a tensor of arbitrary shape
        :return: G(x,o) as a tensor of the same shape as x
        """
        assert all(logical_and(x >= tensor([0]), x <= tensor([self.genome_length])))
        # Tensor dist_ij = x[j] - g_xgrid[i] (j can comprise multiple indices)
        dist = x[None,...] - self.g_xgrid[(...,) + (None,) * x.dim()]
        with no_grad():
            # Tensors with the same shape as dist which contain 0 respectively g_xgrid[i+1] - g_xgrid[i].
            zeros = zeros_like(dist)
            length = ones_like(dist) * self.g_xgrid_len[(...,) + (None,) * x.dim()]
            # Tensor containing the length of the intersection of [0, t[j]] and [h_tgrid[i], h_tgrid[i+1]]
            c = maximum(zeros, minimum(dist, length))
        # Create vector of values of g(x) at the grid points g_xgrid
        g_log_inczero = cat((dtensor([0.0]), flatten(stack((self.g_log, zeros_like(self.g_log)), dim=1))))
        # G(x[j]) = sum_i g[i] * c[i,j]
        return sum(c * exp(g_log_inczero[(...,) + (None,) * x.dim()]), dim=0)

    def G(self, x: Tensor, o: Tensor, d: Tensor) -> Tensor:
        """
        Evaluates G(x,o) = | int_p g(u) du | for the triples (x, o, d) where p is a path from o to x in direction d.

        g(x) is piece-wise constant and = 1 except on the intervals [g_xranges_l[i], g_xranges_u[i]] where g(x) = g[i].

        :param x:  Genomic locations as a tensor of arbitrary shape
        :param o: Origins as tensor of the same shape as x
        :return: G(x,o) as a tensor of the same shape as x
        """
        result = (self.G_0(x) - self.G_0(o)) * sign(d)
        wrap = logical_or(logical_and(x < o, d > tensor([0])), logical_and(o < x, d < tensor([0])))
        result = where(wrap, result + self.G_0(self.genome_length), result)
        return result

    def surrounding_origins(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Returns a pair (l, r) of tensors of the same shape as x containing the closest left and right origin location

        :param x: tensor containing genomic locations
        :return: pair (l, r) of tensors containg the closest left and right origin location
        """
        assert all(logical_and(x >= tensor([0]), x <= tensor([self.genome_length])))
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
        assert all((x >= tensor([0], dtype=torch.int32)) & (x <= tensor([self.genome_length], dtype=torch.int32)))
        with no_grad():
            (i_left, i_right) = self.surrounding_origins(x)
        return VelocityModel.X(x, i_left, i_right)

    def T(self, x: Tensor | X) -> Tensor:
        """Returns the virtual replication time T(x)

        T(x) = min { Delta_l + int_(o_l)^x g(u)/v0 du, Delta_r + int_x^(o_r) g(u)/v0 du }, which is the virtual time
        at which the location x, surrounded by origins o_l on the left and o_r on the right is replicated. Delta_l,
        Delta_r are the delays with which these origins fire.

        :param x: Genomic locations as a tensor of arbitrary shape or as a prepapred input returned by self.prepare_input
        :return: Virtual replication times as a tensor of the same shape
        """
        x_prep = x if isinstance(x, VelocityModel.X) else self.prepare_input(x)
        t_left = self.delay[x_prep.origin_index_left] + self.G(x_prep.x, self.origins[x_prep.origin_index_left], dtensor([+1.0])) / self.v0
        t_right = self.delay[x_prep.origin_index_right] + self.G(x_prep.x, self.origins[x_prep.origin_index_right], dtensor([-1.0])) / self.v0
        return minimum(t_left, t_right)

    def P(self, x: Tensor | X) -> Tensor:
        """Returns the predicted DNA abundance P(x)
        :param x: Genomic locations as a tensor of arbitrary shape or as a prepapred input returned by self.prepare_input
        :return: Predicted DNA abundance
        """
        return exp(- self.growthrate * self.Hinv(self.T(x)))

    def forward(self, x: Tensor | X):
        return self.P(x)

    def h_switch_basis(self, newbasis: str):
        """Switch the basis in which h_log is represented

        The represented h_log stays the same, i.e. h_log("canonical") does not change
        (except for numerical errors). The possible bases are

        (1) 'canonical'. log(h(t)) is represented by the vector log(h(h_tgrid[1])), ..., log(h(h_tgrid[k]))

        (2) 'eigenbasis'. log(h(t)) is represented by the vector [c1, ..., ck] such that
        sum_i c[i] * v[i] = is log(h(h_tgrid[1])), ..., log(h(h_tgrid[kp])) where v[i] are the eigenvectors of
        the covariance matrix of log(h(h_tgrid[1])), ..., log(h(h_tgrid[k])). Eigenvectors whose eigenvalue is
        negative (due to rounding errors) or less than the eigenvalue threshold are ignored, in that case kp < k.

        :param newbasis: The basis to switch to ('canonical' or 'eigenbasis')
        """
        if self.h_basis != newbasis:
            with no_grad():
                if isinstance(self.h_log_raw, nn.Parameter):
                    self.h_log_raw = nn.Parameter(self.h_log(newbasis))
                else:
                    self.h_log_raw = self.h_log(newbasis)
            self.h_basis = newbasis

    def h_log(self, basis: str):
        """Returns the vector log(h(h_tgrid)) in the requested basis.

        See `h_switch_basis` for a discussion of the possible bases.

        :param basis: The basis to return log(h(h_tgrid)) in, either 'canonical' or 'eigenbasis'
        :return: The vector log(h(h_tgrid)) in the specified basis
        """
        if (self.h_basis == "canonical") and (basis == "eigenbasis"):
            return matmul(self.h_log_raw, self.h_eigenbasis())
        elif (self.h_basis == "eigenbasis") and (basis == "canonical"):
            return matmul(self.h_eigenbasis(), self.h_log_raw)
        elif self.h_basis == basis:
            return self.h_log_raw
        else:
            raise ValueError("invalid basis " + self.h_basis + " or " + basis)

    def h_cov(self, include_sigma=True) -> Tensor:
        """The covariance matrix of log(h(h_log_raw[1])), ..., log(h(h_log_raw[k]))

        :param  include_sigma: Whether to Don't include the pre-factor h_sigma ** 2
        :return: the covariance matrix
        """
        if self.h_basis == "canonical":
            s = exp(-((self.h_tgrid[None, :] - self.h_tgrid[:, None]) ** 2) / (2 * (self.h_timescale ** 2)))
            return (self.h_sigma ** 2) * s if include_sigma else s
        elif self.h_basis == "eigenbasis":
            s = diagflat(diagflat(self.h_eigenvalues()))
            return (self.h_sigma ** 2) * s if include_sigma else s
        else:
            raise ValueError("invalid basis " + self.h_basis + " or " + self.h_basis)

    def h_cov_inv(self, include_sigma=True) -> Tensor:
        """The inverse of the covariance matrix of log(h(h_tgrid[1])), ..., log(h(h_tgrid[k]))

        :param  include_sigma: Whether to Don't include the pre-factor 1 / h_sigma ** 2
        :return: the covariance matrix
        """
        if self.h_basis == "canonical":
            s = cholesky_inverse(exp(-((self.h_tgrid[None, :] - self.h_tgrid[:, None]) ** 2) / (2 * (self.h_timescale ** 2))))
            return s / (self.h_sigma ** 2) if include_sigma else s
        elif self.h_basis == "eigenbasis":
            s = diagflat(dtensor([1.0]) / self.h_eigenvalues())
            return s / (self.h_sigma ** 2) if include_sigma else s
        else:
            raise ValueError("invalid basis " + self.h_basis + " or " + self.h_basis)

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
            e = linalg.eigh(self.h_cov(include_sigma=False))
            # Remove the first i eigenvalues (the smallest) where is chosen such that no
            # more than a fraction of self.H_EIGENFUNCTION_VARIANCE_THRESHOLD of the total
            # variance (i.e. sum of eigenvalues) is removed. This essentially removes
            # components for which even small changes would have very drastic effects on
            # the prior likelihood, which poses a challenge for the optimizer.
            variance_fraction = cumsum(e.eigenvalues, dim=0) / sum(e.eigenvalues)
            i = argmax((variance_fraction >= dtensor([self.H_EIGENFUNCTION_VARIANCE_THRESHOLD])).view(torch.int8))
            variance_threshold = e.eigenvalues[i-1] if i > 0 else dtensor([0.0])
            self.h_cov_eig = self.Eigenbasis(e.eigenvalues[i:], e.eigenvectors[:, i:], i.item(), variance_threshold)
            self.h_cov_eig_timescale = self.h_timescale
            # Update the non-h_log-dependent prefactor of the multivariate normal density of h_log
            # We thus compute h_prior_prefactor = -log(sqrt[(2pi)^k * |Sigma|]) where
            # k is the dimension of h_log and Sigma the covariance matrix. Since we have
            # already computed the eigenbasis of Sigma (but without the pre-factor h_sigma!),
            # we use that |Sigma| = h_sigma^(2k) * lambda_1 * ... * lambda_n where
            # lambda_1, ..., lambda_n are the eigenvalues we computed above. We cannot, however,
            # simply ignore eigenvalues less than the variance threshold here, because
            # that would lead to discontinuities if the number of these eigenvalues changes. We
            # thus replace those eigenvalues by the threshold, thus essentially bounding the
            # eigenvalues from below.
            self.h_prior_prefactor = (
                dtensor([- 0.5 * math.log(2*math.pi) * len(self.h_tgrid)])
                - log(self.h_sigma) * len(self.h_tgrid)
                - 0.5 * sum(log(self.h_cov_eig.eigenvalues))
                - 0.5 * log(self.h_cov_eig.threshold) * (len(self.h_tgrid) - self.h_cov_eig.k)
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

    def prior_hlog(self, hyperparameter_dependence):
        """The prior log-probability of self.h_log as a realization of a Gaussian process

        Returns the log of the probability (or, rather, density) of self.h_log as a sample of a multivariate
        Gaussian distribution with mean zero and covariance matrix self.h_sigma^2 *  self.h_cov().

        :param hdependent_only: Whether to return the full density, or only the h_log-dependent terms
        """
        # In the eigenbasis, the covariance matrix diagonalizes, so we just sum the squared entries of h_log
        # (since the expected value is zero) scaled by the diagonal element (the eigenvector). Have to manually
        # add the h_sigma pre-factor since we left that out when computing the covariance matrix.
        # NOTE: The h_prior_prefactor also includes terms for those eigenvalues which were included
        # from the Eigenbasis because they lie below the variance threshold. Here, we don't explicitly account
        # for these components, which amounts to treating them as being zero (and thus they don't contribute
        # to the squared-distance from the mean)
        self.update_h_cov_eig()
        r = self.h_prior_prefactor if hyperparameter_dependence else 0
        r -= 0.5 * sum((self.h_log("eigenbasis") ** 2) / ((self.h_sigma ** 2) * self.h_eigenvalues()))
        return r

    def sample_hlog(self, basis : str, sample_shape : torch.Size = torch.Size([1])):
        self.update_h_cov_eig()
        # Generate as many standard-normal random variables for each sample as the covariance matrix of h
        # has eigenvectors that are significantly larger than zero.
        shape = (self.h_cov_eig.eigenvalues.size()[0],) + sample_shape
        z = torch.distributions.normal.Normal(dtensor([0.0]), dtensor([1.0])).rsample(shape)
        # Generate random h_log in the eigenbasis representation
        z = z * (self.h_sigma ** 2) * self.h_cov_eig.eigenvalues[:, (None,) * len(sample_shape)]
        if basis == "canonical":
            return matmul(self.h_eigenbasis(), z)
        elif basis == "eigenbasis":
            return z
        else:
            raise ValueError("Invalid basis " + basis)

    def prior(self, parameter, hyperparameter_dependence):
        if parameter is self.v0:
            return 0
        elif parameter is self.delay:
            return 0
        elif parameter is self.h_log_raw:
            return self.prior_hlog(hyperparameter_dependence=hyperparameter_dependence)
        elif parameter is self.g_log:
            return 0
        else:
            raise RuntimeError("no prior defined for this parameter")

class VelocityModelLoss(nn.Module):
    def __init__(self, x: Tensor, y_obs: Tensor, s_obs: Tensor, model: VelocityModel):
        super().__init__()
        self.model = model
        self.x = x
        self.x.requires_grad = False
        self.x_prep = self.model.prepare_input(self.x)
        assert self.x.dim() == 1
        self.y_obs = y_obs
        self.y_obs.requires_grad = False
        self.s_obs = s_obs
        self.s_obs.requires_grad = False
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
        return 0.5 * sum(self.weights * ((self.y_obs - self.y_pred()) ** 2))

    def jacobian(self, params: List[nn.Parameter]) -> Tuple[Tensor, Dict[str, nn.Parameter]]:
        # Resolve parameter names
        names = {}
        for name, param in self.model.named_parameters():
            names[id(param)] = name
        # The i-th parameter tensor corresponds to columns c_i,...,c_i + n_i of the
        # jacobian, where n_i is the number of elements of the i-th parameter tensor
        c = [0] + np.cumsum([ numel(p) for i, p in enumerate(params) ]).tolist()
        pmap = { names[id(pi)]: ci for (ci, pi) in zip(c[:-1], params) }
        # Evaluate model for all locations to find scaling factor
        _, scale = self.abundance_and_scale()
        # Prepare output matrix
        n =  self.x.size()[0]
        J = zeros(Size([n, c[-1]]), dtype=torch.float64)
        # Evaluate at locations x_1,..., x_N individually to build jacobian
        for k in builtins.range(0,n):
            # Zero gradients
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()
            # Evaluate model at position x_i and compute gradients
            y_pred = scale * self.model(self.x[k])
            y_pred.backward(inputs=params, retain_graph=True)
            # Copy gradients into the k-th row of the jacobian
            for (ci, pi) in zip(c[:-1], params):
                ni = numel(pi)
                J[k, ci:(ci+ni)] = reshape(pi.grad, Size([ni]))
        return J, pmap
