from __future__ import annotations
from typing import Any, Dict, Optional, Union, Tuple, List
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

class VelocityModel(nn.Module):
    int32_min : int = -2**31
    int32_max : int = 2**31 - 1

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
            i = argmax((variance_fraction >= dtensor([VelocityModel.H_EIGENFUNCTION_VARIANCE_THRESHOLD])).view(torch.int8))
            log_variance_ignored = log(sum(e.eigenvalues[:(i - 1)])) if i > 0 else dtensor([0.0])
            n = e.eigenvalues.size()[0]
            return VelocityModel.Eigenbasis(e.eigenvalues[i:], e.eigenvectors[:, i:], n, n - i, log_variance_ignored)

    @dataclass
    class X:
        """ Represents a set of genomic positions, pre-processes for faster model evaluation """
        x: Tensor
        origin_index_left: Tensor
        origin_index_right: Tensor

    def __init__(self, genome_length: int, is_circular: bool,
                 origins: IntTensor, origin_names: list[str],
                 h_tgrid: DoubleTensor,
                 g_xranges_l: IntTensor, g_xranges_u: IntTensor,
                 v0_min: DoubleTensor, v0_max: DoubleTensor,
                 g_delayscale: DoubleTensor | None = None):
        super().__init__()
        with no_grad():
            # Genome
            self.genome_length = tensor([genome_length], dtype=torch.int32, requires_grad=False)
            self.is_circular = is_circular
            # Growth rate
            self.growthrate = dtensor([1.0/3600.0], requires_grad=False)
            # Baseline velocity
            self.v0_min = v0_min
            self.v0_max = v0_max
            self.v0_log_min = log(self.v0_min)
            self.v0_log_max = log(self.v0_max)
            self.v0_log = 0.5*self.v0_log_min + 0.5*self.v0_log_max
            # Gaussian process hyperparameters
            self.h_cov_function = "sqexp"
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
                self.origins_augmented = cat((self.origins[-1] - dtensor([genome_length]), self.origins, dtensor([genome_length]) + self.origins[0]))
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
            self.g_xranges_len = self.g_xranges_u - self.g_xranges_l
            # Grid on which g(x) is defined, i.e. the grid
            # [ l_1, u_1, l_2, u_2, .... ] where l = from g_xranges_l, u = g_xranges_u
            self.g_xgrid = cat((tensor([0]), flatten(stack((self.g_xranges_l, self.g_xranges_u), dim=1))))
            self.g_xgrid.requires_grad = False
            self.g_xgrid_len = cat((self.g_xgrid[1:], dtensor([genome_length]))) - self.g_xgrid
            self.g_xgrid_len.requires_grad = False
            self.g_delta_scaled = zeros(self.g_xranges_l.size(), dtype=torch.float64)
            # If provided, g_delayscale specifies for each g range, the magnitudes of the expected delays,
            # measured relative to the time a single replisome would take to replicate the genome.
            if g_delayscale is None:
                self.g_delta_prefactor = ones_like(self.g_delta_scaled)
            else:
                # Check that we have as many delayscales as g ranges
                assert (g_delayscale.dim() == 1) and (g_delayscale.size() == self.g_xranges_l.size())
                self.g_delta_prefactor = g_delayscale.double() * self.genome_length.double() / self.g_xranges_len.double()
            self.g_delta_scaled.requires_grad = False

    def clamp_parameters(self):
        """ Clamp parameters to their respective domains """

        # We must update parameters in-place since optimizes identify parameters
        # by the *identity* of the corresponding nn.Parameter object. In-place
        # updates are forbidding while auto-grad is active, but that's OK, this
        # function should be called *before* the model is evaluated.
        with no_grad():
            assert all(isfinite(self.v0_log))
            self.v0_log.clamp_(self.v0_log_min, self.v0_log_max)
            assert all(isfinite(self.delay_scaled))
            self.delay_scaled.clamp_(0.0, math.inf)
            self.delay_scaled[:] -= min(self.delay_scaled)
            assert all(isfinite(self.g_delta_scaled))
            self.g_delta_scaled.clamp_(0.0, 10)
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

    @property
    def delay(self) -> Tensor:
        """ The current value of the origin delays """
        return self.delay_scaled / (10 * self.growthrate)

    @delay.setter
    def delay(self, value: Tensor):
        """ Sets the current value of the origin delays """
        assert(value.shape == self.delay_scaled.shape)
        self.delay_scaled.copy_(value * 10.0 * self.growthrate)

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
    def genome_length_virtual(self):
        """ Returns the length of the genome in virtual coordinates """
        return self.genome_length.double() + sum(self.g_delta * self.g_xranges_len)

    def replisome_position(self, t: Tensor, origin: Tensor, dir: Tensor, delay: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # Compute distance travelled in virtual coordinates
        # TODO: We ignore diffusion here atm
        z = self.G_0(origin) + dir * self.v0 * maximum(self.H(t) - self.H(delay), dtensor([0.0]))
        # Translate distance travelled to position (plus wrap-around count)
        if self.is_circular:
            z_wrap = div(z, self.genome_length_virtual, rounding_mode="floor")
            z = remainder(z, self.genome_length_virtual)
        else:
            z_wrap = zeros_like(z, dtype=bool)
            z = max(dtensor([0.0]), min(z, self.genome_length_virtual))
        # Translate position to physical coordinates
        x = self.G_0_inv(z)
        # Translate position and wrap-around count to distance travelled in physical coordinates
        dx = x - origin
        dx += self.genome_length * z_wrap
        # Compute instantaneous velocities
        v = self.v_t(t) / self.g_fun(x)
        return x, dx, v

    def trajectories(self, t: Tensor = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Use grid on which h() is defined if no output times are specified
        if t is None:
            t = self.h_tgrid
        # Create output tensors, one for replisome positions, or for velocities
        k = 2*numel(self.origins)
        tab_state = ones(t.shape + (k,), dtype=torch.int32)
        tab_x = zeros(t.shape + (k,), dtype=torch.float64)
        tab_dx = zeros(t.shape + (k,), dtype=torch.float64)
        tab_v = zeros(t.shape + (k,), dtype=torch.float64)
        for o_i in builtins.range(0, numel(self.origins)):
            for d_i, d in enumerate([-1, 1]):
                x, dx, v = self.replisome_position(t, origin=self.origins[o_i], dir=d, delay=self.delay[o_i])
                tab_x[:, 2*o_i + d_i] = x
                tab_dx[:, 2*o_i + d_i] = dx
                tab_v[:, 2*o_i + d_i] = v
        # Determine chromosome meeting points
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
            # Get left and right origins and compute time at which the replisomes meet
            o_left = self.origins[oi_left] if oi_left >= 0 else dtensor([0.0])
            d_left = self.delay[oi_left] if oi_left >= 0 else dtensor([0.0])
            o_right = self.origins[oi_right] if oi_right < numel(self.origins) else self.genome_length
            d_right = self.delay[oi_right] if oi_right< numel(self.origins) else dtensor([0.0])
            # Before replisome creation, state = 0, position = origin, distance = 0.0, v = nan
            if ri_left >= 0:
                w = t < d_left
                tab_state[w, ri_left] = 0
                tab_x[w, ri_left] = o_left
                tab_dx[w, ri_left] = 0.0
                tab_v[w, ri_left] = nan
            if ri_right >= 0:
                w = t < d_right
                tab_state[w, ri_right] = 0
                tab_x[w, ri_right] = o_right
                tab_dx[w, ri_right] = 0.0
                tab_v[w, ri_right] = nan
            # Compute meeting time.
            # NOTE: Must be careful on a circular genome with a single origin since G(x, x) == 0
            fudge = tensor(1 if self.is_circular and (o_left == o_right).item() else 0, dtype=torch.int32)
            # TODO: We ignore diffusion here atm
            # TODO: This has to be adapted if G(x,o,d) ever becomes direction-dependent.
            t_meet = self.Hinv((self.G(o_right - fudge, o_left, d=dtensor([1.0]))/self.v0 + self.H(d_left) + self.H(d_right))/2.0)
            x_meet_left, dx_meet_left, _ = self.replisome_position(t_meet, origin=o_left, dir=dtensor([+1.0]), delay=d_left)
            x_meet_right, dx_meet_right, _ = self.replisome_position(t_meet, origin=o_right, dir=dtensor([-1.0]), delay=d_right)
            # After replisomes meet, state = 2, position = meeting point, distance = dist. travelled, v = nan
            if ri_left >= 0:
                w = t > t_meet
                tab_state[w, ri_left] = 2
                tab_x[w, ri_left] = x_meet_left
                tab_dx[w, ri_left] = dx_meet_left
                tab_v[w, ri_left] = nan
            if ri_right >= 0:
                w = t > t_meet
                tab_state[w, ri_right] = 2
                tab_x[w, ri_right] = x_meet_right
                tab_dx[w, ri_right] = dx_meet_right
                tab_v[w, ri_right] = nan
        # Remove first row which ri_right to the entry t=0 we added at the top
        return tab_state, tab_x, tab_dx, tab_v

    def v_t_posterior_stderr(self, t: Tensor,
                             loss: VelocityModelLoss | None = None,
                             posterior_cov: Tensor | None = None,
                             pmap: dict[str, int] | None = None) -> Tensor:
        """
        The standard deviation of the posterior distribution of v(t)

        NOTE: the posterior covariance matrix posterior_cov should *not*
        include the parameter v0_log. In our model, the only part of
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
            posterior_cov, pmap = self.posterior_cov(loss, ["h_log_raw"])
        elif (posterior_cov is not None) and (pmap is not None) and (loss is None):
            pass
        else:
            raise RuntimeError("Either posterior_cov and pmap, or loss must be specififed (but not both)")
        if "v0_log" in pmap:
            print("WARNING: Parameter v0_log should be excluded from the covariance " +
                  "matrix used with v_t_posterior_stderr(), otherwise the posterior std. " +
                  "err. is inflated due to the anti-correlation of v0 and h(t).")

        # Extract covariances of h_log_raw
        assert "h_log_raw" in pmap
        i, l = pmap["h_log_raw"], numel(self.h_log_raw)
        pc = posterior_cov[i:(i + l), i:(i + l)]

        # Return scaled posterior standard deviation of h(t)
        return self.v0 * self.h_posterior_stderr(t, pc)

    def h_indices(self, t: Tensor) -> Tensor:
        """
        Map times `t` to their corresponding indices in `h_tgrid`
        :param t: times to map to indices
        :return: the indices in `h_tgrid` that correspond to the given times
        """

        assert all(t >= self.h_tgrid[0])
        # Compute H(t) on the (fixed) grid h_tgrid (we could call H here, but this is faster)
        # NOTE: We required h_tgrid to contain +inf in its last element, since that guarantees that
        #       the lookup below always finds at least one grid cell where u < H.
        h_tgrid_augmented = cat((self.h_tgrid, dtensor([float('inf')])))
        # Find i where h_tgrid[i] <= u but h_tgrid[i+1] > u
        with no_grad():
            lt = t[None, ...] < h_tgrid_augmented[(...,) + (None,) * t.dim()]
            return argmax(lt.view(torch.int8), dim=0) - tensor([1])

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

        NOTE: The covariance matrix should not include the parameter v0, see discussion in `self.vt_posterior_stderr()`

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
            # Add Sigma_h^-1 for h_log_raw, which has a GP prior
            # We do this by querying the parameter map to find the
            # sub-matrix of u that corresponds to the parameter h_log_raw,
            # and adding the inverse of its prior covariance matrix
            if "h_log_raw" in pmap:
                i, l = pmap["h_log_raw"], numel(self.h_log_raw)
                u[i:(i+l), i:(i+l)] += self.h_cov_inv(include_sigma=True)
            # Add prior for g_delta
            # NOTE: This is only a very rough approximation since the prior
            # on g_delta IS NOT Gaussian.
            if "g_delta_scaled" in pmap:
                i, l = pmap["g_delta_scaled"], numel(self.g_delta_scaled)
                u[i:(i+l), i:(i+l)] += diagflat(dtensor([1.0]) / self.g_delta_prefactor ** 2)
        return u, pmap

    def posterior_cov(self, loss,
                      params: List[nn.Parameter | str] | List[Tuple[str, nn.Parameter]] | None,
                      ignore_priors=False) -> Tuple[Tensor, Dict[str, int]]:
        # Obtain inverse posterior cov matrix and invert
        u, pmap = self.posterior_cov_inv(loss, params, ignore_priors)
        return inverse(u), pmap
        
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
        Evaluates G_0(x) = int_0^x g(u) du

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
        g = cat((dtensor([1.0]), flatten(stack((self.g, ones_like(self.g_delta_scaled)), dim=1))))
        # G(x[j]) = sum_i g[i] * c[i,j]
        return sum(c * g[(...,) + (None,) * x.dim()], dim=0)

    def G_0_inv(self, z: Tensor):
        """
        Evaluates G_0^-1(z) for the specified points z

        :param u: Points at which to evaluate G_0^-1
        :return: G_0^-1(u) at the specified points u
        """

        assert all(z >= 0)
        # Create vector of values of g(x) at the grid points g_xgrid
        g = cat((dtensor([1.0]), flatten(stack((self.g, ones_like(self.g_delta_scaled)), dim=1))))
        # Create vector of values of G_0(x) at the grid points g_xgrid
        # Note: We add an extra point at the end to account for numerical
        # inaccuracies in z. Other than x, which is an integer tensor, z
        # is a floating-point tensor!
        Gp = torch.cumsum(self.g_xgrid_len * g, dim=0)
        G = cat((dtensor([0.0]), Gp, Gp[-1, None]+1), dim=0)
        assert all(z < G[-1])
        # Find x_i where G(x_i) <= z but G(x_(i+1)) > z
        with no_grad():
            lt = z[None, ...] < G[(...,) + (None,) * z.dim()]
            idx = argmax(lt.view(torch.int8), dim=0) - tensor([1])
        # Within [z_i, z_(i+1)] the function G is linear, i.e.
        #   G(t) = G(z_i) + (z - z_i) * g[i]
        # and its inverse is thus
        #   G^-1(z) = z_i + (z - G(z_i)) / g[i]
        return self.g_xgrid[idx] + (z - G[idx]) / g[idx]

    def G(self, x: Tensor, o: Tensor, d: Tensor) -> Tensor:
        """
        Evaluates G(x,o) = | int_p g(u) du | for the triples (x, o, d) where p is a path from o to x in direction d.

        g(x) is piece-wise constant and = 1 except on the intervals [g_xranges_l[i], g_xranges_u[i]] where g(x) = g[i].

        :param x:  Genomic locations as a tensor of arbitrary shape
        :param o: Origins as tensor of the same shape as x
        :return: G(x,o) as a tensor of the same shape as x
        """
        result = (self.G_0(x) - self.G_0(o)) * sign(d)
        if self.is_circular:
            wrap = logical_or(logical_and(x < o, d > tensor([0])), logical_and(o < x, d < tensor([0])))
            result = where(wrap, result + self.G_0(self.genome_length), result)
        return result

    def g_fun(self, x: Tensor):
        # Create vector of values of g(x) at the grid points g_xgrid
        g = cat((dtensor([1.0]), flatten(stack((self.g, ones_like(self.g_delta_scaled)), dim=1))))
        # Find i where x_grid[i] <= z but x_grid[i+1 > z
        with no_grad():
            lt = x[None, ...] < self.g_xgrid[(...,) + (None,) * x.dim()]
            idx = argmax(lt.view(torch.int8), dim=0) - tensor([1])
        # Return g[i]
        return g[idx]


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
        """ Performs parameter-independent computations on the model input (i.e. tensor containing genomic locations),
        and returns a `VelocityModel.X` instance that represents the processed input. Passing this to `self.forward()`
        speeds up the computation.

        :param x: tensor containing genomic locations
        :return: `VelocityModel.X` instance containing pre-processed input
        """
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
        if self.h_cov_function == "sqexp":            
            return exp(- (d ** 2) / 2)
        elif self.h_cov_function == "matern1/2":
            return exp(-d)
        elif self.h_cov_function == "matern3/2":
            dp = math.sqrt(3) * d
            return (1 + dp) * exp(-dp)
        elif self.h_cov_function == "matern5/2":
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
            self.h_cov_eig = VelocityModel.Eigenbasis.make(self.h_cov_reduced_canonical())
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
        if (self.h_basis == "cov"):
            self.update_h_cov_chol()
            r = self.h_prior_prefactor if hyperparameter_dependence else zeros(1)
            r += 0.5 * sum(self.h_log_raw * matmul(self.h_cov_reduced_canonical(), self.h_log_raw))
            return r
        elif (self.h_basis == "col"):
            self.update_h_cov_chol()
            r = self.h_prior_prefactor if hyperparameter_dependence else zeros(1)
            r += 0.5 * sum(self.h_log("col") ** 2)
            return r
        else:
            self.update_h_cov_eig()
            r = self.h_prior_prefactor if hyperparameter_dependence else zeros(1)
            r += 0.5 * sum(self.h_log("eigenbasis_scaled") ** 2)
            return r

    def prior_gdelta(self, hyperparameter_dependence: bool):
        """The prior log-probability of self.g_delta as a realization of independent Gaussians

        Returns the negative log-probability (or, rather, log-density) of self.g_delta as samples of independent
        Exponential distribution with mean self.g_delta_prefactor. Since g_delta = g_delta_scaled * g_delta_prefactor, if
        g_delta follows an exp. dist. with mean self.g_delta_prefactor, g_delta_scaled follows an exp. dist with mean 1.0.

        :param hdependent_only: Whether to return the full density, or only the g_delta-dependent terms

        TODO: Is there really no hyperparameter-dependent term?
        """
        return sum(abs(self.g_delta_scaled))

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

    def prior(self, parameter, hyperparameter_dependence: bool):
        if parameter is self.v0_log:
            return zeros([])
        elif parameter is self.delay_scaled:
            return zeros([])
        elif parameter is self.h_log_raw:
            return self.prior_hlog(hyperparameter_dependence=hyperparameter_dependence)
        elif parameter is self.g_delta_scaled:
            return self.prior_gdelta(hyperparameter_dependence=hyperparameter_dependence)
        else:
            raise RuntimeError("no prior defined for this parameter")

    def sample_prior(self, parameter):
        if parameter is self.v0_log:
            return torch.distributions.uniform.Uniform(low=self.v0_log_min, high=self.v0_log_max).sample(parameter.shape)
        elif parameter is self.delay_scaled:
            return torch.distributions.exponential.Exponential(dtensor(1.0)).sample(parameter.shape)
        elif parameter is self.h_log_raw:
            return self.sample_hlog(self.h_basis)
        elif parameter is self.g_delta_scaled:
            return torch.distributions.exponential.Exponential(dtensor(1.0)).sample(parameter.shape)
        else:
            raise RuntimeError("no prior distribution defined for this parameter")

class VelocityModelLoss(nn.Module):
    def __init__(self, x: Tensor, y_obs: Tensor, s_obs: Tensor, model: VelocityModel, params: List[nn.Parameter]):
        super().__init__()
        self.model = model
        self.params = params
        self.x = x
        self.x.requires_grad = False
        self.x_prep = self.model.prepare_input(self.x)
        assert self.x.dim() == 1
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
        l = 0.5 * sum(self.weights * ((self.y_obs - self.y_pred()) ** 2))
        for n, p in self.model.resolve_params(self.params, strict=False):
            l += self.model.prior(p, False).reshape(())
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
            # TODO: It's unclear why retain_graph=True is necessary here
            #       Getting rid of this would allow us to re-enable donated buffers
            y_pred = scale * self.model(self.x[k])
            y_pred.backward(inputs=[p for n, p in params], retain_graph=True)
            J[k, :] = self.model.pgradient(params, (c, pmap))
        return J, pmap

    def newton_step(self, params: List[nn.Parameter | str]) -> Tensor:
        # NOTE: This is based on the observation that we expect 
        #   tilde-Sigma_h dM^T Sigma_y^-1 (y - m) - tilde-Pi_h Theta
        # to vanish when Theta is a MAP estimate of the parameters.
        # However, the updates this suggests for Theta do not seem to be reasonable,
        # it's unclear whether the problem is numerical or or principal issue.
        # Code left in for now because it may be useful in the future
        # The i-th parameter tensor corresponds to columns c_i,...,c_i + n_i of the
        # jacobian, where n_i is the number of elements of the i-th parameter tensor
        c, pmap = self.model.flatmap(params)
        # Compute jacobian dM
        dM = self.jacobian(params)
        # Construct covariance matrix tilde-Sigma_h = Sigma_h ^ -1
        tS_h = zeros(Size([c[-1], c[-1]]), dtype=torch.float64)
        i, l = pmap["h_log_raw"], numel(self.model.h_log_raw)
        tS_h[i:(i+l), i:(i+l)] = self.model.h_cov(include_sigma=True)
        # Compute tSigma_h * dM^T * Sigma_y^-1 * (y - m)
        r1 = self.weights * ((self.y_obs - self.y_pred()) ** 2)
        r2 = t(dM) @ r1
        r3 = tS_h @ r2
        return r3
        
        
