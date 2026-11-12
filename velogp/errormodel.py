"""
Overdispersion error model for observed abundances.

The standard errors q_i reported by the read mapping pipeline assume purely Poissonian read counts.
To account for additional (overdispersed) noise, we model the variance of the abundance y_i of window i as

    sigma_i^2 = q_i^2 + d * y_i^2,

i.e. as the Poissonian variance plus a noise component whose standard deviation is proportional to the
abundance itself, with a single genome-wide overdispersion parameter d >= 0.

d is estimated from the scatter between adjacent windows. For every pair of windows i and i+1 that are both
valid and adjacent on the same chromosome, we use

    s^2 = 1/2 * (y_(i+1) - y_i)^2

as an estimate of the variance of the pair's mean abundance ybar = (y_i + y_(i+1)) / 2. If the true abundance
varies little between adjacent windows, then E[(y_(i+1) - y_i)^2] = sigma_i^2 + sigma_(i+1)^2, so s^2 estimates
the mean (sigma_i^2 + sigma_(i+1)^2) / 2 of the two windows' variances, which we model as

    sigma^2 = qbar^2 + d * ybar^2   with   qbar^2 = (q_i^2 + q_(i+1)^2) / 2.

d is estimated by maximum quasi-likelihood, treating s^2 as sigma^2 * chi^2_1 (i.e. Gamma-distributed with
shape 1/2 and mean sigma^2). The resulting estimating equation

    sum ybar^2 * (s^2 - sigma^2) / sigma^4 = 0      (summing over pairs)

is unbiased whenever E[s^2] = sigma^2, so the estimate is consistent even if the distribution of s^2 is not
exactly chi^2_1. (The shape of the assumed distribution only rescales the quasi-likelihood, and so does not
affect the estimate.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NeighbourPairs:
    """ Pairs of adjacent windows, see neighbour_pairs() """
    # Mean abundance ybar = (y_i + y_(i+1)) / 2 of each pair
    y: np.ndarray
    # Mean Poissonian std. error qbar = sqrt((q_i^2 + q_(i+1)^2) / 2) of each pair
    q: np.ndarray
    # Observed variance s^2 = (y_(i+1) - y_i)^2 / 2 of each pair
    s2: np.ndarray


def neighbour_pairs(seq: np.ndarray, window: np.ndarray, y: np.ndarray, q: np.ndarray) -> NeighbourPairs:
    """
    Find all pairs of adjacent windows and compute their mean abundance, mean Poissonian std. error and
    observed variance, see the module documentation.

    Expects the table of windows after unreliable windows have been removed. Two windows form a pair if they
    lie on the same chromosome and their indices differ by one. Windows are not wrapped around the ends of
    circular chromosomes, since the last window of a chromosome is usually shorter than the others.

    :param seq: chromosome of each window
    :param window: index of each window, numbered consecutively by position within each chromosome
                   before filtering
    :param y: abundance of each window
    :param q: Poissonian standard error of each window's abundance
    """
    seq, window = np.asarray(seq), np.asarray(window)
    y, q = np.asarray(y, dtype=np.float64), np.asarray(q, dtype=np.float64)
    order = np.lexsort((window, seq))
    seq, window, y, q = seq[order], window[order], y[order], q[order]
    # Remaining windows k and k+1 (in sorted order) form a pair if they are adjacent on the same chromosome
    k = np.flatnonzero((seq[1:] == seq[:-1]) & (window[1:] - window[:-1] == 1))
    return NeighbourPairs(y=0.5 * (y[k] + y[k + 1]),
                          q=np.sqrt(0.5 * (q[k] ** 2 + q[k + 1] ** 2)),
                          s2=0.5 * (y[k + 1] - y[k]) ** 2)


def model_variance(y: np.ndarray, q: np.ndarray, d: float) -> np.ndarray:
    """ Variance sigma^2 = q^2 + d y^2 of the abundances y under the error model """
    return np.asarray(q, dtype=np.float64) ** 2 + d * np.asarray(y, dtype=np.float64) ** 2


def _score(d: float, y2: np.ndarray, q2: np.ndarray, s2: np.ndarray) -> float:
    # Derivative of the quasi-log-likelihood -sum(log sigma^2 + s^2 / sigma^2) / 2 with respect to d
    sigma2 = q2 + d * y2
    return float(np.sum(y2 * (s2 - sigma2) / sigma2 ** 2))


def fit_overdispersion(y: np.ndarray, q: np.ndarray, s2: np.ndarray, rtol: float = 1e-10) -> float:
    """
    Estimate the overdispersion d >= 0 in sigma^2 = q^2 + d y^2 from observed variances s^2.

    Only data points for which y, q and s2 are finite and q > 0 are used.

    :param y: abundances (for pairs of windows, their mean abundance)
    :param q: Poissonian standard errors of the abundances
    :param s2: observed variances, see neighbour_pairs()
    :param rtol: relative tolerance of the estimate
    :return: the estimate of d
    """
    y, q, s2 = (np.asarray(a, dtype=np.float64) for a in (y, q, s2))
    use = np.isfinite(y) & np.isfinite(q) & np.isfinite(s2) & (q > 0)
    if not use.any():
        raise ValueError("no pairs of adjacent windows available to estimate the overdispersion")
    y2, q2, s2 = y[use] ** 2, q[use] ** 2, s2[use]
    # If the likelihood decreases at d=0, the (constrained) maximum lies at d=0
    if _score(0.0, y2, q2, s2) <= 0:
        return 0.0
    # For d >= max(s^2 / y^2) every term of the score is negative, so the maximum lies in (0, d_hi)
    pos = y2 > 0
    d_lo, d_hi = 0.0, float(np.max(s2[pos] / y2[pos]))
    # Bisection on the score (no external dependencies, and robust)
    while d_hi - d_lo > rtol * d_hi:
        d_mid = 0.5 * (d_lo + d_hi)
        if _score(d_mid, y2, q2, s2) > 0:
            d_lo = d_mid
        else:
            d_hi = d_mid
    return 0.5 * (d_lo + d_hi)


@dataclass
class ErrorModelFit:
    """ Result of fitting the error model """
    d: float
    # Pairs of adjacent windows the model was fitted to
    pairs: NeighbourPairs
    # Number of pairs used for the fit
    n: int
    # Mean of s^2/qbar^2 and of s^2/sigma^2 over the pairs used for the fit. Under the correct error
    # model, the latter should be close to one.
    mean_ratio_poisson: float
    mean_ratio_model: float

    def stderr(self, y: np.ndarray, q: np.ndarray) -> np.ndarray:
        """ Standard errors sqrt(q^2 + d y^2) of the abundances y under the fitted model """
        return np.sqrt(model_variance(y, q, self.d))


def fit(seq: np.ndarray, window: np.ndarray, y: np.ndarray, q: np.ndarray) -> ErrorModelFit:
    """
    Estimate the overdispersion d from pairs of adjacent windows, see the module documentation.

    Expects the table of windows after unreliable windows have been removed.

    :param seq: chromosome of each window
    :param window: index of each window, numbered consecutively by position within each chromosome
                   before filtering
    :param y: abundance of each window
    :param q: Poissonian standard error of each window's abundance
    """
    pairs = neighbour_pairs(seq, window, y, q)
    d = fit_overdispersion(pairs.y, pairs.q, pairs.s2)
    use = np.isfinite(pairs.y) & np.isfinite(pairs.q) & np.isfinite(pairs.s2) & (pairs.q > 0)
    y, q, s2 = pairs.y[use], pairs.q[use], pairs.s2[use]
    return ErrorModelFit(
        d=d, pairs=pairs, n=int(use.sum()),
        mean_ratio_poisson=float(np.mean(s2 / q ** 2)),
        mean_ratio_model=float(np.mean(s2 / model_variance(y, q, d))))


def plot(filename: str, result: ErrorModelFit, bins: int = 20,
         xlabel: str = "abundance", qlabel: str = "Poissonian"):
    """
    Plot the observed errors s of pairs of adjacent windows against their mean abundance, together with the
    fitted model.

    Since s^2 scatters strongly around sigma^2, the plot also shows the root-mean-square of s within abundance
    bins, which should agree with the root-mean-square of the model's sigma within the same bins.

    :param xlabel: label of the abundance axis (usually the name of the abundance column)
    :param qlabel: label of the line showing the input std. errors q (usually the name of the std. error column)
    """
    import matplotlib.pyplot as pp

    p = result.pairs
    use = np.isfinite(p.y) & np.isfinite(p.q) & np.isfinite(p.s2)
    y, q, s2 = p.y[use], p.q[use], p.s2[use]
    sigma2 = model_variance(y, q, result.d)
    # Bins containing equal numbers of pairs
    edges = np.unique(np.quantile(y, np.linspace(0, 1, bins + 1)))
    b = np.clip(np.searchsorted(edges, y, side="right") - 1, 0, len(edges) - 2)
    nonempty = np.bincount(b, minlength=len(edges) - 1) > 0
    def binned_rms(v):
        return np.sqrt(np.bincount(b, weights=v, minlength=len(edges) - 1)[nonempty] /
                       np.bincount(b, minlength=len(edges) - 1)[nonempty])
    y_bin = np.bincount(b, weights=y, minlength=len(edges) - 1)[nonempty] / \
            np.bincount(b, minlength=len(edges) - 1)[nonempty]

    fig, ax = pp.subplots(figsize=(3.5, 2.8))
    ax.plot(y, np.sqrt(s2), marker=".", markersize=1, linestyle="", color="lightgrey", label="observed error",
            zorder=1)
    ax.plot(y, np.sqrt(sigma2), marker=".", markersize=0.5, linestyle="", color="lightblue", zorder=2)
    ax.plot(y_bin, binned_rms(s2), marker="o", markersize=2.5, linestyle="", color="black",
            label="observed error (binned)", zorder=4)
    ax.plot(y_bin, binned_rms(sigma2), linewidth=1, color="navy",
            label=f"error model, overdispersion = {result.d:.3g}", zorder=3)
    ax.plot(y_bin, binned_rms(q ** 2), linewidth=1, linestyle="dashed", color="darkorange",
            label=qlabel, zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("error")
    ax.set_ylim(0, np.quantile(np.sqrt(s2), 0.99) * 1.1)
    ax.legend(fontsize="x-small", frameon=False)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    pp.close(fig)
