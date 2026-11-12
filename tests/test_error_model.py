"""
Unit tests for `velogp.errormodel`.

Run with `python -m pytest` from the repository root.
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from velogp import errormodel


def test_neighbour_pairs_hand_computed():
    seq = np.array(["a"] * 4)
    y = np.array([1.0, 2.0, 4.0, 3.0])
    q = np.array([0.1, 0.2, 0.3, 0.4])
    p = errormodel.neighbour_pairs(seq, np.arange(4), y, q)
    np.testing.assert_allclose(p.y, [1.5, 3.0, 3.5])
    np.testing.assert_allclose(p.q, np.sqrt([(0.01 + 0.04) / 2, (0.04 + 0.09) / 2, (0.09 + 0.16) / 2]))
    np.testing.assert_allclose(p.s2, [0.5, 2.0, 0.5])


def test_neighbour_pairs_order_chromosomes_and_filtered_windows():
    # Two chromosomes, numbered before filtering, rows shuffled. Window b:2 (index 6) was filtered out, so
    # b:1 and b:3 do not form a pair, and neither do the last window of a and the first window of b
    seq = np.array(["a", "a", "a", "b", "b", "b", "b"])
    window = np.array([0, 1, 2, 3, 4, 6, 7])
    y = np.array([1, 2, 4, 10, 11, 13, 16], dtype=float)
    perm = np.random.default_rng(0).permutation(len(y))
    p = errormodel.neighbour_pairs(seq[perm], window[perm], y[perm], np.ones(len(y)))
    np.testing.assert_allclose(p.y, [1.5, 3.0, 10.5, 14.5])
    np.testing.assert_allclose(p.s2, [0.5, 2.0, 0.5, 4.5])


def simulate(d, n=20000, seed=1):
    rng = np.random.default_rng(seed)
    pos = np.arange(n) * 1000.0
    # Smoothly varying true abundance between 0.5 and 1
    mu = 0.75 + 0.25 * np.cos(2 * np.pi * pos / pos[-1])
    # Poissonian error of a ratio of counts with ~5000 reads per window
    q = mu * np.sqrt(1 / (5000 * mu) + 1 / 5000)
    y = mu + rng.normal(size=n) * np.sqrt(q ** 2 + d * mu ** 2)
    return np.array(["chr"] * n), pos, y, q


@pytest.mark.parametrize("d", [1e-4, 4e-4, 1e-3])
def test_fit_recovers_overdispersion(d):
    seq, pos, y, q = simulate(d)
    res = errormodel.fit(seq, np.arange(len(y)), y, q)
    assert res.d == pytest.approx(d, rel=0.15)
    assert res.mean_ratio_model == pytest.approx(1.0, abs=0.05)
    assert res.mean_ratio_poisson > res.mean_ratio_model


def test_fit_without_overdispersion():
    seq, pos, y, q = simulate(0.0)
    res = errormodel.fit(seq, np.arange(len(y)), y, q)
    # Poisson variance alone is q^2 ~ 3e-4 * mu^2, so d should be negligible compared to that
    assert res.d < 2e-5
    assert res.mean_ratio_poisson == pytest.approx(1.0, abs=0.05)


def test_fit_underdispersed_data_gives_zero():
    y = np.ones(10)
    assert errormodel.fit_overdispersion(y, q=np.ones(10), s2=np.full(10, 0.5)) == 0.0


def test_fit_without_usable_windows_raises():
    with pytest.raises(ValueError):
        errormodel.fit_overdispersion(np.ones(3), np.ones(3), np.full(3, np.nan))


def test_stderr():
    res = errormodel.ErrorModelFit(d=0.01, pairs=None, n=0, mean_ratio_poisson=1, mean_ratio_model=1)
    np.testing.assert_allclose(res.stderr(np.array([2.0]), np.array([0.3])), [np.sqrt(0.09 + 0.04)])


def test_plot(tmp_path):
    seq, pos, y, q = simulate(4e-4, n=2000)
    res = errormodel.fit(seq, np.arange(len(y)), y, q)
    out = tmp_path / "errormodel.pdf"
    errormodel.plot(str(out), res)
    assert out.stat().st_size > 0
