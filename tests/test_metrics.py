"""Portfolio statistics, and the two unit errors the original made."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data as datamod
import metrics


def test_sharpe_divides_by_volatility_not_variance():
    """The original divided excess return by w' Sigma w.

    That is the variance. The Sharpe ratio divides by its square root. On these
    numbers the difference is a factor of five, and on daily data it is far larger,
    because dividing by a small number squared inflates the result.
    """
    weights = np.array([0.5, 0.5])
    mu = np.array([0.10, 0.10])
    sigma = np.diag([0.08, 0.08])

    variance = metrics.portfolio_variance(weights, sigma)
    volatility = metrics.portfolio_volatility(weights, sigma)
    assert volatility == pytest.approx(np.sqrt(variance))

    correct = metrics.sharpe_ratio(weights, mu, sigma, risk_free_rate=0.03)
    assert correct == pytest.approx((0.10 - 0.03) / volatility)

    wrong = (0.10 - 0.03) / variance
    assert wrong > correct * 4, "the original's version is inflated, not merely different"


def test_volatility_is_the_square_root_of_variance():
    rng = np.random.default_rng(0)
    for _ in range(5):
        w = rng.uniform(0, 1, 4)
        w /= w.sum()
        a = rng.uniform(0.001, 0.01, (4, 4))
        sigma = (a + a.T) / 2
        np.fill_diagonal(sigma, rng.uniform(0.005, 0.02, 4))
        sigma = datamod.nearest_psd(sigma)
        assert metrics.portfolio_volatility(w, sigma) == pytest.approx(
            np.sqrt(metrics.portfolio_variance(w, sigma)))


def test_annualization_uses_a_consistent_basis():
    """Returns scale by the period count, covariance by the same count.

    The original compared a daily mean return against an annual risk-free rate, so
    the excess return was wrong before the division even happened.
    """
    daily_mu = np.array([0.0004, 0.0006])
    daily_cov = np.array([[1e-4, 2e-5], [2e-5, 1.5e-4]])

    annual_mu = metrics.annualize_returns(daily_mu)
    annual_cov = metrics.annualize_covariance(daily_cov)

    assert annual_mu == pytest.approx(daily_mu * 252)
    assert annual_cov == pytest.approx(daily_cov * 252)
    # Volatility scales with the square root of time, which falls out of the above.
    w = np.array([0.5, 0.5])
    assert metrics.portfolio_volatility(w, annual_cov) == pytest.approx(
        metrics.portfolio_volatility(w, daily_cov) * np.sqrt(252))


def test_sharpe_on_mixed_units_differs_materially():
    """Shows the unit mismatch is not a rounding issue."""
    daily_mu = np.array([0.0006, 0.0006])
    daily_cov = np.diag([1e-4, 1e-4])
    w = np.array([0.5, 0.5])

    mixed = metrics.sharpe_ratio(w, daily_mu, daily_cov, risk_free_rate=0.03)
    proper = metrics.sharpe_ratio(w, metrics.annualize_returns(daily_mu),
                                  metrics.annualize_covariance(daily_cov),
                                  risk_free_rate=0.03)
    assert mixed < 0 < proper, "daily return against an annual rate looks like a loss"


def test_zero_volatility_gives_nan_not_a_division_error():
    w = np.zeros(3)
    mu = np.array([0.1, 0.1, 0.1])
    assert np.isnan(metrics.sharpe_ratio(w, mu, np.eye(3) * 0.01))


def test_approximation_ratio_is_one_at_the_optimum():
    assert metrics.approximation_ratio(-0.5, -0.5) == pytest.approx(1.0)
    assert metrics.approximation_ratio(-0.4, -0.5) == pytest.approx(0.8)
    # Works with a negative optimum, which a minimized objective usually is.
    assert metrics.approximation_ratio(-0.6, -0.5) == pytest.approx(0.8)


def test_summarize_reports_holdings():
    w = np.array([0.5, 0.0, 0.5])
    mu = np.array([0.1, 0.2, 0.15])
    sigma = np.diag([0.04, 0.09, 0.04])
    s = metrics.summarize(w, mu, sigma)
    assert s["n_holdings"] == 2
    assert s["return"] == pytest.approx(0.125)
    assert s["volatility"] == pytest.approx(np.sqrt(0.02))


# --- covariance validity --------------------------------------------------------

def test_the_originals_random_covariance_is_not_psd():
    """Pinned because it is a real defect in the original's synthetic data.

    A symmetrized matrix of uniform random entries is generally indefinite, so it is
    not a covariance matrix at all. With this seed it has two negative eigenvalues.
    """
    rng = np.random.RandomState(128)
    n = 10
    rng.uniform(0.05, 0.2, n)                    # consumed as the original does
    a = rng.uniform(0.001, 0.01, (n, n))
    sigma = (a + a.T) / 2
    np.fill_diagonal(sigma, rng.uniform(0.005, 0.02, n))

    assert not datamod.is_psd(sigma)
    assert np.linalg.eigvalsh(sigma).min() < 0


def test_nearest_psd_repairs_it_and_leaves_valid_matrices_alone():
    rng = np.random.RandomState(128)
    n = 10
    rng.uniform(0.05, 0.2, n)
    a = rng.uniform(0.001, 0.01, (n, n))
    sigma = (a + a.T) / 2
    np.fill_diagonal(sigma, rng.uniform(0.005, 0.02, n))

    repaired = datamod.nearest_psd(sigma)
    assert datamod.is_psd(repaired)
    # Already-valid matrices come back unchanged.
    valid = np.diag([0.01, 0.02, 0.03])
    assert datamod.nearest_psd(valid) == pytest.approx(valid)


def test_synthetic_moments_are_psd_by_construction():
    for seed in (1, 2, 128):
        _, sigma = datamod.synthetic_moments(10, seed=seed)
        assert datamod.is_psd(sigma)
