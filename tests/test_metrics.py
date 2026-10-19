"""Portfolio statistics and market data."""

import numpy as np
import pytest

from qportfolio import data, metrics


def test_sharpe_divides_by_volatility():
    w = np.array([0.5, 0.5])
    mu = np.array([0.10, 0.10])
    sigma = np.diag([0.08, 0.08])
    vol = metrics.portfolio_volatility(w, sigma)
    assert vol == pytest.approx(np.sqrt(metrics.portfolio_variance(w, sigma)))
    assert metrics.sharpe_ratio(w, mu, sigma, 0.03) == pytest.approx(0.07 / vol)


def test_annualization_is_consistent():
    daily_mu = np.array([0.0004, 0.0006])
    daily_cov = np.array([[1e-4, 2e-5], [2e-5, 1.5e-4]])
    w = np.array([0.5, 0.5])
    assert metrics.annualize_returns(daily_mu) == pytest.approx(daily_mu * 252)
    assert metrics.portfolio_volatility(w, metrics.annualize_covariance(daily_cov)) == \
        pytest.approx(metrics.portfolio_volatility(w, daily_cov) * np.sqrt(252))


def test_zero_volatility_gives_nan():
    assert np.isnan(metrics.sharpe_ratio(np.zeros(3), np.full(3, 0.1), np.eye(3) * 0.01))


def test_nearest_psd():
    rng = np.random.default_rng(128)
    a = rng.uniform(0.001, 0.01, (10, 10))
    s = (a + a.T) / 2
    np.fill_diagonal(s, rng.uniform(0.005, 0.02, 10))
    assert not data.is_psd(s)
    assert data.is_psd(data.nearest_psd(s))
    valid = np.diag([0.01, 0.02])
    assert data.nearest_psd(valid) == pytest.approx(valid)


def test_synthetic_market_is_deterministic_and_well_formed():
    a, b = data.synthetic_market(10, seed=4), data.synthetic_market(10, seed=4)
    assert np.array_equal(a.returns, b.returns)
    assert data.is_psd(a.sigma)
    assert a.benchmark.sum() == pytest.approx(1.0)
    assert a.coskewness.shape == (10, 10, 10)
    assert len(a.sector_groups()) == 5
    sub = a.subset([0, 2, 4])
    assert sub.n == 3 and sub.benchmark.sum() == pytest.approx(1.0)


def test_lottery_names_are_right_skewed():
    m = data.synthetic_market(12, seed=7)
    d = m.returns - m.returns.mean(0)
    skew = (d ** 3).mean(0) / d.std(0) ** 3
    assert skew.max() > 0.5
