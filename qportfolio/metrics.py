"""Portfolio statistics on an annual basis."""

import numpy as np

TRADING_DAYS = 252


def portfolio_return(weights, mu):
    return float(np.asarray(weights, dtype=float) @ np.asarray(mu, dtype=float))


def portfolio_volatility(weights, sigma):
    w = np.asarray(weights, dtype=float)
    return float(np.sqrt(max(w @ np.asarray(sigma, dtype=float) @ w, 0.0)))


def sharpe_ratio(weights, mu, sigma, risk_free_rate=0.03):
    """(return - risk-free) / volatility; nan when nothing is held."""
    vol = portfolio_volatility(weights, sigma)
    if vol == 0:
        return float("nan")
    return (portfolio_return(weights, mu) - risk_free_rate) / vol
