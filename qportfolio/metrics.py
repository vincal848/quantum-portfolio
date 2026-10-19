"""Portfolio statistics on one consistent annual basis.

Sharpe divides excess return by volatility (not variance), and daily moments are
annualized before they meet an annual risk-free rate.
"""

import numpy as np

TRADING_DAYS = 252


def annualize_returns(daily_mean, periods=TRADING_DAYS):
    """Daily arithmetic mean return to an annual figure."""
    return np.asarray(daily_mean, dtype=float) * periods


def annualize_covariance(daily_cov, periods=TRADING_DAYS):
    """Daily covariance to annual. Scales with time, so it is a straight multiple."""
    return np.asarray(daily_cov, dtype=float) * periods


def portfolio_return(weights, mu):
    """Expected return of a weighted portfolio. Same basis as mu."""
    return float(np.asarray(weights, dtype=float) @ np.asarray(mu, dtype=float))


def portfolio_variance(weights, sigma):
    """w' Sigma w."""
    w = np.asarray(weights, dtype=float)
    return float(w @ np.asarray(sigma, dtype=float) @ w)


def portfolio_volatility(weights, sigma):
    """Standard deviation, which is what the Sharpe ratio needs."""
    return float(np.sqrt(max(portfolio_variance(weights, sigma), 0.0)))


def sharpe_ratio(weights, mu, sigma, risk_free_rate=0.03):
    """(return - risk free) / volatility, all on the same annual basis.

    Returns nan for a zero-volatility portfolio rather than dividing by zero, which
    happens when nothing is selected.
    """
    vol = portfolio_volatility(weights, sigma)
    if vol == 0:
        return float("nan")
    return (portfolio_return(weights, mu) - risk_free_rate) / vol


def summarize(weights, mu, sigma, risk_free_rate=0.03):
    """Every statistic for one portfolio, as a dict."""
    return {
        "return": portfolio_return(weights, mu),
        "variance": portfolio_variance(weights, sigma),
        "volatility": portfolio_volatility(weights, sigma),
        "sharpe": sharpe_ratio(weights, mu, sigma, risk_free_rate),
        "n_holdings": int(np.count_nonzero(weights)),
    }


def approximation_ratio(solver_objective, optimal_objective):
    """1 - |gap| / |optimum|. Solvers are scored with Reference.ratio instead, which
    normalizes by the feasible range and so works for objectives of either sign."""
    denom = abs(optimal_objective)
    if denom == 0:
        return float("nan")
    return 1.0 - abs(solver_objective - optimal_objective) / denom
