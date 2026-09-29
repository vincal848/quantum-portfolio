"""Portfolio statistics, computed on a consistent time basis.

Two things the original got wrong, both of which made the Sharpe ratio meaningless:

1. It divided the excess return by the portfolio VARIANCE. The Sharpe ratio divides
   by the standard deviation. Variance is a small number squared, so dividing by it
   inflates the ratio by roughly 1/sigma -- on daily data that is a factor of tens.

2. It compared a DAILY mean return against an ANNUAL risk-free rate. The returns
   came from `data.pct_change().mean()`, which is a daily figure, and was then
   measured against 0.03. Those are not the same units, so the excess return was
   wrong before the division even happened.

Everything here is annualized before any comparison, and the trading-day count is
explicit rather than implied.
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
    """How close a solver got to the true optimum, as a fraction.

    1.0 means it found the optimum. The objective can be negative, so this is a gap
    relative to the magnitude of the optimum rather than a plain ratio, which would
    flip sign and become unreadable.
    """
    denom = abs(optimal_objective)
    if denom == 0:
        return float("nan")
    return 1.0 - abs(solver_objective - optimal_objective) / denom
