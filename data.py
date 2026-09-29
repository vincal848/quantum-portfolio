"""Price data and the moment estimates the optimizer needs.

Real prices are cached to CSV so results are reproducible and a re-run does not
depend on Yahoo being up or on it having revised its history.
"""

import os

import numpy as np
import pandas as pd

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "JPM", "XOM", "JNJ",
                   "PG", "NVDA"]
START = "2020-01-01"
END = "2023-01-01"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_prices(tickers=None, start=START, end=END, use_cache=True):
    """Adjusted closes as a DataFrame, one column per ticker."""
    tickers = list(tickers or DEFAULT_TICKERS)
    cache = os.path.join(CACHE_DIR, "prices_%s_%s.csv" % (start, end))

    if use_cache and os.path.exists(cache):
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        missing = [t for t in tickers if t not in df.columns]
        if not missing:
            return df[tickers].dropna()

    # Imported here so the rest of the module can be used, and tested, without
    # yfinance installed.
    import yfinance as yf

    # auto_adjust=True is the current yfinance default and folds dividends and
    # splits into Close, so there is no separate 'Adj Close' column any more. The
    # original indexed data['Adj Close'] and now raises KeyError on a fresh install.
    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close = close[tickers].dropna()

    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        close.to_csv(cache)
    return close


def moments(prices, annualized=True):
    """Mean return vector and covariance matrix from a price frame.

    Returns (mu, sigma, tickers). Annualized by default: the optimizer is compared
    against an annual risk-free rate downstream, and mixing a daily mean with an
    annual rate is how the original produced Sharpe ratios that could not be read.
    """
    from metrics import annualize_covariance, annualize_returns

    returns = prices.pct_change().dropna()
    mu = returns.mean().values
    sigma = returns.cov().values

    if annualized:
        mu = annualize_returns(mu)
        sigma = annualize_covariance(sigma)

    return mu, sigma, list(prices.columns)


def synthetic_moments(n_assets, seed=128, annualized=True):
    """Randomly generated moments, matching the original's example.

    Kept so the repository runs with no network access, and used by the scaling
    experiment where the point is problem size rather than any particular market.

    The covariance is built symmetric and then given a positive diagonal. It is not
    guaranteed positive semi-definite, so it is repaired below -- a random symmetric
    matrix generally is not a valid covariance, and an indefinite one can make the
    'minimum variance' portfolio unbounded.
    """
    rng = np.random.default_rng(seed)

    mu = rng.uniform(0.05, 0.20, n_assets)
    a = rng.uniform(0.001, 0.01, (n_assets, n_assets))
    sigma = (a + a.T) / 2.0
    np.fill_diagonal(sigma, rng.uniform(0.005, 0.02, n_assets))
    sigma = nearest_psd(sigma)

    if annualized:
        # The synthetic numbers are already on an annual scale, so nothing to do.
        pass
    return mu, sigma


def nearest_psd(matrix):
    """Clip negative eigenvalues to zero to make a symmetric matrix a valid covariance.

    A matrix built by symmetrizing random entries is usually indefinite. Left alone
    it lets x' Sigma x go negative, so the 'risk' term rewards risk and the
    optimizer chases it.
    """
    matrix = np.asarray(matrix, dtype=float)
    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if np.all(eigenvalues >= 0):
        return symmetric
    clipped = np.clip(eigenvalues, 0.0, None)
    return eigenvectors @ np.diag(clipped) @ eigenvectors.T


def is_psd(matrix, tol=1e-10):
    """True if the matrix is a valid covariance."""
    eigenvalues = np.linalg.eigvalsh((np.asarray(matrix, float) + np.asarray(matrix, float).T) / 2)
    return bool(np.all(eigenvalues >= -tol))
