"""Market data. Daily returns go in; the moments the models need come out.

synthetic_market builds a seeded factor-model market: five sectors, a market factor
that occasionally crashes, and a few right-skewed "lottery" names. It needs no
network, so every number in the README reproduces anywhere.

real_market downloads adjusted closes from Yahoo Finance and caches them to CSV.

Besides mean and covariance, the models use sectors (rebalance), a benchmark
(tracking) and third co-moments (skew_aware), so a Market carries all of them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .metrics import TRADING_DAYS

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "TSLA", "JPM", "GS",
                   "XOM", "CVX", "JNJ", "PFE", "PG", "KO"]
TICKER_SECTORS = {"AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "GOOG": "Tech",
                  "AMZN": "Consumer", "TSLA": "Consumer", "JPM": "Financials",
                  "GS": "Financials", "XOM": "Energy", "CVX": "Energy",
                  "JNJ": "Health", "PFE": "Health", "PG": "Staples", "KO": "Staples"}
START, END = "2020-01-01", "2023-01-01"
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SECTORS = ["Tech", "Energy", "Health", "Financials", "Staples"]


@dataclass
class Market:
    names: list[str]
    sectors: list[str]
    returns: np.ndarray          # daily, T x n
    benchmark: np.ndarray        # weights summing to 1, used by index tracking

    @property
    def n(self) -> int:
        return len(self.names)

    @property
    def mu(self) -> np.ndarray:
        return self.returns.mean(axis=0) * TRADING_DAYS

    @property
    def sigma(self) -> np.ndarray:
        return nearest_psd(np.cov(self.returns, rowvar=False) * TRADING_DAYS)

    @property
    def correlation(self) -> np.ndarray:
        s = self.sigma
        d = np.sqrt(np.diag(s))
        return s / np.outer(d, d)

    @property
    def coskewness(self) -> np.ndarray:
        """Third central co-moment tensor of daily returns.

        Kept daily on purpose. For independent days, annualizing shrinks skewness by
        sqrt(252), which would wash out the crashes and jumps skew_aware looks for.
        """
        d = self.returns - self.returns.mean(axis=0)
        return np.einsum("ti,tj,tk->ijk", d, d, d) / len(d)

    def sector_groups(self) -> list[list[int]]:
        groups = {}
        for i, s in enumerate(self.sectors):
            groups.setdefault(s, []).append(i)
        return list(groups.values())

    def subset(self, idx) -> "Market":
        idx = list(idx)
        b = self.benchmark[idx]
        return Market([self.names[i] for i in idx], [self.sectors[i] for i in idx],
                      self.returns[:, idx], b / b.sum())


def synthetic_market(n: int = 12, seed: int = 7, days: int = 3 * TRADING_DAYS) -> Market:
    rng = np.random.default_rng(seed)
    sectors = [SECTORS[i % len(SECTORS)] for i in range(n)]
    sector_id = np.array([SECTORS.index(s) for s in sectors])

    # Market factor: calm most days, a crash about 3% of the time.
    crash = rng.random(days) < 0.03
    market = np.where(crash, rng.normal(-0.025, 0.02, days), rng.normal(0.0009, 0.009, days))
    sector_f = rng.normal(0, 0.007, (days, len(SECTORS)))

    beta = rng.uniform(0.6, 1.4, n)
    load = rng.uniform(0.5, 1.5, n)
    idio = rng.uniform(0.008, 0.02, n)
    alpha = rng.uniform(-0.0003, 0.0006, n)
    eps = rng.normal(0, 1, (days, n)) * idio

    # Lottery names: a rare 4-10% jump up, with the drift lowered to pay for it.
    lottery = rng.random(n) < 0.3
    jumps = (rng.random((days, n)) < 0.015) * rng.uniform(0.04, 0.10, (days, n))
    eps = eps + jumps * lottery - (0.015 * 0.07) * lottery

    returns = alpha + market[:, None] * beta + sector_f[:, sector_id] * load + eps
    caps = rng.lognormal(0, 1, n)
    names = ["%s%d" % (s[:3].upper(), i) for i, s in enumerate(sectors)]
    return Market(names, sectors, returns, caps / caps.sum())


def real_market(tickers=None, start=START, end=END, use_cache=True) -> Market:
    """Real daily returns. The benchmark is equal-weight across the universe."""
    tickers = list(tickers or DEFAULT_TICKERS)
    prices = load_prices(tickers, start, end, use_cache)
    returns = prices.pct_change().dropna().values
    n = len(tickers)
    return Market(tickers, [TICKER_SECTORS.get(t, "Other") for t in tickers], returns,
                  np.full(n, 1 / n))


def load_prices(tickers, start=START, end=END, use_cache=True):
    import pandas as pd

    cache = os.path.join(CACHE_DIR, "prices_%s_%s.csv" % (start, end))
    if use_cache and os.path.exists(cache):
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        if all(t in df.columns for t in tickers):
            return df[tickers].dropna()

    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close = close[tickers].dropna()
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        close.to_csv(cache)
    return close


def nearest_psd(matrix):
    """Clip negative eigenvalues so the matrix is a valid covariance."""
    matrix = np.asarray(matrix, dtype=float)
    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if np.all(eigenvalues >= 0):
        return symmetric
    clipped = np.clip(eigenvalues, 0.0, None)
    return eigenvectors @ np.diag(clipped) @ eigenvectors.T


def is_psd(matrix, tol=1e-10):
    m = np.asarray(matrix, float)
    return bool(np.all(np.linalg.eigvalsh((m + m.T) / 2) >= -tol))
