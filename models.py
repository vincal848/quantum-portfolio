"""The market and the seven portfolio models.

A model is a cost and a penalty, both evaluated on a matrix X of bitstrings (one
row per basis state, one column per qubit). Each penalty is a polynomial in the
bits that is zero exactly when the portfolio is valid, so cost + P * penalty is a
diagonal Ising Hamiltonian. Models that fix the number of holdings set k; the XY
mixer keeps that count by itself, so their (sum(x) - k)^2 term can be dropped.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Callable

import numpy as np

DAYS = 252
SECTORS = ["Tech", "Energy", "Health", "Financials", "Staples"]
TICKERS = {"AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "GOOG": "Tech",
           "XOM": "Energy", "CVX": "Energy", "JNJ": "Health", "PFE": "Health",
           "JPM": "Financials", "GS": "Financials", "PG": "Staples", "KO": "Staples"}


@dataclass
class Market:
    names: list
    sectors: list
    returns: np.ndarray      # daily, days x assets
    benchmark: np.ndarray    # weights summing to 1

    @property
    def n(self):
        return len(self.names)

    @property
    def mu(self):
        return self.returns.mean(0) * DAYS

    @property
    def sigma(self):
        return np.cov(self.returns, rowvar=False) * DAYS

    def first(self, n):
        b = self.benchmark[:n]
        return Market(self.names[:n], self.sectors[:n], self.returns[:, :n], b / b.sum())


def synthetic_market(n=12, seed=7, days=3 * DAYS):
    """Factor model: five sectors, a market factor that crashes ~3% of days, and a
    few "lottery" names with rare 4-10% jumps up."""
    rng = np.random.default_rng(seed)
    sectors = [SECTORS[i % len(SECTORS)] for i in range(n)]
    sector_id = np.array([SECTORS.index(s) for s in sectors])

    crash = rng.random(days) < 0.03
    market = np.where(crash, rng.normal(-0.025, 0.02, days), rng.normal(0.0009, 0.009, days))
    sector_f = rng.normal(0, 0.007, (days, len(SECTORS)))
    beta = rng.uniform(0.6, 1.4, n)
    load = rng.uniform(0.5, 1.5, n)
    idio = rng.uniform(0.008, 0.02, n)
    alpha = rng.uniform(-0.0003, 0.0006, n)
    eps = rng.normal(0, 1, (days, n)) * idio
    lottery = rng.random(n) < 0.3
    jumps = (rng.random((days, n)) < 0.015) * rng.uniform(0.04, 0.10, (days, n))
    eps = eps + jumps * lottery - (0.015 * 0.07) * lottery   # drift pays for the jumps

    returns = alpha + market[:, None] * beta + sector_f[:, sector_id] * load + eps
    caps = rng.lognormal(0, 1, n)
    names = ["%s%d" % (s[:3].upper(), i) for i, s in enumerate(sectors)]
    return Market(names, sectors, returns, caps / caps.sum())


def real_market(start="2020-01-01", end="2023-01-01"):
    """Daily returns for TICKERS from Yahoo Finance. Equal-weight benchmark."""
    import yfinance as yf

    names = list(TICKERS)
    prices = yf.download(names, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    returns = prices[names].dropna().pct_change().dropna().values
    return Market(names, list(TICKERS.values()), returns, np.full(len(names), 1 / len(names)))


@dataclass
class Model:
    name: str
    n: int                              # qubits
    cost: Callable                      # X -> objective for each row
    weights: Callable                   # one bitstring -> portfolio weights
    labels: list
    k: int = None                       # exact number of ones, if the model fixes it
    other_penalty: Callable = None      # any further constraint, zero when met

    def penalty(self, X, count_enforced=False):
        p = np.zeros(len(X))
        if self.k is not None and not count_enforced:
            p += (X.sum(1) - self.k) ** 2
        if self.other_penalty is not None:
            p += self.other_penalty(X)
        return p

    def describe(self, x):
        w = self.weights(np.asarray(x, float))
        return ", ".join("%s %+.0f%%" % (self.labels[i], 100 * w[i])
                         for i in np.argsort(-np.abs(w)) if abs(w[i]) > 1e-12)


def quad(X, M):
    """x' M x for every row of X."""
    return np.einsum("si,ij,sj->s", X, M, X)


def mean_variance(m, k, lam):
    mu, sigma = m.mu, m.sigma
    return lambda X: -(X @ mu) / k + lam * quad(X, sigma) / k ** 2


def selection(m, k=3, lam=1.0):
    """k names at equal weight, maximizing return minus lam * variance."""
    return Model("selection", m.n, mean_variance(m, k, lam), lambda x: x / k, m.names, k=k)


def allocation(m, lam=1.0):
    """Real weights in quarters: two bits per asset count 0-3 units, units sum to 4."""
    mu, sigma = m.mu, m.sigma
    w = lambda X: (X[:, 0::2] + 2 * X[:, 1::2]) / 4
    return Model("allocation", 2 * m.n,
                 lambda X: -(w(X) @ mu) + lam * quad(w(X), sigma),
                 lambda x: w(x[None])[0], m.names,
                 other_penalty=lambda X: (4 * w(X).sum(1) - 4) ** 2)


def tracking(m, k=4):
    """k names at equal weight, minimizing tracking variance against the benchmark."""
    sigma, b = m.sigma, m.benchmark
    return Model("tracking", m.n, lambda X: quad(X / k - b, sigma), lambda x: x / k,
                 m.names, k=k)


def long_short(m, lam=1.0):
    """Dollar-neutral: each name long or short at 1/n, half on each side.

    With s = 2x - 1, the risk s' Sigma s is a constant minus 4 x the covariance
    between the two books, so this is a balanced MaxCut on the covariance graph.
    """
    n, mu, sigma = m.n, m.mu, m.sigma
    w = lambda X: (2 * X - 1) / n
    return Model("long_short", n, lambda X: -(w(X) @ mu) + lam * quad(w(X), sigma),
                 w, m.names, k=n // 2)


def correlated_pairs(m, threshold):
    sd = np.sqrt(np.diag(m.sigma))
    corr = m.sigma / np.outer(sd, sd)
    return [(i, j) for i, j in combinations(range(m.n), 2) if corr[i, j] > threshold]


def diversified(m, k=3, lam=1.0, threshold=0.35):
    """Selection where no two holdings correlate above threshold: a maximum
    independent set on the correlation graph, one x_i x_j penalty per edge."""
    A = np.zeros((m.n, m.n))
    for i, j in correlated_pairs(m, threshold):
        A[i, j] = 1
    return Model("diversified", m.n, mean_variance(m, k, lam), lambda x: x / k, m.names,
                 k=k, other_penalty=lambda X: quad(X, A))


def rebalance(m, current, k=4, lam=1.0, cost_per_trade=0.01):
    """Move from `current` to k new names, paying per name traded, at most 2 per sector.

    "At most 2" is one cubic term per triple in a sector; for 0/1 bits that sum is
    C(count, 3), so no slack qubits are needed.
    """
    x0 = np.asarray(current, float)
    base = mean_variance(m, k, lam)
    groups = [[i for i, s in enumerate(m.sectors) if s == sec] for sec in dict.fromkeys(m.sectors)]

    def triples(X):
        c = np.stack([X[:, g].sum(1) for g in groups], 1)
        return (c * (c - 1) * (c - 2) / 6).sum(1)

    return Model("rebalance", m.n, lambda X: base(X) + cost_per_trade * np.abs(X - x0).sum(1) / k,
                 lambda x: x / k, m.names, k=k, other_penalty=triples)


def skew_aware(m, k=3, lam=1.0, kappa=1.0):
    """Mean-variance minus kappa x daily skewness of the portfolio.

    The third moment is cubic in x (three-body Z terms). It's scaled by the average
    daily volatility cubed so kappa has no units.
    """
    base = mean_variance(m, k, lam)
    R = m.returns
    s3 = R.std(0).mean() ** 3

    def cost(X):
        r = R @ (X / k).T                       # daily returns of every candidate
        m3 = ((r - r.mean(0)) ** 3).mean(0)
        return base(X) - kappa * m3 / s3

    return Model("skew_aware", m.n, cost, lambda x: x / k, m.names, k=k)


def all_models(m):
    current = np.zeros(m.n)
    current[:4] = 1
    return [selection(m), allocation(m.first(5)), tracking(m), long_short(m.first(10)),
            diversified(m), rebalance(m, current), skew_aware(m)]
