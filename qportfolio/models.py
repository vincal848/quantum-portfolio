"""Portfolio models, each compiled to a binary polynomial `Problem`.

| model         | qubits     | degree | constraint(s)                 | XY mixer |
|---------------|------------|--------|-------------------------------|----------|
| selection     | n          | 2      | hold exactly k                | yes      |
| allocation    | n * bits   | 2      | integer units sum to budget   | no       |
| tracking      | n          | 2      | hold exactly k                | yes      |
| long_short    | n          | 2      | dollar neutral (n/2 long)     | yes      |
| diversified   | n          | 2      | k held, no correlated pair    | yes      |
| rebalance     | n          | 3      | k held, <= 2 per sector       | yes      |
| skew_aware    | n          | 3      | hold exactly k                | yes      |

Weights for the fixed-cardinality models are equal across holdings, so the cost is
a real portfolio statistic (variance of the equal-weight portfolio, not of the raw
0/1 vector) and objective values are comparable across k.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .data import Market
from .problem import (Constraint, Polynomial, Problem, at_most_two_penalty,
                      equality_penalty, pair_penalty, safe_penalty_weight)


def _cardinality(n, k, cost, scale):
    return Constraint("cardinality", equality_penalty(np.ones(n), k, n),
                      safe_penalty_weight(cost, scale))


def _equal_weight_fn(k):
    return lambda x: np.asarray(x, float) / k


def mean_variance_cost(mu, sigma, lam, k) -> Polynomial:
    """-mu.w + lam w'Sigma w with w = x / k."""
    n = len(mu)
    return Polynomial(n, c=-np.asarray(mu) / k, Q=lam * np.asarray(sigma) / k ** 2)


def selection(market: Market, k: int = 3, lam: float = 1.0,
              penalty_scale: float = 1.0) -> Problem:
    """Pick k assets, equal weight, maximize return minus lam * variance."""
    n = market.n
    cost = mean_variance_cost(market.mu, market.sigma, lam, k)
    return Problem(
        "selection", cost, [_cardinality(n, k, cost, penalty_scale)], _equal_weight_fn(k),
        market.names, "Mean-variance selection of %d of %d assets" % (k, n),
        hamming_weight=k, objective_label="-return + %.2g * variance" % lam,
        meta={"k": k, "lam": lam})


def allocation(market: Market, lam: float = 1.0, bits: int = 2, budget: int = 4,
               penalty_scale: float = 1.0) -> Problem:
    """Unequal weights. Each asset gets `bits` qubits encoding 0..2^bits-1 units.

    Weight of asset i is units_i / budget, and the units must add up to the budget,
    so with bits=2, budget=4 the portfolio is built from quarters, any single asset
    holding 0 to 75%. Qubit count is n * bits, which is why this model is run on a
    smaller universe than the others.
    """
    n = market.n
    nq = n * bits
    A = np.zeros((n, nq))
    for i in range(n):
        for j in range(bits):
            A[i, i * bits + j] = 2 ** j / budget
    mu, sigma = market.mu, market.sigma
    cost = Polynomial(nq, c=-(mu @ A), Q=lam * A.T @ sigma @ A)
    units = A.sum(axis=0) * budget
    con = Constraint("budget", equality_penalty(units, budget, nq),
                     safe_penalty_weight(cost, penalty_scale))
    labels = ["%s.b%d" % (market.names[i], j) for i in range(n) for j in range(bits)]
    return Problem(
        "allocation", cost, [con], lambda x: A @ np.asarray(x, float), market.names,
        "Integer allocation in 1/%d units, %d bits per asset" % (budget, bits),
        variable_labels=labels, objective_label="-return + %.2g * variance" % lam,
        meta={"bits": bits, "budget": budget, "lam": lam, "A": A})


def tracking(market: Market, k: int = 4, penalty_scale: float = 1.0) -> Problem:
    """Replicate the benchmark with k equally weighted names.

    Minimizes tracking variance (w - b)' Sigma (w - b), the variance of the
    difference between portfolio and benchmark returns.
    """
    n = market.n
    sigma, b = market.sigma, market.benchmark
    cost = Polynomial(n, offset=float(b @ sigma @ b), c=-2 * (sigma @ b) / k,
                      Q=sigma / k ** 2)
    return Problem(
        "tracking", cost, [_cardinality(n, k, cost, penalty_scale)], _equal_weight_fn(k),
        market.names, "Track the cap-weighted benchmark with %d names" % k,
        hamming_weight=k, objective_label="tracking variance", meta={"k": k})


def long_short(market: Market, lam: float = 1.0, penalty_scale: float = 1.0) -> Problem:
    """Dollar-neutral long/short: every asset is long or short at 1/n.

    Spin s_i = +1 long, -1 short, weight s_i / n, and s = 2x - 1. The variance term
    s' Sigma s equals a constant minus four times the covariance-weighted cut between
    the long and short books, so minimizing risk is a weighted MaxCut on the
    covariance graph: put assets that move together on opposite sides. Dollar
    neutrality (sum s = 0) makes it a balanced cut, i.e. exactly n/2 qubits set,
    which the XY mixer preserves. This is the graph picture the project started
    from, made exact.
    """
    n = market.n
    if n % 2:
        raise ValueError("long_short needs an even number of assets")
    mu, sigma = market.mu, market.sigma
    ones = np.ones(n)
    cost = Polynomial(n, offset=float(mu.sum() / n + lam * ones @ sigma @ ones / n ** 2),
                      c=-2 * mu / n - 4 * lam * (sigma @ ones) / n ** 2,
                      Q=4 * lam * sigma / n ** 2)
    return Problem(
        "long_short", cost, [_cardinality(n, n // 2, cost, penalty_scale)],
        lambda x: (2 * np.asarray(x, float) - 1) / n, market.names,
        "Dollar-neutral long/short across %d assets (balanced MaxCut)" % n,
        hamming_weight=n // 2, objective_label="-return + %.2g * variance" % lam,
        meta={"lam": lam})


def correlation_edges(market: Market, threshold: float):
    corr = market.correlation
    return [(i, j) for i, j in combinations(range(market.n), 2) if corr[i, j] > threshold]


def diversified(market: Market, k: int = 3, lam: float = 1.0, threshold: float = 0.35,
                penalty_scale: float = 1.0) -> Problem:
    """Selection on the correlation graph: no two held assets may correlate above
    `threshold`. Nodes are assets, edges join correlated pairs, and the holdings
    must form an independent set of size k - a weighted maximum independent set
    with a cardinality constraint."""
    n = market.n
    cost = mean_variance_cost(market.mu, market.sigma, lam, k)
    edges = correlation_edges(market, threshold)
    cons = [_cardinality(n, k, cost, penalty_scale),
            Constraint("independent_set", pair_penalty(edges, n),
                       safe_penalty_weight(cost, penalty_scale))]
    return Problem(
        "diversified", cost, cons, _equal_weight_fn(k), market.names,
        "Selection of %d with no pair correlated above %.2f (%d edges)"
        % (k, threshold, len(edges)),
        hamming_weight=k, objective_label="-return + %.2g * variance" % lam,
        meta={"k": k, "lam": lam, "threshold": threshold, "edges": edges})


def rebalance(market: Market, current, k: int = 4, lam: float = 1.0,
              cost_per_trade: float = 0.01, sector_cap: int = 2,
              penalty_scale: float = 1.0) -> Problem:
    """Move from current holdings to a new k-asset portfolio, paying for turnover,
    with at most `sector_cap` names per sector.

    Turnover is linear in binary variables: |x_i - x0_i| = x_i (1 - 2 x0_i) + x0_i,
    and each changed name moves 1/k of capital. A cap of 2 per sector is a cubic
    penalty (one term per triple in a sector) and needs no slack qubits.
    """
    n = market.n
    x0 = np.asarray(current, float)
    cost = mean_variance_cost(market.mu, market.sigma, lam, k)
    cost = cost + Polynomial(n, offset=cost_per_trade * x0.sum() / k,
                             c=cost_per_trade * (1 - 2 * x0) / k)
    groups = market.sector_groups()
    if sector_cap == 2:
        cap = at_most_two_penalty(groups, n)
    elif sector_cap == 1:
        cap = pair_penalty([e for g in groups for e in combinations(g, 2)], n)
    else:
        raise ValueError("sector_cap must be 1 or 2")
    cons = [_cardinality(n, k, cost, penalty_scale),
            Constraint("sector_cap", cap, safe_penalty_weight(cost, penalty_scale))]
    held = [market.names[i] for i in np.flatnonzero(x0)]
    return Problem(
        "rebalance", cost, cons, _equal_weight_fn(k), market.names,
        "Rebalance from %s with %.0f bp per trade, <= %d per sector"
        % ("/".join(held), cost_per_trade * 1e4, sector_cap),
        hamming_weight=k, objective_label="-return + variance + turnover cost",
        meta={"k": k, "lam": lam, "current": x0.astype(int).tolist(),
              "cost_per_trade": cost_per_trade, "sector_cap": sector_cap})


def skew_aware(market: Market, k: int = 3, lam: float = 1.0, kappa: float = 1.0,
               penalty_scale: float = 1.0) -> Problem:
    """Mean-variance-skewness: -mu.w + lam w'Sigma w - kappa * m3(w) / s^3.

    m3(w) = sum_ijk M_ijk w_i w_j w_k is the portfolio's third central moment of
    daily returns and s the average daily single-asset volatility, so m3(w) / s^3 is
    on the scale of a daily skewness and kappa is unitless. The objective prefers right-skewed portfolios
    and penalizes crash exposure. With
    w = x / k this is a cubic in x: three-body Z terms in the Hamiltonian, which the
    phase layer applies as easily as two-body ones, but which no QUBO solver accepts
    without auxiliary variables.
    """
    n = market.n
    cost = mean_variance_cost(market.mu, market.sigma, lam, k)
    s3 = float(np.mean(market.returns.std(axis=0))) ** 3
    cost = cost + Polynomial(n, T=-kappa * market.coskewness / (k ** 3 * s3))
    return Problem(
        "skew_aware", cost, [_cardinality(n, k, cost, penalty_scale)], _equal_weight_fn(k),
        market.names, "Mean-variance-skewness selection of %d (kappa=%g)" % (k, kappa),
        hamming_weight=k, objective_label="-return + variance - %g * skew" % kappa,
        meta={"k": k, "lam": lam, "kappa": kappa})


def continuous_markowitz(market: Market, lam: float = 1.0, max_weight: float = 1.0):
    """Long-only continuous optimum, for measuring discretization loss in allocation."""
    from scipy.optimize import minimize

    mu, sigma = market.mu, market.sigma
    n = market.n
    out = minimize(lambda w: -mu @ w + lam * w @ sigma @ w, np.full(n, 1 / n),
                   jac=lambda w: -mu + 2 * lam * sigma @ w,
                   bounds=[(0, max_weight)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   method="SLSQP")
    return out.x, float(out.fun)
