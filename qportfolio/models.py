"""The seven portfolio models. Each one returns a Problem.

model        qubits     order   constraints                      XY mixer
selection    n          2       hold exactly k                   yes
allocation   n * bits   2       units add up to the budget       no
tracking     n          2       hold exactly k                   yes
long_short   n          2       n/2 long, n/2 short              yes
diversified  n          2       k held, no correlated pair       yes
rebalance    n          3       k held, at most 2 per sector     yes
skew_aware   n          3       hold exactly k                   yes

The fixed-cardinality models weight holdings equally (w = x / k), so the cost is
the actual return and variance of that portfolio rather than of the raw 0/1 vector.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.optimize import minimize

from .data import Market
from .problem import (Constraint, Polynomial, Problem, at_most_one_penalty,
                      at_most_two_penalty, equality_penalty, pair_penalty,
                      safe_penalty_weight)


def _hold_exactly(n, k, cost, scale):
    penalty = equality_penalty(np.ones(n), k, n)
    return Constraint("cardinality", penalty, safe_penalty_weight(cost, scale))


def _equal_weights(k):
    return lambda x: np.asarray(x, dtype=float) / k


def mean_variance_cost(mu, sigma, lam, k) -> Polynomial:
    """-mu.w + lam w'Sigma w, with w = x / k."""
    return Polynomial(len(mu), c=-np.asarray(mu) / k, Q=lam * np.asarray(sigma) / k ** 2)


def selection(market: Market, k=3, lam=1.0, penalty_scale=1.0) -> Problem:
    """Pick k names at equal weight to maximize return minus lam * variance."""
    n = market.n
    cost = mean_variance_cost(market.mu, market.sigma, lam, k)
    return Problem(
        "selection", cost, [_hold_exactly(n, k, cost, penalty_scale)], _equal_weights(k),
        market.names, "Mean-variance selection of %d of %d assets" % (k, n),
        hamming_weight=k, objective_label="-return + %.2g * variance" % lam,
        meta={"k": k, "lam": lam})


def allocation(market: Market, lam=1.0, bits=2, budget=4, penalty_scale=1.0) -> Problem:
    """Real (unequal) weights, built from whole units.

    Each asset gets `bits` qubits that count 0 to 2^bits - 1 units, and the units
    have to add up to `budget`. With the defaults that's a portfolio in quarters
    where no name holds more than 75%. It costs n * bits qubits, so it runs on a
    smaller universe than the other models.
    """
    n, nq = market.n, market.n * bits
    # A maps the bits to weights: w = A @ x.
    A = np.zeros((n, nq))
    for i in range(n):
        for j in range(bits):
            A[i, i * bits + j] = 2 ** j / budget

    cost = Polynomial(nq, c=-(market.mu @ A), Q=lam * A.T @ market.sigma @ A)
    units_per_bit = A.sum(axis=0) * budget
    budget_rule = Constraint("budget", equality_penalty(units_per_bit, budget, nq),
                             safe_penalty_weight(cost, penalty_scale))
    labels = ["%s.b%d" % (market.names[i], j) for i in range(n) for j in range(bits)]
    return Problem(
        "allocation", cost, [budget_rule], lambda x: A @ np.asarray(x, dtype=float),
        market.names, "Integer allocation in 1/%d units, %d bits per asset" % (budget, bits),
        variable_labels=labels, objective_label="-return + %.2g * variance" % lam,
        meta={"bits": bits, "budget": budget, "lam": lam, "A": A})


def tracking(market: Market, k=4, penalty_scale=1.0) -> Problem:
    """Copy the benchmark as closely as possible with k equally weighted names.

    The cost is tracking variance, (w - b)' Sigma (w - b): the variance of the gap
    between the portfolio's return and the benchmark's.
    """
    n = market.n
    sigma, b = market.sigma, market.benchmark
    cost = Polynomial(n, offset=float(b @ sigma @ b), c=-2 * (sigma @ b) / k, Q=sigma / k ** 2)
    return Problem(
        "tracking", cost, [_hold_exactly(n, k, cost, penalty_scale)], _equal_weights(k),
        market.names, "Track the cap-weighted benchmark with %d names" % k,
        hamming_weight=k, objective_label="tracking variance", meta={"k": k})


def long_short(market: Market, lam=1.0, penalty_scale=1.0) -> Problem:
    """Dollar-neutral long/short, every name held at 1/n on one side or the other.

    Write s_i = +1 for long and -1 for short, so w = s / n and s = 2x - 1. The risk
    term s' Sigma s is a constant minus 4 times the total covariance between the
    long book and the short book. Minimizing risk is therefore weighted MaxCut on
    the covariance graph: put names that move together on opposite sides. Dollar
    neutrality means exactly n/2 are long, a balanced cut, which the XY mixer
    keeps for free.
    """
    n = market.n
    if n % 2:
        raise ValueError("long_short needs an even number of assets")
    mu, sigma = market.mu, market.sigma
    ones = np.ones(n)
    cost = Polynomial(n,
                      offset=float(mu.sum() / n + lam * ones @ sigma @ ones / n ** 2),
                      c=-2 * mu / n - 4 * lam * (sigma @ ones) / n ** 2,
                      Q=4 * lam * sigma / n ** 2)
    return Problem(
        "long_short", cost, [_hold_exactly(n, n // 2, cost, penalty_scale)],
        lambda x: (2 * np.asarray(x, dtype=float) - 1) / n, market.names,
        "Dollar-neutral long/short across %d assets (balanced MaxCut)" % n,
        hamming_weight=n // 2, objective_label="-return + %.2g * variance" % lam,
        meta={"lam": lam})


def correlation_edges(market: Market, threshold):
    corr = market.correlation
    return [(i, j) for i, j in combinations(range(market.n), 2) if corr[i, j] > threshold]


def diversified(market: Market, k=3, lam=1.0, threshold=0.35, penalty_scale=1.0) -> Problem:
    """Selection where no two holdings may be correlated above `threshold`.

    Draw a graph with an edge between every correlated pair. The holdings then have
    to be an independent set of size k, so this is a weighted maximum independent
    set problem with a cardinality constraint.
    """
    n = market.n
    cost = mean_variance_cost(market.mu, market.sigma, lam, k)
    edges = correlation_edges(market, threshold)
    rules = [_hold_exactly(n, k, cost, penalty_scale),
             Constraint("independent_set", pair_penalty(edges, n),
                        safe_penalty_weight(cost, penalty_scale))]
    return Problem(
        "diversified", cost, rules, _equal_weights(k), market.names,
        "Selection of %d with no pair correlated above %.2f (%d edges)"
        % (k, threshold, len(edges)),
        hamming_weight=k, objective_label="-return + %.2g * variance" % lam,
        meta={"k": k, "lam": lam, "threshold": threshold, "edges": edges})


def rebalance(market: Market, current, k=4, lam=1.0, cost_per_trade=0.01, sector_cap=2,
              penalty_scale=1.0) -> Problem:
    """Move from the current holdings to a new k-name portfolio, paying per trade,
    with at most `sector_cap` names from any one sector.

    For binary x, |x_i - x0_i| = x_i (1 - 2 x0_i) + x0_i, so turnover is linear and
    costs no extra qubits. Each changed name moves 1/k of the capital.
    """
    n = market.n
    x0 = np.asarray(current, dtype=float)
    cost = mean_variance_cost(market.mu, market.sigma, lam, k)
    cost = cost + Polynomial(n, offset=cost_per_trade * x0.sum() / k,
                             c=cost_per_trade * (1 - 2 * x0) / k)

    groups = market.sector_groups()
    if sector_cap == 1:
        cap = at_most_one_penalty(groups, n)
    elif sector_cap == 2:
        cap = at_most_two_penalty(groups, n)
    else:
        raise ValueError("sector_cap must be 1 or 2")
    rules = [_hold_exactly(n, k, cost, penalty_scale),
             Constraint("sector_cap", cap, safe_penalty_weight(cost, penalty_scale))]

    held = "/".join(market.names[i] for i in np.flatnonzero(x0))
    return Problem(
        "rebalance", cost, rules, _equal_weights(k), market.names,
        "Rebalance from %s with %.0f bp per trade, <= %d per sector"
        % (held, cost_per_trade * 1e4, sector_cap),
        hamming_weight=k, objective_label="-return + variance + turnover cost",
        meta={"k": k, "lam": lam, "current": x0.astype(int).tolist(),
              "cost_per_trade": cost_per_trade, "sector_cap": sector_cap})


def skew_aware(market: Market, k=3, lam=1.0, kappa=1.0, penalty_scale=1.0) -> Problem:
    """Mean-variance-skewness: -mu.w + lam w'Sigma w - kappa m3(w) / s^3.

    m3(w) is the portfolio's third central moment of daily returns and s is the
    average daily volatility of a single name, so m3(w) / s^3 reads like a daily
    skewness and kappa has no units. Positive kappa rewards right skew (occasional
    big up days) and punishes crash exposure.

    With w = x / k the skew term is cubic in x. In the Hamiltonian that's three-body
    Z terms, which the phase layer handles directly; a QUBO solver would need extra
    variables to represent them.
    """
    n = market.n
    cost = mean_variance_cost(market.mu, market.sigma, lam, k)
    s3 = float(np.mean(market.returns.std(axis=0))) ** 3
    cost = cost + Polynomial(n, T=-kappa * market.coskewness / (k ** 3 * s3))
    return Problem(
        "skew_aware", cost, [_hold_exactly(n, k, cost, penalty_scale)], _equal_weights(k),
        market.names, "Mean-variance-skewness selection of %d (kappa=%g)" % (k, kappa),
        hamming_weight=k, objective_label="-return + variance - %g * skew" % kappa,
        meta={"k": k, "lam": lam, "kappa": kappa})


def continuous_markowitz(market: Market, lam=1.0, max_weight=1.0):
    """The long-only continuous optimum. Shows what `allocation` loses by rounding."""
    mu, sigma, n = market.mu, market.sigma, market.n
    out = minimize(lambda w: -mu @ w + lam * w @ sigma @ w, np.full(n, 1 / n),
                   jac=lambda w: -mu + 2 * lam * sigma @ w,
                   bounds=[(0, max_weight)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   method="SLSQP")
    return out.x, float(out.fun)
