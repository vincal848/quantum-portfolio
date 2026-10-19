"""Each model's polynomial must equal the portfolio statistic it claims to be."""

from itertools import product
from math import comb

import numpy as np
import pytest

from qportfolio import data, models, solvers
from qportfolio.problem import bit_table


@pytest.fixture(scope="module")
def market():
    return data.synthetic_market(8, seed=3)


def feasible_bits(problem):
    return bit_table(problem.n_qubits, problem.feasible_states())


def test_selection_is_equal_weight_mean_variance(market):
    p = models.selection(market, k=3, lam=2.0)
    for x in feasible_bits(p)[:20]:
        w = x / 3
        assert p.objective(x) == pytest.approx(-market.mu @ w + 2.0 * w @ market.sigma @ w)
    assert len(p.feasible_states()) == comb(8, 3)


def test_allocation_weights_sum_to_one_and_cover_every_split(market):
    m = market.subset(range(4))
    p = models.allocation(m, bits=2, budget=4)
    X = feasible_bits(p)
    for x in X:
        w = p.weights(x)
        assert w.sum() == pytest.approx(1.0)
        assert p.objective(x) == pytest.approx(-m.mu @ w + w @ m.sigma @ w)
    # Integer vectors of 4 entries in 0..3 summing to 4.
    expected = sum(1 for u in product(range(4), repeat=4) if sum(u) == 4)
    assert len(X) == expected


def test_tracking_is_tracking_variance(market):
    p = models.tracking(market, k=4)
    b = market.benchmark
    for x in feasible_bits(p)[:20]:
        d = x / 4 - b
        assert p.objective(x) == pytest.approx(d @ market.sigma @ d)
        assert p.objective(x) >= 0


def test_long_short_is_dollar_neutral_and_is_a_maxcut(market):
    p = models.long_short(market, lam=1.0)
    n = market.n
    sigma = market.sigma
    for x in feasible_bits(p)[:30]:
        w = p.weights(x)
        assert w.sum() == pytest.approx(0.0)
        assert np.abs(w).sum() == pytest.approx(1.0)
        assert p.objective(x) == pytest.approx(-market.mu @ w + w @ sigma @ w)
        # Variance = (total - 4 * cut weight) / n^2, the MaxCut identity.
        s = 2 * x - 1
        cut = sum(sigma[i, j] for i in range(n) for j in range(n) if s[i] != s[j]) / 2
        assert s @ sigma @ s == pytest.approx(sigma.sum() - 4 * cut)


def test_long_short_needs_an_even_universe(market):
    with pytest.raises(ValueError):
        models.long_short(market.subset(range(7)))


def test_diversified_forbids_correlated_pairs(market):
    p = models.diversified(market, k=3, threshold=0.3)
    edges = p.meta["edges"]
    assert edges, "threshold chosen so the constraint binds"
    for x in feasible_bits(p):
        assert not any(x[i] and x[j] for i, j in edges)


def test_rebalance_turnover_and_sector_cap(market):
    current = np.zeros(market.n)
    current[:4] = 1
    p = models.rebalance(market, current, k=4, cost_per_trade=0.02, sector_cap=2)
    base = models.selection(market, k=4)
    groups = market.sector_groups()
    for x in feasible_bits(p):
        trades = np.abs(x - current).sum()
        assert p.objective(x) == pytest.approx(base.objective(x) + 0.02 * trades / 4)
        assert all(x[g].sum() <= 2 for g in groups)


def test_skew_term_is_the_portfolio_third_moment(market):
    kappa = 1.5
    p = models.skew_aware(market, k=3, kappa=kappa)
    base = models.selection(market, k=3)
    R = market.returns
    s3 = np.mean(R.std(axis=0)) ** 3
    for x in feasible_bits(p)[:20]:
        r = R @ (x / 3)
        m3 = np.mean((r - r.mean()) ** 3)
        assert p.objective(x) == pytest.approx(base.objective(x) - kappa * m3 / s3)


@pytest.mark.parametrize("build", [
    lambda m: models.selection(m, k=3),
    lambda m: models.allocation(m.subset(range(4))),
    lambda m: models.tracking(m, k=3),
    lambda m: models.long_short(m),
    lambda m: models.diversified(m, k=3, threshold=0.3),
    lambda m: models.rebalance(m, np.eye(m.n)[:3].sum(0), k=3),
    lambda m: models.skew_aware(m, k=3),
])
def test_penalized_ground_state_is_the_feasible_optimum(market, build):
    """The penalty weight must be large enough that no infeasible state wins."""
    p = build(market)
    _, ref = solvers.exact(p)
    energies = p.hamiltonian().evaluate_states(np.arange(2 ** p.n_qubits))
    ground = int(np.argmin(energies))
    assert ground in set(ref.optimal_states)
    assert energies[ground] == pytest.approx(ref.best)


def test_continuous_markowitz_is_a_lower_bound_on_allocation(market):
    m = market.subset(range(4))
    _, ref = solvers.exact(models.allocation(m, bits=2, budget=4))
    w, value = models.continuous_markowitz(m, max_weight=0.75)
    assert w.sum() == pytest.approx(1.0)
    assert value <= ref.best + 1e-9
