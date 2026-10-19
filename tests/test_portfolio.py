from itertools import product

import numpy as np
import pytest
from scipy.linalg import expm

import models
import qaoa

X1 = np.array([[0, 1], [1, 0]], dtype=complex)
Y1 = np.array([[0, -1j], [1j, 0]])


def op(n, ops):
    """Dense operator with ops[q] on qubit q (qubit 0 is the last Kronecker factor)."""
    out = np.eye(1)
    for q in reversed(range(n)):
        out = np.kron(out, ops.get(q, np.eye(2)))
    return out


@pytest.fixture(scope="module")
def market():
    return models.synthetic_market(8, seed=3)


def valid_states(model):
    X = qaoa.bits(model.n, qaoa.states(model.n, model.k))
    return X[model.penalty(X) == 0]


def test_bit_order_matches_qiskit():
    assert list(qaoa.bits(4, [1])[0]) == [1, 0, 0, 0]
    assert list(qaoa.bits(4, [8])[0]) == [0, 0, 0, 1]


def test_costs_are_the_statistics_they_claim(market):
    mu, sigma, b, R = market.mu, market.sigma, market.benchmark, market.returns
    sel, trk, skew = models.selection(market), models.tracking(market, 3), models.skew_aware(market, kappa=1.5)
    M = np.einsum("ti,tj,tk->ijk", *(3 * [R - R.mean(0)])) / len(R)    # coskewness tensor
    for x in valid_states(sel)[:15]:
        w = x / 3
        assert sel.cost(x[None])[0] == pytest.approx(-mu @ w + w @ sigma @ w)
        m3 = np.einsum("ijk,i,j,k->", M, w, w, w)
        assert skew.cost(x[None])[0] == pytest.approx(sel.cost(x[None])[0] - 1.5 * m3 / R.std(0).mean() ** 3)
    for x in valid_states(trk)[:15]:
        d = x / 3 - b
        assert trk.cost(x[None])[0] == pytest.approx(d @ sigma @ d)


def test_long_short_is_a_balanced_maxcut(market):
    m = models.long_short(market)
    sigma = market.sigma
    for x in valid_states(m)[:20]:
        s = 2 * x - 1
        cut = sum(sigma[i, j] for i in range(8) for j in range(8) if s[i] != s[j]) / 2
        assert s @ sigma @ s == pytest.approx(sigma.sum() - 4 * cut)
        assert m.weights(x).sum() == pytest.approx(0)


def test_allocation_covers_every_split_into_quarters(market):
    m = models.allocation(market.first(4))
    X = valid_states(m)
    assert len(X) == sum(1 for u in product(range(4), repeat=4) if sum(u) == 4)
    assert all(m.weights(x).sum() == pytest.approx(1) for x in X)


def test_penalties_are_zero_exactly_on_valid_portfolios(market):
    current = np.eye(8)[:3].sum(0)
    reb = models.rebalance(market, current, k=3, cost_per_trade=0.02)
    div = models.diversified(market, threshold=0.3)
    edges = models.correlated_pairs(market, 0.3)
    groups = [[i for i, s in enumerate(market.sectors) if s == sec] for sec in set(market.sectors)]
    assert edges
    for x in qaoa.bits(8, qaoa.states(8, 3)):
        assert (reb.penalty(x[None])[0] == 0) == all(x[g].sum() <= 2 for g in groups)
        assert (div.penalty(x[None])[0] == 0) == (not any(x[i] and x[j] for i, j in edges))
        trades = np.abs(x - current).sum()
        assert reb.cost(x[None])[0] == pytest.approx(models.selection(market).cost(x[None])[0] + 0.02 * trades / 3)


@pytest.mark.parametrize("i", range(7))
def test_penalized_ground_state_is_the_best_valid_portfolio(market, i):
    model = models.all_models(models.synthetic_market(12, seed=3))[i]
    ref = qaoa.exact(model)
    X = qaoa.bits(model.n, qaoa.states(model.n))
    E = model.cost(X) + qaoa.penalty_weight(model) * model.penalty(X)
    assert model.penalty(X[[E.argmin()]])[0] == 0
    assert E.min() == pytest.approx(ref["best"])


def test_x_mixer_matches_the_matrix_exponential():
    rng = np.random.default_rng(0)
    psi = rng.normal(size=16) + 1j * rng.normal(size=16)
    H = sum(op(4, {q: X1}) for q in range(4))
    assert qaoa.XMixer(4).apply(psi, 0.37) == pytest.approx(expm(-0.37j * H) @ psi)


def test_xy_mixer_matches_dense_gates_and_keeps_the_count():
    mix = qaoa.XYMixer(5, 2)
    rng = np.random.default_rng(1)
    sub = rng.normal(size=10) + 1j * rng.normal(size=10)
    full = np.zeros(32, dtype=complex)
    full[mix.states] = sub
    for a, b in [(0, 1), (2, 3), (0, 4), (1, 2), (3, 4)]:
        full = expm(-0.61j * (op(5, {a: X1, b: X1}) + op(5, {a: Y1, b: Y1})) / 2) @ full
    assert mix.apply(sub, 0.61) == pytest.approx(full[mix.states])
    assert np.abs(np.delete(full, mix.states)).max() < 1e-12


def test_warm_start_begins_at_the_relaxed_solution():
    c = np.array([0.3, 0.6, 0.5])
    probs = np.abs(qaoa.WarmMixer(3, c).start()) ** 2
    assert qaoa.bits(3, range(8)).T @ probs == pytest.approx(c)


def test_cvar():
    p, e = np.array([0.1, 0.2, 0.3, 0.4]), np.array([4.0, 1.0, 3.0, 2.0])
    assert qaoa.cvar(p, e, 1) == pytest.approx(p @ e)
    assert qaoa.cvar(p, e, 0.4) == pytest.approx((0.2 * 1 + 0.2 * 2) / 0.4)


def test_xy_beats_x_and_every_solver_finds_the_optimum(market):
    model = models.selection(market)
    ref = qaoa.exact(model)
    x, xy = qaoa.qaoa(model, ref, "x"), qaoa.qaoa(model, ref, "xy")
    assert xy["p_valid"] == pytest.approx(1)
    assert xy["p_optimal"] > 5 * x["p_optimal"]
    for r in (x, xy, qaoa.qaoa(model, ref, "warm"), qaoa.annealing(model)):
        assert r["objective"] == pytest.approx(ref["best"])


def test_slower_anneal_gets_closer(market):
    model = models.selection(market)
    ref = qaoa.exact(model)
    p = [qaoa.adiabatic(model, ref, "xy", T, 4 * T)["p_optimal"] for T in (2, 10, 100)]
    assert p[0] < p[1] < p[2]
