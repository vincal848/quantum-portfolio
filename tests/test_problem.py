"""The binary polynomial layer: evaluation, Ising form, penalties, bit order."""

from math import comb

import numpy as np
import pytest

from qportfolio.problem import (Polynomial, at_most_one_penalty, at_most_two_penalty,
                                bit_table, equality_penalty, pair_penalty, weight_k_states)


def random_poly(n, seed, cubic=False):
    rng = np.random.default_rng(seed)
    return Polynomial(n, offset=rng.normal(), c=rng.normal(size=n), Q=rng.normal(size=(n, n)),
                      T=rng.normal(size=(n, n, n)) if cubic else None)


def direct(poly, x):
    x = np.asarray(x, float)
    v = poly.offset + poly.c @ x + x @ poly.Q @ x
    if poly.T is not None:
        v += np.einsum("ijk,i,j,k->", poly.T, x, x, x)
    return v


def test_bit_order_is_little_endian():
    assert list(bit_table(4, [1])[0]) == [1, 0, 0, 0]
    assert list(bit_table(4, [8])[0]) == [0, 0, 0, 1]
    assert list(bit_table(3, [6])[0]) == [0, 1, 1]


@pytest.mark.parametrize("cubic", [False, True])
def test_batch_evaluation_matches_direct(cubic):
    poly = random_poly(5, 1, cubic)
    X = bit_table(5)
    batch = poly.evaluate(X)
    assert batch == pytest.approx([direct(poly, x) for x in X])
    assert poly.evaluate_states(np.arange(32), chunk=7) == pytest.approx(batch)


def test_ising_form_reproduces_every_state():
    poly = random_poly(6, 2)
    h, J, const = poly.to_ising()
    Z = 1 - 2 * bit_table(6)
    ising = const + Z @ h + np.einsum("si,ij,sj->s", Z, J, Z)
    assert ising == pytest.approx(poly.evaluate(bit_table(6)))


def test_ising_rejects_cubic_terms():
    with pytest.raises(ValueError):
        random_poly(3, 0, cubic=True).to_ising()


def test_gradient_matches_finite_differences():
    poly = random_poly(4, 3, cubic=True)
    x = np.array([0.2, 0.7, 0.4, 0.9])
    eps = 1e-6
    fd = [(direct(poly, x + eps * e) - direct(poly, x - eps * e)) / (2 * eps)
          for e in np.eye(4)]
    assert poly.gradient(x) == pytest.approx(fd, rel=1e-5)


def test_equality_penalty_is_zero_exactly_on_the_constraint():
    X = bit_table(5)
    pen = equality_penalty(np.ones(5), 2, 5).evaluate(X)
    assert np.all((pen == 0) == (X.sum(1) == 2))
    assert np.all(pen[X.sum(1) != 2] >= 1)


def test_group_penalties():
    X = bit_table(6)
    groups = [[0, 1, 2], [3, 4, 5]]
    one = at_most_one_penalty(groups, 6).evaluate(X)
    two = at_most_two_penalty(groups, 6).evaluate(X)
    counts = np.stack([X[:, g].sum(1) for g in groups], axis=1)
    assert np.all((one == 0) == np.all(counts <= 1, axis=1))
    assert np.all((two == 0) == np.all(counts <= 2, axis=1))
    assert np.all(two >= 0) and np.all(one >= 0)


def test_pair_penalty_is_an_independent_set_check():
    edges = [(0, 1), (1, 2)]
    X = bit_table(3)
    pen = pair_penalty(edges, 3).evaluate(X)
    independent = [not any(x[i] and x[j] for i, j in edges) for x in X]
    assert list(pen == 0) == independent


@pytest.mark.parametrize("n,k", [(5, 2), (8, 3), (10, 5)])
def test_weight_k_states(n, k):
    s = weight_k_states(n, k)
    assert len(s) == comb(n, k)
    assert np.all(bit_table(n, s).sum(1) == k)
