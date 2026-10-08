"""Problem construction, bit decoding and the solvers, against brute force."""

import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import solvers
from portfolio import decode, equal_weights, objective


# --- the bit-ordering bug -------------------------------------------------------

def test_decode_is_little_endian():
    """Qiskit puts qubit 0 in the RIGHTMOST position.

    The original did `enumerate(bitstring)`, which maps index 0 to the leftmost
    character - the most significant bit, i.e. the LAST qubit. The selection came
    back mirrored.
    """
    assert decode("0001", 4) == [0]
    assert decode("1000", 4) == [3]
    assert decode("0011", 4) == [0, 1]
    assert decode("1010", 4) == [1, 3]
    assert decode("0000", 4) == []
    assert decode("1111", 4) == [0, 1, 2, 3]


def test_decode_disagrees_with_the_original_reading():
    """Confirms the test above is testing something real.

    On a palindromic bitstring the two readings agree, which is exactly why the bug
    survived: the toy example in the original had a symmetric answer.
    """
    naive = lambda bits, n: [i for i, b in enumerate(bits) if b == "1"]
    assert decode("0001", 4) != naive("0001", 4)
    assert decode("1111", 4) == naive("1111", 4)   # invisible here
    assert decode("0110", 4) == naive("0110", 4)   # and here


@pytest.mark.parametrize("best_index", [0, 1, 2, 3])
def test_decoded_selection_matches_brute_force(best_index):
    """Make one asset clearly the only one worth holding, then check we name it.

    Run for each position in turn, so a mirrored decode cannot pass by coincidence.
    """
    n = 4
    mu = np.full(n, 0.02)
    mu[best_index] = 0.30
    variances = np.full(n, 0.30)
    variances[best_index] = 0.01
    sigma = np.diag(variances)

    exact = solvers.brute_force(mu, sigma, lam=1.0)
    assert exact.selected == [best_index]

    # And the bitstring for that selection round-trips.
    bits = "".join("1" if i in exact.selected else "0" for i in range(n))[::-1]
    assert decode(bits, n) == [best_index]


# --- the objective --------------------------------------------------------------

def test_objective_matches_the_formula():
    mu = np.array([0.1, 0.2])
    sigma = np.array([[0.005, 0.002], [0.002, 0.006]])
    x = np.array([1.0, 1.0])
    expected = -(0.1 + 0.2) + 0.5 * (0.005 + 0.002 + 0.002 + 0.006)
    assert objective(x, mu, sigma, 0.5) == pytest.approx(expected)


def test_empty_selection_scores_zero():
    mu = np.array([0.1, 0.2])
    sigma = np.eye(2) * 0.01
    assert objective([0, 0], mu, sigma, 0.5) == pytest.approx(0.0)


def test_higher_risk_aversion_prefers_lower_variance():
    """Sanity on the sign of lambda: more risk aversion should shed the risky asset."""
    mu = np.array([0.25, 0.20])
    sigma = np.diag([0.40, 0.01])

    low = solvers.brute_force(mu, sigma, lam=0.1, cardinality=1)
    high = solvers.brute_force(mu, sigma, lam=10.0, cardinality=1)
    assert low.selected == [0], "at low risk aversion, take the higher return"
    assert high.selected == [1], "at high risk aversion, take the lower variance"


# --- solvers --------------------------------------------------------------------

def test_brute_force_is_actually_exhaustive():
    """Cross-check against an independent enumeration."""
    rng = np.random.default_rng(3)
    n = 8
    mu = rng.uniform(0.05, 0.2, n)
    a = rng.uniform(0.001, 0.01, (n, n))
    sigma = (a + a.T) / 2
    np.fill_diagonal(sigma, rng.uniform(0.005, 0.02, n))

    got = solvers.brute_force(mu, sigma, lam=0.5, cardinality=3)
    best = min((c for c in itertools.product([0, 1], repeat=n) if sum(c) == 3),
               key=lambda c: objective(c, mu, sigma, 0.5))
    assert got.selected == [i for i, b in enumerate(best) if b == 1]


def test_brute_force_respects_cardinality():
    rng = np.random.default_rng(11)
    mu = rng.uniform(0.05, 0.2, 7)
    sigma = np.diag(rng.uniform(0.005, 0.02, 7))
    for k in (1, 2, 3, 4):
        assert len(solvers.brute_force(mu, sigma, 0.5, k).selected) == k


def test_impossible_cardinality_raises():
    mu = np.array([0.1, 0.2])
    sigma = np.eye(2) * 0.01
    with pytest.raises(ValueError):
        solvers.brute_force(mu, sigma, 0.5, cardinality=5)


def test_greedy_is_never_better_than_exact():
    """Greedy may tie the optimum but can never beat it, by definition."""
    rng = np.random.default_rng(5)
    for _ in range(10):
        n = 8
        mu = rng.uniform(0.05, 0.2, n)
        a = rng.uniform(0.001, 0.01, (n, n))
        sigma = (a + a.T) / 2
        np.fill_diagonal(sigma, rng.uniform(0.005, 0.02, n))

        exact = solvers.brute_force(mu, sigma, 0.5, 3)
        heuristic = solvers.greedy(mu, sigma, 0.5, 3)
        assert heuristic.objective >= exact.objective - 1e-12
        assert len(heuristic.selected) == 3


# --- weights --------------------------------------------------------------------

def test_equal_weights_sum_to_one():
    w = equal_weights([0, 2, 4], 6)
    assert w.sum() == pytest.approx(1.0)
    assert w[0] == w[2] == w[4] == pytest.approx(1 / 3)
    assert w[1] == w[3] == w[5] == 0.0


def test_equal_weights_of_nothing_is_all_zero():
    w = equal_weights([], 4)
    assert w.sum() == pytest.approx(0.0)


def test_solvers_return_typed_solutions_without_timing():
    """Timing belongs to run.py; a solver returns only the answer."""
    mu = np.array([0.1, 0.2, 0.15])
    sigma = np.diag([0.01, 0.02, 0.015])
    for solve in (solvers.brute_force, solvers.greedy):
        got = solve(mu, sigma, 0.5, 2)
        assert isinstance(got, solvers.Solution)
        assert not hasattr(got, "seconds")
