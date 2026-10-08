"""QAOA against the exact optimum.

Skipped when qiskit is not installed, which is how CI runs: the simulator is slow
and the rest of the suite covers the encoding and the scoring. Run locally with
`pip install -r requirements.txt` to exercise these.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import solvers
from portfolio import build_program, to_ising

qiskit = pytest.importorskip("qiskit", reason="qiskit not installed")
pytest.importorskip("qiskit_optimization", reason="qiskit-optimization not installed")

pytestmark = pytest.mark.slow


def toy_problem(n=5, seed=1):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.05, 0.20, n)
    sigma = np.diag(rng.uniform(0.005, 0.02, n))
    return mu, sigma


def test_cardinality_constraint_adds_no_slack_qubits():
    """An equality constraint needs no slack; an inequality does.

    This is why build_program uses '==': sum(x) <= k would cost extra qubits, which
    on a statevector simulator is the difference between running and not.
    """
    mu, sigma = toy_problem(8)
    _, _, qubits = to_ising(build_program(mu, sigma, 0.5, cardinality=3))
    assert qubits == 8


def test_ising_encoding_agrees_with_the_objective():
    """The exact eigensolver and brute force must find the same selection.

    If they disagree, the Hamiltonian is not encoding the problem the objective
    describes, and every result built on it is meaningless.
    """
    pytest.importorskip("qiskit_algorithms")
    mu, sigma = toy_problem(6, seed=4)
    exact = solvers.brute_force(mu, sigma, 0.5, cardinality=2)
    viaising = solvers.exact_eigensolver(mu, sigma, 0.5, cardinality=2)
    assert viaising.selected == exact.selected


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_qaoa_finds_the_exact_optimum_on_small_problems(seed):
    mu, sigma = toy_problem(5, seed=seed)
    exact = solvers.brute_force(mu, sigma, 0.5, cardinality=2)
    approx = solvers.qaoa(mu, sigma, 0.5, cardinality=2, seed=seed)

    assert approx.feasible
    assert len(approx.selected) == 2
    assert approx.objective >= exact.objective - 1e-9
    assert approx.selected == exact.selected


def test_qaoa_respects_the_cardinality_constraint():
    """The constraint is only a penalty, so feasibility has to be checked, not assumed."""
    mu, sigma = toy_problem(6, seed=9)
    result = solvers.qaoa(mu, sigma, 0.5, cardinality=3, seed=9)
    assert len(result.selected) == 3
    assert result.feasible


def test_qaoa_is_reproducible_under_a_fixed_seed():
    mu, sigma = toy_problem(5, seed=2)
    first = solvers.qaoa(mu, sigma, 0.5, cardinality=2, seed=42)
    second = solvers.qaoa(mu, sigma, 0.5, cardinality=2, seed=42)
    assert first.selected == second.selected
    assert first.objective == pytest.approx(second.objective)
