"""The selection problem, and how it becomes a QUBO.

The problem is mean-variance selection over a binary choice vector x, where x_i = 1
means asset i is held:

    minimize   -mu . x  +  lambda * x' Sigma x        subject to  sum(x) = k

This is the standard cardinality-constrained formulation. Note what it is NOT: the
objective is over the raw 0/1 selection, so it does not price the fact that holding
five assets at a fifth each is different from holding five assets at full weight.
Once k is fixed by the constraint, scaling is common to every feasible point and the
ranking is unaffected -- but the objective value itself is not a portfolio variance,
and reported portfolio statistics are computed separately in metrics.py from equal
weights across the chosen assets. Keeping those two things distinct matters; the
original mixed them.
"""

import numpy as np


def objective(selection, mu, sigma, lam):
    """Value of the mean-variance objective for a 0/1 selection vector.

    This is the single definition of the objective. Every solver is scored with it,
    so a solver cannot look good by optimizing something slightly different.
    """
    x = np.asarray(selection, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    return float(-mu @ x + lam * x @ sigma @ x)


def build_program(mu, sigma, lam, cardinality=None):
    """Build the QuadraticProgram for the selection problem.

    `cardinality` adds sum(x) == k. Equality is used rather than <= on purpose:
    an inequality needs slack variables, and the converter encodes those as extra
    qubits, so a 10-asset problem with sum(x) <= 3 costs 12 qubits instead of 10.
    On a simulator that is the difference between a problem that runs and one that
    does not.
    """
    # Imported here, not at module scope, so decode(), objective() and
    # equal_weights() stay usable - and testable - without the qiskit stack.
    from qiskit_optimization import QuadraticProgram

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = len(mu)
    if sigma.shape != (n, n):
        raise ValueError("sigma must be %dx%d, got %s" % (n, n, sigma.shape))
    if cardinality is not None and not 1 <= cardinality <= n:
        raise ValueError("cardinality must be between 1 and %d, got %r" % (n, cardinality))

    qp = QuadraticProgram()
    for i in range(n):
        qp.binary_var("x%d" % i)

    qp.minimize(
        linear={"x%d" % i: -mu[i] for i in range(n)},
        quadratic={("x%d" % i, "x%d" % j): lam * sigma[i][j]
                   for i in range(n) for j in range(n)},
    )

    if cardinality is not None:
        qp.linear_constraint(linear={"x%d" % i: 1 for i in range(n)},
                             sense="==", rhs=cardinality, name="cardinality")
    return qp


def to_ising(qp):
    """Convert to an Ising Hamiltonian. Returns (operator, offset, n_qubits).

    The constraint becomes a penalty term here, which is why the constrained problem
    can still come back with an infeasible bitstring: nothing forbids violating it,
    it is only expensive. Solvers have to be checked for feasibility afterwards.
    """
    from qiskit_optimization.converters import QuadraticProgramToQubo

    qubo = QuadraticProgramToQubo().convert(qp)
    operator, offset = qubo.to_ising()
    return operator, offset, operator.num_qubits


def decode(bitstring, n_assets):
    """Turn a measured bitstring into a list of asset indices.

    Qiskit is little-endian: qubit 0 is the RIGHTMOST character. The original read
    the string left to right with enumerate(), which maps index 0 to the most
    significant bit, so the selection came back mirrored -- asset 3 reported when
    the answer was asset 0. On a symmetric solution it is invisible, which is why
    it survived. tests/test_portfolio.py pins this against brute force.
    """
    bits = str(bitstring)[::-1]
    return [i for i in range(n_assets) if i < len(bits) and bits[i] == "1"]


def equal_weights(selected, n_assets):
    """Capital split evenly across the selected assets, zero elsewhere."""
    w = np.zeros(n_assets, dtype=float)
    if selected:
        w[list(selected)] = 1.0 / len(selected)
    return w
