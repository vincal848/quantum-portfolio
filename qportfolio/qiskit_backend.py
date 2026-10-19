"""The same QAOA circuit built with Qiskit.

Used two ways. The tests run it against quantum.evolve with the same angles and
require the output probabilities to agree, which checks bit order, signs and the
Ising conversion in one go. And it's the starting point for running any of the
quadratic models on real hardware.
"""

from __future__ import annotations

import numpy as np

from .problem import Polynomial


def ising_operator(poly: Polynomial):
    """SparsePauliOp for the polynomial, plus the constant it drops."""
    from qiskit.quantum_info import SparsePauliOp

    h, J, const = poly.to_ising()
    terms = [("Z", [i], h[i]) for i in range(poly.n) if h[i]]
    terms += [("ZZ", [i, j], J[i, j])
              for i in range(poly.n) for j in range(i + 1, poly.n) if J[i, j]]
    if not terms:
        terms = [("I", [0], 0.0)]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=poly.n), const


def qaoa_circuit(poly: Polynomial, gammas, betas):
    from qiskit.circuit.library import QAOAAnsatz

    operator, _ = ising_operator(poly)
    ansatz = QAOAAnsatz(cost_operator=operator, reps=len(gammas))
    # Qiskit names its parameters β[0], β[1], ..., γ[0], γ[1], ...
    values = {}
    for param in ansatz.parameters:
        symbol, index = param.name.rstrip("]").split("[")
        values[param] = (gammas if symbol == "γ" else betas)[int(index)]
    return ansatz.assign_parameters(values)


def qaoa_probabilities(poly: Polynomial, gammas, betas) -> np.ndarray:
    """Exact output probabilities, indexed by integer state."""
    from qiskit.quantum_info import Statevector

    return np.abs(Statevector(qaoa_circuit(poly, gammas, betas)).data) ** 2
