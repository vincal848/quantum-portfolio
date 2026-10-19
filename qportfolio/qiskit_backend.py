"""The same QAOA circuit built in Qiskit, as an independent check on the engine.

`qaoa_probabilities` builds the Ising Hamiltonian as a SparsePauliOp, wraps it in
Qiskit's QAOAAnsatz and returns the exact output distribution. The tests require it
to match `quantum.evolve` to numerical precision for the same angles, which pins the
bit ordering, the sign conventions and the Ising conversion at once. It is also the
starting point for running any quadratic model here on real hardware.
"""

from __future__ import annotations

import numpy as np

from .problem import Polynomial


def ising_operator(poly: Polynomial):
    from qiskit.quantum_info import SparsePauliOp

    h, J, const = poly.to_ising()
    n = poly.n
    terms = []
    for i in range(n):
        if h[i]:
            terms.append(("Z", [i], h[i]))
    for i in range(n):
        for j in range(i + 1, n):
            if J[i, j]:
                terms.append(("ZZ", [i, j], J[i, j]))
    if not terms:
        terms = [("I", [0], 0.0)]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n), const


def qaoa_circuit(poly: Polynomial, gammas, betas):
    from qiskit.circuit.library import QAOAAnsatz

    op, _ = ising_operator(poly)
    ansatz = QAOAAnsatz(cost_operator=op, reps=len(gammas))
    by_name = {}
    for prm in ansatz.parameters:
        kind, idx = prm.name.split("[")
        idx = int(idx.rstrip("]"))
        by_name[prm] = gammas[idx] if kind.startswith("γ") or kind.startswith("g") else betas[idx]
    return ansatz.assign_parameters(by_name)


def qaoa_probabilities(poly: Polynomial, gammas, betas) -> np.ndarray:
    """Exact probabilities indexed by integer state (little-endian, as everywhere)."""
    from qiskit.quantum_info import Statevector

    circuit = qaoa_circuit(poly, gammas, betas)
    return np.abs(Statevector(circuit).data) ** 2
