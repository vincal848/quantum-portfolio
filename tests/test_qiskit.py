"""The numpy engine against the same circuit built in Qiskit."""

import numpy as np
import pytest

from qportfolio import data, models
from qportfolio.quantum import XMixer, evolve

pytest.importorskip("qiskit", reason="qiskit not installed")
pytestmark = pytest.mark.slow

from qportfolio.qiskit_backend import ising_operator, qaoa_probabilities  # noqa: E402


@pytest.mark.parametrize("build", [
    lambda m: models.selection(m, k=2),
    lambda m: models.tracking(m, k=3),
    lambda m: models.long_short(m),
    lambda m: models.diversified(m, k=2, threshold=0.3),
])
def test_engine_matches_qiskit_qaoa_ansatz(build):
    m = data.synthetic_market(6, seed=3)
    H = build(m).hamiltonian()
    gammas, betas = [0.31, 0.72, 0.15], [0.52, 0.21, 0.66]
    ours = np.abs(evolve(H.evaluate_states(np.arange(64)), XMixer(6), gammas, betas)) ** 2
    theirs = qaoa_probabilities(H, gammas, betas)
    assert ours == pytest.approx(theirs, abs=1e-10)


def test_operator_diagonal_is_the_hamiltonian():
    from qiskit.quantum_info import Operator

    m = data.synthetic_market(5, seed=2)
    H = models.selection(m, k=2).hamiltonian()
    op, const = ising_operator(H)
    diag = np.real(np.diag(Operator(op).data)) + const
    assert diag == pytest.approx(H.evaluate_states(np.arange(32)))
