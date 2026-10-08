"""Four ways to solve the selection problem, all returning the same shape.

Each solver returns a `Solution` (or a subclass with solver-specific extras), so they
can be compared directly. Timing is the caller's job -- see run.py. Brute force is the reference: at these sizes the problem is
small enough to enumerate exactly, which means every other solver can be scored
against the true optimum rather than against the others.

That reference is the thing the original was missing. It solved the problem with
NumPyMinimumEigensolver and reported the answer, with nothing to say whether the
answer was right.
"""

import itertools
from dataclasses import dataclass

import numpy as np

from portfolio import decode, objective


@dataclass(frozen=True)
class Solution:
    selected: list[int]
    objective: float
    feasible: bool


@dataclass(frozen=True)
class QaoaSolution(Solution):
    top_sample_feasible: bool
    n_qubits: int
    energy: float
    shots_on_top: int


@dataclass(frozen=True)
class EigenSolution(Solution):
    energy: float


def brute_force(mu: np.ndarray, sigma: np.ndarray, lam: float,
                cardinality: int | None = None) -> Solution:
    """Exact optimum by enumerating every subset. The reference answer.

    2^n subsets, so this is only usable while n is small -- which it is here, and
    that is the point: on problems a quantum heuristic can actually run on today,
    the exact answer is cheap. Any claim of quantum advantage has to survive that.
    """
    n = len(mu)

    best_x, best_value = None, float("inf")
    for combo in itertools.product([0, 1], repeat=n):
        if cardinality is not None and sum(combo) != cardinality:
            continue
        value = objective(combo, mu, sigma, lam)
        if value < best_value:
            best_value, best_x = value, combo

    if best_x is None:
        raise ValueError("no feasible selection for cardinality=%r" % (cardinality,))

    return Solution([i for i, b in enumerate(best_x) if b == 1], best_value, True)


def greedy(mu: np.ndarray, sigma: np.ndarray, lam: float,
           cardinality: int | None = None) -> Solution:
    """Classical baseline: add whichever asset improves the objective most.

    Included because 'QAOA beat random' is not interesting. The question is whether
    it beats something anyone would actually reach for, and a greedy pass takes
    microseconds.
    """
    n = len(mu)
    target = cardinality if cardinality is not None else n

    chosen = []
    for _ in range(target):
        candidates = [i for i in range(n) if i not in chosen]
        if not candidates:
            break
        scored = []
        for i in candidates:
            trial = np.zeros(n)
            trial[chosen + [i]] = 1
            scored.append((objective(trial, mu, sigma, lam), i))
        value, pick = min(scored)
        # Without a cardinality target, stop once nothing improves the objective.
        if cardinality is None and chosen:
            current = np.zeros(n)
            current[chosen] = 1
            if value >= objective(current, mu, sigma, lam):
                break
        chosen.append(pick)

    x = np.zeros(n)
    x[chosen] = 1
    return Solution(sorted(chosen), objective(x, mu, sigma, lam),
                    cardinality is None or len(chosen) == cardinality)


def qaoa(mu: np.ndarray, sigma: np.ndarray, lam: float,
         cardinality: int | None = None, reps: int = 2, restarts: int = 3,
         maxiter: int = 400, shots: int = 8192, seed: int = 7) -> QaoaSolution:
    """Quantum Approximate Optimization Algorithm on a statevector simulator.

    Written against Qiskit 2.x primitives directly rather than through
    qiskit-algorithms, which no longer works with Qiskit 2 -- the original imported
    `Sampler` and `BackendSamplerV2` from qiskit.primitives, neither of which still
    exists, so it raises on import. This needs only qiskit and scipy.

    The classical optimizer is restarted from several random points because the QAOA
    energy landscape is non-convex and a single COBYLA run lands in a local minimum
    often enough to matter.
    """
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.primitives import StatevectorEstimator, StatevectorSampler
    from scipy.optimize import minimize

    from portfolio import build_program, to_ising

    n = len(mu)

    qp = build_program(mu, sigma, lam, cardinality)
    operator, offset, n_qubits = to_ising(qp)

    ansatz = QAOAAnsatz(cost_operator=operator, reps=reps).decompose(reps=3)
    estimator = StatevectorEstimator()

    def energy(theta):
        result = estimator.run([(ansatz, operator, [theta])]).result()
        # The estimator returns an array even for a single observable.
        return float(np.asarray(result[0].data.evs).reshape(-1)[0])

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        x0 = rng.uniform(0, np.pi, ansatz.num_parameters)
        out = minimize(energy, x0, method="COBYLA", options={"maxiter": maxiter})
        if best is None or out.fun < best.fun:
            best = out

    bound = ansatz.assign_parameters(best.x)
    bound.measure_all()
    sampler = StatevectorSampler(seed=seed)
    counts = sampler.run([bound], shots=shots).result()[0].data.meas.get_counts()

    # The constraint is only a penalty in the Hamiltonian, so the most-sampled
    # bitstring can be infeasible. Take the best feasible one that was actually
    # measured, and record whether the top sample was feasible at all.
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    top_selected = decode(ranked[0][0], n)
    top_feasible = cardinality is None or len(top_selected) == cardinality

    chosen, chosen_value = None, float("inf")
    for bitstring, _ in ranked:
        selected = decode(bitstring, n)
        if cardinality is not None and len(selected) != cardinality:
            continue
        x = np.zeros(n)
        x[selected] = 1
        value = objective(x, mu, sigma, lam)
        if value < chosen_value:
            chosen_value, chosen = value, selected

    if chosen is None:
        # Nothing feasible was sampled at all.
        chosen, chosen_value = top_selected, objective(
            np.isin(np.arange(n), top_selected).astype(float), mu, sigma, lam)

    return QaoaSolution(
        selected=sorted(chosen),
        objective=chosen_value,
        feasible=cardinality is None or len(chosen) == cardinality,
        top_sample_feasible=top_feasible,
        n_qubits=n_qubits,
        energy=float(best.fun) + offset,
        shots_on_top=ranked[0][1],
    )


def exact_eigensolver(mu: np.ndarray, sigma: np.ndarray, lam: float,
                      cardinality: int | None = None) -> EigenSolution:
    """Classical exact diagonalization of the Ising Hamiltonian.

    This is what the original used and called quantum. It is not: it builds the same
    Hamiltonian QAOA would minimize and then diagonalizes it exactly on a classical
    computer, which is an exhaustive 2^n search wearing different notation. Kept
    because it is a useful cross-check that the Hamiltonian encodes the intended
    problem -- if this disagrees with brute_force, the encoding is wrong, not the
    solver.
    """
    from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver

    from portfolio import build_program, to_ising

    n = len(mu)
    qp = build_program(mu, sigma, lam, cardinality)
    operator, offset, _ = to_ising(qp)

    result = NumPyMinimumEigensolver().compute_minimum_eigenvalue(operator)
    state = result.eigenstate
    amplitudes = state.data if hasattr(state, "data") else np.asarray(state)
    index = int(np.argmax(np.abs(amplitudes) ** 2))
    bitstring = format(index, "0%db" % operator.num_qubits)

    selected = decode(bitstring, n)
    x = np.zeros(n)
    x[selected] = 1

    return EigenSolution(
        selected=selected,
        objective=objective(x, mu, sigma, lam),
        feasible=cardinality is None or len(selected) == cardinality,
        energy=float(np.real(result.eigenvalue)) + offset,
    )
