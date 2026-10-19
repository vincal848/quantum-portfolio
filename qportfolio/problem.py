"""Binary polynomials, constraints, and the Problem type every model builds.

A model is a cost polynomial over x in {0,1}^n plus a list of constraints. Each
constraint is itself a polynomial that is >= 0 everywhere and 0 exactly when the
constraint holds, so feasibility is just "every penalty is zero".

Bit order: integer state b has x_i = (b >> i) & 1, so variable i is qubit i. This
is Qiskit's little-endian convention, where qubit 0 is the rightmost character.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable

import numpy as np

MAX_ENUMERATION_QUBITS = 24


def bit_table(n: int, states=None) -> np.ndarray:
    """0/1 rows for the given integer states, or for all 2^n of them."""
    if states is None:
        states = np.arange(2 ** n, dtype=np.int64)
    states = np.asarray(states, dtype=np.int64)
    return ((states[:, None] >> np.arange(n)) & 1).astype(np.int8)


def weight_k_states(n: int, k: int) -> np.ndarray:
    """Every integer state with exactly k ones, in ascending order."""
    states = np.arange(2 ** n, dtype=np.int64)
    return states[np.bitwise_count(states) == k]


@dataclass
class Polynomial:
    """offset + c.x + x'Qx + sum T_ijk x_i x_j x_k.

    Everything is stored dense. At the sizes a statevector can hold that costs
    nothing, and evaluation stays a couple of einsums.
    """

    n: int
    offset: float = 0.0
    c: np.ndarray = None
    Q: np.ndarray = None
    T: np.ndarray | None = None

    def __post_init__(self):
        self.c = np.zeros(self.n) if self.c is None else np.array(self.c, dtype=float)
        self.Q = np.zeros((self.n, self.n)) if self.Q is None else np.array(self.Q, dtype=float)
        if self.T is not None:
            self.T = np.array(self.T, dtype=float)

    def __add__(self, other: "Polynomial") -> "Polynomial":
        T = None
        if self.T is not None or other.T is not None:
            T = np.zeros((self.n,) * 3)
            if self.T is not None:
                T += self.T
            if other.T is not None:
                T += other.T
        return Polynomial(self.n, self.offset + other.offset, self.c + other.c,
                          self.Q + other.Q, T)

    def __mul__(self, s: float) -> "Polynomial":
        T = None if self.T is None else self.T * s
        return Polynomial(self.n, self.offset * s, self.c * s, self.Q * s, T)

    __rmul__ = __mul__

    @property
    def degree(self) -> int:
        if self.T is not None and self.T.any():
            return 3
        if self.Q.any():
            return 2
        return 1 if self.c.any() else 0

    def evaluate(self, X):
        """Value at one 0/1 vector, or at each row of a matrix of them."""
        X = np.asarray(X, dtype=float)
        single = X.ndim == 1
        X = np.atleast_2d(X)
        values = self.offset + X @ self.c + np.einsum("si,ij,sj->s", X, self.Q, X)
        if self.T is not None:
            XT = (X @ self.T.reshape(self.n, -1)).reshape(len(X), self.n, self.n)
            values = values + np.einsum("sjk,sj,sk->s", XT, X, X)
        return values[0] if single else values

    def evaluate_states(self, states, chunk: int = 1 << 16) -> np.ndarray:
        """Value at each integer state. Chunked so 2^20 states fit in memory."""
        states = np.asarray(states, dtype=np.int64)
        values = np.empty(len(states))
        for start in range(0, len(states), chunk):
            block = states[start:start + chunk]
            values[start:start + len(block)] = self.evaluate(bit_table(self.n, block))
        return values

    def gradient(self, x) -> np.ndarray:
        """Gradient over the continuous box [0,1]^n (used for warm starts)."""
        x = np.asarray(x, dtype=float)
        g = self.c + (self.Q + self.Q.T) @ x
        if self.T is not None:
            g = g + (np.einsum("ajk,j,k->a", self.T, x, x)
                     + np.einsum("jak,j,k->a", self.T, x, x)
                     + np.einsum("jka,j,k->a", self.T, x, x))
        return g

    def coefficient_bound(self) -> float:
        """An upper bound on max - min over {0,1}^n: the sum of |coefficients|."""
        total = np.abs(self.c).sum() + np.abs(self.Q).sum()
        if self.T is not None:
            total += np.abs(self.T).sum()
        return float(total)

    def to_ising(self):
        """(h, J, const) such that value = const + h.z + sum_{i<j} J_ij z_i z_j.

        Uses x = (1 - z) / 2, so x = 0 is z = +1, the |0> state. Quadratic only.
        """
        if self.degree > 2:
            raise ValueError("to_ising only handles degree <= 2")
        c = self.c + np.diag(self.Q)               # x_i^2 = x_i for binary x
        S = self.Q + self.Q.T
        np.fill_diagonal(S, 0)
        const = self.offset + c.sum() / 2 + S.sum() / 8
        h = -c / 2 - S.sum(axis=1) / 4
        J = np.triu(S, 1) / 4
        return h, J, float(const)


def equality_penalty(a, b: float, n: int) -> Polynomial:
    """(a.x - b)^2."""
    a = np.asarray(a, dtype=float)
    return Polynomial(n, offset=b * b, c=-2 * b * a, Q=np.outer(a, a))


def pair_penalty(edges, n: int) -> Polynomial:
    """Sum of x_i x_j over the edges: zero when no edge has both ends selected."""
    Q = np.zeros((n, n))
    for i, j in edges:
        Q[min(i, j), max(i, j)] += 1.0
    return Polynomial(n, Q=Q)


def at_most_one_penalty(groups, n: int) -> Polynomial:
    return pair_penalty([pair for g in groups for pair in combinations(sorted(g), 2)], n)


def at_most_two_penalty(groups, n: int) -> Polynomial:
    """One cubic term per triple in each group: zero when no group has three ones.

    The quadratic way to say "at most 2" needs slack qubits. A cubic term needs
    none, and the QAOA phase layer doesn't care about the order of its terms.
    """
    T = np.zeros((n, n, n))
    for g in groups:
        for i, j, k in combinations(sorted(g), 3):
            T[i, j, k] += 1.0
    return Polynomial(n, T=T)


@dataclass
class Constraint:
    name: str
    penalty: Polynomial      # unweighted, integer-valued, zero iff satisfied
    weight: float            # multiplier when it goes into the Hamiltonian


@dataclass
class Problem:
    """A model compiled to binary form.

    If `hamming_weight` is set, every feasible state has exactly that many ones and
    the constraint enforcing it is called "cardinality". An XY mixer can then hold
    that constraint by itself, so the solver leaves its penalty out of H.
    """

    name: str
    cost: Polynomial
    constraints: list[Constraint]
    weights_fn: Callable[[np.ndarray], np.ndarray]
    labels: list[str]
    description: str = ""
    hamming_weight: int | None = None
    objective_label: str = "objective"

    @property
    def n_qubits(self) -> int:
        return self.cost.n

    def hamiltonian(self, preserve_hamming_weight: bool = False) -> Polynomial:
        """cost + weighted penalties, minus the cardinality penalty if a mixer holds it."""
        H = self.cost
        for con in self.constraints:
            if preserve_hamming_weight and con.name == "cardinality":
                continue
            H = H + con.penalty * con.weight
        return H

    def feasible(self, X):
        X = np.asarray(X)
        single = X.ndim == 1
        X = np.atleast_2d(X)
        ok = np.ones(len(X), dtype=bool)
        for con in self.constraints:
            ok &= np.abs(con.penalty.evaluate(X)) < 1e-9
        return ok[0] if single else ok

    def feasible_states(self) -> np.ndarray:
        if self.hamming_weight is not None:
            states = weight_k_states(self.n_qubits, self.hamming_weight)
        elif self.n_qubits <= MAX_ENUMERATION_QUBITS:
            states = np.arange(2 ** self.n_qubits, dtype=np.int64)
        else:
            raise ValueError("too many qubits to enumerate")
        keep = np.ones(len(states), dtype=bool)
        for con in self.constraints:
            keep &= np.abs(con.penalty.evaluate_states(states)) < 1e-9
        return states[keep]

    def objective(self, x) -> float:
        return float(self.cost.evaluate(np.asarray(x)))

    def weights(self, x) -> np.ndarray:
        return self.weights_fn(np.asarray(x))

    def describe(self, x) -> str:
        """Holdings as text, largest first: 'STA4 +75%, HEA2 +25%'."""
        w = self.weights(x)
        order = np.argsort(-np.abs(w))
        return ", ".join("%s %+.0f%%" % (self.labels[i], 100 * w[i])
                         for i in order if abs(w[i]) > 1e-12)


def safe_penalty_weight(cost: Polynomial, scale: float = 1.0) -> float:
    """A penalty weight large enough that the penalized ground state is feasible.

    Every penalty here is a whole number when violated, so a violation costs at
    least `weight`. If that beats the largest possible swing in the cost, no
    infeasible state can undercut the best feasible one. The bound is loose; a
    smaller weight is kinder to QAOA, and `scale` lets you try that.
    """
    return scale * max(cost.coefficient_bound(), 1e-12)
