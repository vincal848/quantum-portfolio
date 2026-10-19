"""Binary polynomial optimization problems: the format every model compiles to.

A model is a polynomial over binary variables x in {0,1}^n (the cost), plus a list
of constraints, each written as a non-negative polynomial that is zero exactly when
the constraint holds. That covers QUBOs (quadratic) and HUBOs (cubic here), and it
is all a solver needs:

* classical solvers evaluate the polynomial directly;
* quantum solvers turn it into a diagonal Hamiltonian, cost + weight * penalties,
  or, when the only coupling constraint is a fixed number of ones, keep the
  constraint out of the Hamiltonian and enforce it with an XY mixer instead.

Bit convention throughout: integer state b has x_i = (b >> i) & 1, so variable i is
qubit i. That is Qiskit's little-endian ordering (qubit 0 is the rightmost character
of a bitstring).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable

import numpy as np

MAX_ENUMERATION_QUBITS = 24


# --- bit utilities ------------------------------------------------------------

def bit_table(n: int, states: np.ndarray | None = None) -> np.ndarray:
    """Rows of 0/1 bits for each integer state (all 2^n when `states` is None)."""
    idx = np.arange(2 ** n, dtype=np.int64) if states is None else np.asarray(states, np.int64)
    return ((idx[:, None] >> np.arange(n, dtype=np.int64)) & 1).astype(np.int8)


def to_index(x) -> int:
    return int(sum(int(b) << i for i, b in enumerate(x)))


def to_bitstring(x) -> str:
    """Qiskit-style bitstring: variable 0 is the rightmost character."""
    return "".join(str(int(b)) for b in x)[::-1]


def from_bitstring(bits: str) -> np.ndarray:
    return np.array([int(c) for c in bits[::-1]], dtype=np.int8)


def weight_k_states(n: int, k: int) -> np.ndarray:
    """All integer states with exactly k ones, ascending."""
    if n <= MAX_ENUMERATION_QUBITS:
        idx = np.arange(2 ** n, dtype=np.int64)
        return idx[np.bitwise_count(idx) == k]
    return np.array(sorted(sum(1 << i for i in c) for c in combinations(range(n), k)),
                    dtype=np.int64)


# --- polynomials ----------------------------------------------------------------

@dataclass
class Polynomial:
    """offset + c.x + x'Qx + sum_ijk T_ijk x_i x_j x_k over binary x.

    Q and T are dense and not required to be symmetric. Dense is the right trade at
    the sizes a statevector can hold, and it keeps evaluation a few einsums.
    """

    n: int
    offset: float = 0.0
    c: np.ndarray = None
    Q: np.ndarray = None
    T: np.ndarray | None = None

    def __post_init__(self):
        self.c = np.zeros(self.n) if self.c is None else np.asarray(self.c, float).copy()
        self.Q = np.zeros((self.n, self.n)) if self.Q is None else np.asarray(self.Q, float).copy()
        if self.T is not None:
            self.T = np.asarray(self.T, float).copy()

    # arithmetic
    def __add__(self, other: "Polynomial") -> "Polynomial":
        if other is None:
            return self.copy()
        T = None
        if self.T is not None or other.T is not None:
            T = np.zeros((self.n,) * 3)
            for t in (self.T, other.T):
                if t is not None:
                    T = T + t
        return Polynomial(self.n, self.offset + other.offset, self.c + other.c,
                          self.Q + other.Q, T)

    __radd__ = lambda self, other: self if other in (0, None) else self.__add__(other)

    def __mul__(self, s: float) -> "Polynomial":
        return Polynomial(self.n, self.offset * s, self.c * s, self.Q * s,
                          None if self.T is None else self.T * s)

    __rmul__ = __mul__

    def copy(self) -> "Polynomial":
        return self * 1.0

    @property
    def degree(self) -> int:
        if self.T is not None and np.any(self.T):
            return 3
        if np.any(self.Q):
            return 2
        return 1 if np.any(self.c) else 0

    # evaluation
    def evaluate(self, X) -> np.ndarray:
        """Value for each row of a (states x n) 0/1 matrix, or a single vector."""
        X = np.asarray(X, dtype=float)
        single = X.ndim == 1
        X = np.atleast_2d(X)
        out = self.offset + X @ self.c + np.einsum("si,ij,sj->s", X, self.Q, X)
        if self.T is not None:
            XT = (X @ self.T.reshape(self.n, -1)).reshape(len(X), self.n, self.n)
            out = out + np.einsum("sjk,sj,sk->s", XT, X, X)
        return out[0] if single else out

    def evaluate_states(self, states: np.ndarray, chunk: int = 1 << 16) -> np.ndarray:
        """Value for integer states, in chunks so 2^20 states fit in memory."""
        states = np.asarray(states, np.int64)
        out = np.empty(len(states))
        for start in range(0, len(states), chunk):
            block = states[start:start + chunk]
            out[start:start + len(block)] = self.evaluate(bit_table(self.n, block))
        return out

    def gradient(self, x) -> np.ndarray:
        """Gradient of the continuous extension, used for warm-start relaxations."""
        x = np.asarray(x, float)
        g = self.c + (self.Q + self.Q.T) @ x
        if self.T is not None:
            T = self.T
            g = g + (np.einsum("ajk,j,k->a", T, x, x) + np.einsum("jak,j,k->a", T, x, x)
                     + np.einsum("jka,j,k->a", T, x, x))
        return g

    def coefficient_bound(self) -> float:
        """Upper bound on (max - min) of the polynomial over {0,1}^n."""
        total = np.abs(self.c).sum() + np.abs(self.Q).sum()
        if self.T is not None:
            total += np.abs(self.T).sum()
        return float(total)

    # Ising form
    def to_ising(self):
        """Return (h, J, const) with value = const + sum h_i z_i + sum_{i<j} J_ij z_i z_j.

        Substitutes x = (1 - z) / 2, so x = 0 is spin up (z = +1), matching the
        computational basis. Quadratic polynomials only.
        """
        if self.degree > 2:
            raise ValueError("Ising form is only defined here for degree <= 2")
        c = self.c + np.diag(self.Q)              # x_i^2 = x_i
        Q = self.Q - np.diag(np.diag(self.Q))
        S = Q + Q.T                               # symmetric off-diagonal couplings
        const = self.offset + c.sum() / 2 + S.sum() / 8
        h = -c / 2 - S.sum(axis=1) / 4
        J = np.triu(S, 1) / 4
        return h, J, float(const)


# --- constraints ----------------------------------------------------------------

def equality_penalty(a, b: float, n: int) -> Polynomial:
    """(a.x - b)^2 as a polynomial. Zero exactly when a.x = b."""
    a = np.asarray(a, float)
    return Polynomial(n, offset=b * b, c=-2 * b * a, Q=np.outer(a, a))


def at_most_one_penalty(groups, n: int) -> Polynomial:
    """sum over pairs inside each group of x_i x_j. Zero iff no group has two ones."""
    Q = np.zeros((n, n))
    for g in groups:
        for i, j in combinations(sorted(g), 2):
            Q[i, j] += 1.0
    return Polynomial(n, Q=Q)


def at_most_two_penalty(groups, n: int) -> Polynomial:
    """sum over triples inside each group of x_i x_j x_k. Zero iff no group has three.

    A cubic penalty with no slack qubits. The quadratic route to "at most m" needs
    log2(m) slack bits per group; a higher-order term needs none, and a diagonal
    Hamiltonian does not care what order its terms are.
    """
    T = np.zeros((n, n, n))
    for g in groups:
        for i, j, k in combinations(sorted(g), 3):
            T[i, j, k] += 1.0
    return Polynomial(n, T=T)


def pair_penalty(edges, n: int) -> Polynomial:
    """sum over edges of x_i x_j. Zero iff the selection is an independent set."""
    Q = np.zeros((n, n))
    for i, j in edges:
        Q[min(i, j), max(i, j)] += 1.0
    return Polynomial(n, Q=Q)


@dataclass
class Constraint:
    name: str
    penalty: Polynomial          # unweighted, >= 0, zero iff satisfied, integer-valued
    weight: float                # multiplier used when the constraint goes into H


# --- problems -------------------------------------------------------------------

@dataclass
class Problem:
    """A portfolio model compiled to binary form.

    `hamming_weight`: if every feasible state has exactly this many ones and that
    constraint is the one named "cardinality", a Hamming-weight-preserving mixer can
    enforce it, so the solver drops that penalty from the Hamiltonian.
    """

    name: str
    cost: Polynomial
    constraints: list[Constraint]
    weights_fn: Callable[[np.ndarray], np.ndarray]
    labels: list[str]
    description: str = ""
    hamming_weight: int | None = None
    variable_labels: list[str] | None = None
    objective_label: str = "objective"
    meta: dict = field(default_factory=dict)

    @property
    def n_qubits(self) -> int:
        return self.cost.n

    def penalty(self, exclude: tuple[str, ...] = ()) -> Polynomial:
        total = Polynomial(self.n_qubits)
        for con in self.constraints:
            if con.name not in exclude:
                total = total + con.penalty * con.weight
        return total

    def hamiltonian(self, preserve_hamming_weight: bool = False) -> Polynomial:
        """cost + weighted penalties. Drops the cardinality term if a mixer enforces it."""
        exclude = ("cardinality",) if preserve_hamming_weight else ()
        return self.cost + self.penalty(exclude)

    def feasible(self, X) -> np.ndarray:
        X = np.asarray(X)
        single = X.ndim == 1
        X = np.atleast_2d(X)
        ok = np.ones(len(X), dtype=bool)
        for con in self.constraints:
            ok &= np.abs(con.penalty.evaluate(X)) < 1e-9
        return ok[0] if single else ok

    def feasible_states(self) -> np.ndarray:
        """Every feasible integer state. Uses the Hamming-weight shortcut when it can."""
        if self.hamming_weight is not None:
            states = weight_k_states(self.n_qubits, self.hamming_weight)
        else:
            if self.n_qubits > MAX_ENUMERATION_QUBITS:
                raise ValueError("too many qubits to enumerate")
            states = np.arange(2 ** self.n_qubits, dtype=np.int64)
        keep = np.ones(len(states), dtype=bool)
        for con in self.constraints:
            keep &= np.abs(con.penalty.evaluate_states(states)) < 1e-9
        return states[keep]

    def objective(self, x) -> float:
        return float(self.cost.evaluate(np.asarray(x)))

    def weights(self, x) -> np.ndarray:
        return self.weights_fn(np.asarray(x))

    def describe(self, x) -> str:
        w = self.weights(x)
        parts = ["%s %+.0f%%" % (self.labels[i], 100 * w[i])
                 for i in np.argsort(-np.abs(w)) if abs(w[i]) > 1e-12]
        return ", ".join(parts)


def safe_penalty_weight(cost: Polynomial, scale: float = 1.0) -> float:
    """A penalty weight that guarantees the penalized ground state is feasible.

    Every penalty here is integer-valued, so a violation costs at least `weight`.
    Making that exceed the largest possible swing in the cost means no infeasible
    state can undercut the feasible optimum. It is a sufficient bound, not a tight
    one: a smaller weight gives QAOA a gentler landscape, a larger one is safer.
    `scale` trades between them.
    """
    return scale * max(cost.coefficient_bound(), 1e-12)
