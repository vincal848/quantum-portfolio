"""A small, exact statevector engine for QAOA-type circuits on diagonal Hamiltonians.

Every portfolio model compiles to a diagonal cost Hamiltonian, so the cost layer is
an elementwise phase, exp(-i gamma E). What changes between variants is the mixer
and the initial state:

* XMixer           - standard QAOA (Farhi et al.). |+>^n, exp(-i beta sum X).
* XYMixer          - Quantum Alternating Operator Ansatz (Hadfield et al.). Starts in
                     the Dicke state over all weight-k strings, mixes with XX+YY
                     swaps, so the number of ones never changes. The engine stores
                     only the C(n,k) amplitudes of that subspace, which is what lets
                     it reach 20 qubits cheaply.
* WarmStartMixer   - warm-start QAOA (Egger et al.). Each qubit starts tilted toward
                     a continuous relaxation's answer, and the mixer is rotated to
                     match, so p = 1 already reproduces the relaxation.

All three share `evolve`, which is the whole circuit: alternate phase and mixer.
Probabilities are exact, so anything derived from them (expected energy, CVaR, the
probability of measuring the optimum) is exact too, with no shot noise.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .problem import weight_k_states


def _ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(phi):
    return np.diag([np.exp(-0.5j * phi), np.exp(0.5j * phi)])


def apply_single_qubit(psi: np.ndarray, n: int, q: int, U: np.ndarray) -> np.ndarray:
    """Apply a 2x2 unitary to qubit q of a full 2^n statevector."""
    v = psi.reshape(-1, 2, 2 ** q)
    a, b = v[:, 0, :], v[:, 1, :]
    out = np.empty_like(v)
    out[:, 0, :] = U[0, 0] * a + U[0, 1] * b
    out[:, 1, :] = U[1, 0] * a + U[1, 1] * b
    return out.reshape(-1)


class XMixer:
    """Transverse-field mixer on the full Hilbert space."""

    name = "x"

    def __init__(self, n: int):
        self.n = n
        self.states = np.arange(2 ** n, dtype=np.int64)

    def initial(self) -> np.ndarray:
        return np.full(2 ** self.n, 2 ** (-self.n / 2), dtype=complex)

    def apply(self, psi, beta):
        U = np.array([[np.cos(beta), -1j * np.sin(beta)],
                      [-1j * np.sin(beta), np.cos(beta)]])
        for q in range(self.n):
            psi = apply_single_qubit(psi, self.n, q, U)
        return psi


class WarmStartMixer(XMixer):
    """Warm-start mixer: RY(theta_i) RZ(-2 beta) RY(-theta_i) on each qubit."""

    name = "warm"

    def __init__(self, n: int, relaxed: np.ndarray, epsilon: float = 0.25):
        super().__init__(n)
        c = np.clip(np.asarray(relaxed, float), epsilon, 1 - epsilon)
        self.relaxed = c
        self.thetas = 2 * np.arcsin(np.sqrt(c))

    def initial(self):
        psi = np.zeros(2 ** self.n, dtype=complex)
        psi[0] = 1.0
        for q, th in enumerate(self.thetas):
            psi = apply_single_qubit(psi, self.n, q, _ry(th))
        return psi

    def apply(self, psi, beta):
        for q, th in enumerate(self.thetas):
            U = _ry(th) @ _rz(-2 * beta) @ _ry(-th)
            psi = apply_single_qubit(psi, self.n, q, U)
        return psi


class XYMixer:
    """Hamming-weight-preserving XY mixer, simulated inside the weight-k subspace.

    topology "ring": the parity-ordered ring of Hadfield et al., n two-qubit gates
        per layer, nearest-neighbour on a line plus one wrap-around.
    topology "complete": every pair. Its continuous-time generator has the Dicke
        state as ground state, which is what an adiabatic schedule needs.
    """

    name = "xy"

    def __init__(self, n: int, k: int, topology: str = "ring"):
        self.n, self.k, self.topology = n, k, topology
        self.states = weight_k_states(n, k)
        position = {int(s): i for i, s in enumerate(self.states)} if n > 24 else None
        if position is None:
            lookup = np.full(2 ** n, -1, dtype=np.int64)
            lookup[self.states] = np.arange(len(self.states))
        if topology == "ring":
            pairs = [(q, (q + 1) % n) for q in range(0, n, 2)] + \
                    [(q, (q + 1) % n) for q in range(1, n, 2)]
            pairs = [p for p in pairs if p[0] != p[1]]
            if n == 2:
                pairs = [(0, 1)]
        elif topology == "complete":
            pairs = list(combinations(range(n), 2))
        else:
            raise ValueError(topology)
        self.pairs = []
        for a, b in pairs:
            bit_a = (self.states >> a) & 1
            bit_b = (self.states >> b) & 1
            src = np.nonzero((bit_a == 1) & (bit_b == 0))[0]
            partner_states = self.states[src] ^ ((1 << a) | (1 << b))
            dst = (np.array([position[int(s)] for s in partner_states])
                   if position is not None else lookup[partner_states])
            self.pairs.append((src, dst))

    def initial(self):
        return np.full(len(self.states), 1 / np.sqrt(len(self.states)), dtype=complex)

    def apply(self, psi, beta):
        # exp(-i beta (XX + YY)/2) acts as a rotation on each |01>, |10> pair.
        c, s = np.cos(beta), -1j * np.sin(beta)
        psi = psi.copy()
        for src, dst in self.pairs:
            a, b = psi[src], psi[dst]
            psi[src], psi[dst] = c * a + s * b, s * a + c * b
        return psi


def evolve(energies: np.ndarray, mixer, gammas, betas) -> np.ndarray:
    """|psi(gamma, beta)> = prod_l U_M(beta_l) U_C(gamma_l) |psi_0>."""
    psi = mixer.initial()
    for g, b in zip(gammas, betas):
        psi = psi * np.exp(-1j * g * energies)
        psi = mixer.apply(psi, b)
    return psi


def expectation(probs, energies) -> float:
    return float(probs @ energies)


def cvar(probs, energies, alpha: float, order: np.ndarray | None = None) -> float:
    """Conditional value at risk: mean energy of the lowest alpha of probability mass.

    Barkoutsos et al. (2020). alpha = 1 is the plain expectation. Small alpha tells
    the optimizer to care only about the good tail of the distribution, which is
    what you keep when you sample and take the best shot.
    """
    if alpha >= 1:
        return expectation(probs, energies)
    order = np.argsort(energies) if order is None else order
    p = probs[order]
    e = energies[order]
    cum = np.cumsum(p)
    cut = int(np.searchsorted(cum, alpha))
    mass_before = cum[cut - 1] if cut > 0 else 0.0
    total = (p[:cut] @ e[:cut]) + (alpha - mass_before) * e[cut]
    return float(total / alpha)
