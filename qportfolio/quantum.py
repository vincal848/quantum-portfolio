"""Exact statevector simulation of QAOA-style circuits with a diagonal cost.

Every model here gives a diagonal Hamiltonian, so the cost layer is just an
elementwise phase exp(-i gamma E). The variants differ only in the mixer and the
starting state:

XMixer
    Standard QAOA (Farhi et al. 2014). Starts in |+>^n, mixes with exp(-i beta X)
    on every qubit.
XYMixer
    The alternating operator ansatz of Hadfield et al. (2019). Starts in the Dicke
    state (equal superposition of all strings with k ones) and mixes with XX + YY
    swaps, which never change the number of ones. We store only those C(n,k)
    amplitudes, which is why 20 qubits runs in seconds.
WarmStartMixer
    Egger et al. (2021). Each qubit starts tilted toward the answer of a continuous
    relaxation, and the mixer is rotated to match.

Probabilities come out exact, so P(optimal), P(feasible) and CVaR have no shot noise.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .problem import weight_k_states


def ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(phi):
    return np.diag([np.exp(-0.5j * phi), np.exp(0.5j * phi)])


def apply_single_qubit(psi, n, q, U):
    """Apply a 2x2 unitary to qubit q of a 2^n statevector."""
    # Index b splits as (high bits, bit q, low bits), which is exactly this reshape.
    v = psi.reshape(-1, 2, 2 ** q)
    out = np.empty_like(v)
    out[:, 0, :] = U[0, 0] * v[:, 0, :] + U[0, 1] * v[:, 1, :]
    out[:, 1, :] = U[1, 0] * v[:, 0, :] + U[1, 1] * v[:, 1, :]
    return out.reshape(-1)


class XMixer:
    name = "x"

    def __init__(self, n):
        self.n = n
        self.states = np.arange(2 ** n, dtype=np.int64)

    def initial(self):
        return np.full(2 ** self.n, 2 ** (-self.n / 2), dtype=complex)

    def apply(self, psi, beta):
        c, s = np.cos(beta), -1j * np.sin(beta)
        U = np.array([[c, s], [s, c]])
        for q in range(self.n):
            psi = apply_single_qubit(psi, self.n, q, U)
        return psi


class WarmStartMixer(XMixer):
    """Mixer RY(theta) RZ(-2 beta) RY(-theta) per qubit, theta from the relaxation."""

    name = "warm"

    def __init__(self, n, relaxed, epsilon=0.25):
        super().__init__(n)
        # Clipping keeps every qubit away from |0> or |1>; a fully pinned qubit
        # can never be moved by the mixer again.
        self.relaxed = np.clip(np.asarray(relaxed, dtype=float), epsilon, 1 - epsilon)
        self.thetas = 2 * np.arcsin(np.sqrt(self.relaxed))

    def initial(self):
        psi = np.zeros(2 ** self.n, dtype=complex)
        psi[0] = 1.0
        for q, theta in enumerate(self.thetas):
            psi = apply_single_qubit(psi, self.n, q, ry(theta))
        return psi

    def apply(self, psi, beta):
        for q, theta in enumerate(self.thetas):
            psi = apply_single_qubit(psi, self.n, q, ry(theta) @ rz(-2 * beta) @ ry(-theta))
        return psi


class XYMixer:
    """Hamming-weight-preserving mixer, simulated inside the weight-k subspace.

    "ring" is the parity-ordered ring from Hadfield et al.: n two-qubit gates per
    layer, nearest neighbours plus one wrap-around. "complete" uses every pair. Its
    ground state is the Dicke state, which is what an adiabatic schedule needs.
    """

    name = "xy"

    def __init__(self, n, k, topology="ring"):
        self.n, self.k, self.topology = n, k, topology
        self.states = weight_k_states(n, k)

        if topology == "ring":
            if n == 2:
                pairs = [(0, 1)]
            else:
                even = [(q, (q + 1) % n) for q in range(0, n, 2)]
                odd = [(q, (q + 1) % n) for q in range(1, n, 2)]
                pairs = even + odd
        elif topology == "complete":
            pairs = list(combinations(range(n), 2))
        else:
            raise ValueError("unknown topology %r" % topology)

        # For each pair, precompute which subspace entries swap with which.
        position = np.full(2 ** n, -1, dtype=np.int64)
        position[self.states] = np.arange(len(self.states))
        self.pairs = []
        for a, b in pairs:
            has_a = (self.states >> a) & 1
            has_b = (self.states >> b) & 1
            src = np.flatnonzero((has_a == 1) & (has_b == 0))
            dst = position[self.states[src] ^ ((1 << a) | (1 << b))]
            self.pairs.append((src, dst))

    def initial(self):
        return np.full(len(self.states), 1 / np.sqrt(len(self.states)), dtype=complex)

    def apply(self, psi, beta):
        # exp(-i beta (XX + YY)/2) rotates each |01>, |10> pair and leaves the rest.
        c, s = np.cos(beta), -1j * np.sin(beta)
        psi = psi.copy()
        for src, dst in self.pairs:
            a, b = psi[src], psi[dst]
            psi[src], psi[dst] = c * a + s * b, s * a + c * b
        return psi


def evolve(energies, mixer, gammas, betas):
    """Run the circuit: alternate cost phase and mixer, one pair per layer."""
    psi = mixer.initial()
    for gamma, beta in zip(gammas, betas):
        psi = mixer.apply(psi * np.exp(-1j * gamma * energies), beta)
    return psi


def cvar(probs, energies, alpha, order=None):
    """Mean energy of the lowest `alpha` of the probability mass.

    Barkoutsos et al. (2020). alpha = 1 is the ordinary expectation; a small alpha
    rewards the circuit for its best outcomes only, which matches how you use it
    (sample, keep the best shot).
    """
    if alpha >= 1:
        return float(probs @ energies)
    order = np.argsort(energies) if order is None else order
    p, e = probs[order], energies[order]
    cum = np.cumsum(p)
    cut = int(np.searchsorted(cum, alpha))
    before = cum[cut - 1] if cut > 0 else 0.0
    return float((p[:cut] @ e[:cut] + (alpha - before) * e[cut]) / alpha)
