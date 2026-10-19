"""Solvers: exact enumeration, simulated annealing, and an exact statevector
simulation of QAOA (X, XY and warm-start mixers, mean or CVaR loss) and adiabatic
evolution.

Bit order: state b has x_i = (b >> i) & 1, so asset i is qubit i (Qiskit's order).
"""

from itertools import combinations

import numpy as np
from scipy.optimize import minimize


def states(n, k=None):
    s = np.arange(2 ** n)
    return s if k is None else s[np.bitwise_count(s) == k]


def bits(n, s):
    return ((np.asarray(s)[:, None] >> np.arange(n)) & 1).astype(float)


def penalty_weight(model):
    # Any violation costs >= 1 x weight, more than the whole cost range, so the
    # penalized ground state is always the best valid portfolio.
    c = model.cost(bits(model.n, states(model.n)))
    return 1.01 * (c.max() - c.min()) + 1e-12


def exact(model):
    X = bits(model.n, states(model.n, model.k))
    X = X[model.penalty(X) == 0]
    c = model.cost(X)
    best = c.min()
    return {"x": X[c.argmin()], "best": best, "worst": c.max(), "n_valid": len(X),
            "n_optimal": int((np.abs(c - best) < 1e-9).sum())}


def ratio(ref, value):
    """1 at the best valid portfolio, 0 at the worst."""
    return (ref["worst"] - value) / (ref["worst"] - ref["best"])


# --- simulated annealing --------------------------------------------------------

def annealing(model, sweeps=300, restarts=4, seed=0):
    """Metropolis on the penalized energy. For fixed-k models a move swaps a held
    name for an unheld one, so the count never changes (the XY mixer's twin)."""
    rng = np.random.default_rng(seed)
    swap = model.k is not None
    P = penalty_weight(model)
    energy = lambda x: (model.cost(x[None]) + P * model.penalty(x[None], swap))[0]
    steps = sweeps * model.n
    best_x, best = None, np.inf

    for _ in range(restarts):
        if swap:
            x = np.zeros(model.n)
            x[rng.choice(model.n, model.k, replace=False)] = 1
        else:
            x = rng.integers(0, 2, model.n).astype(float)
        e = energy(x)
        t0 = max(np.mean([abs(energy(_move(x, rng, swap)) - e) for _ in range(20)]), 1e-12)
        for step in range(steps):
            y = _move(x, rng, swap)
            ey = energy(y)
            if ey <= e or rng.random() < np.exp(-(ey - e) / (t0 * 1e-3 ** (step / steps))):
                x, e = y, ey
                value = model.cost(x[None])[0]
                if model.penalty(x[None])[0] == 0 and value < best:
                    best_x, best = x.copy(), value
    return {"x": best_x, "objective": best}


def _move(x, rng, swap):
    y = x.copy()
    if swap:
        y[rng.choice(np.flatnonzero(x == 1))] = 0
        y[rng.choice(np.flatnonzero(x == 0))] = 1
    else:
        i = rng.integers(len(y))
        y[i] = 1 - y[i]
    return y


# --- mixers ---------------------------------------------------------------------

def _single_qubit(psi, q, U):
    v = psi.reshape(-1, 2, 2 ** q)          # (higher bits, bit q, lower bits)
    return np.stack([U[0, 0] * v[:, 0] + U[0, 1] * v[:, 1],
                     U[1, 0] * v[:, 0] + U[1, 1] * v[:, 1]], 1).reshape(-1)


def _ry(t):
    return np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]])


class XMixer:
    """Standard QAOA: start in |+>^n, mix with exp(-i beta X) on every qubit."""

    def __init__(self, n):
        self.n, self.states = n, states(n)

    def start(self):
        return np.full(2 ** self.n, 2 ** (-self.n / 2), dtype=complex)

    def apply(self, psi, beta):
        U = np.array([[np.cos(beta), -1j * np.sin(beta)], [-1j * np.sin(beta), np.cos(beta)]])
        for q in range(self.n):
            psi = _single_qubit(psi, q, U)
        return psi


class WarmMixer(XMixer):
    """Warm start (Egger et al. 2021): each qubit tilted toward a relaxed solution c,
    mixer RY(t) RZ(-2 beta) RY(-t) with t = 2 arcsin(sqrt(c))."""

    def __init__(self, n, relaxed):
        super().__init__(n)
        self.t = 2 * np.arcsin(np.sqrt(np.clip(relaxed, 0.25, 0.75)))   # never pin a qubit

    def start(self):
        psi = np.zeros(2 ** self.n, dtype=complex)
        psi[0] = 1
        for q, t in enumerate(self.t):
            psi = _single_qubit(psi, q, _ry(t))
        return psi

    def apply(self, psi, beta):
        rz = np.diag([np.exp(1j * beta), np.exp(-1j * beta)])
        for q, t in enumerate(self.t):
            psi = _single_qubit(psi, q, _ry(t) @ rz @ _ry(-t))
        return psi


class XYMixer:
    """Hadfield et al. 2019: start in the Dicke state (every string with k ones,
    equal weight) and mix with (XX + YY)/2 swaps, which never change the count.
    Only the C(n, k) valid amplitudes are stored."""

    def __init__(self, n, k, ring=True):
        self.states = states(n, k)
        if ring:
            pairs = [(q, (q + 1) % n) for q in range(0, n, 2)] + [(q, (q + 1) % n) for q in range(1, n, 2)]
        else:
            pairs = list(combinations(range(n), 2))
        index = np.full(2 ** n, -1)
        index[self.states] = np.arange(len(self.states))
        self.swaps = []
        for a, b in dict.fromkeys(tuple(sorted(p)) for p in pairs):
            src = np.flatnonzero(((self.states >> a) & 1) > ((self.states >> b) & 1))
            self.swaps.append((src, index[self.states[src] ^ (1 << a | 1 << b)]))

    def start(self):
        return np.full(len(self.states), len(self.states) ** -0.5, dtype=complex)

    def apply(self, psi, beta):
        c, s = np.cos(beta), -1j * np.sin(beta)
        psi = psi.copy()
        for src, dst in self.swaps:
            psi[src], psi[dst] = c * psi[src] + s * psi[dst], s * psi[src] + c * psi[dst]
        return psi


def evolve(energies, mixer, gammas, betas):
    psi = mixer.start()
    for g, b in zip(gammas, betas):
        psi = mixer.apply(psi * np.exp(-1j * g * energies), b)
    return psi


def cvar(probs, energies, alpha):
    """Mean energy of the lowest alpha of the probability mass (Barkoutsos et al. 2020)."""
    if alpha >= 1:
        return probs @ energies
    order = np.argsort(energies)
    p, e = probs[order], energies[order]
    cut = np.searchsorted(np.cumsum(p), alpha)
    return (p[:cut] @ e[:cut] + (alpha - p[:cut].sum()) * e[cut]) / alpha


# --- quantum solvers ------------------------------------------------------------

def _setup(model, mixer, ring=True):
    """Mixer plus energies, costs and validity for every state it can reach."""
    P = penalty_weight(model)
    if mixer == "xy":
        assert model.k is not None, "the XY mixer needs a fixed number of holdings"
        mix = XYMixer(model.n, model.k, ring)
    elif mixer == "warm":
        H = lambda x: model.cost(x[None])[0] + P * model.penalty(x[None])[0]
        relaxed = minimize(H, np.full(model.n, 0.5), bounds=[(0, 1)] * model.n).x
        mix = WarmMixer(model.n, relaxed)
    else:
        mix = XMixer(model.n)
    X = bits(model.n, mix.states)
    cost = model.cost(X)
    E = cost + P * model.penalty(X, count_enforced=mixer == "xy")
    # Standardized so the same angle range works for every model.
    return mix, X, cost, model.penalty(X) == 0, (E - E.mean()) / (E.std() or 1)


def qaoa(model, ref, mixer="x", p=3, alpha=1.0, restarts=4, seed=0):
    """Train depth p. Depth 1 from a ramp and random starts; each extra layer starts
    from the last optimum stretched onto one more layer (INTERP, Zhou et al. 2020)."""
    rng = np.random.default_rng(seed)
    mix, X, cost, valid, E = _setup(model, mixer)
    loss = lambda t: cvar(np.abs(evolve(E, mix, t[:len(t) // 2], t[len(t) // 2:])) ** 2, E, alpha)

    starts = [np.array([0.4, 0.4])] + [rng.uniform(0, np.pi / 2, 2) for _ in range(restarts - 1)]
    best = min((minimize(loss, x0, method="COBYLA", options={"maxiter": 400}) for x0 in starts),
               key=lambda r: r.fun)
    for depth in range(1, p):
        x0 = np.concatenate([_interp(best.x[:depth]), _interp(best.x[depth:])])
        best = minimize(loss, x0, method="COBYLA", options={"maxiter": 400})

    half = len(best.x) // 2
    probs = np.abs(evolve(E, mix, best.x[:half], best.x[half:])) ** 2
    return _read_out(ref, X, cost, valid, probs, rng)


def _interp(params):
    p = len(params)
    padded = np.concatenate([[0], params, [0]])
    return np.array([(i * padded[i] + (p - i) * padded[i + 1]) / p for i in range(p + 1)])


def adiabatic(model, ref, mixer="x", total_time=20.0, steps=40, seed=0):
    """Slow linear sweep from the mixer to the cost: QAOA with fixed angles,
    gamma = s dt and beta = -(1 - s) dt. The XY version uses every pair, because
    that mixer's ground state is the Dicke state the circuit starts in."""
    mix, X, cost, valid, E = _setup(model, mixer, ring=False)
    dt = total_time / steps
    s = (np.arange(steps) + 0.5) / steps
    betas = -(1 - s) * dt
    if mixer == "xy":
        betas *= 2 / model.n          # every pair hops; keep dt comparable to the X case
    probs = np.abs(evolve(E, mix, s * dt, betas)) ** 2
    return _read_out(ref, X, cost, valid, probs, np.random.default_rng(seed))


def _read_out(ref, X, cost, valid, probs, rng, shots=1024):
    """Best valid shot out of 1024, plus exact P(valid) and P(optimal)."""
    probs = probs / probs.sum()
    shots = rng.choice(len(probs), shots, p=probs)
    good = shots[valid[shots]]
    pick = good[np.argmin(cost[good])] if len(good) else shots[0]
    p_opt = probs[valid & (np.abs(cost - ref["best"]) < 1e-9)].sum()
    return {"x": X[pick], "objective": cost[pick], "valid": bool(valid[pick]),
            "p_valid": probs[valid].sum(), "p_optimal": p_opt,
            "amplification": p_opt * ref["n_valid"] / ref["n_optimal"], "dim": len(probs)}
