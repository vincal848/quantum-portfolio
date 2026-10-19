"""Solvers. Each takes a Problem and returns a Result.

exact                 enumerate every feasible portfolio (the reference)
simulated_annealing   the classical heuristic you'd actually reach for
qaoa                  QAOA with an X, XY or warm-start mixer, mean or CVaR loss
adiabatic             a slow linear anneal through the same circuit, no training

The quantum solvers also report numbers read straight off the exact output
distribution: p_feasible, p_optimal, and amplification (p_optimal divided by the
odds of hitting the optimum by picking a feasible portfolio at random).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .problem import Problem, bit_table
from .quantum import WarmStartMixer, XMixer, XYMixer, cvar, evolve


@dataclass
class Result:
    solver: str
    x: np.ndarray
    objective: float
    feasible: bool
    seconds: float
    info: dict = field(default_factory=dict)

    @property
    def bitstring(self) -> str:
        return "".join(str(int(b)) for b in self.x)[::-1]


@dataclass
class Reference:
    """What exact enumeration knows about the feasible set."""

    x: np.ndarray
    best: float
    worst: float
    n_feasible: int
    optimal_states: np.ndarray

    def ratio(self, value: float) -> float:
        """1 at the best feasible portfolio, 0 at the worst."""
        span = self.worst - self.best
        return 1.0 if span <= 0 else float((self.worst - value) / span)


def exact(problem: Problem):
    """Returns (Result, Reference)."""
    start = time.perf_counter()
    states = problem.feasible_states()
    if len(states) == 0:
        raise ValueError("%s has no feasible portfolio" % problem.name)
    values = problem.cost.evaluate_states(states)
    i = int(np.argmin(values))
    best = float(values[i])
    x = bit_table(problem.n_qubits, states[i:i + 1])[0]
    ref = Reference(x, best, float(values.max()), len(states),
                    states[np.abs(values - best) < 1e-9])
    result = Result("exact", x, best, True, time.perf_counter() - start,
                    {"n_feasible": len(states)})
    return result, ref


def simulated_annealing(problem: Problem, sweeps=300, restarts=4, seed=0) -> Result:
    """Metropolis annealing on the penalized energy.

    For fixed-cardinality models a move swaps one held asset for one unheld asset,
    so the walk never changes how many names it holds. That's the classical twin of
    the XY mixer, and the fair thing to compare it with.
    """
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = problem.n_qubits
    swap = problem.hamming_weight is not None
    H = problem.hamiltonian(preserve_hamming_weight=swap)
    steps = sweeps * n

    best_x, best_val = None, np.inf
    for _ in range(restarts):
        if swap:
            x = np.zeros(n, dtype=np.int8)
            x[rng.choice(n, problem.hamming_weight, replace=False)] = 1
        else:
            x = rng.integers(0, 2, n).astype(np.int8)
        e = H.evaluate(x)

        # Start hot enough that a typical move is usually accepted.
        t0 = max(np.mean([abs(H.evaluate(_move(x, rng, swap)) - e) for _ in range(20)]), 1e-12)
        for step in range(steps):
            t = t0 * 1e-3 ** (step / max(steps - 1, 1))
            y = _move(x, rng, swap)
            ey = H.evaluate(y)
            if ey <= e or rng.random() < np.exp(-(ey - e) / t):
                x, e = y, ey
                if problem.feasible(x) and problem.objective(x) < best_val:
                    best_x, best_val = x.copy(), problem.objective(x)

    if best_x is None:
        return Result("simulated_annealing", x, problem.objective(x), False,
                      time.perf_counter() - start, {"steps": steps * restarts})
    return Result("simulated_annealing", best_x, best_val, True,
                  time.perf_counter() - start, {"steps": steps * restarts})


def _move(x, rng, swap):
    y = x.copy()
    if not swap:
        y[rng.integers(len(y))] ^= 1
        return y
    ones, zeros = np.flatnonzero(y), np.flatnonzero(y == 0)
    if len(ones) and len(zeros):
        y[rng.choice(ones)] = 0
        y[rng.choice(zeros)] = 1
    return y


def _relaxation(problem: Problem):
    """Minimize the penalized energy over the box [0,1]^n, for warm starts."""
    H = problem.hamiltonian()
    n = problem.n_qubits
    out = minimize(lambda x: float(H.evaluate(x)), np.full(n, 0.5), jac=H.gradient,
                   bounds=[(0, 1)] * n, method="L-BFGS-B")
    return out.x


def _make_mixer(problem: Problem, mixer: str, topology="ring"):
    """Returns (mixer, whether it enforces the cardinality constraint itself)."""
    n = problem.n_qubits
    if mixer == "x":
        return XMixer(n), False
    if mixer == "warm":
        return WarmStartMixer(n, _relaxation(problem)), False
    if mixer == "xy":
        if problem.hamming_weight is None:
            raise ValueError("%s has no fixed number of ones, so the XY mixer "
                             "doesn't apply" % problem.name)
        return XYMixer(n, problem.hamming_weight, topology), True
    raise ValueError("unknown mixer %r" % mixer)


class _Landscape:
    """Energies, feasibility and optimality for every state the mixer can reach."""

    def __init__(self, problem, mixer, preserve, ref):
        states = mixer.states
        self.states = states
        self.energies = problem.hamiltonian(preserve).evaluate_states(states)
        # Rescale so circuit angles mean the same thing whatever the units are.
        self.scaled = (self.energies - self.energies.mean()) / (self.energies.std() or 1.0)
        self.order = np.argsort(self.scaled)
        self.objective = problem.cost.evaluate_states(states)
        self.optimal = np.isin(states, ref.optimal_states)
        self.feasible = np.zeros(len(states), dtype=bool)
        chunk = 1 << 16
        for start in range(0, len(states), chunk):
            bits = bit_table(problem.n_qubits, states[start:start + chunk])
            self.feasible[start:start + len(bits)] = problem.feasible(bits)


def qaoa(problem: Problem, ref: Reference, p=3, mixer="x", alpha=1.0, restarts=4,
         shots=1024, maxiter=400, seed=0, topology="ring", label=None) -> Result:
    """Train a depth-p QAOA circuit and sample it.

    p = 1 is trained from a small ramp and from random starting points. Each deeper
    circuit starts from the previous optimum stretched onto one more layer (INTERP,
    Zhou et al. 2020), so going deeper never makes the loss worse.
    """
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    mix, preserve = _make_mixer(problem, mixer, topology)
    land = _Landscape(problem, mix, preserve, ref)

    def loss(theta):
        half = len(theta) // 2
        psi = evolve(land.scaled, mix, theta[:half], theta[half:])
        return cvar(np.abs(psi) ** 2, land.scaled, alpha, land.order)

    starts = [np.array([0.4, 0.4])]
    starts += [rng.uniform(0, np.pi / 2, 2) for _ in range(restarts - 1)]
    runs = [minimize(loss, x0, method="COBYLA", options={"maxiter": maxiter}) for x0 in starts]
    best = min(runs, key=lambda r: r.fun)
    evaluations = sum(r.nfev for r in runs)
    history = [float(best.fun)]

    for depth in range(2, p + 1):
        gammas, betas = best.x[:depth - 1], best.x[depth - 1:]
        x0 = np.concatenate([_interp(gammas), _interp(betas)])
        best = minimize(loss, x0, method="COBYLA", options={"maxiter": maxiter})
        evaluations += best.nfev
        history.append(float(best.fun))

    half = len(best.x) // 2
    probs = np.abs(evolve(land.scaled, mix, best.x[:half], best.x[half:])) ** 2
    result = _read_out(problem, ref, land, probs, shots, rng)
    result.solver = label or "qaoa_%s%s" % (mixer, "_cvar" if alpha < 1 else "")
    result.seconds = time.perf_counter() - start
    result.info.update(p=p, mixer=mixer, alpha=alpha, evaluations=evaluations,
                       gammas=best.x[:half].tolist(), betas=best.x[half:].tolist(),
                       loss_by_depth=history)
    return result


def _interp(params):
    """Angles for p + 1 layers from the best angles at p, by linear interpolation."""
    p = len(params)
    padded = np.concatenate([[0.0], params, [0.0]])
    return np.array([(i / p) * padded[i] + ((p - i) / p) * padded[i + 1]
                     for i in range(p + 1)])


def adiabatic(problem: Problem, ref: Reference, steps=40, total_time=20.0, mixer="x",
              shots=1024, seed=0) -> Result:
    """Trotterized anneal from the mixer's ground state to the cost's.

    H(s) = (1 - s) H_M + s H_C with s going linearly from 0 to 1. If that happens
    slowly compared with the smallest energy gap, the system ends near the ground
    state of H_C. In circuit terms it is QAOA with fixed angles,
    gamma_j = s_j dt and beta_j = -(1 - s_j) dt. The XY version uses the
    complete-graph mixer, since that's the one whose ground state is the Dicke
    state we start in.
    """
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    mix, preserve = _make_mixer(problem, mixer, "complete")
    land = _Landscape(problem, mix, preserve, ref)

    dt = total_time / steps
    s = (np.arange(steps) + 0.5) / steps
    gammas, betas = s * dt, -(1 - s) * dt
    if mixer == "xy":
        # Every pair hops in the complete graph; divide by n/2 so dt is comparable.
        betas = betas * 2 / problem.n_qubits

    probs = np.abs(evolve(land.scaled, mix, gammas, betas)) ** 2
    result = _read_out(problem, ref, land, probs, shots, rng)
    result.solver = "adiabatic_%s" % mixer
    result.seconds = time.perf_counter() - start
    result.info.update(steps=steps, total_time=total_time, mixer=mixer)
    return result


def _read_out(problem, ref, land, probs, shots, rng) -> Result:
    """Sample shots, keep the best feasible one, and record the distribution's stats."""
    probs = probs / probs.sum()
    draws = rng.choice(len(probs), size=shots, p=probs)
    good = draws[land.feasible[draws]]
    if len(good):
        pick, feasible = good[np.argmin(land.objective[good])], True
    else:
        pick, feasible = np.bincount(draws).argmax(), False

    p_feasible = float(probs[land.feasible].sum())
    p_optimal = float(probs[land.optimal].sum())
    if p_feasible > 0:
        mean_feasible = probs[land.feasible] @ land.objective[land.feasible] / p_feasible
        expected_ratio = ref.ratio(mean_feasible)
    else:
        expected_ratio = float("nan")

    x = bit_table(problem.n_qubits, land.states[pick:pick + 1])[0]
    return Result("", x, float(land.objective[pick]), feasible, 0.0, {
        "p_feasible": p_feasible,
        "p_optimal": p_optimal,
        "amplification": p_optimal * ref.n_feasible / len(ref.optimal_states),
        "expected_ratio_given_feasible": expected_ratio,
        "most_likely_feasible": bool(land.feasible[int(np.argmax(probs))]),
        "n_qubits": problem.n_qubits,
        "hilbert_dim": len(land.states),
    })
