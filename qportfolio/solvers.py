"""Solvers. Every one takes a `Problem` and returns a `Result` in the same shape.

Classical:
    exact                 enumerate the feasible set; the reference answer
    simulated_annealing   the classical heuristic a practitioner would reach for

Quantum (exact statevector simulation):
    qaoa                  QAOA with an X, XY or warm-start mixer, expectation or CVaR
                          objective, layer-by-layer INTERP parameter growth
    adiabatic             Trotterized adiabatic evolution: a fixed linear schedule
                          through the same circuit, nothing optimized

Quantum results also carry the exact output distribution's key numbers:
p_feasible (probability a shot is a valid portfolio), p_optimal (probability a shot
is THE optimal portfolio) and amplification (p_optimal divided by what uniform
sampling of the feasible set would give).
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
    """Exact facts about a problem's feasible set, used to score other solvers."""

    x: np.ndarray
    best: float
    worst: float
    n_feasible: int
    optimal_states: np.ndarray

    def ratio(self, value: float) -> float:
        """1 at the optimum, 0 at the worst feasible portfolio."""
        span = self.worst - self.best
        return 1.0 if span <= 0 else float((self.worst - value) / span)


# --- classical ------------------------------------------------------------------

def exact(problem: Problem) -> tuple[Result, Reference]:
    start = time.perf_counter()
    states = problem.feasible_states()
    if len(states) == 0:
        raise ValueError("%s has no feasible portfolio" % problem.name)
    values = problem.cost.evaluate_states(states)
    i = int(np.argmin(values))
    best = float(values[i])
    x = bit_table(problem.n_qubits, states[i:i + 1])[0]
    seconds = time.perf_counter() - start
    ref = Reference(x=x, best=best, worst=float(values.max()), n_feasible=len(states),
                    optimal_states=states[np.abs(values - best) < 1e-9])
    return Result("exact", x, best, True, seconds, {"n_feasible": len(states)}), ref


def simulated_annealing(problem: Problem, sweeps: int = 300, restarts: int = 4,
                        seed: int = 0) -> Result:
    """Metropolis annealing on the penalized energy.

    For fixed-cardinality models it moves by swapping a held asset for an unheld one,
    so it never leaves the feasible cardinality, which is the classical analogue of
    the XY mixer and the fair comparison for it.
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

        # Temperature scale from the typical size of a move.
        deltas = []
        for _ in range(20):
            y = _move(x, rng, swap)
            deltas.append(abs(H.evaluate(y) - e))
        t0 = max(np.mean(deltas), 1e-12)
        for step in range(steps):
            t = t0 * (1e-3 ** (step / max(steps - 1, 1)))
            y = _move(x, rng, swap)
            ey = H.evaluate(y)
            if ey <= e or rng.random() < np.exp(-(ey - e) / t):
                x, e = y, ey
                if problem.feasible(x):
                    v = problem.objective(x)
                    if v < best_val:
                        best_x, best_val = x.copy(), v

    feasible = best_x is not None
    if not feasible:
        best_x = x
        best_val = problem.objective(x)
    return Result("simulated_annealing", best_x, best_val, feasible,
                  time.perf_counter() - start, {"steps": steps * restarts})


def _move(x, rng, swap):
    y = x.copy()
    if swap:
        ones, zeros = np.flatnonzero(y), np.flatnonzero(y == 0)
        if len(ones) and len(zeros):
            y[rng.choice(ones)] = 0
            y[rng.choice(zeros)] = 1
    else:
        y[rng.integers(len(y))] ^= 1
    return y


# --- quantum ----------------------------------------------------------------------

def _relaxation(problem: Problem) -> np.ndarray:
    """Continuous relaxation of the penalized energy over [0,1]^n, for warm starts."""
    H = problem.hamiltonian()
    n = problem.n_qubits
    out = minimize(lambda x: float(H.evaluate(x)), np.full(n, 0.5),
                   jac=H.gradient, bounds=[(0, 1)] * n, method="L-BFGS-B")
    return out.x


def _build_mixer(problem: Problem, mixer: str, topology: str = "ring"):
    n = problem.n_qubits
    if mixer == "x":
        return XMixer(n), False
    if mixer == "xy":
        if problem.hamming_weight is None:
            raise ValueError("%s has no fixed Hamming weight; the XY mixer does not apply"
                             % problem.name)
        return XYMixer(n, problem.hamming_weight, topology), True
    if mixer == "warm":
        return WarmStartMixer(n, _relaxation(problem)), False
    raise ValueError(mixer)


class _Landscape:
    """Energies, feasibility and the reference, laid out over the mixer's state space."""

    def __init__(self, problem: Problem, mixer, preserve: bool, ref: Reference):
        H = problem.hamiltonian(preserve_hamming_weight=preserve)
        states = mixer.states
        self.states = states
        self.energies = H.evaluate_states(states)
        # Normalize so circuit angles are O(1) whatever the units of the objective.
        self.scale = float(np.std(self.energies)) or 1.0
        self.shift = float(np.mean(self.energies))
        self.scaled = (self.energies - self.shift) / self.scale
        self.order = np.argsort(self.scaled)
        bits = None
        self.feasible = np.zeros(len(states), dtype=bool)
        for start in range(0, len(states), 1 << 16):
            bits = bit_table(problem.n_qubits, states[start:start + (1 << 16)])
            self.feasible[start:start + len(bits)] = problem.feasible(bits)
        self.objective = problem.cost.evaluate_states(states)
        self.optimal = np.isin(states, ref.optimal_states)


def qaoa(problem: Problem, ref: Reference, p: int = 3, mixer: str = "x",
         alpha: float = 1.0, restarts: int = 4, shots: int = 1024, maxiter: int = 400,
         seed: int = 0, topology: str = "ring", label: str | None = None) -> Result:
    """QAOA with layerwise INTERP growth (Zhou et al. 2020).

    p = 1 is optimized from a ramp and from `restarts - 1` random points; each deeper
    level starts from the previous optimum interpolated onto one more layer. That
    makes the energy non-increasing in p, which random initialization at every depth
    does not guarantee.
    """
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    mix, preserve = _build_mixer(problem, mixer, topology)
    land = _Landscape(problem, mix, preserve, ref)

    def loss(theta):
        k = len(theta) // 2
        psi = evolve(land.scaled, mix, theta[:k], theta[k:])
        return cvar(np.abs(psi) ** 2, land.scaled, alpha, land.order)

    evals = 0
    best = None
    inits = [np.array([0.4, 0.4])] + [rng.uniform(0, np.pi / 2, 2) for _ in range(restarts - 1)]
    for x0 in inits:
        out = minimize(loss, x0, method="COBYLA", options={"maxiter": maxiter})
        evals += out.nfev
        if best is None or out.fun < best.fun:
            best = out
    history = [float(best.fun)]
    for depth in range(2, p + 1):
        k = depth - 1
        g, b = best.x[:k], best.x[k:]
        x0 = np.concatenate([_interp(g), _interp(b)])
        out = minimize(loss, x0, method="COBYLA", options={"maxiter": maxiter})
        evals += out.nfev
        best = out
        history.append(float(best.fun))

    k = len(best.x) // 2
    probs = np.abs(evolve(land.scaled, mix, best.x[:k], best.x[k:])) ** 2
    result = _readout(problem, ref, land, probs, shots, rng)
    result.solver = label or ("qaoa_%s%s" % (mixer, "" if alpha >= 1 else "_cvar"))
    result.seconds = time.perf_counter() - start
    result.info.update({"p": p, "mixer": mixer, "alpha": alpha, "evaluations": evals,
                        "gammas": best.x[:k].tolist(), "betas": best.x[k:].tolist(),
                        "loss_by_depth": history})
    return result


def _interp(params):
    """INTERP: values for p+1 layers from the optimum at p, by linear interpolation."""
    p = len(params)
    padded = np.concatenate([[0.0], params, [0.0]])
    return np.array([(i / p) * padded[i] + ((p - i) / p) * padded[i + 1]
                     for i in range(p + 1)])


def adiabatic(problem: Problem, ref: Reference, steps: int = 40, total_time: float = 20.0,
              mixer: str = "x", shots: int = 1024, seed: int = 0) -> Result:
    """Trotterized adiabatic evolution, H(s) = (1 - s) H_M + s H_C, s linear in time.

    Starts in the mixer's ground state and moves slowly toward the cost Hamiltonian;
    by the adiabatic theorem it ends near the cost ground state if the schedule is
    slow relative to the minimum spectral gap. It is QAOA with fixed angles:
    gamma_j = s_j dt, beta_j = -(1 - s_j) dt. For the XY version the complete-graph
    mixer is used, because its ground state is the Dicke state the circuit starts in.
    """
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    mix, preserve = _build_mixer(problem, mixer, "complete")
    land = _Landscape(problem, mix, preserve, ref)
    dt = total_time / steps
    s = (np.arange(steps) + 0.5) / steps
    gammas, betas = s * dt, -(1 - s) * dt
    if mixer == "xy":
        # Complete-graph XY hopping, normalized by n so dt means the same thing.
        betas = betas / problem.n_qubits * 2
    probs = np.abs(evolve(land.scaled, mix, gammas, betas)) ** 2
    result = _readout(problem, ref, land, probs, shots, rng)
    result.solver = "adiabatic_%s" % mixer
    result.seconds = time.perf_counter() - start
    result.info.update({"steps": steps, "total_time": total_time, "mixer": mixer})
    return result


def _readout(problem, ref, land, probs, shots, rng) -> Result:
    """Sample shots, keep the best feasible one, and record exact distribution stats."""
    probs = probs / probs.sum()
    draws = rng.choice(len(probs), size=shots, p=probs)
    feasible_draws = draws[land.feasible[draws]]
    if len(feasible_draws):
        pick = feasible_draws[np.argmin(land.objective[feasible_draws])]
        feasible = True
    else:
        pick = draws[np.argmax(np.bincount(draws))]
        feasible = False
    x = bit_table(problem.n_qubits, land.states[pick:pick + 1])[0]
    p_opt = float(probs[land.optimal].sum())
    p_feas = float(probs[land.feasible].sum())
    uniform = len(ref.optimal_states) / ref.n_feasible
    exp_ratio = (float(probs[land.feasible] @ land.objective[land.feasible]) / p_feas
                 if p_feas > 0 else float("nan"))
    return Result("", x, float(land.objective[pick]), feasible, 0.0, {
        "p_feasible": p_feas,
        "p_optimal": p_opt,
        "amplification": p_opt / uniform,
        "expected_ratio_given_feasible": ref.ratio(exp_ratio) if p_feas > 0 else float("nan"),
        "most_likely_feasible": bool(land.feasible[int(np.argmax(probs))]),
        "n_qubits": problem.n_qubits,
        "hilbert_dim": len(land.states),
    })
