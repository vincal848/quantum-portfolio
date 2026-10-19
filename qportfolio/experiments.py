"""The experiments behind the README tables and figures."""

from __future__ import annotations

import numpy as np

from . import models, solvers
from .data import Market

QUANTUM = [
    ("qaoa_x", dict(mixer="x")),
    ("qaoa_x_cvar", dict(mixer="x", alpha=0.1)),
    ("qaoa_warm", dict(mixer="warm")),
    ("qaoa_xy", dict(mixer="xy")),
]


def default_problems(market: Market) -> list:
    """The seven models at 10-12 qubits on one market."""
    n = market.n
    current = np.zeros(n)
    current[:4] = 1                                   # an arbitrary existing book
    return [
        models.selection(market, k=3),
        models.allocation(market.subset(range(5)), bits=2, budget=4),
        models.tracking(market, k=4),
        models.long_short(market.subset(range(10))),
        models.diversified(market, k=3, threshold=0.35),
        models.rebalance(market, current, k=4),
        models.skew_aware(market, k=3, kappa=1.0),
    ]


def _row(problem, ref, result):
    info = result.info
    return {
        "model": problem.name, "solver": result.solver, "qubits": problem.n_qubits,
        "selection": problem.describe(result.x), "objective": result.objective,
        "ratio": ref.ratio(result.objective) if result.feasible else float("nan"),
        "found_optimum": bool(result.feasible and abs(result.objective - ref.best) < 1e-9),
        "p_feasible": info.get("p_feasible"), "p_optimal": info.get("p_optimal"),
        "amplification": info.get("amplification"), "seconds": result.seconds,
    }


def model_zoo(market: Market, p: int = 3, seed: int = 0, verbose: bool = True,
              only: str | None = None) -> dict:
    out = {"problems": [], "rows": []}
    for problem in default_problems(market):
        if only and problem.name != only:
            continue
        exact, ref = solvers.exact(problem)
        out["problems"].append({
            "model": problem.name, "description": problem.description,
            "qubits": problem.n_qubits, "degree": problem.cost.degree,
            "feasible_portfolios": ref.n_feasible, "optimum": ref.best,
            "selection": problem.describe(exact.x),
            "xy_mixer": problem.hamming_weight is not None})
        results = [exact, solvers.simulated_annealing(problem, seed=seed)]
        for label, kw in QUANTUM:
            if kw["mixer"] == "xy" and problem.hamming_weight is None:
                continue
            results.append(solvers.qaoa(problem, ref, p=p, seed=seed, label=label, **kw))
        results.append(solvers.adiabatic(problem, ref, mixer="x"))
        if problem.hamming_weight is not None:
            results.append(solvers.adiabatic(problem, ref, mixer="xy"))
        rows = [_row(problem, ref, r) for r in results]
        out["rows"].extend(rows)
        if verbose:
            print_rows(problem, ref, rows)
    return out


def print_rows(problem, ref, rows):
    print("\n%s - %s  [%d qubits, %d feasible portfolios]"
          % (problem.name, problem.description, problem.n_qubits, ref.n_feasible))
    print("  optimum: %s  (%s = %.5f)" % (rows[0]["selection"], problem.objective_label,
                                          ref.best))
    print("  %-20s%8s%10s%10s%10s%9s" % ("solver", "ratio", "P(valid)", "P(opt)",
                                          "x random", "seconds"))
    for r in rows:
        fmt = lambda v, f: ("%" + f) % v if v is not None else "-"
        print("  %-20s%8.4f%10s%10s%10s%9.2f" % (
            r["solver"], r["ratio"], fmt(r["p_feasible"], ".3f"), fmt(r["p_optimal"], ".4f"),
            fmt(r["amplification"], ".1f"), r["seconds"]))


def depth_study(market: Market, depths=range(1, 7), k: int = 3, seed: int = 0,
                verbose: bool = True) -> list:
    """P(optimal) and P(feasible) against circuit depth, for each mixer."""
    problem = models.selection(market, k=k)
    _, ref = solvers.exact(problem)
    rows = []
    for mixer in ("x", "warm", "xy"):
        for p in depths:
            r = solvers.qaoa(problem, ref, p=p, mixer=mixer, seed=seed)
            rows.append({"mixer": mixer, "p": p, "p_optimal": r.info["p_optimal"],
                         "p_feasible": r.info["p_feasible"],
                         "amplification": r.info["amplification"],
                         "expected_ratio": r.info["expected_ratio_given_feasible"]})
            if verbose:
                print("  %-5s p=%d  P(opt) %.4f  P(valid) %.3f  E[ratio|valid] %.3f"
                      % (mixer, p, r.info["p_optimal"], r.info["p_feasible"],
                         r.info["expected_ratio_given_feasible"]))
    return rows


def scaling_study(make_market, sizes=range(6, 21, 2), seeds=(1, 2, 3), p: int = 3,
                  max_x_qubits: int = 16, verbose: bool = True) -> list:
    """How the probability of the optimum and its amplification hold up with size."""
    rows = []
    for n in sizes:
        k = max(2, n // 4)
        for mixer in ("x", "xy"):
            if mixer == "x" and n > max_x_qubits:
                continue
            p_opt, amp, secs, hits = [], [], [], 0
            for seed in seeds:
                problem = models.selection(make_market(n, seed), k=k)
                _, ref = solvers.exact(problem)
                r = solvers.qaoa(problem, ref, p=p, mixer=mixer, seed=seed, restarts=3)
                p_opt.append(r.info["p_optimal"])
                amp.append(r.info["amplification"])
                secs.append(r.seconds)
                hits += int(abs(r.objective - ref.best) < 1e-9)
            rows.append({"n": n, "k": k, "mixer": mixer, "feasible": ref.n_feasible,
                         "hilbert_dim": r.info["hilbert_dim"],
                         "p_optimal": float(np.mean(p_opt)),
                         "amplification": float(np.mean(amp)),
                         "found_in_1024_shots": hits, "trials": len(seeds),
                         "seconds": float(np.mean(secs))})
            if verbose:
                x = rows[-1]
                print("  n=%2d k=%d %-3s states %7d  P(opt) %.4f  x%6.1f over random  "
                      "found %d/%d  %.1fs" % (n, k, mixer, x["hilbert_dim"], x["p_optimal"],
                                             x["amplification"], hits, len(seeds),
                                             x["seconds"]))
    return rows


def penalty_study(market: Market, scales=(0.05, 0.1, 0.25, 0.5, 1.0), k: int = 3,
                  p: int = 3, seed: int = 0, verbose: bool = True) -> list:
    """X-mixer QAOA against the penalty weight: the knob the XY mixer removes."""
    rows = []
    for s in scales:
        problem = models.selection(market, k=k, penalty_scale=s)
        _, ref = solvers.exact(problem)
        r = solvers.qaoa(problem, ref, p=p, mixer="x", seed=seed)
        H = problem.hamiltonian()
        ground = int(np.argmin(H.evaluate_states(np.arange(2 ** problem.n_qubits))))
        rows.append({"scale": s, "p_optimal": r.info["p_optimal"],
                     "p_feasible": r.info["p_feasible"],
                     "ground_state_is_optimal": bool(ground in set(ref.optimal_states))})
        if verbose:
            x = rows[-1]
            print("  scale %.2f  P(opt) %.4f  P(valid) %.3f  ground state optimal: %s"
                  % (s, x["p_optimal"], x["p_feasible"], x["ground_state_is_optimal"]))
    return rows


def anneal_study(market: Market, times=(5, 10, 20, 40, 80, 160), k: int = 3,
                 verbose: bool = True) -> list:
    """Adiabatic evolution: P(optimal) against total anneal time, for both mixers."""
    problem = models.selection(market, k=k)
    _, ref = solvers.exact(problem)
    rows = []
    for T in times:
        for mixer in ("x", "xy"):
            r = solvers.adiabatic(problem, ref, steps=int(4 * T), total_time=T, mixer=mixer)
            rows.append({"time": T, "mixer": mixer, "p_optimal": r.info["p_optimal"],
                         "p_feasible": r.info["p_feasible"]})
        if verbose:
            print("  T=%4d  X mixer P(opt) %.4f   XY mixer P(opt) %.4f"
                  % (T, rows[-2]["p_optimal"], rows[-1]["p_optimal"]))
    return rows


def frontier(market: Market, k: int = 3, lams=None) -> list:
    """Exact selection across risk aversion, with the XY-QAOA answer alongside."""
    from .metrics import portfolio_return, portfolio_volatility

    lams = np.geomspace(0.1, 30, 16) if lams is None else lams
    points = []
    for lam in lams:
        problem = models.selection(market, k=k, lam=float(lam))
        best, ref = solvers.exact(problem)
        q = solvers.qaoa(problem, ref, p=3, mixer="xy", restarts=2)
        w = problem.weights(best.x)
        points.append({"lambda": float(lam), "selection": problem.describe(best.x),
                       "return": portfolio_return(w, market.mu),
                       "volatility": portfolio_volatility(w, market.sigma),
                       "qaoa_xy_found": bool(abs(q.objective - ref.best) < 1e-9),
                       "qaoa_xy_p_optimal": q.info["p_optimal"]})
    return points
