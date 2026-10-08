"""Run the solver comparison and write results/ and docs/img/.

    python run.py                  # real data, 10 tickers, pick 3
    python run.py --synthetic      # no network needed
    python run.py --scaling        # does QAOA find the optimum as n grows?
    python run.py --frontier       # efficient frontier over the risk aversion
"""

import argparse
import dataclasses
import json
import os
import time

import numpy as np

import data as datamod
import metrics
import solvers
from portfolio import equal_weights

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
IMG = os.path.join(HERE, "docs", "img")

LAMBDA = 0.5
CARDINALITY = 3
RISK_FREE = 0.03


def timed(solver, *args, **kwargs) -> dict:
    """Run a solver and return its fields plus wall-clock `seconds`."""
    start = time.perf_counter()
    solution = solver(*args, **kwargs)
    return {**dataclasses.asdict(solution), "seconds": time.perf_counter() - start}


def get_moments(synthetic: bool, n_assets: int = 10, seed: int = 128):
    if synthetic:
        mu, sigma = datamod.synthetic_moments(n_assets, seed=seed)
        return mu, sigma, ["A%d" % i for i in range(n_assets)]
    prices = datamod.load_prices()
    return datamod.moments(prices)


def compare(mu, sigma, lam=LAMBDA, cardinality=CARDINALITY, include_qaoa=True,
            include_eigensolver=False):
    """Every solver on the same problem, scored against the exact optimum."""
    out = {"brute_force": timed(solvers.brute_force, mu, sigma, lam, cardinality),
           "greedy": timed(solvers.greedy, mu, sigma, lam, cardinality)}
    if include_qaoa:
        out["qaoa"] = timed(solvers.qaoa, mu, sigma, lam, cardinality)
    if include_eigensolver:
        try:
            out["exact_eigensolver"] = timed(
                solvers.exact_eigensolver, mu, sigma, lam, cardinality)
        except ImportError:
            pass

    optimum = out["brute_force"]["objective"]
    for name, row in out.items():
        row["approximation_ratio"] = metrics.approximation_ratio(row["objective"], optimum)
        row["found_optimum"] = row["selected"] == out["brute_force"]["selected"]
    return out


def print_comparison(results, tickers, mu, sigma, lam, cardinality):
    print("\n%d assets, pick %d, lambda=%s, risk-free %.0f%%"
          % (len(mu), cardinality, lam, RISK_FREE * 100))
    print("%-18s%-28s%12s%10s%10s%9s"
          % ("solver", "selected", "objective", "approx", "optimal?", "seconds"))
    for name, row in results.items():
        names = ", ".join(tickers[i] for i in row["selected"])
        if len(names) > 26:
            names = names[:23] + "..."
        print("%-18s%-28s%12.6f%10.4f%10s%9.3f"
              % (name, names, row["objective"], row["approximation_ratio"],
                 "yes" if row["found_optimum"] else "no", row["seconds"]))

    print("\nportfolio statistics, equal weight across the selection (annualized)")
    print("%-18s%10s%12s%10s" % ("solver", "return", "volatility", "sharpe"))
    for name, row in results.items():
        w = equal_weights(row["selected"], len(mu))
        s = metrics.summarize(w, mu, sigma, RISK_FREE)
        print("%-18s%9.2f%%%11.2f%%%10.3f"
              % (name, s["return"] * 100, s["volatility"] * 100, s["sharpe"]))


def scaling_study(max_assets=12, cardinality=3, lam=LAMBDA, seeds=(1, 2, 3, 4, 5)):
    """Does QAOA find the exact optimum, and does that hold as the problem grows?

    This is the experiment the original could not run, because it had nothing to
    compare against.
    """
    rows = []
    for n in range(4, max_assets + 1):
        hits, ratios, qubits, seconds = 0, [], None, []
        for seed in seeds:
            mu, sigma = datamod.synthetic_moments(n, seed=seed)
            exact = solvers.brute_force(mu, sigma, lam, cardinality)
            q = timed(solvers.qaoa, mu, sigma, lam, cardinality, seed=seed)
            hits += int(q["selected"] == exact["selected"])
            ratios.append(metrics.approximation_ratio(q["objective"], exact["objective"]))
            qubits = q["n_qubits"]
            seconds.append(q["seconds"])
        rows.append({"n_assets": n, "n_qubits": qubits, "trials": len(seeds),
                     "found_optimum": hits, "hit_rate": hits / len(seeds),
                     "mean_approximation_ratio": float(np.mean(ratios)),
                     "mean_seconds": float(np.mean(seconds))})
        print("  n=%2d  qubits=%2d  found optimum %d/%d  approx %.4f  %.1fs"
              % (n, qubits, hits, len(seeds), rows[-1]["mean_approximation_ratio"],
                 rows[-1]["mean_seconds"]))
    return rows


def frontier(mu, sigma, cardinality=CARDINALITY, n_points=20):
    """Efficient frontier by sweeping the risk-aversion parameter.

    Values stay numeric here. The original formatted them into strings before
    appending, so the frontier plot was drawing text labels on both axes and the
    curve it produced was meaningless.
    """
    points = []
    for lam in np.linspace(0.1, 20.0, n_points):
        best = solvers.brute_force(mu, sigma, lam, cardinality)
        w = equal_weights(best.selected, len(mu))
        s = metrics.summarize(w, mu, sigma, RISK_FREE)
        points.append({"lambda": float(lam), "selected": best.selected,
                       "return": s["return"], "volatility": s["volatility"],
                       "sharpe": s["sharpe"]})
    return points


def figures(results, tickers, mu, sigma, frontier_points, scaling_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(IMG, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9})

    # Allocation per solver.
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / len(results)
    x = np.arange(len(tickers))
    for k, (name, row) in enumerate(results.items()):
        w = equal_weights(row["selected"], len(mu))
        ax.bar(x + k * width, w, width, label=name, alpha=0.85)
    ax.set(xticks=x + 0.4 - width / 2, ylabel="capital weight",
           title="Selection by solver (equal weight across holdings)")
    ax.set_xticklabels(tickers, rotation=45, ha="right")
    ax.legend()
    fig.savefig(os.path.join(IMG, "allocation.png"))
    plt.close(fig)

    # Frontier, with numbers on both axes.
    if frontier_points:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        vols = [p["volatility"] * 100 for p in frontier_points]
        rets = [p["return"] * 100 for p in frontier_points]
        sc = ax.scatter(vols, rets, c=[p["lambda"] for p in frontier_points],
                        cmap="viridis", zorder=3)
        ax.plot(vols, rets, color="0.7", lw=1, zorder=2)
        fig.colorbar(sc, ax=ax, label="risk aversion (lambda)")
        ax.set(xlabel="annualized volatility (%)", ylabel="annualized return (%)",
               title="Frontier over risk aversion, %d holdings" % CARDINALITY)
        ax.grid(alpha=0.3)
        fig.savefig(os.path.join(IMG, "frontier.png"))
        plt.close(fig)

    # Whether QAOA keeps finding the optimum as the problem grows.
    if scaling_rows:
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        ns = [r["n_assets"] for r in scaling_rows]
        ax1.plot(ns, [r["hit_rate"] * 100 for r in scaling_rows], "o-",
                 color="tab:blue", label="found exact optimum")
        ax1.set(xlabel="number of assets", ylabel="% of trials finding the optimum",
                ylim=(-5, 105), title="QAOA against the exact optimum")
        ax1.grid(alpha=0.3)
        ax2 = ax1.twinx()
        ax2.plot(ns, [r["mean_seconds"] for r in scaling_rows], "s--",
                 color="tab:red", label="QAOA seconds")
        ax2.set_ylabel("mean seconds per solve", color="tab:red")
        ax1.legend(loc="lower left")
        fig.savefig(os.path.join(IMG, "scaling.png"))
        plt.close(fig)

    print("wrote figures to %s" % IMG)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synthetic", action="store_true", help="skip the download")
    p.add_argument("--assets", type=int, default=10)
    p.add_argument("--cardinality", type=int, default=CARDINALITY)
    p.add_argument("--lam", type=float, default=LAMBDA)
    p.add_argument("--no-qaoa", action="store_true", help="skip the simulator")
    p.add_argument("--scaling", action="store_true", help="run the scaling study")
    p.add_argument("--frontier", action="store_true", help="sweep risk aversion")
    p.add_argument("--all", action="store_true", help="everything, and write figures")
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args()

    run_scaling = args.scaling or args.all
    run_frontier = args.frontier or args.all

    mu, sigma, tickers = get_moments(args.synthetic, args.assets)
    print("covariance positive semi-definite: %s" % datamod.is_psd(sigma))

    results = compare(mu, sigma, args.lam, args.cardinality,
                      include_qaoa=not args.no_qaoa, include_eigensolver=True)
    print_comparison(results, tickers, mu, sigma, args.lam, args.cardinality)

    frontier_points = []
    if run_frontier:
        print("\nfrontier over risk aversion")
        frontier_points = frontier(mu, sigma, args.cardinality)
        for pnt in frontier_points[::5]:
            print("  lambda %5.2f  return %6.2f%%  vol %6.2f%%  sharpe %6.3f  %s"
                  % (pnt["lambda"], pnt["return"] * 100, pnt["volatility"] * 100,
                     pnt["sharpe"], [tickers[i] for i in pnt["selected"]]))

    scaling_rows = []
    if run_scaling:
        print("\nscaling study: QAOA against the exact optimum")
        scaling_rows = scaling_study(cardinality=args.cardinality, lam=args.lam)

    os.makedirs(RESULTS, exist_ok=True)
    payload = {
        "setup": {"tickers": tickers, "lambda": args.lam,
                  "cardinality": args.cardinality, "risk_free": RISK_FREE,
                  "synthetic": bool(args.synthetic),
                  "covariance_psd": datamod.is_psd(sigma)},
        "solvers": {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
        "frontier": frontier_points,
        "scaling": scaling_rows,
    }
    with open(os.path.join(RESULTS, "results.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print("\nwrote %s" % os.path.join(RESULTS, "results.json"))

    if not args.no_figures:
        figures(results, tickers, mu, sigma, frontier_points, scaling_rows)


if __name__ == "__main__":
    main()
