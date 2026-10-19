"""Run the experiments and write results/ and docs/img/.

    python run.py                  # all seven models, every solver (synthetic market)
    python run.py --all            # plus depth, anneal, penalty, scaling and frontier
    python run.py --real           # real prices (needs network or data/ cache)
    python run.py --model tracking # one model
"""

import argparse
import json
import os

import numpy as np

from qportfolio import data, experiments, models, solvers

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
IMG = os.path.join(HERE, "docs", "img")


def get_market(real: bool, n: int = 12, seed: int = 7):
    return data.real_market(data.DEFAULT_TICKERS[:n]) if real else data.synthetic_market(n, seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", action="store_true", help="real prices instead of synthetic")
    ap.add_argument("--assets", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--p", type=int, default=3, help="QAOA depth")
    ap.add_argument("--model", choices=[f.__name__ for f in (
        models.selection, models.allocation, models.tracking, models.long_short,
        models.diversified, models.rebalance, models.skew_aware)])
    ap.add_argument("--all", action="store_true", help="every study, and figures")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    market = get_market(args.real, args.assets, args.seed)
    print("market: %d assets, %d days, %s" % (market.n, len(market.returns),
                                             "real" if args.real else "synthetic"))

    payload = {"setup": {"assets": market.names, "sectors": market.sectors,
                         "real": args.real, "seed": args.seed, "p": args.p}}

    if args.model:
        problem = next(p for p in experiments.default_problems(market)
                       if p.name == args.model)
        exact, ref = solvers.exact(problem)
        rows = [experiments._row(problem, ref, exact)]
        for label, kw in experiments.QUANTUM:
            if kw["mixer"] == "xy" and problem.hamming_weight is None:
                continue
            rows.append(experiments._row(problem, ref, solvers.qaoa(
                problem, ref, p=args.p, label=label, **kw)))
        experiments.print_rows(problem, ref, rows)
        return

    print("\n== model zoo ==")
    payload["zoo"] = experiments.model_zoo(market, p=args.p)

    if args.all:
        print("\n== depth: selection, 12 qubits ==")
        payload["depth"] = experiments.depth_study(market)
        print("\n== adiabatic: anneal time ==")
        payload["anneal"] = experiments.anneal_study(market)
        print("\n== penalty weight, X mixer ==")
        payload["penalty"] = experiments.penalty_study(market)
        print("\n== scaling: selection, k = n/4 ==")
        payload["scaling"] = experiments.scaling_study(
            lambda n, s: data.synthetic_market(n, seed=100 + s))
        print("\n== frontier ==")
        payload["frontier"] = experiments.frontier(market)
        for pt in payload["frontier"][::3]:
            print("  lambda %6.2f  return %6.1f%%  vol %5.1f%%  %s  (XY-QAOA found: %s)"
                  % (pt["lambda"], 100 * pt["return"], 100 * pt["volatility"],
                     pt["selection"], pt["qaoa_xy_found"]))

    os.makedirs(RESULTS, exist_ok=True)
    name = "results_real.json" if args.real else "results.json"
    with open(os.path.join(RESULTS, name), "w") as fh:
        json.dump(payload, fh, indent=1, default=float)
    print("\nwrote results/%s" % name)

    if args.all and not args.no_figures:
        figures(payload)


def figures(payload):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(IMG, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 140, "savefig.bbox": "tight", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    colors = {"x": "#8a8f98", "warm": "#d08c3a", "xy": "#2b6cb0",
              "qaoa_x": "#8a8f98", "qaoa_x_cvar": "#b9bcc2", "qaoa_warm": "#d08c3a",
              "qaoa_xy": "#2b6cb0", "adiabatic_x": "#5d6168", "adiabatic_xy": "#163d6b"}

    # 1. P(optimal) by model and quantum solver.
    zoo = payload["zoo"]["rows"]
    model_names = list(dict.fromkeys(r["model"] for r in zoo))
    solvers_q = ["qaoa_x", "qaoa_x_cvar", "qaoa_warm", "qaoa_xy", "adiabatic_x", "adiabatic_xy"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    width = 0.8 / len(solvers_q)
    for j, s in enumerate(solvers_q):
        vals = [next((r["amplification"] for r in zoo if r["model"] == m and r["solver"] == s),
                     np.nan) for m in model_names]
        ax.bar(np.arange(len(model_names)) + j * width, vals, width, label=s, color=colors[s])
    ax.axhline(1, color="k", lw=0.8, ls=":")
    ax.text(len(model_names) - 0.35, 1.15, "uniform random", fontsize=7, ha="right")
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(model_names)) + 0.4 - width / 2)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("P(optimal) / P(random feasible pick)")
    ax.set_title("How much each circuit concentrates on the optimal portfolio (p = 3)")
    ax.legend(ncol=3, fontsize=7, frameon=False)
    fig.savefig(os.path.join(IMG, "models.png"))
    plt.close(fig)

    # 2. Depth.
    depth = payload["depth"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    for mixer in ("x", "warm", "xy"):
        rows = [r for r in depth if r["mixer"] == mixer]
        a1.plot([r["p"] for r in rows], [r["p_optimal"] for r in rows], "o-",
                color=colors[mixer], label=mixer)
        a2.plot([r["p"] for r in rows], [r["p_feasible"] for r in rows], "o-",
                color=colors[mixer], label=mixer)
    a1.set(xlabel="QAOA layers p", ylabel="P(optimal portfolio)", title="Probability of the optimum")
    a2.set(xlabel="QAOA layers p", ylabel="P(valid portfolio)", ylim=(0, 1.05),
           title="Probability a shot satisfies the constraint")
    a1.legend(frameon=False)
    fig.savefig(os.path.join(IMG, "depth.png"))
    plt.close(fig)

    # 3. Anneal time.
    anneal = payload["anneal"]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for mixer in ("x", "xy"):
        rows = [r for r in anneal if r["mixer"] == mixer]
        ax.plot([r["time"] for r in rows], [r["p_optimal"] for r in rows], "o-",
                color=colors[mixer], label="%s mixer" % mixer)
    ax.set(xscale="log", xlabel="total anneal time (normalized units)",
           ylabel="P(optimal portfolio)", ylim=(0, 1.02),
           title="Adiabatic evolution, 12-asset selection")
    ax.legend(frameon=False)
    fig.savefig(os.path.join(IMG, "anneal.png"))
    plt.close(fig)

    # 4. Scaling.
    scaling = payload["scaling"]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for mixer in ("x", "xy"):
        rows = [r for r in scaling if r["mixer"] == mixer]
        ax.plot([r["n"] for r in rows], [r["amplification"] for r in rows], "o-",
                color=colors[mixer], label="%s mixer" % mixer)
    ax.axhline(1, color="k", lw=0.8, ls=":")
    ax.set(yscale="log", xlabel="assets = qubits (k = n/4 held)",
           ylabel="P(optimal) / P(random feasible pick)", title="Scaling, QAOA p = 3")
    ax.legend(frameon=False)
    fig.savefig(os.path.join(IMG, "scaling.png"))
    plt.close(fig)

    # 5. Frontier.
    fr = payload["frontier"]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    sc = ax.scatter([100 * p["volatility"] for p in fr], [100 * p["return"] for p in fr],
                    c=np.log10([p["lambda"] for p in fr]), cmap="viridis", zorder=3)
    ax.plot([100 * p["volatility"] for p in fr], [100 * p["return"] for p in fr],
            color="0.75", lw=1, zorder=2)
    fig.colorbar(sc, ax=ax, label="log10 risk aversion")
    ax.set(xlabel="annualized volatility (%)", ylabel="annualized return (%)",
           title="Selection frontier, 3 of 12")
    fig.savefig(os.path.join(IMG, "frontier.png"))
    plt.close(fig)
    print("wrote figures to docs/img/")


if __name__ == "__main__":
    main()
