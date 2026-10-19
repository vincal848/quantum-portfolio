"""Run every solver on every model, then the scaling study. Writes results/ and docs/img/.

    python run.py            # synthetic market (no network)
    python run.py --real     # 12 real tickers, 2020-2022, from Yahoo Finance
"""

import json
import sys
import time

import numpy as np

import models
import qaoa

SOLVERS = [
    ("annealing", lambda m, ref: qaoa.annealing(m)),
    ("qaoa_x", lambda m, ref: qaoa.qaoa(m, ref, "x")),
    ("qaoa_x_cvar", lambda m, ref: qaoa.qaoa(m, ref, "x", alpha=0.1)),
    ("qaoa_warm", lambda m, ref: qaoa.qaoa(m, ref, "warm")),
    ("qaoa_xy", lambda m, ref: qaoa.qaoa(m, ref, "xy")),
    ("adiabatic_x", lambda m, ref: qaoa.adiabatic(m, ref, "x")),
    ("adiabatic_xy", lambda m, ref: qaoa.adiabatic(m, ref, "xy")),
]


def compare(market):
    rows = []
    for model in models.all_models(market):
        ref = qaoa.exact(model)
        print("\n%s  [%d qubits, %d valid portfolios]  optimum: %s"
              % (model.name, model.n, ref["n_valid"], model.describe(ref["x"])))
        print("  %-14s%8s%10s%9s%10s%9s" % ("solver", "ratio", "P(valid)", "P(opt)", "x random", "seconds"))
        for name, solve in SOLVERS:
            if name.endswith("xy") and model.k is None:
                continue
            start = time.perf_counter()
            r = solve(model, ref)
            r["seconds"] = time.perf_counter() - start
            r["ratio"] = qaoa.ratio(ref, r["objective"])
            rows.append({"model": model.name, "solver": name, "selection": model.describe(r["x"]),
                         **{k: v for k, v in r.items() if k != "x"}})
            print("  %-14s%8.4f%10s%9s%10s%9.2f" % (
                name, r["ratio"], "%.3f" % r["p_valid"] if "p_valid" in r else "-",
                "%.4f" % r["p_optimal"] if "p_optimal" in r else "-",
                "%.1f" % r["amplification"] if "amplification" in r else "-", r["seconds"]))
    return rows


def scaling(sizes=range(6, 21, 2), seeds=(1, 2, 3)):
    """Selection with k = n/4 on random markets. The X mixer stops at 16 qubits (65,536
    amplitudes); the XY mixer only stores C(n, k)."""
    rows = []
    for n in sizes:
        for mixer in ("x", "xy"):
            if mixer == "x" and n > 16:
                continue
            runs = []
            for seed in seeds:
                model = models.selection(models.synthetic_market(n, seed=100 + seed), k=max(2, n // 4))
                ref = qaoa.exact(model)
                start = time.perf_counter()
                r = qaoa.qaoa(model, ref, mixer, restarts=3, seed=seed)
                runs.append((r["amplification"], r["p_optimal"], abs(r["objective"] - ref["best"]) < 1e-9,
                             time.perf_counter() - start))
            amp, p_opt, found, secs = map(np.mean, zip(*runs))
            rows.append({"n": n, "k": max(2, n // 4), "mixer": mixer, "amplification": amp,
                         "p_optimal": p_opt, "found": int(found * len(seeds)), "trials": len(seeds),
                         "seconds": secs})
            print("  n=%2d %-3s P(opt) %.4f  x%6.1f over random  found %d/%d  %.1fs"
                  % (n, mixer, p_opt, amp, rows[-1]["found"], len(seeds), secs))
    return rows


def figures(rows, scale):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 140, "savefig.bbox": "tight", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    colors = {"qaoa_x": "#8a8f98", "qaoa_x_cvar": "#b9bcc2", "qaoa_warm": "#d08c3a",
              "qaoa_xy": "#2b6cb0", "adiabatic_x": "#5d6168", "adiabatic_xy": "#163d6b"}
    names = list(dict.fromkeys(r["model"] for r in rows))
    width = 0.8 / len(colors)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    for j, solver in enumerate(colors):
        amp = [next((r["amplification"] for r in rows if r["model"] == m and r["solver"] == solver), np.nan)
               for m in names]
        ax.bar(np.arange(len(names)) + j * width, amp, width, label=solver, color=colors[solver])
    ax.axhline(1, color="k", lw=0.8, ls=":")
    ax.text(-0.3, 1.15, "uniform random", fontsize=7)
    ax.set(yscale="log", ylabel="P(optimal) / P(random valid pick)",
           title="How much each circuit concentrates on the optimal portfolio (p = 3)")
    ax.set_xticks(np.arange(len(names)) + 0.4 - width / 2, names)
    ax.legend(ncol=3, fontsize=7, frameon=False)
    fig.savefig("docs/img/models.png")

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for mixer, color in (("x", "#8a8f98"), ("xy", "#2b6cb0")):
        pts = [r for r in scale if r["mixer"] == mixer]
        ax.plot([r["n"] for r in pts], [r["amplification"] for r in pts], "o-", color=color,
                label="%s mixer" % mixer)
    ax.axhline(1, color="k", lw=0.8, ls=":")
    ax.set(yscale="log", xlabel="assets = qubits (k = n/4 held)",
           ylabel="P(optimal) / P(random valid pick)", title="Scaling, QAOA p = 3")
    ax.legend(frameon=False)
    fig.savefig("docs/img/scaling.png")


if __name__ == "__main__":
    market = models.real_market() if "--real" in sys.argv else models.synthetic_market()
    rows = compare(market)
    print("\nscaling: selection, k = n/4")
    scale = scaling()
    with open("results/results.json", "w") as f:
        json.dump({"comparison": rows, "scaling": scale}, f, indent=1, default=float)
    figures(rows, scale)
