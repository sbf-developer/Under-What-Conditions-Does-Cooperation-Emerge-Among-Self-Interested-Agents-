"""
Generate publication-quality figures for the cooperation emergence study.

Run from project root:
    python src/generate_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from simulation import (
    Payoffs,
    grim_trigger_min_discount,
    simulate_kin_selection_groups,
    simulate_spatial_pd,
    simulate_well_mixed_replicator,
    sweep_discount_repeated,
    sweep_noise_spatial,
    sweep_relatedness_kin,
    sweep_temptation_spatial,
)

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Academic styling
rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)

COLORS = {
    "coop": "#2166ac",
    "defect": "#b2182b",
    "accent": "#4daf4a",
    "neutral": "#636363",
}


def fig_payoff_matrix() -> None:
    p = Payoffs()
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    labels = ["Cooperate (C)", "Defect (D)"]
    data = np.array([[p.R, p.S], [p.T, p.P]])
    im = ax.imshow(data, cmap="YlGnBu", vmin=0, vmax=p.T)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Opponent")
    ax.set_ylabel("Player")
    ax.set_title("Prisoner's Dilemma payoff matrix\n($T=5, R=3, P=1, S=0$)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Payoff")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig01_payoff_matrix.pdf")
    fig.savefig(FIG_DIR / "fig01_payoff_matrix.png")
    plt.close(fig)


def fig_grim_trigger() -> None:
    p = Payoffs()
    delta = np.linspace(0, 1, 200)
    feasible = sweep_discount_repeated(delta, p)
    threshold = grim_trigger_min_discount(p)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.fill_between(delta, 0, feasible, alpha=0.25, color=COLORS["coop"], label="Cooperation sustainable")
    ax.fill_between(delta, feasible, 1, alpha=0.15, color=COLORS["defect"], label="Defection dominates")
    ax.axvline(threshold, color=COLORS["defect"], ls="--", lw=1.5, label=rf"$\delta^* = (T-R)/(T-P) = {threshold:.2f}$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"Discount factor $\delta$")
    ax.set_ylabel("Grim-trigger feasibility")
    ax.set_title("Repeated game: when can cooperation be an equilibrium?")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig02_grim_trigger.pdf")
    fig.savefig(FIG_DIR / "fig02_grim_trigger.png")
    plt.close(fig)


def fig_spatial_snapshots() -> None:
    p = Payoffs()
    rng_seed = 7
    grids = []
    for steps in (0, 200, 800):
        g, _ = simulate_spatial_pd(size=60, p=p, noise=0.01, steps=steps, seed=rng_seed)
        grids.append(g)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    titles = ["Initial ($t=0$)", "Intermediate ($t=200$)", "Final ($t=800$)"]
    cmap = plt.cm.colors.ListedColormap([COLORS["defect"], COLORS["coop"]])
    for ax, g, title in zip(axes, grids, titles):
        ax.imshow(g, cmap=cmap, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Spatial evolutionary Prisoner's Dilemma (Nowak--May style)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig03_spatial_snapshots.pdf")
    fig.savefig(FIG_DIR / "fig03_spatial_snapshots.png")
    plt.close(fig)


def fig_spatial_timeseries() -> None:
    _, hist = simulate_spatial_pd(size=50, noise=0.01, steps=1000, seed=11)
    _, hist_noisy = simulate_spatial_pd(size=50, noise=0.08, steps=1000, seed=11)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(hist, color=COLORS["coop"], lw=1.5, label=r"Low noise ($\varepsilon=0.01$)")
    ax.plot(hist_noisy, color=COLORS["defect"], lw=1.5, label=r"High noise ($\varepsilon=0.08$)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cooperation fraction")
    ax.set_title("Spatial model: cooperation dynamics over time")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig04_spatial_timeseries.pdf")
    fig.savefig(FIG_DIR / "fig04_spatial_timeseries.png")
    plt.close(fig)


def fig_temptation_sweep() -> None:
    T_vals = np.linspace(3.05, 5.0, 20)
    T, means, stds = sweep_temptation_spatial(T_vals, n_replicates=12)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.errorbar(T, means, yerr=1.96 * stds / np.sqrt(12), fmt="o-", color=COLORS["coop"], capsize=3, ms=4)
    ax.axhline(0.5, color=COLORS["neutral"], ls=":", lw=1)
    ax.set_xlabel(r"Temptation payoff $T$ (holding $R=3, P=1, S=0$)")
    ax.set_ylabel("Final cooperation fraction")
    ax.set_title("Comparative static: cooperation vs. temptation to defect")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig05_temptation_sweep.pdf")
    fig.savefig(FIG_DIR / "fig05_temptation_sweep.png")
    plt.close(fig)


def fig_noise_sweep() -> None:
    noise_vals = np.linspace(0, 0.15, 16)
    n, means, stds = sweep_noise_spatial(noise_vals, n_replicates=10)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.errorbar(n, means, yerr=1.96 * stds / np.sqrt(10), fmt="s-", color=COLORS["accent"], capsize=3, ms=4)
    ax.set_xlabel(r"Strategy noise $\varepsilon$")
    ax.set_ylabel("Final cooperation fraction")
    ax.set_title("Effect of mutation / execution errors on spatial cooperation")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig06_noise_sweep.pdf")
    fig.savefig(FIG_DIR / "fig06_noise_sweep.png")
    plt.close(fig)


def fig_well_mixed_vs_spatial() -> None:
    _, hist_spatial = simulate_spatial_pd(size=50, steps=1500, seed=3)
    hist_mixed = simulate_well_mixed_replicator(steps=1500, seed=3)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(hist_spatial, color=COLORS["coop"], lw=1.5, label="Spatial (local interaction)")
    ax.plot(hist_mixed, color=COLORS["defect"], lw=1.5, label="Well-mixed (Moran process)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cooperation fraction")
    ax.set_title("Structure matters: spatial vs. well-mixed population")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig07_mixed_vs_spatial.pdf")
    fig.savefig(FIG_DIR / "fig07_mixed_vs_spatial.png")
    plt.close(fig)


def fig_relatedness_sweep() -> None:
    r_vals = np.linspace(0, 1, 21)
    r, means, stds = sweep_relatedness_kin(r_vals, n_replicates=10)

    p = Payoffs()
    # Hamilton's rule threshold: rb > c with b=R-P, c=T-R for PD donation framing
    b = p.R - p.P
    c = p.T - p.R
    r_star = c / b if b > 0 else np.nan

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.errorbar(r, means, yerr=1.96 * stds / np.sqrt(10), fmt="^-", color=COLORS["coop"], capsize=3, ms=4)
    if np.isfinite(r_star):
        ax.axvline(r_star, color=COLORS["defect"], ls="--", lw=1.5, label=rf"Hamilton threshold $r^* = c/b = {r_star:.2f}$")
    ax.set_xlabel(r"Relatedness / assortment $r$")
    ax.set_ylabel("Final cooperation fraction")
    ax.set_title("Group structure and positive assortment\n(dashed line: Hamilton threshold $r^*=c/b$ under donation framing)")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.subplots_adjust(top=0.82)
    fig.savefig(FIG_DIR / "fig08_relatedness_sweep.pdf")
    fig.savefig(FIG_DIR / "fig08_relatedness_sweep.png")
    plt.close(fig)


def fig_mechanisms_summary() -> None:
    """Schematic bar chart of illustrative cooperation levels by mechanism."""
    mechanisms = [
        "One-shot PD",
        "Well-mixed\nevolution",
        "Repeated game\n($\\delta=0.8$)",
        "Spatial\nstructure",
        "Positive\nassortment",
    ]
    # Illustrative final fractions from simulations (single seed, documented in paper)
    np.random.seed(99)
    _, h_sp = simulate_spatial_pd(steps=800, seed=99)
    h_mixed = simulate_well_mixed_replicator(steps=800, seed=99)
    h_kin = simulate_kin_selection_groups(relatedness=0.6, steps=800, seed=99)

    values = [0.0, h_mixed[-1], 1.0, h_sp[-1], h_kin[-1]]
    colors = [COLORS["defect"], COLORS["defect"], COLORS["coop"], COLORS["coop"], COLORS["coop"]]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars = ax.bar(mechanisms, values, color=colors, edgecolor="black", linewidth=0.6, alpha=0.85)
    ax.set_ylabel("Cooperation (illustrative)")
    ax.set_title("Summary: cooperation emerges only under qualifying conditions")
    ax.set_ylim(0, 1.1)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig09_mechanisms_summary.pdf")
    fig.savefig(FIG_DIR / "fig09_mechanisms_summary.png")
    plt.close(fig)


def main() -> None:
    print("Generating figures...")
    fig_payoff_matrix()
    fig_grim_trigger()
    fig_spatial_snapshots()
    fig_spatial_timeseries()
    fig_temptation_sweep()
    fig_noise_sweep()
    fig_well_mixed_vs_spatial()
    fig_relatedness_sweep()
    fig_mechanisms_summary()
    print(f"Figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
