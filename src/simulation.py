"""
Agent-based simulations of cooperation emergence in the Prisoner's Dilemma.

Models implemented (illustrative, not empirical identification):
  1. One-shot Nash equilibrium (defection dominates).
  2. Infinitely repeated game: grim-trigger feasibility region.
  3. Spatial evolutionary PD (Nowak & May, 1992 style).
  4. Well-mixed replicator dynamics with selection-mutation.
  5. Parameter sweeps over temptation, noise, and relatedness.

Payoff convention (standard PD): T > R > P > S and 2R > T + S.
  C,C -> (R,R); C,D -> (S,T); D,C -> (T,S); D,D -> (P,P)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Strategy = Literal["C", "D"]


@dataclass(frozen=True)
class Payoffs:
    """Prisoner's Dilemma payoffs with T > R > P > S."""

    T: float = 5.0  # temptation to defect
    R: float = 3.0  # reward for mutual cooperation
    P: float = 1.0  # punishment for mutual defection
    S: float = 0.0  # sucker's payoff

    def __post_init__(self) -> None:
        if not (self.T > self.R > self.P > self.S):
            raise ValueError("Payoffs must satisfy T > R > P > S.")
        if not (2 * self.R > self.T + self.S):
            raise ValueError("Must satisfy 2R > T + S (social dilemma).")

    def matrix(self) -> np.ndarray:
        """Return 2x2 payoff matrix [row=player, col=opponent] for (C,D)."""
        return np.array([[self.R, self.S], [self.T, self.P]])


def one_shot_pd_payoffs(p: Payoffs) -> dict[tuple[Strategy, Strategy], tuple[float, float]]:
    """Stage-game payoffs; unique Nash equilibrium is (D, D)."""
    m = p.matrix()
    idx = {"C": 0, "D": 1}
    out: dict[tuple[Strategy, Strategy], tuple[float, float]] = {}
    for s1 in ("C", "D"):
        for s2 in ("C", "D"):
            v = m[idx[s1], idx[s2]]
            out[(s1, s2)] = (v, v)
    return out


def grim_trigger_min_discount(p: Payoffs) -> float:
    """
    Minimum discount factor delta such that mutual cooperation is a
    subgame-perfect equilibrium under grim trigger in the infinitely
    repeated Prisoner's Dilemma (two symmetric players).

    Condition: R/(1-delta) >= T + delta*P/(1-delta)  =>  delta >= (T-R)/(T-P).
    """
    return (p.T - p.R) / (p.T - p.P)


def repeated_pd_cooperation_feasible(p: Payoffs, delta: float) -> bool:
    return delta >= grim_trigger_min_discount(p)


def _neighbor_payoffs(grid: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Vectorized average PD payoff against four von Neumann neighbors."""
    payoff_c = mat[0]
    payoff_d = mat[1]

    def vs_neighbors(nbr: np.ndarray) -> np.ndarray:
        return np.where(grid == 0, payoff_c[nbr], payoff_d[nbr])

    total = (
        vs_neighbors(np.roll(grid, -1, axis=0))
        + vs_neighbors(np.roll(grid, 1, axis=0))
        + vs_neighbors(np.roll(grid, -1, axis=1))
        + vs_neighbors(np.roll(grid, 1, axis=1))
    )
    return total / 4.0


def _best_neighbor_strategy(grid: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Strategy of the highest-scoring von Neumann neighbor (ties: last wins)."""
    best_score = scores.copy()
    best_strategy = grid.copy()
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        n_score = np.roll(np.roll(scores, di, axis=0), dj, axis=1)
        n_strat = np.roll(np.roll(grid, di, axis=0), dj, axis=1)
        improve = n_score > best_score
        best_score = np.where(improve, n_score, best_score)
        best_strategy = np.where(improve, n_strat, best_strategy)
    return best_strategy


def simulate_spatial_pd(
    size: int = 50,
    p: Payoffs | None = None,
    noise: float = 0.01,
    steps: int = 500,
    init_coop_frac: float = 0.5,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Spatial evolutionary Prisoner's Dilemma on a toroidal lattice.

    Each cell plays PD with its four von Neumann neighbors; fitness is average
    payoff. Synchronous update: all sites simultaneously copy the strategy of
    the highest-scoring neighbor with probability 1 - noise, else randomize.

    Returns (final_grid, cooperation_history) where cooperation_history[t] is
    the fraction of cooperators at step t.
    """
    if p is None:
        p = Payoffs()
    rng = np.random.default_rng(seed)
    grid = (rng.random((size, size)) < init_coop_frac).astype(np.int8)

    mat = p.matrix()
    history = np.empty(steps + 1)
    history[0] = grid.mean()

    for t in range(steps):
        scores = _neighbor_payoffs(grid, mat)
        best_strategy = _best_neighbor_strategy(grid, scores)
        adopt = rng.random((size, size)) >= noise
        random_strat = rng.integers(0, 2, size=(size, size))
        grid = np.where(adopt, best_strategy, random_strat).astype(np.int8)
        history[t + 1] = grid.mean()

    return grid, history


def simulate_well_mixed_replicator(
    p: Payoffs | None = None,
    n: int = 500,
    mutation_rate: float = 0.01,
    steps: int = 2000,
    init_coop_frac: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """
    Wright-Fisher style Moran process with two strategies (C, D).

    Fitness is expected payoff against a random opponent from the population.
    Returns cooperation fraction history.
    """
    if p is None:
        p = Payoffs()
    rng = np.random.default_rng(seed)
    mat = p.matrix()

    n_coop = int(round(init_coop_frac * n))
    history = np.empty(steps + 1)
    history[0] = n_coop / n

    for t in range(steps):
        frac = n_coop / n
        # Expected payoff
        pi_c = frac * mat[0, 0] + (1 - frac) * mat[0, 1]
        pi_d = frac * mat[1, 0] + (1 - frac) * mat[1, 1]
        # Moran: death replaced by copy proportional to fitness (non-negative shift)
        shift = abs(min(pi_c, pi_d, 0)) + 1e-9
        w_c = n_coop * (pi_c + shift)
        w_d = (n - n_coop) * (pi_d + shift)
        total = w_c + w_d

        if rng.random() < mutation_rate:
            n_coop += rng.choice([-1, 1])
        else:
            if rng.random() < w_c / total:
                n_coop = min(n_coop + 1, n)
            else:
                n_coop = max(n_coop - 1, 0)
        history[t + 1] = n_coop / n

    return history


def simulate_kin_selection_groups(
    p: Payoffs | None = None,
    n_groups: int = 100,
    group_size: int = 5,
    relatedness: float = 0.5,
    steps: int = 500,
    seed: int | None = None,
) -> np.ndarray:
    """
    Group-structured one-shot PD with positive assortment (relatedness r).

    With probability r an individual's strategy is copied from a randomly chosen
    group member (positive assortment); otherwise it is copied fitness-proportionally
    from a group member. This is a stylized illustration of Hamilton's rb > c logic,
    not an empirical estimate of genetic relatedness.
    """
    if p is None:
        p = Payoffs()
    rng = np.random.default_rng(seed)
    mat = p.matrix()
    groups = (rng.random((n_groups, group_size)) < 0.5).astype(np.int8)
    history = np.empty(steps + 1)
    history[0] = groups.mean()

    idx = np.arange(n_groups)

    for t in range(steps):
        c_count = groups.sum(axis=1, keepdims=True)
        d_count = group_size - c_count
        payoffs = np.where(
            groups == 0,
            ((c_count - 1) * mat[0, 0] + d_count * mat[0, 1]) / max(group_size - 1, 1),
            (c_count * mat[1, 0] + (d_count - 1) * mat[1, 1]) / max(group_size - 1, 1),
        )

        weights = payoffs - payoffs.min(axis=1, keepdims=True) + 1e-6
        weights /= weights.sum(axis=1, keepdims=True)

        assortment = rng.random((n_groups, group_size)) < relatedness
        rand_parent = rng.integers(0, group_size, size=(n_groups, group_size))
        u = rng.random((n_groups, group_size))
        fit_parent = (u[:, :, np.newaxis] < np.cumsum(weights, axis=1)[:, np.newaxis, :]).argmax(axis=2)

        parent_idx = np.where(assortment, rand_parent, fit_parent)
        groups = groups[idx[:, np.newaxis], parent_idx]
        history[t + 1] = groups.mean()

    return history


def sweep_temptation_spatial(
    T_values: np.ndarray,
    size: int = 40,
    steps: int = 400,
    noise: float = 0.01,
    n_replicates: int = 20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cooperation fraction vs temptation T (spatial model, multiple seeds)."""
    rng = np.random.default_rng(seed)
    means = np.zeros(len(T_values))
    stds = np.zeros(len(T_values))
    for i, T in enumerate(T_values):
        fracs = []
        for _ in range(n_replicates):
            p = Payoffs(T=float(T))
            _, hist = simulate_spatial_pd(
                size=size, p=p, noise=noise, steps=steps, seed=int(rng.integers(1e9))
            )
            fracs.append(hist[-1])
        means[i] = np.mean(fracs)
        stds[i] = np.std(fracs, ddof=1)
    return T_values, means, stds


def sweep_discount_repeated(
    delta_values: np.ndarray,
    p: Payoffs | None = None,
) -> np.ndarray:
    """Binary indicator: 1 if grim-trigger cooperation is feasible, else 0."""
    if p is None:
        p = Payoffs()
    threshold = grim_trigger_min_discount(p)
    return (delta_values >= threshold).astype(float)


def sweep_noise_spatial(
    noise_values: np.ndarray,
    size: int = 40,
    steps: int = 400,
    n_replicates: int = 15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    means = np.zeros(len(noise_values))
    stds = np.zeros(len(noise_values))
    for i, noise in enumerate(noise_values):
        fracs = []
        for _ in range(n_replicates):
            _, hist = simulate_spatial_pd(
                size=size, noise=float(noise), steps=steps, seed=int(rng.integers(1e9))
            )
            fracs.append(hist[-1])
        means[i] = np.mean(fracs)
        stds[i] = np.std(fracs, ddof=1)
    return noise_values, means, stds


def sweep_relatedness_kin(
    r_values: np.ndarray,
    steps: int = 250,
    n_replicates: int = 15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    means = np.zeros(len(r_values))
    stds = np.zeros(len(r_values))
    for i, r in enumerate(r_values):
        fracs = []
        for _ in range(n_replicates):
            hist = simulate_kin_selection_groups(
                relatedness=float(r), steps=steps, seed=int(rng.integers(1e9))
            )
            fracs.append(hist[-1])
        means[i] = np.mean(fracs)
        stds[i] = np.std(fracs, ddof=1)
    return r_values, means, stds
