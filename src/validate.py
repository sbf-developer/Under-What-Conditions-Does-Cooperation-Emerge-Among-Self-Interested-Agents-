"""
Pre-push validation: math, PD constraints, and qualitative simulation checks.

Run from project root:
    python src/validate.py
"""

from __future__ import annotations

import sys

import numpy as np

from simulation import (
    Payoffs,
    grim_trigger_min_discount,
    repeated_pd_cooperation_feasible,
    simulate_spatial_pd,
    simulate_well_mixed_replicator,
    sweep_noise_spatial,
    sweep_temptation_spatial,
)


def test_payoff_constraints() -> None:
    p = Payoffs()
    assert p.T > p.R > p.P > p.S
    assert 2 * p.R > p.T + p.S
    m = p.matrix()
    assert m[1, 0] > m[0, 0] and m[1, 1] > m[0, 1]  # D dominates C


def test_grim_trigger() -> None:
    p = Payoffs()
    assert grim_trigger_min_discount(p) == 0.5
    assert repeated_pd_cooperation_feasible(p, 0.5)
    assert repeated_pd_cooperation_feasible(p, 0.8)
    assert not repeated_pd_cooperation_feasible(p, 0.49)


def test_hamilton_threshold() -> None:
    p = Payoffs()
    b, c = p.R - p.P, p.T - p.R
    assert b == 2.0 and c == 2.0
    assert c / b == 1.0


def test_spatial_vs_mixed_direction() -> None:
    """Spatial structure should sustain more cooperation than well-mixed (typical run)."""
    _, hist_sp = simulate_spatial_pd(size=40, steps=800, seed=42)
    hist_mixed = simulate_well_mixed_replicator(steps=800, seed=42)
    assert hist_sp[-1] > hist_mixed[-1], (
        f"Expected spatial > mixed; got {hist_sp[-1]:.3f} vs {hist_mixed[-1]:.3f}"
    )


def test_temptation_comparative_static() -> None:
    T, means, _ = sweep_temptation_spatial(
        np.array([3.2, 4.0, 4.8]), size=30, steps=200, n_replicates=5, seed=1
    )
    assert means[0] >= means[-1] - 0.15, "Cooperation should fall as T rises (allow noise)"


def test_noise_comparative_static() -> None:
    n, means, _ = sweep_noise_spatial(
        np.array([0.0, 0.08, 0.15]), size=30, steps=200, n_replicates=5, seed=1
    )
    assert means[0] >= means[-1] - 0.2, "Cooperation should fall as noise rises (allow noise)"


def main() -> int:
    tests = [
        test_payoff_constraints,
        test_grim_trigger,
        test_hamilton_threshold,
        test_spatial_vs_mixed_direction,
        test_temptation_comparative_static,
        test_noise_comparative_static,
    ]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print("\nAll validation checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
