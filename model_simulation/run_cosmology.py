"""
Big Bang to "current universe" simulation: start from a concentrated
"primordial" region (e.g. left fraction of the chain) with all energy
and excitations; run the cascade dynamics so diffusion and interactions
spread and evolve the system until absorption.  Produces (1) global
time series and (2) spatial snapshots at several times to show
expansion and structure formation.
"""

import argparse
from typing import Optional, List, Tuple
import numpy as np
from cascade_model import CascadeModel


def run_cosmology(
    P: int = 100,
    max_steps: int = 8000,
    primordial_frac: float = 0.25,
    X0_per_site: float = 4.0,
    Y0_per_site: float = 4.0,
    E0_total: float = 400.0,
    seed: Optional[int] = None,
    num_snapshots: int = 6,
    **model_kw,
):
    """
    Run from concentrated initial conditions. Left (primordial_frac * P)
    sites get all X, Y, E; rest are empty. Returns model, history dict,
    and snapshots = list of (n, X, Y, Beta) arrays.
    """
    m = CascadeModel(P=P, seed=seed, **model_kw)
    P_prim = max(1, int(P * primordial_frac))
    # Put all mass/energy in the "primordial" region
    m.X[:P_prim] = X0_per_site
    m.X[P_prim:] = 0.0
    m.Y[:P_prim] = Y0_per_site
    m.Y[P_prim:] = 0.0
    m.Beta[:P_prim] = E0_total / P_prim
    m.Beta[P_prim:] = 0.0
    m.S[:] = 0.0
    m.S_prev1[:] = 0.0
    m.S_prev2[:] = 0.0

    history = {
        "n": [],
        "E_total": [],
        "X_total": [],
        "Y_total": [],
        "S_total": [],
        "F_avg": [],
    }
    # Record state at every step so we can subsample to num_snapshots later (same realisation)
    all_snapshots: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []

    for _ in range(max_steps):
        n = m.n
        history["n"].append(n)
        history["E_total"].append(m.total_energy())
        tx, ty = m.total_XY()
        history["X_total"].append(tx)
        history["Y_total"].append(ty)
        history["S_total"].append(m.total_structure())
        history["F_avg"].append(m.mean_ripple())
        all_snapshots.append((n, m.X.copy(), m.Y.copy(), m.Beta.copy()))
        active = m.step()
        if not active or m.is_absorbing():
            break

    # Subsample to num_snapshots (include t=0 and spread over time)
    if len(all_snapshots) <= num_snapshots:
        snapshots = all_snapshots
    else:
        indices = np.linspace(0, len(all_snapshots) - 1, num_snapshots, dtype=int)
        snapshots = [all_snapshots[i] for i in indices]

    return m, history, snapshots


def main():
    ap = argparse.ArgumentParser(description="Big Bang to current universe: concentrated init + spatial evolution")
    ap.add_argument("--sites", type=int, default=100, help="Number of sites (chain)")
    ap.add_argument("--steps", type=int, default=8000, help="Max time steps")
    ap.add_argument("--primordial-frac", type=float, default=0.25, help="Fraction of sites in initial hot region (e.g. 0.25 = left 25%%)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--output", "-o", type=str, default="", help="Base path for figures (e.g. ../figures/cosmology)")
    ap.add_argument("--no-plot", action="store_true", help="Skip plots")
    args = ap.parse_args()

    model, history, snapshots = run_cosmology(
        P=args.sites,
        max_steps=args.steps,
        primordial_frac=args.primordial_frac,
        X0_per_site=4.0,
        Y0_per_site=4.0,
        E0_total=600.0,
        seed=args.seed,
        num_snapshots=6,
    )

    n_final = model.n
    E_final = model.total_energy()
    X_final, Y_final = model.total_XY()
    print("Cosmology run (Big Bang -> current analogue)")
    print(f"  Primordial region: left {max(1, int(args.sites * args.primordial_frac))} of {args.sites} sites")
    print(f"  Steps: {n_final}")
    print(f"  Final total energy: {E_final:.2f}")
    print(f"  Final X_total: {X_final:.0f}, Y_total: {Y_final:.0f}")
    print(f"  Absorbing: {model.is_absorbing()}")

    if args.no_plot:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return

    base = args.output.rstrip("/").rstrip("\\") if args.output else "cosmology"
    if not base.endswith("cosmology"):
        base = base + "_cosmology" if not ("cosmology" in base) else base

    # Figure 1: global time series (like bigbang)
    fig1, axes = plt.subplots(4, 1, sharex=True, figsize=(9, 8))
    steps = history["n"]
    axes[0].plot(steps, history["E_total"], color="C0")
    axes[0].set_ylabel("Total energy")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(steps, history["X_total"], label="X", color="C1")
    axes[1].plot(steps, history["Y_total"], label="Y", color="C2")
    axes[1].set_ylabel("Total X, Y")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(steps, history["S_total"], color="C4")
    axes[2].set_ylabel("Total structure S")
    axes[2].grid(True, alpha=0.3)
    axes[3].plot(steps, history["F_avg"], color="C3")
    axes[3].set_ylabel("Mean ripple F")
    axes[3].set_xlabel("Step (time)")
    axes[3].grid(True, alpha=0.3)
    fig1.suptitle("Big Bang to current: global evolution (concentrated initial region)")
    plt.tight_layout()
    path1 = base + "_timeseries.png" if args.output else "cosmology_timeseries.png"
    plt.savefig(path1, dpi=150)
    print(f"Saved {path1}")
    if not args.output:
        plt.show()
    else:
        plt.close()

    # Figure 2: spatial snapshots at several times (X, Y, E vs site p)
    n_snap = len(snapshots)
    fig2, axes = plt.subplots(n_snap, 1, sharex=True, figsize=(10, 2 * n_snap))
    if n_snap == 1:
        axes = [axes]
    P = model.P
    for i, (n, X, Y, Beta) in enumerate(snapshots):
        ax = axes[i]
        scale_E = max(np.max(X), np.max(Y), 1.0) / (np.max(Beta) + 1e-9)
        ax.plot(range(P), X, label="X", color="C1", alpha=0.9)
        ax.plot(range(P), Y, label="Y", color="C2", alpha=0.9)
        ax.plot(range(P), Beta * scale_E * 0.6, label="E (scaled)", color="C0", alpha=0.8)
        ax.set_ylabel(f"t = {n}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, None)
    axes[-1].set_xlabel("Site p")
    fig2.suptitle("Spatial distribution at different times (expansion from primordial region)")
    plt.tight_layout()
    path2 = base + "_snapshots.png" if args.output else "cosmology_snapshots.png"
    plt.savefig(path2, dpi=150)
    print(f"Saved {path2}")
    if not args.output:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
