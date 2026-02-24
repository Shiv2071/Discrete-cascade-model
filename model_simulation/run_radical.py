"""
Radical simulation (unproven regime): two "universes" (two chains)
connected by a single bridge. All initial mass and energy live in
Universe A; Universe B is empty. Same cascade dynamics (diffusion
can cross the bridge). Question: does matter/energy cross? How much
"leaks" before the system freezes?

Physics translation: Can one universe transfer energy/matter to
another through a narrow channel? The model doesn't assume or prove
this; we just run and observe.
"""

import argparse
from typing import List, Tuple, Optional
import numpy as np
from cascade_model import CascadeModel


def two_chains_bridge_neighbors(P1: int, P2: int) -> List[List[int]]:
    """Chain 1: sites 0..P1-1. Chain 2: sites P1..P1+P2-1. One edge between P1-1 and P1."""
    P = P1 + P2
    neighbors = []
    for p in range(P):
        if p < P1:
            nb = []
            if p > 0:
                nb.append(p - 1)
            if p < P1 - 1:
                nb.append(p + 1)
            elif P2 > 0:
                nb.append(P1)
            neighbors.append(nb)
        else:
            nb = []
            if p > P1:
                nb.append(p - 1)
            elif P1 > 0:
                nb.append(P1 - 1)
            if p < P - 1:
                nb.append(p + 1)
            neighbors.append(nb)
    return neighbors


def run_two_universes(
    P1: int = 50,
    P2: int = 50,
    max_steps: int = 10000,
    X0: float = 5.0,
    Y0: float = 5.0,
    E0: float = 200.0,
    seed: Optional[int] = None,
    **model_kw,
) -> Tuple[CascadeModel, dict]:
    """Universe A = sites 0..P1-1 (gets all initial mass/energy). Universe B = sites P1..P1+P2-1 (empty)."""
    P = P1 + P2
    neighbors = two_chains_bridge_neighbors(P1, P2)
    m = CascadeModel(P=P, seed=seed, neighbors=neighbors, **model_kw)
    m.X[:P1] = X0
    m.X[P1:] = 0.0
    m.Y[:P1] = Y0
    m.Y[P1:] = 0.0
    m.Beta[:P1] = E0 / P1
    m.Beta[P1:] = 0.0
    m.S[:] = 0.0
    m.S_prev1[:] = 0.0
    m.S_prev2[:] = 0.0

    history = {
        "n": [],
        "E_A": [],
        "E_B": [],
        "X_A": [],
        "X_B": [],
        "Y_A": [],
        "Y_B": [],
        "E_total": [],
        "F_avg": [],
    }

    for _ in range(max_steps):
        n = m.n
        E_A = float(np.sum(m.Beta[:P1]))
        E_B = float(np.sum(m.Beta[P1:]))
        X_A = float(np.sum(m.X[:P1]))
        X_B = float(np.sum(m.X[P1:]))
        Y_A = float(np.sum(m.Y[:P1]))
        Y_B = float(np.sum(m.Y[P1:]))
        history["n"].append(n)
        history["E_A"].append(E_A)
        history["E_B"].append(E_B)
        history["X_A"].append(X_A)
        history["X_B"].append(X_B)
        history["Y_A"].append(Y_A)
        history["Y_B"].append(Y_B)
        history["E_total"].append(E_A + E_B)
        history["F_avg"].append(m.mean_ripple())
        active = m.step()
        if not active or m.is_absorbing():
            break

    return m, history


def main():
    ap = argparse.ArgumentParser(description="Two universes connected by a bridge; all mass starts in A")
    ap.add_argument("--P1", type=int, default=50, help="Sites in Universe A")
    ap.add_argument("--P2", type=int, default=50, help="Sites in Universe B")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--output", "-o", type=str, default="", help="Save plot path (e.g. ../figures/radical_two_universes.png)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    # Unproven regime: two universes, one bridge. Default params often show negligible transfer (A freezes first).
    # Try higher D_X/D_Y and lower alpha_XY/alpha_XX in run_two_universes() to see mass cross.
    model, history = run_two_universes(
        P1=args.P1,
        P2=args.P2,
        max_steps=args.steps,
        X0=5.0,
        Y0=5.0,
        E0=200.0,
        seed=args.seed,
    )
    P1, P2 = args.P1, args.P2
    n_final = model.n
    E_A_f = np.sum(model.Beta[:P1])
    E_B_f = np.sum(model.Beta[P1:])
    X_A_f = np.sum(model.X[:P1])
    X_B_f = np.sum(model.X[P1:])
    Y_A_f = np.sum(model.Y[:P1])
    Y_B_f = np.sum(model.Y[P1:])

    print("Radical run: Two universes (A and B) connected by one bridge")
    print(f"  Universe A: sites 0..{P1-1} (all initial mass).  Universe B: sites {P1}..{P1+P2-1} (empty at t=0)")
    print(f"  Steps: {n_final}")
    print(f"  Final — Universe A: E={E_A_f:.2f}, X={X_A_f:.0f}, Y={Y_A_f:.0f}")
    print(f"  Final — Universe B: E={E_B_f:.2f}, X={X_B_f:.0f}, Y={Y_B_f:.0f}")
    print(f"  Fraction of energy in B at end: {E_B_f/(E_A_f+E_B_f+1e-12):.2%}")
    print(f"  Absorbing: {model.is_absorbing()}")

    if args.no_plot:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return

    steps = history["n"]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 7))
    fig.suptitle("Radical: Two universes, one bridge — does energy/matter cross? (unproven regime)")

    axes[0].plot(steps, history["E_A"], label="Energy in A", color="C0")
    axes[0].plot(steps, history["E_B"], label="Energy in B", color="C1")
    axes[0].set_ylabel("Total energy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, history["X_A"], label="X in A", color="C2")
    axes[1].plot(steps, history["X_B"], label="X in B", color="C3")
    axes[1].set_ylabel("Total X")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, history["Y_A"], label="Y in A", color="C4")
    axes[2].plot(steps, history["Y_B"], label="Y in B", color="C5")
    axes[2].set_ylabel("Total Y")
    axes[2].set_xlabel("Step")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out = args.output if args.output else "radical_two_universes.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    if args.output:
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
