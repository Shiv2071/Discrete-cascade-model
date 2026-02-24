"""
Big Bang analogue: run the discrete cascade from hot, dense initial conditions
and observe the arc — burst of activity, structure formation (S, F), irreversible
energy depletion, then decay toward absorption.

Usage: python run_bigbang.py [--steps N] [--sites P] [--seed S]
"""

import argparse
from cascade_model import run_simulation


def main():
    ap = argparse.ArgumentParser(description="Big Bang analogue simulation")
    ap.add_argument("--steps", type=int, default=10000, help="Max time steps")
    ap.add_argument("--sites", type=int, default=100, help="Number of sites (chain)")
    ap.add_argument("--seed", type=int, default=2024, help="RNG seed")
    ap.add_argument("--no-plot", action="store_true", help="Skip saving plot")
    ap.add_argument("--output", "-o", type=str, default="", help="Save plot to path (e.g. ../figures/bigbang_analogue.png)")
    args = ap.parse_args()

    # Hot, dense "pre-bang" initial conditions: high energy, high X,Y
    # — evokes a burst of interactions, explosions, structure, then decay
    E0 = 500.0
    X0 = 8.0
    Y0 = 8.0

    model, history = run_simulation(
        P=args.sites,
        max_steps=args.steps,
        X0=X0,
        Y0=Y0,
        E0=E0,
        seed=args.seed,
    )

    n = model.n
    E_final = model.total_energy()
    X_final, Y_final = model.total_XY()
    S_final = model.total_structure()
    print("Big Bang analogue - run complete")
    print(f"  Steps: {n}")
    print(f"  Final total energy (Beta): {E_final:.2f}")
    print(f"  Final X_total: {X_final:.0f}, Y_total: {Y_final:.0f}")
    print(f"  Final total structure S: {S_final:.2f}")
    print(f"  Absorbing: {model.is_absorbing()}")

    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping plot")
            return
        steps = history["n"]
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(9, 8))
        fig.suptitle("Big Bang analogue: discrete cascade dynamics", fontsize=11)

        axes[0].plot(steps, history["E_total"], color="C0")
        axes[0].set_ylabel("Total energy (Beta)")
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

        plt.tight_layout()
        outpath = args.output if args.output else "bigbang_analogue.png"
        plt.savefig(outpath, dpi=150)
        print(f"Saved {outpath}")
        if args.output:
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    main()
