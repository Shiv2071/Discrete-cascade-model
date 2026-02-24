"""
Run discrete cascade simulation and optionally plot diagnostics.
Usage: python run_simulation.py [--steps N] [--sites P] [--seed S] [--plot]
"""

import argparse
from cascade_model import run_simulation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000, help="Max time steps")
    ap.add_argument("--sites", type=int, default=50, help="Number of sites (chain)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--plot", action="store_true", help="Plot E_tot, X+Y, F_avg vs step")
    ap.add_argument("--output", "-o", type=str, default="", help="Save plot to path (e.g. ../figures/diagnostics.png)")
    args = ap.parse_args()

    model, history = run_simulation(
        P=args.sites,
        max_steps=args.steps,
        X0=3.0,
        Y0=3.0,
        E0=100.0,
        seed=args.seed,
    )

    n = model.n
    E_final = model.total_energy()
    X_final, Y_final = model.total_XY()
    print(f"Steps: {n}")
    print(f"Final total energy (Beta): {E_final:.2f}")
    print(f"Final X_total: {X_final:.0f}, Y_total: {Y_final:.0f}")
    print(f"Absorbing: {model.is_absorbing()}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping plot")
            return
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
        steps = history["n"]
        axes[0].plot(steps, history["E_total"], color="C0")
        axes[0].set_ylabel("Total energy (Beta)")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(steps, history["X_total"], label="X", color="C1")
        axes[1].plot(steps, history["Y_total"], label="Y", color="C2")
        axes[1].set_ylabel("Total X, Y")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[2].plot(steps, history["F_avg"], color="C3")
        axes[2].set_ylabel("Mean ripple F")
        axes[2].set_xlabel("Step")
        axes[2].grid(True, alpha=0.3)
        plt.tight_layout()
        outpath = args.output if args.output else "diagnostics.png"
        plt.savefig(outpath, dpi=150)
        print(f"Saved {outpath}")
        if args.output:
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    main()
