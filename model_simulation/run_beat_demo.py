"""
Beat-frequency demonstration (Paper II, Section 5).

Runs the cascade model twice with identical parameters except the frequency
mismatch: once with omega_X = omega_Y (no beat) and once with
omega_X != omega_Y (beat period 2*pi/|wX - wY|). Slow interaction rates keep
the populations alive across several beat periods so the modulation of the
structural increment, and hence the ripple F = |Delta^2 S|, is visible.

Usage: python run_beat_demo.py [-o ../figures/beat_demo.png]
"""

import argparse

import numpy as np

from cascade_model import CascadeModel


def run_case(omega_Y: float, steps: int, seed: int):
    m = CascadeModel(
        P=50,
        alpha_XY=0.004,
        alpha_XX=0.0005,
        omega_X=1.0,
        omega_Y=omega_Y,
        theta_floor=0.05,
        k_XY=0.02,
        k_XX=0.02,
        eta=0.5,
        kappa=0.2,
        C=2.0,          # high threshold: stay in the sub-explosive band
        Delta=1.0,
        lambda_=0.02,
        gamma_1=1.0,
        gamma_XX=0.5,
        gamma_2=0.5,
        D_X=0.05,
        D_Y=0.05,
        seed=seed,
    )
    m.X[:] = 12.0
    m.Y[:] = 12.0
    m.Beta[:] = 200.0

    F_avg, dS_avg = [], []
    S_prev = m.S.copy()
    for _ in range(steps):
        m.step()
        F_avg.append(m.mean_ripple())
        dS_avg.append(float(np.mean(m.S - S_prev)))
        S_prev = m.S.copy()
    return np.array(dS_avg), np.array(F_avg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output", "-o", type=str, default="../figures/beat_demo.png")
    args = ap.parse_args()

    d_omega = 0.4  # beat period 2*pi/0.4 ~ 16 steps
    dS_eq, F_eq = run_case(omega_Y=1.0, steps=args.steps, seed=args.seed)
    dS_mm, F_mm = run_case(omega_Y=1.0 + d_omega, steps=args.steps, seed=args.seed)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    steps = np.arange(1, args.steps + 1)

    axes[0].plot(steps, dS_eq, color="C0", label=r"$\omega_X = \omega_Y$ (no beat)")
    axes[0].plot(steps, dS_mm, color="C3",
                 label=r"$\omega_Y - \omega_X = 0.4$ (beat $\approx 16$ steps)")
    axes[0].set_ylabel(r"Mean structural increment $\overline{\Delta S}$")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, F_eq, color="C0")
    axes[1].plot(steps, F_mm, color="C3")
    axes[1].set_ylabel(r"Mean ripple $\overline{F} = |\Delta^2 S|$")
    axes[1].set_xlabel("Step")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Phase-interference beat in the structural increment and ripple")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")
    print(f"  equal frequencies : max F = {F_eq.max():.3f}")
    print(f"  mismatch 0.4      : max F = {F_mm.max():.3f}")


if __name__ == "__main__":
    main()
