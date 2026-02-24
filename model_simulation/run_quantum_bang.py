"""
UNPROVEN SIMULATION — X and Y as quantum excitations whose fluctuation
triggers the Big Bang and whose dynamics produce the observable universe.
Uses the PAPER PRINCIPLES exactly (no ad-hoc parameter overrides).

================================================================================
PAPER PRINCIPLES APPLIED (Discrete Cascade Thesis + mathematical_model.txt)
================================================================================

1. TWO SPECIES, ASYMMETRIC RULES (Paper I, §2.3)
   - X and Y: two excitation species. Cross-interaction X–Y and self-interaction
     X–X are allowed. Y–Y is FORBIDDEN (N_YY = 0). Implemented in CascadeModel:
     only N_XY and N_XX are sampled; no Y–Y channel.

2. INTRINSIC FREQUENCIES ω_X ≠ ω_Y (Paper I; Paper II Beat Frequency Theorem)
   - Rates: R_XY = α_XY ω_X ω_Y X Y, R_XX = α_XX ω_X^2 X(X−1)/2. Defaults
     ω_X=1.0, ω_Y=1.2 so species are non-interchangeable.
   - Paper II: (i) Symmetric Collapse — if α_XX=α_YY and ω_X=ω_Y, hierarchy
     collapses to single-species. (ii) Differential Depletion — forbidden Y–Y
     gives δμ > 0, X depletes faster than Y, Y/X grows; necessary for bonds.
     (iii) Beat Frequency — ω_X ≠ ω_Y necessary for sustained ripple F and
     hence for explosive regime/cascades. Both asymmetries necessary.

3. UPDATE ORDER (Paper I, §2.8 Complete Update Rule)
   Exactly: (1) Interaction → (2) Ripple → (3) Regime (leakage/explosion) →
   (4) Bond formation (Landau) → (5) Energy update → (6) Structural update →
   (7) Diffusion. Same order in CascadeModel.step().

4. RIPPLE F = |Δ²S| (Paper I, §2.5; mathematical_model §12.4)
   F(p,n) = |S(p,n) − 2S(p,n−1) + S(p,n−2)|. Drives regime and bonds.

5. THREE REGIMES (Paper I, Def. regimes; mathematical_model §V)
   - Quiescent: F ≤ C → L=0, M=0.
   - Leakage: C < F < C+Δ → L = λF, M=0.
   - Explosive: F ≥ C+Δ → m = floor((F−C)/Δ), M = ηm; create m X,Y pairs;
     if Beta < M then m = floor(Beta/η). Leakage L = λF still applies.

6. ENERGY (BETA) IRREVERSIBLE (Paper I, Thm. Energy Monotonicity)
   Beta(p,n+1) = Beta(p,n) − k_XY N_XY − k_XX N_XX − L − M − κB, clamped ≥ 0.
   Total Beta strictly decreases when active. Beta does NOT diffuse.

7. STRUCTURAL EVOLUTION (Paper I, eq. S update; mathematical_model §12.5)
   S(p,n+1) = S(p,n) + γ_1 N_XY + γ_XX N_XX + γ_2 B. No decay.

8. BOND FORMATION (Landau, Paper I §2.7)
   ψ = √(XY), T_eff = F/C. Bond only when T_eff < T_c − (2b/a_0)ψ² and
   dF/dψ < 0; B = min(ψ²|dF/dψ|/b, floor(Beta/κ), X, Y). X,Y and Beta reduced by B.

9. DIFFUSION (Paper I, Def. diffusion)
   Only X and Y diffuse (nearest-neighbour Laplacian); Beta does not.

10. ABSORBING STATE (Paper I, Def. absorbing)
    X·Y = 0 everywhere, X ≤ 1 (if α_XX>0), F ≤ C. Process reaches it a.s.

11. ASYMMETRY INDEX & CRITICAL THRESHOLD (Paper II, §6)
    A = A_c × A_f (channel asymmetry × frequency asymmetry). A = 0 iff
    symmetric. Paper II: if A < A_crit no explosions/cascades; if A ≥ A_crit
    full hierarchy (excitation → pair → bond → clump) can emerge. Default
    parameters (α_YY=0, ω_X≠ω_Y) give A > 0 so cascades are possible.

This script uses CascadeModel with DEFAULT parameters (Paper I + II). The
only choice is the INITIAL CONDITION: a single small “quantum fluctuation”
(one region with X,Y,E; rest empty). Narrative (unproven): that fluctuation
triggers the cascade (Big Bang); late-time state = observable universe.
"""

import argparse
from typing import Optional, List, Tuple
import numpy as np
from cascade_model import CascadeModel


def run_quantum_bang(
    P: int = 200,
    max_steps: int = 12000,
    seed_site: int = 0,
    seed_radius: int = 1,
    X_seed: float = 3.0,
    Y_seed: float = 3.0,
    E_seed: float = 60.0,
    seed: Optional[int] = None,
    **model_kw,
) -> Tuple[CascadeModel, dict, List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Pre-bang: only sites in [seed_site - seed_radius, seed_site + seed_radius]
    get X_seed, Y_seed, and E_seed (total energy over that patch). Rest is 0.
    Dynamics use the full paper update (interaction → ripple → regime → bonds →
    energy → structure → diffusion); no parameter overrides. Returns model,
    history, and snapshots at narrative eras.
    """
    m = CascadeModel(P=P, seed=seed, **model_kw)
    lo = max(0, seed_site - seed_radius)
    hi = min(P, seed_site + seed_radius + 1)
    n_seed = hi - lo
    m.X[:] = 0.0
    m.Y[:] = 0.0
    m.Beta[:] = 0.0
    m.X[lo:hi] = X_seed
    m.Y[lo:hi] = Y_seed
    m.Beta[lo:hi] = E_seed / n_seed
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
    all_snapshots: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    peak_XY_step = 0
    peak_XY = 0.0

    for _ in range(max_steps):
        n = m.n
        tx = float(np.sum(m.X))
        ty = float(np.sum(m.Y))
        history["n"].append(n)
        history["E_total"].append(m.total_energy())
        history["X_total"].append(tx)
        history["Y_total"].append(ty)
        history["S_total"].append(m.total_structure())
        history["F_avg"].append(m.mean_ripple())
        if tx + ty > peak_XY:
            peak_XY = tx + ty
            peak_XY_step = n
        all_snapshots.append((n, m.X.copy(), m.Y.copy(), m.Beta.copy()))
        active = m.step()
        if not active or m.is_absorbing():
            break

    # Record final state so the plot has at least two points (initial + final)
    if history["n"] and m.n != history["n"][-1]:
        history["n"].append(m.n)
        history["E_total"].append(m.total_energy())
        history["X_total"].append(float(np.sum(m.X)))
        history["Y_total"].append(float(np.sum(m.Y)))
        history["S_total"].append(m.total_structure())
        history["F_avg"].append(m.mean_ripple())
        all_snapshots.append((m.n, m.X.copy(), m.Y.copy(), m.Beta.copy()))

    steps = history["n"]
    # Narrative snapshots: t=0 (fluctuation), t=peak (Bang), 1/3, 2/3, end
    if len(steps) <= 1:
        idxs = [0]
    else:
        n_end = steps[-1]
        n_bang = min(peak_XY_step, n_end)
        idxs = [0]
        for t in [n_bang, n_end // 3, 2 * n_end // 3, n_end]:
            if t != 0 and t != n_end:
                # find closest step index
                i = min(range(len(steps)), key=lambda i: abs(steps[i] - t))
                if steps[i] not in [steps[j] for j in idxs]:
                    idxs.append(i)
        idxs.append(len(steps) - 1)
        idxs = sorted(set(idxs))
    snapshots = [all_snapshots[i] for i in idxs]

    return m, history, snapshots


def main():
    ap = argparse.ArgumentParser(description="Unproven: quantum X,Y fluctuation as Big Bang trigger")
    ap.add_argument("--sites", type=int, default=200, help="Universe size (chain length)")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--seed-site", type=int, default=0, help="Center of initial fluctuation")
    ap.add_argument("--X", type=float, default=3.0, help="X in fluctuation")
    ap.add_argument("--Y", type=float, default=3.0, help="Y in fluctuation")
    ap.add_argument("--E", type=float, default=60.0, help="Total energy in fluctuation")
    ap.add_argument("--rng", type=int, default=42)
    ap.add_argument("--output", "-o", type=str, default="", help="Base path for figures, e.g. ../figures/quantum_bang")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    # Paper principles: use model defaults (no overrides). Only initial condition is the "quantum fluctuation."
    model, history, snapshots = run_quantum_bang(
        P=args.sites,
        max_steps=args.steps,
        seed_site=args.seed_site,
        seed_radius=1,
        X_seed=args.X,
        Y_seed=args.Y,
        E_seed=args.E,
        seed=args.rng,
    )

    n_final = model.n
    print("Unproven run: Quantum fluctuation -> Big Bang -> observable universe")
    print(f"  Initial: one small region with X={args.X}, Y={args.Y}, E={args.E}; rest empty")
    print(f"  Steps: {n_final}")
    print(f"  Final total energy: {model.total_energy():.2f}")
    print(f"  Final X_total: {np.sum(model.X):.0f}, Y_total: {np.sum(model.Y):.0f}")
    print(f"  Final total structure S: {model.total_structure():.2f}")
    print(f"  Absorbing: {model.is_absorbing()}")

    if args.no_plot:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return

    base = args.output.rstrip("/").rstrip("\\") if args.output else "quantum_bang"
    if base and not base.endswith("quantum_bang"):
        base = base + "_quantum_bang"
    steps = history["n"]

    # Figure 1: global evolution with narrative labels
    fig1, axes = plt.subplots(4, 1, sharex=True, figsize=(9, 8))
    fig1.suptitle(
        "Unproven narrative: X,Y as quantum particles — fluctuation triggers Big Bang, then observable universe",
        fontsize=10,
    )
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
    # Label eras (approximate)
    if len(steps) > 2:
        ax = axes[0]
        ax.annotate("quantum\nfluctuation", xy=(0, history["E_total"][0]), fontsize=7, color="gray")
        mid = len(steps) // 2
        ax.annotate("structure era", xy=(steps[mid], history["E_total"][min(mid, len(history["E_total"])-1)]), fontsize=7, color="gray")
        ax.annotate("current universe", xy=(steps[-1], history["E_total"][-1]), fontsize=7, color="gray")
    plt.tight_layout()
    path1 = base + "_timeseries.png"
    plt.savefig(path1, dpi=150)
    print(f"Saved {path1}")
    if args.output:
        plt.close()
    else:
        plt.show()

    # Figure 2: spatial spread at narrative times (X, Y, E vs site)
    n_snap = len(snapshots)
    fig2, axes = plt.subplots(n_snap, 1, sharex=True, figsize=(10, 2 * n_snap))
    if n_snap == 1:
        axes = [axes]
    labels = ["t=0 (fluctuation)", "Bang / burst", "structure", "late", "current"][:n_snap]
    P = model.P
    for i, (n, X, Y, Beta) in enumerate(snapshots):
        ax = axes[i]
        lab = labels[i] if i < len(labels) else f"t={n}"
        scale_E = max(np.max(X), np.max(Y), 1.0) / (np.max(Beta) + 1e-9) * 0.5
        ax.plot(range(P), X, label="X", color="C1", alpha=0.9)
        ax.plot(range(P), Y, label="Y", color="C2", alpha=0.9)
        ax.plot(range(P), Beta * scale_E, label="E (scaled)", color="C0", alpha=0.7)
        ax.set_ylabel(lab)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, None)
    axes[-1].set_xlabel("Site (space)")
    fig2.suptitle("Spatial evolution: fluctuation -> expansion -> current", fontsize=10)
    plt.tight_layout()
    path2 = base + "_snapshots.png"
    plt.savefig(path2, dpi=150)
    print(f"Saved {path2}")
    if args.output:
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
