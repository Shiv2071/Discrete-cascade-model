# Unproven simulation: X,Y as quantum particles triggering the Big Bang

**This is not proved; it is a narrative interpretation.** The simulation applies the **paper’s principles in full** (same equations, same update order, same asymmetry); the only special choice is the **initial condition** (a single small “quantum fluctuation”).

---

## Paper principles applied in detail

The run uses `CascadeModel` with **thesis/default parameters**. No ad-hoc overrides. Every principle from the discrete cascade thesis and `mathematical_model.txt` is implemented as follows.

| # | Principle (paper) | In this simulation |
|---|-------------------|---------------------|
| 1 | **Two species, asymmetric rules** (Paper I §2.3): X–Y and X–X allowed; **Y–Y forbidden** (N_YY = 0). | `CascadeModel` samples only N_XY and N_XX; no Y–Y channel. |
| 2 | **Intrinsic frequencies ω_X ≠ ω_Y** (Paper I; Paper II): R_XY = α_XY ω_X ω_Y X Y; R_XX = α_XX ω_X² X(X−1)/2. **Paper II** proves: (i) **Symmetric Collapse** — if α_XX=α_YY and ω_X=ω_Y, hierarchy collapses. (ii) **Differential Depletion** — forbidden Y–Y ⇒ X depletes faster, Y/X grows; necessary for bonds. (iii) **Beat Frequency Theorem** — ω_X ≠ ω_Y necessary for sustained ripple F and hence explosions/cascades. | Defaults ω_X=1.0, ω_Y=1.2. Same rates; both asymmetries present so hierarchy and cascades are possible. |
| 3 | **Update order** (Paper I §2.8): (1) Interaction (2) Ripple (3) Regime (4) Bonds (5) Energy (6) Structure (7) Diffusion. | `CascadeModel.step()` follows this order exactly. |
| 4 | **Ripple** F(p,n) = |S(p,n) − 2S(p,n−1) + S(p,n−2)| = |Δ²S| (Paper I §2.5; mathematical_model §12.4). | Computed after S history; drives regime and Landau bonds. |
| 5 | **Three regimes** (Paper I Def. regimes; mathematical_model §V): Quiescent (F≤C), Leakage (C<F<C+Δ, L=λF), Explosive (F≥C+Δ, m=⌊(F−C)/Δ⌋, M=ηm). | Same thresholds C, Δ; leakage and explosion as in the paper. |
| 6 | **Energy (Beta) irreversible** (Paper I Thm. Energy Monotonicity): Beta decreases by k_XY N_XY + k_XX N_XX + L + M + κB; total strictly decreases when active; **Beta does not diffuse**. | Per-site update and clamp; no diffusion term for Beta. |
| 7 | **Structural evolution** (Paper I eq. S update): S(p,n+1) = S(p,n) + γ₁ N_XY + γ_XX N_XX + γ₂ B. | Same increments; no decay. |
| 8 | **Bond formation** (Landau, Paper I §2.7): ψ=√(XY), T_eff=F/C; bond when T_eff < T_c − (2b/a₀)ψ² and dF/dψ<0; B capped by energy and min(X,Y). | Landau condition and B formula as in thesis. |
| 9 | **Diffusion**: Only **X and Y** diffuse (nearest-neighbour); Beta does not. | Diffusion step acts only on X, Y. |
| 10 | **Absorbing state** (Paper I Def.): X·Y=0, X≤1 (if α_XX>0), F≤C. Process reaches it a.s. | Run stops when no activity; final state is absorbing. |
| 11 | **Asymmetry index & critical threshold** (Paper II §6): 𝒜 = 𝒜_c × 𝒜_f. If 𝒜 < 𝒜_crit, no explosions/cascades; if 𝒜 ≥ 𝒜_crit, full hierarchy (excitation→pair→bond→clump) can emerge. | Defaults give 𝒜 > 0 (α_YY=0, ω_X≠ω_Y), so the fluctuation can trigger cascades when F ≥ C+Δ. |

So: **what the simulation does** is the full Paper I model, with both asymmetries required by **Paper II** (asymmetry_necessity_paper2.tex) so that hierarchy and cascades are possible. The only “radical” part is the **initial condition**: a single small region (the “quantum fluctuation”) has X, Y, and Beta; the rest of the graph is empty. That choice is **unproven**; the dynamics are not.

---

## Paper II (asymmetry_necessity_paper2.tex) — what this simulation relies on

Paper II proves that **both** asymmetries are **necessary** for the structural hierarchy and for non-trivial dynamics. The quantum-bang run uses the Paper I model with **both** asymmetries present:

- **Symmetric Collapse:** If α_XX = α_YY and ω_X = ω_Y, the two-species system reduces to a single-species process; the Landau order parameter degenerates; the four-level hierarchy (excitation → pair → bond → clump) collapses.
- **Differential Depletion (forbidden Y–Y necessary):** With α_YY = 0, X depletes faster than Y; the ratio Y/X grows; bonds are X-limited. This concentration allows the hierarchy. If Y–Y were allowed, the mechanism would break.
- **Beat Frequency Theorem (ω_X ≠ ω_Y necessary):** When ω_X = ω_Y, ΔS is monotone so ripple F → 0 and the system stays quiescent—no explosions, no cascades. When ω_X ≠ ω_Y, cross- and self-interaction contribute at different rates; the evolving species ratio produces **modulation** (beat), hence non-zero F and the possibility of F ≥ C+Δ. So the frequency mismatch is what allows the fluctuation to **trigger** the Bang.
- **Asymmetry index 𝒜 and critical threshold:** Below 𝒜_crit no explosions; above it, full hierarchy can emerge. Defaults give 𝒜 > 0 so cascades are possible.

The simulation is built on **Paper I dynamics** and **Paper II necessity**; without both asymmetries, the “Bang” and the hierarchy would not exist in the model.

---

## Narrative (unproven)

- **X and Y** = two species of excitations in the model (paper: asymmetric, ω_X ≠ ω_Y, Y–Y forbidden). Interpreted here as the “quantum” degrees of freedom that can trigger the Bang.
- **Pre-bang:** Almost all sites empty; one small **fluctuation** (a few sites with X, Y, and Beta).
- **What causes the Big Bang:** The fluctuation is unstable under the **same** cascade rules (interactions, ripple, regimes, bonds, energy, structure, diffusion). When F crosses threshold, explosion creates more X,Y; that burst is the narrative “Big Bang.” So **what caused it** = the same X,Y dynamics; the fluctuation is the trigger.
- **Why the Big Bang:** The update rule plus this initial condition **necessarily** produce a burst (no external cause in the model).
- **Observable universe:** Late-time state = structure S frozen, Beta depleted, asymmetric X,Y (or zero), F≤C everywhere. Interpreted as the “current” universe.

---

## What the simulation does (concrete)

- **Initial condition:** Sites in `[seed_site − seed_radius, seed_site + seed_radius]` get X_seed, Y_seed, and E_seed (total energy shared over that patch). All other sites: 0 X, 0 Y, 0 Beta, 0 S.
- **Run:** One step = full paper update (1→2→3→4→5→6→7). Repeated until no activity or max_steps.
- **Output:** Time series (total Beta, total X, total Y, total S, mean F) and spatial snapshots at narrative times (fluctuation, burst, structure, current).

---

## How to run

```bash
python run_quantum_bang.py --sites 200 --steps 10000 --E 100 -o ../figures/quantum_bang
```

Optional: `--X`, `--Y`, `--E`, `--seed-site`, `--rng`. **Do not** change interaction/diffusion/regime parameters in the script if you want to keep the paper principles exactly; the defaults are the thesis values.

---

## Disclaimer

We do **not** claim this is real cosmology or QFT. We run the **exact** discrete cascade model from the paper with one unproven initial condition and **interpret** it as: quantum X,Y fluctuation → Big Bang → observable universe. The math is the paper; the story is optional.
