# Minimal-seed simulation: fluctuation-triggered cascade

**This is an exploratory regime, not a proven result.** The simulation applies the full Paper I model (same equations, same update order, same asymmetry); the only special choice is the initial condition: a single small region is seeded with excitations and energy while the rest of the graph starts empty.

---

## Paper principles applied

The run uses `CascadeModel` with thesis/default parameters. No ad-hoc overrides. Every principle from the discrete cascade model is implemented as follows.

| # | Principle (paper) | In this simulation |
|---|-------------------|---------------------|
| 1 | **Two species, asymmetric rules** (Paper I S2.3): X-Y and X-X allowed; **Y-Y forbidden** (N_YY = 0). | `CascadeModel` samples only N_XY and N_XX; no Y-Y channel. |
| 2 | **Intrinsic frequencies omega_X != omega_Y** (Paper I; Paper II): different rate contributions. Paper II proves both asymmetries are necessary for hierarchy. | Defaults omega_X=1.0, omega_Y=1.2. Both asymmetries present so hierarchy and cascades are possible. |
| 3 | **Update order** (Paper I S2.8): (1) Interaction (2) Ripple (3) Regime (4) Bonds (5) Energy (6) Structure (7) Diffusion. | `CascadeModel.step()` follows this order exactly. |
| 4 | **Ripple** F(p,n) = |Delta^2 S| (Paper I S2.5). | Computed after S history; drives regime and Landau bonds. |
| 5 | **Three regimes** (Paper I): Quiescent (F<=C), Leakage (C<F<C+Delta), Explosive (F>=C+Delta). | Same thresholds C, Delta; leakage and explosion as in the paper. |
| 6 | **Energy irreversible** (Paper I Thm. Energy Monotonicity): Beta strictly decreases when active; does not diffuse. | Per-site update and clamp; no diffusion term for Beta. |
| 7 | **Structural evolution** (Paper I): S(p,n+1) = S(p,n) + gamma_1 N_XY + gamma_XX N_XX + gamma_2 B. | Same increments; no decay. |
| 8 | **Bond formation** (Landau, Paper I S2.7): psi=sqrt(XY), T_eff=F/C; bond when T_eff < T_c - (2b/a_0)psi^2. | Landau condition and B formula as in thesis. |
| 9 | **Diffusion**: Only X and Y diffuse; Beta does not. | Diffusion step acts only on X, Y. |
| 10 | **Absorbing state** (Paper I): X*Y=0, X<=1 (if alpha_XX>0), F<=C. Reached a.s. | Run stops when no activity; final state is absorbing. |
| 11 | **Asymmetry index** (Paper II S6): A = A_c * A_f. If A >= A_crit, full hierarchy can emerge. | Defaults give A > 0, so cascades are possible. |

---

## What the simulation shows

- **Initial condition:** Sites in a small seed region get X, Y, and Beta. All other sites start empty.
- **Observation:** Even from this minimal seed, the system can trigger the full cascade dynamics: the localised perturbation grows, drives structural accumulation, and eventually decays to absorption.
- **Significance:** This illustrates that the model admits a minimal-seed regime where a small initial perturbation suffices to activate the complete dynamical sequence. The dynamics are proven (Paper I); the choice of initial condition is exploratory.

---

## How to run

```bash
python run_minimal_seed.py --sites 200 --steps 10000 --E 100 -o ../figures/minimal_seed
```

Optional: `--X`, `--Y`, `--E`, `--seed-site`, `--rng`. Do not change interaction/diffusion/regime parameters if you want to keep the paper principles exactly; the defaults are the thesis values.
